import glob
import scipy.io
from typing import Dict, Tuple
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torch.nn import Module
import numpy as np
import random
import pickle
import os
import torch
import torch.nn as nn
from torchvision.transforms import Normalize, Compose, Resize, ToTensor
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import sys
from pathlib import Path
_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
from paths import dataset_root as raw_dataset, legacy_label_path

device = "cuda" if torch.cuda.is_available() else "cpu"



def convert_to_rgb(image):
    return image.convert("RGB")


def get_transform(image_size=384):
    return Compose([
        convert_to_rgb,
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def resolution_transform(image_size=384):
    return Compose([
        convert_to_rgb,
        Resize((image_size, image_size))
    ])


def load_taglist(
        dataset: str
) -> Tuple[Dict]:
    dataset_root = str(raw_dataset(dataset))

    tag_file = dataset_root + f"/{dataset}_ram_taglist.txt"

    with open(tag_file, "r", encoding="utf-8") as f:
        taglist_or = [line.strip() for line in f]
    taglist = taglist_or

    info = {"taglist": taglist}
    return info

def init_model():
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", 
        torch_dtype="auto", 
        device_map="auto"
    )
    
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        use_fast=True
    )
    
    return model, processor

def generate_label(model, processor, pil_image, taglist_label):
    
    options = "\n".join([f"{idx}: {name}" for idx, name in enumerate(taglist_label)])
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": f"""[步骤1]分析图片主体特征
                                            [步骤2]对比选项描述相似度
                                            [步骤3]直接输出最匹配一个的数字编号，无需解释,无需输出类别名称，数字编号必须在0-199之间
                                            可用标签：{options}
                                            最终答案：数字"""
                },
            ],
        }
    ]
    
    # 准备输入
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    
    # 生成描述
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )
    
    return output_text


def load_cifar100():  # 输入你“cifar-100-python”的路径
    with open(raw_dataset("CIFAR100") / "train", 'rb') as f:  #'rb'表示以二进制模式读取文件。
        data_train = pickle.load(f, encoding='latin1')  # 训练集，不同分类的数据，不同类别序号，
    with open(raw_dataset("CIFAR100") / "test", 'rb') as f:
        data_test = pickle.load(f, encoding='latin1')  # 测试集，不同分类的数据，不同类别序号，
    with open(raw_dataset("CIFAR100") / "meta", 'rb') as f:
        data_meta = pickle.load(f, encoding='latin1')  # 100分类与20分类的标签
    return data_train, data_test, data_meta


def read_data_cifar_100():
    # random.seed(1)
    data_train, data_test, data_meta = load_cifar100()
    train_data = data_train['data'].reshape((data_train['data'].shape[0], 3, 32, 32))  # .transpose((0,1,3,2))
    test_data = data_test['data'].reshape((data_test['data'].shape[0], 3, 32, 32))  # .transpose((0,1,3,2))
    train_label = data_train["fine_labels"]
    test_label = data_test["fine_labels"]
    return train_data, train_label, test_data, test_label


def read_data_tiny_imagenet_200():
    id_dict = {}
    for i, line in enumerate(open(raw_dataset("tiny-imagenet-200") / 'wnids.txt', 'r')):  #读取类别映射
        id_dict[line.replace('\n', '')] = i
    num_classes = len(id_dict)
    cls_dic = {}
    for i, line in enumerate(open(raw_dataset("tiny-imagenet-200") / 'val' / 'val_annotations.txt', 'r')):    #将每个验证集图像cls_id映射到其对应的索引
        a = line.split('\t')
        img, cls_id = a[0], a[1]
        cls_dic[img] = id_dict[cls_id]

    tiny_root = raw_dataset("tiny-imagenet-200")
    train_imgs = glob.glob(str(tiny_root / "train" / "*" / "*" / "*.JPEG"))
    test_imgs = glob.glob(str(tiny_root / "val" / "images" / "*.JPEG"))
    train_imgs = [img_path.replace('\\', '/') for img_path in train_imgs]
    test_imgs = [img_path.replace('\\', '/') for img_path in test_imgs]

    train_labels = [id_dict[train_img.split('/')[4]] for train_img in train_imgs]   #对于每张训练集图像，使用图像路径中的第5个元素来提取类别ID，即训练集图像路径中的类别文件夹名称
    test_labels = [cls_dic[os.path.basename(test_img)] for test_img in test_imgs]   #根据文件名查找类别索引

    return train_imgs, train_labels, test_imgs, test_labels, num_classes


def read_data_eruosat():
    id_dict = {}
    for i, line in enumerate(open(raw_dataset("EuroSAT") / 'EuroSAT_ram_taglist.txt', 'r')):
        id_dict[line.replace('\n', '')] = i
    num_classes = len(id_dict)
    EuroSAT_imgs = glob.glob(str(raw_dataset("EuroSAT") / "2750" / "*" / "*.jpg"))
    EuroSAT_imgs = [img_path.replace('\\', '/') for img_path in EuroSAT_imgs]
    EuroSAT_labels = [id_dict[img_path.split('/')[4]] for img_path in EuroSAT_imgs]
    EuroSAT_dataset = list(zip(EuroSAT_imgs, EuroSAT_labels))   #将图像路径和标签打包成一个元组列表，使每个图像路径与对应的标签配对。
    random.seed(0)
    random.shuffle(EuroSAT_dataset) #打乱数据集
    # 划分训练集和测试集，70%训练集，30%测试集
    train_size = int(0.7 * len(EuroSAT_dataset))
    train_set, test_set = EuroSAT_dataset[:train_size], EuroSAT_dataset[train_size:]
    # 分离训练集的图片地址和标签
    train_imgs, train_labels = zip(*train_set)  #将train_set中的图像路径和标签分别解包到train_imgs和train_labels中
    # 分离测试集的图片地址和标签
    test_imgs, test_labels = zip(*test_set)

    return list(train_imgs), list(train_labels), list(test_imgs), list(test_labels), num_classes


def read_data_stanford_cars():
    id_dict = {}
    for i, line in enumerate(open(raw_dataset("stanford_cars") / 'stanford_cars_ram_taglist.txt', 'r')):
        id_dict[line.replace('\n', '')] = i
    num_classes = len(id_dict)
    data = scipy.io.loadmat(raw_dataset("stanford_cars") / 'cars_annos.mat')
    annotations = data['annotations']
    train_imgs = []
    train_labels = []
    test_imgs = []
    test_labels = []
    for i in range(annotations.shape[1]):
        name = str(annotations[0, i][0])[2:-2]
        img_path = str(raw_dataset("stanford_cars") / name).replace('\\', '/')
        clas = int(annotations[0, i][5])
        test = int(annotations[0, i][6])
        if test == 0:
            train_imgs.append(img_path)
            train_labels.append(clas - 1)   #clas从1开始，减一确保从0开始
        elif test == 1:
            test_imgs.append(img_path)
            test_labels.append(clas - 1)
    return train_imgs, train_labels, test_imgs, test_labels, num_classes


def read_data_caltech_101():
    id_dict = {}
    for i, line in enumerate(open(raw_dataset("caltech-101") / 'caltech-101_ram_taglist.txt', 'r')):
        id_dict[line.replace('\n', '')] = i
    num_classes = len(id_dict)
    caltech_101_imgs = glob.glob(str(raw_dataset("caltech-101") / "101_ObjectCategories" / "*" / "*.jpg"))
    caltech_101_imgs = [img_path.replace('\\', '/') for img_path in caltech_101_imgs]
    caltech_101_labels = [id_dict[img_path.split('/')[4]] for img_path in caltech_101_imgs]
    caltech_101_dataset = list(zip(caltech_101_imgs, caltech_101_labels))
    random.seed(0)
    random.shuffle(caltech_101_dataset)
    # 划分训练集和测试集，70%训练集，30%测试集
    train_size = int(0.7 * len(caltech_101_dataset))
    train_set, test_set = caltech_101_dataset[:train_size], caltech_101_dataset[train_size:]
    # 分离训练集的图片地址和标签
    train_imgs, train_labels = zip(*train_set)
    # 分离测试集的图片地址和标签
    test_imgs, test_labels = zip(*test_set)
    return list(train_imgs), list(train_labels), list(test_imgs), list(test_labels), num_classes


def read_data_food_101():
    id_dict = {}
    for i, line in enumerate(open(raw_dataset("food-101") / 'food-101_ram_taglist.txt', 'r')):
        id_dict[line.replace('\n', '')] = i
    num_classes = len(id_dict)
    train_imgs = []
    train_labels = []
    test_imgs = []
    test_labels = []
    with open(raw_dataset("food-101") / 'meta' / 'train.txt', 'r') as f:
        for line in f:
            image = line.replace('\n', '') + '.jpg'
            label = line.split('/')[0]  #获取路径中的类别名称（即文件夹名）
            train_imgs.append(str(raw_dataset("food-101") / "images" / image).replace('\\', '/'))
            train_labels.append(id_dict[label])
    with open(raw_dataset("food-101") / 'meta' / 'test.txt', 'r') as f:
        for line in f:
            image = line.replace('\n', '') + '.jpg'
            label = line.split('/')[0]
            test_imgs.append(str(raw_dataset("food-101") / "images" / image).replace('\\', '/'))
            test_labels.append(id_dict[label])
    return train_imgs, train_labels, test_imgs, test_labels, num_classes


def Generate_cifar_100(input_size, sampler, random_num):
    # random.seed(1)
    transform = get_transform(input_size)
    data_train, data_test, data_meta = load_cifar100()
    test_data = data_test['data'].reshape((data_test['data'].shape[0], 3, 32, 32))  # .transpose((0,1,3,2))通过 reshape 操作，将原始数据从 (num_samples, 3072) 转换为 (num_samples, 3, 32, 32)，即每张图像被重新格式化为 3 个通道（RGB）的 32x32 的图像。

    gen_data = torch.ones(random_num, 3, 224, 224)  #初始化一个大小为 (random_num, 3, 224, 224) 的张量 gen_data，用来存储生成的数据。
    # print([i for i in sampler])
    for i in range(random_num):
        img = Image.fromarray(np.uint8(test_data[sampler[i]]).transpose((1, 2, 0))) #.transpose((1, 2, 0)) 将图像数据的维度从 (3, 32, 32) 转换为 (32, 32, 3)，因为 Image.fromarray 要求输入是 (height, width, channels) 的顺序。
        # print("transform(img) = ", transform(img).size())
        # print("gen_data[i] = ", gen_data[i].size())
        gen_data[i] = transform(img)       #将数组转换为 PIL 图像对象。
    # print("gen_data ", gen_data.size())
    return gen_data


def Generate_tiny_imagenet_200(input_size, sampler, random_num):
    # random.seed(1)
    transform = get_transform(input_size)
    train_imgs, train_labels, test_imgs, test_labels, _ = read_data_tiny_imagenet_200()

    gen_data = torch.ones(random_num, 3, 224, 224)
    for i in range(random_num):
        img = Image.open(test_imgs[sampler[i]])
        gen_data[i] = transform(img)
    return gen_data


def Generate_eurosat(input_size, sampler, random_num):
    # random.seed(1)
    transform = get_transform(input_size)
    train_imgs, train_labels, test_imgs, test_labels, _ = read_data_eruosat()

    gen_data = torch.ones(random_num, 3, 224, 224)
    for i in range(random_num):
        img = Image.open(test_imgs[sampler[i]])
        gen_data[i] = transform(img)
    return gen_data


def Generate_stanford_cars(input_size, sampler, random_num):
    transform = get_transform(input_size)
    train_imgs, train_labels, test_imgs, test_labels, _ = read_data_stanford_cars()

    gen_data = torch.ones(random_num, 3, 224, 224)
    for i in range(random_num):
        img = Image.open(test_imgs[sampler[i]])
        gen_data[i] = transform(img)
    return gen_data


def Generate_caltech_101(input_size, sampler, random_num):
    # random.seed(1)
    transform = get_transform(input_size)
    train_imgs, train_labels, test_imgs, test_labels, _ = read_data_caltech_101()

    gen_data = torch.ones(random_num, 3, 224, 224)
    for i in range(random_num):
        img = Image.open(test_imgs[sampler[i]])
        gen_data[i] = transform(img)
    return gen_data


def Generate_food_101(input_size, sampler, random_num):
    # random.seed(1)
    transform = get_transform(input_size)
    train_imgs, train_labels, test_imgs, test_labels, _ = read_data_food_101()

    gen_data = torch.ones(random_num, 3, 224, 224)
    for i in range(random_num):
        img = Image.open(test_imgs[sampler[i]])
        gen_data[i] = transform(img)
    return gen_data


def Generate_data(input_size, sampler, random_num, dataset):
    if dataset == 'CIFAR100':
        return Generate_cifar_100(input_size, sampler, random_num)
    elif dataset == 'tiny-imagenet-200':
        return Generate_tiny_imagenet_200(input_size, sampler, random_num)
    elif dataset == 'EuroSAT':
        return Generate_eurosat(input_size, sampler, random_num)
    elif dataset == 'stanford_cars':
        return Generate_stanford_cars(input_size, sampler, random_num)
    elif dataset == 'caltech-101':
        return Generate_caltech_101(input_size, sampler, random_num)
    elif dataset == 'food-101':
        return Generate_food_101(input_size, sampler, random_num)


class CIFAR100_handler_train_TF_saved(Dataset):        #clip生成标签与真实标签做对比并写入文件保存
    def __init__(self, X, Y, input_size, model, processor, transform=None):
        self.X = X
        self.Y = Y
        self.YT = torch.empty(len(self.Y))
        self.Y1 = torch.empty(len(self.Y))
        self.transform = get_transform(input_size)
        self.model = model
        self.processor = processor
        info = load_taglist(dataset="CIFAR100")     #加载 CIFAR-100 数据集的标签列表
        taglist_label = info["taglist"]     #提取 CIFAR-100 数据集的标签列表
        for i in range(len(self.X)):
            pil_image = Image.fromarray(np.uint8(self.X[i]).transpose((1, 2, 0)))       #将原始图像数据转换为 PIL 图像对象。
            output_text = generate_label(model, processor, pil_image, taglist_label)
            output_label = int(output_text[0])
            print(f"{i + 1}/{len(self.X)}")
            # print(f"真实标签{self.Y[i]}模型输出{output_text}预测标签{output_label}")
            if self.Y[i] == output_label:  
                self.YT[i] = 1  
                file = open(legacy_label_path('CIFAR100', 'Qwen_VL_7B_label', 'train_label_tf.txt'), 'a')
                file.write("1\n")
                file.close()
            else:
                self.YT[i] = 0  # 否则标记为 0
                file = open(legacy_label_path('CIFAR100', 'Qwen_VL_7B_label', 'train_label_tf.txt'), 'a')
                file.write("0\n")
                file.close()
                file = open(legacy_label_path('CIFAR100', 'Qwen_VL_7B_label', 'train_label_t.txt'), 'a')
            file.write(str(self.Y[i]) + '\n')
            file.close()
            file = open(legacy_label_path('CIFAR100', 'Qwen_VL_7B_label', 'train_label_pre.txt'), 'a')
            file.write(str(output_label) + '\n')  # 将 top2 索引保存到文件中
            file.close()
        print("标记完成")



    def __getitem__(self, index):
        x = Image.fromarray(np.uint8(self.X[index]).transpose((1, 2, 0)))
        x = self.transform(x)
        y = self.Y[index]
        yt = self.YT[index]
        return x, y, yt

    def __len__(self):
        return len(self.X)




class DatasetHandlerTrainClip(Dataset):
    def __init__(self, X, Y, input_size, dataset_name, model, processor, transform=None):
        self.X = X
        self.Y = Y
        self.YT = torch.empty(len(self.Y))
        self.Y1 = torch.empty(len(self.Y))
        self.transform = get_transform(input_size)
        self.resolution_transform = resolution_transform(input_size)
        self.dataset_name = dataset_name
        self.model = model
        self.processor = processor
        torch.manual_seed(1)
        np.random.seed(1)
        info = load_taglist(dataset=dataset_name)
        taglist_label = info["taglist"]
        for i in range(len(self.X)):
            x = Image.open(self.X[i]).convert("RGB")  # 从路径加载图片
            x = self.resolution_transform(x)
            output_text = generate_label(model, processor, x, taglist_label)
            output_label = int(output_text[0])
            print(f"{i + 1}/{len(self.X)}")
            # print(f"真实标签{self.YT[i]}预测文本{output_text}预测标签{output_label}")
            if self.Y[i] == output_label:  
                self.YT[i] = 1  
                file = open(legacy_label_path(dataset_name, 'Qwen_VL_7B_label', 'train_label_tf.txt'), 'a')
                file.write("1\n")
                file.close()
            else:
                self.YT[i] = 0  # 否则标记为 0
                file = open(legacy_label_path(dataset_name, 'Qwen_VL_7B_label', 'train_label_tf.txt'), 'a')
                file.write("0\n")
                file.close()
            file = open(legacy_label_path(dataset_name, 'Qwen_VL_7B_label', 'train_label_t.txt'), 'a')
            file.write(str(self.Y[i]) + '\n')
            file.close()
            file = open(legacy_label_path(dataset_name, 'Qwen_VL_7B_label', 'train_label_pre.txt'), 'a')
            file.write(str(output_label) + '\n')  # 将 top2 索引保存到文件中
            file.close()
        print("标记完成")

    def __getitem__(self, index):
        x = Image.fromarray(np.uint8(self.X[index]).transpose((1, 2, 0)))
        x = self.transform(x)
        y = self.Y[index]
        yt = self.YT[index]
        return x, y, yt

    def __len__(self):
        return len(self.X)



class CIFAR100_handler_test(Dataset):
    def __init__(self, X, Y, input_size, transform=None):
        self.X = X
        self.Y = Y
        self.transform = get_transform(input_size)

    def __getitem__(self, index):
        # print(self.X[index].shape)
        x = Image.fromarray(np.uint8(self.X[index]).transpose((1, 2, 0)))
        # x = Image.open(self.X[index]).convert('RGB')
        x = self.transform(x)
        y = self.Y[index]
        return x, y

    def __len__(self):
        return len(self.X)

class DatasetHandlerTest(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        # print(self.X[index].shape)
        # x = Image.fromarray(np.uint8(self.X[index]).transpose((1, 2, 0)))
        x = Image.open(self.X[index])
        x = self.transform(x)
        y = self.Y[index]
        return x, y

    def __len__(self):
        return len(self.X)


def data_gen_tf(train_set, train_label, test_set, test_label, input_size, batch_size, num_workers):
    train_dataset = CIFAR100_handler_train_gened_clip(train_set, train_label, input_size)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,     #表示每个 epoch 开始时对数据进行随机打乱
                              num_workers=num_workers)
    test_dataset = CIFAR100_handler_test(test_set, test_label, input_size)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=num_workers)
    return train_loader, test_loader


def get_data_handler(dataset, pattern, input_size, model, processor):
    if dataset == 'CIFAR100':
        train_data, train_label, test_data, test_label = read_data_cifar_100()
        if pattern == "train":
            datahandler = CIFAR100_handler_train_TF_saved(train_data, train_label, input_size,model,processor)
        elif pattern == "val":
            datahandler = CIFAR100_handler_test(test_data, test_label, input_size)
    else:
        if dataset == 'tiny-imagenet-200':
            train_data, train_label, test_data, test_label, num_classes = read_data_tiny_imagenet_200()
        elif dataset == 'EuroSAT':
            train_data, train_label, test_data, test_label, num_classes = read_data_eruosat()
        elif dataset == 'stanford_cars':
            train_data, train_label, test_data, test_label, num_classes = read_data_stanford_cars()
        elif dataset == 'caltech-101':
            train_data, train_label, test_data, test_label, num_classes = read_data_caltech_101()
        elif dataset == 'food-101':
            train_data, train_label, test_data, test_label, num_classes = read_data_food_101()

        if pattern == "train":
            datahandler = DatasetHandlerTrainClip(train_data, train_label, input_size, dataset, model, processor)
        elif pattern == "val":
            datahandler = DatasetHandlerTest(test_data, test_label, get_transform(input_size))

    return datahandler


def load_datasets(
        dataset: str,
        model_type: str,
        pattern: str,
        input_size: int,
        batch_size: int,
        num_workers: int,
        model,
        processor
) -> Tuple[DataLoader, Dict]:
    dataset_root = str(raw_dataset(dataset))

    if model_type == "clip":
        tag_file = dataset_root + f"/{dataset}_ram_taglist.txt"

    with open(tag_file, "r", encoding="utf-8") as f:
        taglist_or = [line.strip() for line in f]

    taglist = taglist_or  # + taglist_ot
    # taglist = taglist_ot

    datahandler = get_data_handler(dataset, pattern, input_size, model, processor)
    loader = DataLoader(dataset=datahandler, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # if pattern == "train":
    #     loader, _ = data_gen_tf(train_data, train_label, test_data, test_label, input_size, batch_size)
    # if pattern == "val":
    #     _, loader = data_gen_tf(train_data, train_label, test_data, test_label, input_size, batch_size)

    info = {
        "taglist": taglist
    }

    return loader, info


def divide_labeled_or_not(dataset, input_size):     #根据标签信息，将数据集分为两个子集0，1
    data_handler = get_data_handler(dataset, pattern='train', input_size=input_size)
    indices_yt_0 = torch.nonzero(torch.eq(torch.tensor(data_handler.YT), 0)).squeeze().tolist()
    indices_yt_1 = torch.nonzero(torch.eq(torch.tensor(data_handler.YT), 1)).squeeze().tolist()
    unlabeled_dataset = Subset(data_handler, indices_yt_0)
    labeled_dataset = Subset(data_handler, indices_yt_1)

    return labeled_dataset, unlabeled_dataset


if __name__ == '__main__':


    model, processor = init_model()

    print("==========CIFAR100==========")
    loader, info = load_datasets(
        dataset='CIFAR100',
        model_type='clip',
        pattern='train',
        input_size=224,
        batch_size=1,
        num_workers=0,
        model = model,
        processor = processor
    )

    print("==========caltech-101==========")
    loader, info = load_datasets(
        dataset='caltech-101',
        model_type='clip',
        pattern='train',
        input_size=224,
        batch_size=1,
        num_workers=0,
        model=model,
        processor=processor
    )

    print("==========stanford_cars==========")
    loader, info = load_datasets(
        dataset='stanford_cars',
        model_type='clip',
        pattern='train',
        input_size=224,
        batch_size=1,
        num_workers=0,
        model=model,
        processor=processor
    )

    print("==========food-101==========")
    loader, info = load_datasets(
        dataset='food-101',
        model_type='clip',
        pattern='train',
        input_size=224,
        batch_size=1,
        num_workers=0,
        model=model,
        processor=processor
    )

    print("==========tiny-imagenet-200==========")
    loader, info = load_datasets(
        dataset='tiny-imagenet-200',
        model_type='clip',
        pattern='train',
        input_size=224,
        batch_size=1,
        num_workers=0,
        model=model,
        processor=processor
    )

    print('_________________')
