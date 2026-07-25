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
from clip import clip
import torch.nn as nn
from torchvision.transforms import Normalize, Compose, Resize, ToTensor
import sys
from pathlib import Path
_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
from paths import dataset_root as raw_dataset, legacy_label_path

device = "cuda" if torch.cuda.is_available() else "cpu"

single_template = ["a photo of a {}."]


def convert_to_rgb(image):
    return image.convert("RGB")


def get_transform(image_size=384):
    return Compose([
        convert_to_rgb,
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def article(name):
    return "an" if name[0] in "aeiou" else "a"


def processed_name(name, rm_dot=False):
    # _ for lvis
    # / for obj365
    res = name.replace("_", " ").replace("/", " or ").lower()
    if rm_dot:
        res = res.rstrip(".")
    return res


def load_clip() -> Module:
    model, _ = clip.load("ViT-L/14")
    return model.to(device).eval()


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


def build_clip_label_embedding(model, categories):     #使用 CLIP 模型生成类别标签的嵌入，返回一个包含所有类别标签嵌入的张量。
    # print("Creating pretrained CLIP image model")
    templates = single_template
    run_on_gpu = torch.cuda.is_available()

    with torch.no_grad():
        openset_label_embedding = []
        for category in categories:
            # print("category =", category)
            texts = [
                template.format(
                    processed_name(category, rm_dot=True), article=article(category)
                )
                for template in templates   #对于每个category使用不同模板生成多个text
            ]
            texts = [
                "This is " + text if text.startswith("a") or text.startswith("the") else text
                for text in texts
            ]  # 改造句子
            texts = clip.tokenize(texts)  # tokenize，将文本列表转换为 CLIP 模型需要的 token 格式
            #print("texts =", texts)
            if run_on_gpu:
                texts = texts.cuda()
                model = model.cuda()
            text_embeddings = model.encode_text(texts)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)   #对嵌入进行归一化，确保每个嵌入的范数为 1
            text_embedding = text_embeddings.mean(dim=0)    #对多个生成的文本嵌入取均值
            text_embedding /= text_embedding.norm()     #再次归一化
            openset_label_embedding.append(text_embedding)
        openset_label_embedding = torch.stack(openset_label_embedding, dim=1)   #将所有类别的标签嵌入沿着新维度堆叠成一个张量。最终的形状是(num_categories,embedding_size)
        if run_on_gpu:
            openset_label_embedding = openset_label_embedding.cuda()

    openset_label_embedding = openset_label_embedding.t()   #将张量转置
    return openset_label_embedding


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




class CIFAR100_generate_label_clip(Dataset):        #clip生成标签与真实标签做对比并写入文件保存
    def __init__(self, X, Y, input_size, transform=None):
        self.X = X
        self.Y = Y
        self.YT = torch.empty(len(self.Y))
        self.Y1 = torch.empty(len(self.Y))
        self.transform = get_transform(input_size)
        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6) #定义了一个余弦相似度计算模块，它可以计算两个张量沿指定维度的余弦相似度。
        torch.manual_seed(1)
        np.random.seed(1)
        clip_model = load_clip()
        info = load_taglist(dataset="CIFAR100")     #加载 CIFAR-100 数据集的标签列表
        taglist_label = info["taglist"]     #提取 CIFAR-100 数据集的标签列表
        label_embed_label = build_clip_label_embedding(clip_model, taglist_label)       #使用 CLIP 模型为每个标签构建一个嵌入表示
        label_embed = label_embed_label.repeat(1, 1, 1)     # 对标签嵌入进行复制，使其能够与图像嵌入进行逐一比较。
        label_embed = label_embed.to(device)
        for i in range(len(self.X)):
            x = Image.fromarray(np.uint8(self.X[i]).transpose((1, 2, 0)))       #将原始图像数据转换为 PIL 图像对象。
            imgs = self.transform(x).unsqueeze(0)       #对图像应用预定义的转换，并增加一个批次维度（因为 CLIP 模型一次处理一个批次的图像）
            imgs = imgs.to(device)
            image_embeds = clip_model.encode_image(imgs).unsqueeze(1)   #将图像通过 CLIP 模型进行编码，得到图像的嵌入表示。unsqueeze(1) 是为了添加一个额外的维度，使得嵌入的形状适合后续处理。
            image_embeds = image_embeds.to(device)
            image_to_label = image_embeds.repeat(1, 100, 1)     #将图像嵌入重复 100 次，以与所有标签进行对比（CIFAR-100 有 100 个类别）。
            output = self.cos(image_to_label, label_embed)      #计算余弦相似度
            print(f"{i + 1}/{len(self.X)}")
            _, labels_g = torch.max(output, dim=1)      #从 output 中找到最大值的索引，表示预测的标签。
            if self.Y[i] == labels_g :  # 检查 self.Y[i] 是否在前两个最相似的索引中
                self.YT[i] = 1  # 如果在，则标记为 1
                file = open(legacy_label_path('CIFAR100', 'CLIP-B32', 'train_label_tf.txt'), 'a')
                file.write("1\n")
                file.close()
            else:
                self.YT[i] = 0  # 否则标记为 0
                file = open(legacy_label_path('CIFAR100', 'CLIP-B32', 'train_label_tf.txt'), 'a')
                file.write("0\n")
                file.close()
                file = open(legacy_label_path('CIFAR100', 'CLIP-B32', 'train_label_t.txt'), 'a')
            file.write(str(self.Y[i]) + '\n')
            file.close()
            file = open(legacy_label_path('CIFAR100', 'CLIP-B32', 'train_label_pre.txt'), 'a')
            file.write(str(labels_g.item()) + '\n')
            file.close()
        print("标记完成")



    def __getitem__(self, index):
        # print(self.X[index].shape)
        x = Image.fromarray(np.uint8(self.X[index]).transpose((1, 2, 0)))
        # x = Image.open(self.X[index]).convert('RGB')
        x = self.transform(x)
        y = self.Y[index]
        yt = self.YT[index]
        return x, y, yt

    def __len__(self):
        return len(self.X)





class Dataset_gengerate_label_clip(Dataset):
    def __init__(self, X, Y, dataset_name, num_classes, transform=get_transform()):
        self.X = X
        self.Y = Y
        self.YT = torch.empty(len(self.Y))
        self.transform = transform
        self.dataset_name = dataset_name
        self.class_num = num_classes
        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        torch.manual_seed(1)
        np.random.seed(1)
        clip_model = load_clip()
        info = load_taglist(dataset=dataset_name)
        taglist_label = info["taglist"]
        label_embed_label = build_clip_label_embedding(clip_model, taglist_label)
        label_embed = label_embed_label.repeat(1, 1, 1)     #代表在一二三个维度都复制一次
        label_embed = label_embed.to(device)
        for i in range(len(self.X)):
            print(f"{i + 1}/{len(self.X)}")
            x = Image.open(self.X[i])
            imgs = self.transform(x).unsqueeze(0)
            imgs = imgs.to(device)
            image_embeds = clip_model.encode_image(imgs).unsqueeze(1)
            image_embeds = image_embeds.to(device)
            image_to_label = image_embeds.repeat(1, num_classes, 1)     ##每个图像嵌入需要与每个标签类别的嵌入进行对比。因此，我们需要将图像嵌入扩展，使其能与所有类别的标签嵌入进行比较。
            output = self.cos(image_to_label, label_embed)
            _, labels_g = torch.max(output, dim=1)
            if labels_g == self.Y[i]:
                self.YT[i] = 1
                file = open(legacy_label_path(dataset_name, 'CLIP-B32', 'train_label_tf.txt'), 'a')
                file.write("1\n")
                file.close()
            else:
                self.YT[i] = 0
                file = open(legacy_label_path(dataset_name, 'CLIP-B32', 'train_label_tf.txt'), 'a')
                file.write("0\n")
                file.close()
            file = open(legacy_label_path(dataset_name, 'CLIP-B32', 'train_label_t.txt'), 'a')
            file.write(str(self.Y[i]) + '\n')
            file.close()
            file = open(legacy_label_path(dataset_name, 'CLIP-B32', 'train_label_pre.txt'), 'a')
            file.write(str(labels_g.item()) + '\n')
            file.close()

    def __getitem__(self, index):
        x = Image.open(self.X[index])
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
        x = Image.fromarray(np.uint8(self.X[index]).transpose((1, 2, 0)))
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
        x = Image.open(self.X[index])
        x = self.transform(x)
        y = self.Y[index]
        return x, y

    def __len__(self):
        return len(self.X)


def get_data_handler(dataset, pattern, input_size):
    if dataset == 'CIFAR100':
        train_data, train_label, test_data, test_label = read_data_cifar_100()
        if pattern == "train":
            datahandler = CIFAR100_generate_label_clip(train_data, train_label, input_size)
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
            datahandler = Dataset_gengerate_label_clip(train_data, train_label, dataset_name=dataset, num_classes = num_classes,
                                              transform=get_transform(input_size))
        elif pattern == "val":
            datahandler = DatasetHandlerTest(test_data, test_label, get_transform(input_size))

    return datahandler


def load_datasets(
        dataset: str,
        model_type: str,
        pattern: str,
        input_size: int,
        batch_size: int,
        num_workers: int
) -> Tuple[DataLoader, Dict]:
    dataset_root = str(raw_dataset(dataset))

    if model_type == "clip":
        tag_file = dataset_root + f"/{dataset}_ram_taglist.txt"
 

    with open(tag_file, "r", encoding="utf-8") as f:
        taglist_or = [line.strip() for line in f]

    taglist = taglist_or 

    datahandler = get_data_handler(dataset, pattern, input_size)
    loader = DataLoader(dataset=datahandler, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    info = {
        "taglist": taglist
    }

    return loader, info


def divide_labeled_or_not(dataset, input_size):     #根据标签信息，将数据集分为两个子集0，1
    data_handler = get_data_handler(dataset, pattern='train', input_size=input_size)
    indices_yt_0 = torch.nonzero(torch.eq(data_handler.YT, 0)).squeeze().tolist()
    indices_yt_1 = torch.nonzero(torch.eq(data_handler.YT, 1)).squeeze().tolist()
    unlabeled_dataset = Subset(data_handler, indices_yt_0)
    labeled_dataset = Subset(data_handler, indices_yt_1)

    return labeled_dataset, unlabeled_dataset


if __name__ == '__main__':


    loader, info = load_datasets(
        dataset='CIFAR100',
        model_type='clip',
        pattern='train',
        input_size=224,
        batch_size=64,
        num_workers=0
    )

    print('_________________')
