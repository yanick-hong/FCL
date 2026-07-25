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


def load_taglist(dataset: str) -> Tuple[Dict]:
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
                                            [步骤3]直接输出最匹配一个的数字编号，无需解释,无需输出类别名称，数字编号必须在0-99之间
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


def load_cifar100():
    with open(raw_dataset("CIFAR100") / "train", 'rb') as f:
        data_train = pickle.load(f, encoding='latin1')
    with open(raw_dataset("CIFAR100") / "test", 'rb') as f:
        data_test = pickle.load(f, encoding='latin1')
    with open(raw_dataset("CIFAR100") / "meta", 'rb') as f:
        data_meta = pickle.load(f, encoding='latin1')
    return data_train, data_test, data_meta


def read_data_cifar_100():
    data_train, data_test, data_meta = load_cifar100()
    train_data = data_train['data'].reshape((data_train['data'].shape[0], 3, 32, 32))
    test_data = data_test['data'].reshape((data_test['data'].shape[0], 3, 32, 32))
    train_label = data_train["fine_labels"]
    test_label = data_test["fine_labels"]
    return train_data, train_label, test_data, test_label


class CIFAR100_handler_train_TF_saved(Dataset):
    def __init__(self, X, Y, input_size, model, processor, transform=None):
        self.X = X
        self.Y = Y
        self.YT = torch.empty(len(self.Y))
        self.Y1 = torch.empty(len(self.Y))
        self.transform = get_transform(input_size)
        self.model = model
        self.processor = processor
        info = load_taglist(dataset="CIFAR100")
        taglist_label = info["taglist"]
        for i in range(len(self.X)):
            pil_image = Image.fromarray(np.uint8(self.X[i]).transpose((1, 2, 0)))
            output_text = generate_label(model, processor, pil_image, taglist_label)
            output_label = int(output_text[0])
            print(f"{i + 1}/{len(self.X)}")
            if self.Y[i] == output_label:
                self.YT[i] = 1
                file = open(legacy_label_path('CIFAR100', 'Qwen_VL_7B_label', 'train_label_tf.txt'), 'a')
                file.write("1\n")
                file.close()
            else:
                self.YT[i] = 0
                file = open(legacy_label_path('CIFAR100', 'Qwen_VL_7B_label', 'train_label_tf.txt'), 'a')
                file.write("0\n")
                file.close()
                file = open(legacy_label_path('CIFAR100', 'Qwen_VL_7B_label', 'train_label_t.txt'), 'a')
            file.write(str(self.Y[i]) + '\n')
            file.close()
            file = open(legacy_label_path('CIFAR100', 'Qwen_VL_7B_label', 'train_label_pre.txt'), 'a')
            file.write(str(output_label) + '\n')
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
        x = Image.fromarray(np.uint8(self.X[index]).transpose((1, 2, 0)))
        x = self.transform(x)
        y = self.Y[index]
        return x, y

    def __len__(self):
        return len(self.X)


def get_data_handler(dataset, pattern, input_size, model, processor):
    if dataset == 'CIFAR100':
        train_data, train_label, test_data, test_label = read_data_cifar_100()
        if pattern == "train":
            datahandler = CIFAR100_handler_train_TF_saved(train_data, train_label, input_size, model, processor)
        elif pattern == "val":
            datahandler = CIFAR100_handler_test(test_data, test_label, input_size)
    return datahandler


def load_datasets(dataset: str, model_type: str, pattern: str, input_size: int,
                  batch_size: int, num_workers: int, model, processor) -> Tuple[DataLoader, Dict]:
    dataset_root = str(raw_dataset(dataset))

    if model_type == "clip":
        tag_file = dataset_root + f"/{dataset}_ram_taglist.txt"

    with open(tag_file, "r", encoding="utf-8") as f:
        taglist_or = [line.strip() for line in f]

    taglist = taglist_or

    datahandler = get_data_handler(dataset, pattern, input_size, model, processor)
    loader = DataLoader(dataset=datahandler, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    info = {"taglist": taglist}
    return loader, info


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
        model=model,
        processor=processor
    )
