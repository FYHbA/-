from distutils.command.config import config
import json
import os
import random
import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption
import os
from torchvision.transforms.functional import hflip, resize

import math
import random
from random import random as rand


class DGM4_Dataset(Dataset):
    def __init__(self, config, ann_file, transform, max_words=512, is_train=True,test_json=None):
        # 初始化数据集
        self.root_dir = '../datasets'  # 数据集根目录
        self.ann = []  # 存储注释信息的列表
        self.test_json=[]
        if not test_json:
            for f in ann_file:
                # 从每个注释文件加载数据并合并到 ann 列表中
                self.ann += json.load(open(f, 'r', encoding='utf-8'))
                # 根据配置的 dataset_division 划分数据集
                if 'dataset_division' in config:
                    self.ann = self.ann[:int(len(self.ann) / config['dataset_division'])]
        else:
            self.test_json = test_json
            self.ann=self.test_json


        self.transform = transform  # 图像转换函数
        self.max_words = max_words  # 最大单词数
        self.image_res = config['image_res']  # 图像分辨率

        self.is_train = is_train  # 是否为训练模式

    def __len__(self):
        # 返回数据集的长度
        return len(self.ann)

    def get_bbox(self, bbox):
        # 将边界框坐标转换为整数格式，并计算宽度和高度
        xmin, ymin, xmax, ymax = bbox
        w = xmax - xmin
        h = ymax - ymin
        return int(xmin), int(ymin), int(w), int(h)

    def __getitem__(self, index):
        # 根据索引获取数据
        ann = self.ann[index]  # 获取指定索引的注释
        if not self.test_json:
            img_dir = ann['image']  # 获取图像路径
            image_dir_all = f'{self.root_dir}/{img_dir}'  # 构建完整图像路径

            # 检查图像文件是否存在
            if not img_dir:
                # 如果文件不存在，返回占位图像
                image = Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8))
            else:
                try:
                    # 打开图像文件并转换为 RGB 格式
                    image = Image.open(image_dir_all).convert('RGB')
                except (OSError, IOError) as e:
                    # # 如果无法打开图像，返回占位图像
                    # print(f"Error opening image {image_dir_all}")
                    image = Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8))
        else:
            image_data_list =ann['image']
            if not image_data_list:
                # 如果文件不存在，返回占位图像
                image = Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8))
            else:
                # 将RGB值填充到图像
                # 将图像数据从列表转换为 NumPy 数组
                image_data = np.array(image_data_list, dtype=np.uint8)
                # 使用 NumPy 数组创建一个图像
                image = Image.fromarray(image_data)

        W, H = image.size  # 获取图像的宽度和高度
        has_bbox = False  # 标记是否有边界框
        # try:
        #     # 获取伪造图像的边界框
        #     x, y, w, h = self.get_bbox(ann['fake_image_box'])
        #     has_bbox = True  # 设置标记为 True
        # except:
        #     # 如果没有边界框，设置为零张量
        # fake_image_box = torch.tensor([0, 0, 0, 0], dtype=torch.float)

        do_hflip = False  # 标记是否进行了水平翻转
        if self.is_train:
            # 在训练模式下，随机决定是否进行水平翻转
            if rand() < 0.5:
                image = hflip(image)  # 进行水平翻转
                do_hflip = True

            # 将图像调整到指定的分辨率
            image = resize(image, [self.image_res, self.image_res], interpolation=Image.BICUBIC)

        # 应用其他转换
        image = self.transform(image)

        # if has_bbox:
            # # 如果进行了水平翻转，调整边界框坐标
            # if do_hflip:
            #     x = (W - x) - w  # 更新 x 坐标
            #
            # # 将边界框坐标调整为与图像分辨率相匹配
            # x = self.image_res / W * x
            # w = self.image_res / W * w
            # y = self.image_res / H * y
            # h = self.image_res / H * h
            #
            # # 计算边界框中心点坐标
            # center_x = x + 1 / 2 * w
            # center_y = y + 1 / 2 * h

            # # 创建归一化的边界框张量
            # fake_image_box = torch.tensor([center_x / self.image_res,
            #                                center_y / self.image_res,
            #                                w / self.image_res,
            #                                h / self.image_res],
            #                               dtype=torch.float)

        label = ann['fake_cls']  # 获取伪造图像的类别标签
        caption = pre_caption(ann['text'], self.max_words)  # 预处理文本描述
        # fake_text_pos = ann['fake_text_pos']  # 获取伪造文本位置


        # 返回图像、标签、文本描述、边界框、伪造文本位置列表以及原始图像的宽度和高度
        return image, label, caption, W, H

