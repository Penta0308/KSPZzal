import json
import hashlib
import os
import random
import torch
import numpy as np
import torch.utils.data
from PIL import Image
import torch.optim
import torch.nn.functional as F

SEED = 1
BATCH_SIZE = 4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def config_get(key: str):
    with open("config.json", "r", encoding="UTF-8") as f:
        settings = json.load(f)
    try:
        return settings[key]
    except KeyError:
        config_update(key, None)
        return None


def config_update(key: str, val):
    with open("config.json", "r", encoding="UTF-8") as f:
        settings = json.load(f)
    settings[key] = val
    with open("config.json", "w", encoding="UTF-8") as f:
        json.dump(settings, f, indent=4, ensure_ascii=False)


templatehash = config_get("templateimghash")


def hasmeaning(data) -> bool:
    return str(hashlib.blake2b(data).hexdigest()) not in templatehash


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 합성곱층
        self.conv1 = torch.nn.Conv2d(3, 10, 5)  # 입력 채널 수, 출력 채널 수, 필터 크기
        self.conv2 = torch.nn.Conv2d(10, 20, 5)

        # 전결합층
        self.fc1 = torch.nn.Linear(20 * 29 * 29, 50)  # 29=(((((128-5)+1)/2)-5)+1)/2
        self.fc2 = torch.nn.Linear(50, 2)

    def forward(self, x):
        # 풀링층
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # 풀링 영역 크기
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 20 * 29 * 29)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


class ImgWithoutTemplateIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, dirs: list, label):
        super(ImgWithoutTemplateIterableDataset).__init__()
        self.dirl = []
        for d in dirs:
            self.dirl += [d + "/" + p for p in os.listdir(d)]
        self.dirc = 0
        self.label = label

    def __iter__(self):
        return self

    def __next__(self):
        if self.dirc >= len(self.dirl):
            raise StopIteration
        dir = self.dirl[self.dirc]
        img = Image.open(dir, 'r')

        self.dirc += 1

        r_img = img.resize((128, 128))
        r, g, b = r_img.split()
        r_r_img = np.asarray(np.float32(r) / 255.0)
        b_r_img = np.asarray(np.float32(g) / 255.0)
        g_r_img = np.asarray(np.float32(b) / 255.0)

        rgb_resize_img = np.asarray([r_r_img, b_r_img, g_r_img])
        return rgb_resize_img, self.label

    def __len__(self):
        return len(self.dirl)

    def __getitem__(self, item):
        dir = self.dirl[item]
        img = Image.open(dir, 'r')

        self.dirc += 1

        r_img = img.resize((128, 128))
        r_img = r_img.convert('HSV')
        h, s, v = r_img.split()
        h_r_img = np.asarray(np.float32(h) / 255.0)
        s_r_img = np.asarray(np.float32(s) / 255.0)
        v_r_img = np.asarray(np.float32(v) / 255.0)

        return [torch.tensor([h_r_img, s_r_img, v_r_img]), torch.tensor(self.label, dtype=torch.long)]


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        for p in self.datasets:
            pi = i
            i -= len(p)
            if i < 0:
                return p[pi]

    def __len__(self):
        l = 0
        for d in self.datasets:
            l += len(d)
        return l
