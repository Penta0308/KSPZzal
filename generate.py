import modules
import math
import numpy as np
import torch.utils.data
import os
from PIL import Image


class ImgWithoutTemplateIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, dir: list, label):
        super(ImgWithoutTemplateIterableDataset).__init__()
        self.dirl = []
        for d in dir:
            self.dirl += os.listdir(d)
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
        if not modules.hasmeaning(np.array(img)):
            return self.__next__()

        r_img = img.resize((128, 128))
        r, g, b = r_img.split()
        r_r_img = np.asarray(np.float32(r) / 255.0)
        b_r_img = np.asarray(np.float32(g) / 255.0)
        g_r_img = np.asarray(np.float32(b) / 255.0)

        rgb_resize_img = np.asarray([r_r_img, b_r_img, g_r_img])
        return rgb_resize_img, self.label


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


BATCH_SIZE = 4

train_loader = torch.utils.data.DataLoader(
    ConcatDataset(
        ImgWithoutTemplateIterableDataset(modules.config_get("dirs")["0"], "0"),
        ImgWithoutTemplateIterableDataset(modules.config_get("dirs")["1"], "1")
    ), batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
