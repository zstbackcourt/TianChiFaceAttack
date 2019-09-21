# python3
# -*- coding: utf-8 -*-




import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
from torchvision.transforms import transforms

import numpy as np
import torch
import torch.nn as nn
import random
from PIL import Image

def img2tensor(img_path):
    img = Image.open(img_path)
    img = np.array(img)
    img = img.swapaxes(1, 2).swapaxes(0, 1)
    img = np.reshape(img, [ 3, 112, 112])
    img = np.array(img, dtype=np.float32)
    img = (img - 127.5) / 128.0
    img = torch.from_numpy(img)
    return img

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.paths = []
        picpath = '/notebooks/Workspace/tmp/pycharm_project_314/TianChi/images'
        for root, dirs, files in os.walk(picpath):
            for f in files:
                self.paths.append(os.path.join(root, f))
        # random.shuffle(self.paths)

    def __getitem__(self, index):
        return img2tensor(self.paths[index]) ,self.paths[index]


    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.paths)


# You can then use the prebuilt data loader.

if __name__ == '__main__':
    custom_dataset = CustomDataset()
    train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                               batch_size=64,
                                               shuffle=True)
    for i, (x,path) in enumerate(train_loader):
        print(path)
