from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
from glob import glob
from torchvision.transforms.functional import to_tensor
import random
import torchvision


p = np.random.choice([0, 1])  
hori_flip = torchvision.transforms.RandomHorizontalFlip(p)
class MyDataset(Dataset):
    def __init__(self, root, name, cropSize, mode='train'):
        self.root = root
        self.name = name
        self.mode = mode
        self.cropSize = cropSize

        self.files_A = sorted(glob(os.path.join(self.root, self.mode + "A", "*.*")))
        self.files_B = sorted(glob(os.path.join(self.root, self.mode + "B", "*.*")))
        # self.files = self.files_A

    def __getitem__(self, index):
        if self.name == "HazeCityscapes":
            imgA = np.array(Image.open(self.files_A[index]))
            imgB = np.array(Image.open(self.files_B[index // 3]))

        elif self.name == "RESIDE-6K":
            imgA = np.array(Image.open(self.files_A[index]))
            imgB = np.array(Image.open(self.files_B[index]))

        imgA = to_tensor(imgA)
        imgB = to_tensor(imgB)

        hh, ww = imgA.shape[1], imgB.shape[2]

        if self.mode == "train":
            rr = random.randint(0, hh - self.cropSize)
            cc = random.randint(0, ww - self.cropSize)
            imgA = hori_flip(imgA[:, rr:rr + self.cropSize, cc:cc + self.cropSize])
            imgB = hori_flip(imgB[:, rr:rr + self.cropSize, cc:cc + self.cropSize])

            if np.random.random() > 0.5:
                cut_ratio = np.random.rand() / 2
                ch, cw = np.int(self.cropSize * cut_ratio), np.int(self.cropSize * cut_ratio)
                cy = np.random.randint(0, self.cropSize - ch + 1)
                cx = np.random.randint(0, self.cropSize - cw + 1)

                imgA[:, cy:cy + ch, cx:cx + cw] = imgB[:, cy:cy + ch, cx:cx + cw]

        return imgA, imgB

    def __len__(self):
        return len(self.files_A)
