import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision
import os
import numpy as np

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            ])
black = torchvision.transforms.ToPILImage()(torch.zeros(3, 256, 256))


class MKDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.name = os.listdir(os.path.join(path, 'SegmentationClass'))

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        black1 = torchvision.transforms.ToPILImage()(torch.zeros(3, 256, 256))
        black0 = torchvision.transforms.ToPILImage()(torch.zeros(3, 256, 256))
        name = self.name[index]
        namejpg = name[:-3] + 'jpg'
        img1_path = os.path.join(self.path, 'JPEGImages')
        img0_path = os.path.join(self.path, 'SegmentationClass')
        img1 = Image.open(os.path.join(img1_path, namejpg))
        img0 = Image.open(os.path.join(img0_path, name))
        img1_size = torch.Tensor(img1.size)  # WH
        l_max_index = img1_size.argmax()
        ratio = 256. / img1_size[l_max_index.item()]
        img1_re2size = img1_size * ratio
        img1_re2size = np.array(img1_re2size, dtype=np.int32)
        img1_use = img1.resize(img1_re2size)

        img0_use = img0.resize(img1_re2size)
        w, h = img1_re2size.tolist()
        black1.paste(img1_use, (0, 0, int(w), int(h)))
        black0.paste(img0_use, (0, 0, int(w), int(h)))

        return transform(black1), transform(black0)


if __name__ == '__main__':
    i = 1
    dataset = MKDataset(r'D:\dataSet\VOC2012')
    for a, b in dataset:
        print(i)
        print(a.shape)
        i+=1
