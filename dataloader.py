import torch
from torchvision import transforms

import os
import glob

from PIL import Image
import numpy as np


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, emoji_type, image_size=(32, 32), format="RGB", train=True, device='cuda:0'):
        super(ImageDataset, self).__init__()
        self.data_sub_file = '{}'.format(
            emoji_type) if train else 'Test_{}'.format(emoji_type)
        self.data_dir = os.path.join(
            data_dir, self.data_sub_file, emoji_type)

        self.transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if format == "RGB" else transforms.Normalize(
                (0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))
        ])

        self.data = []
        print(self.data_dir)
        for i, f in enumerate(glob.glob(os.path.join(self.data_dir, '*.png'))):
            image = Image.open(f).convert(format)
            self.data.append(self.transforms(image))

        self.data = torch.stack(self.data, dim=0).to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_data_loader(emoji_type, batch_size, num_workers, format):
    """Creates training and test data loaders"""
    train_dataset = ImageDataset(
        '../emojis', emoji_type, format=format, train=True)
    test_dataset = ImageDataset(
        '../emojis', emoji_type, format=format, train=False)

    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    return trainloader, testloader
