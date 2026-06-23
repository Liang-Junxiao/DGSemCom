import os
import numpy as np
from glob import glob
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from PIL import Image
import torch

class Datasets(Dataset):
    def __init__(self, config, train=False):
        if train:
            self.data_dir = config.train_data_dir
            _, self.im_height, self.im_width = config.image_dims
            transforms_list = [
                transforms.RandomCrop((self.im_height, self.im_width),padding=0,pad_if_needed=True),
                transforms.ToTensor()]
            self.transform = transforms.Compose(transforms_list)
        else:
            if config.phase=='visual':
                self.data_dir = config.visual_data_dir
            else:
                self.data_dir = config.test_data_dir
            _, self.im_height, self.im_width = config.image_dims
            transforms_list = [
                # transforms.RandomCrop((self.im_height, self.im_width), padding=0, pad_if_needed=True),
                transforms.ToTensor()]
            self.transform = transforms.Compose(transforms_list)
        self.imgs = []
        for dir in self.data_dir:
            # self.imgs += glob(os.path.join(dir, '*'))
            self.imgs += glob(os.path.join(dir, '*.jpeg'))
            self.imgs += glob(os.path.join(dir, '*.jpg'))
            self.imgs += glob(os.path.join(dir, '*.png'))

        self.imgs.sort()

    def __getitem__(self, item):
        image_ori = self.imgs[item]
        image = Image.open(image_ori).convert('RGB')
        img = self.transform(image)
        return img

    def __len__(self):
        return len(self.imgs)


def  get_loader(config):
    train_dataset = Datasets(config, train=True)
    test_dataset = Datasets(config)

    def worker_init_fn_seed(worker_id):
        seed = 10
        seed += worker_id
        np.random.seed(seed)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               num_workers=config.num_workers,
                                               pin_memory=True,
                                               batch_size=config.batch_size,
                                               worker_init_fn=worker_init_fn_seed,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              shuffle=False)

    return train_loader, test_loader

def get_test_loader(config, **kwargs):
    test_dataset = Datasets(config)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size= kwargs['batch_size'] if 'batch_size' in kwargs else 1,
                                              shuffle=False)

    return test_loader

def get_visual_loader(config):
    config.phase = 'visual'
    visual_dateset = Datasets(config)
    visual_loader = torch.utils.data.DataLoader(dataset=visual_dateset,
                                                batch_size=1,
                                                shuffle=False)
    return  visual_loader