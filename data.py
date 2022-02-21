import os

import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_path, gt_path, crop_size=(256, 256)):
        super(type(self), self).__init__()
        self.img_path = img_path
        self.gt_path = gt_path
        self.img_list = sorted(os.listdir(img_path))
        self.gt_list = sorted(os.listdir(gt_path))
        self.crop_size = crop_size
        self.to_tensor = transforms.ToTensor()

    def crop_resize_totensor(self, img, crop_location):
        img256 = img.crop(crop_location)
        img128 = img256.resize((self.crop_size[0] // 2, self.crop_size[1] // 2), resample=Image.BILINEAR)
        img64 = img128.resize((self.crop_size[0] // 4, self.crop_size[1] // 4), resample=Image.BILINEAR)
        return self.to_tensor(img256), self.to_tensor(img128), self.to_tensor(img64)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # filename processing
        blurry_img_name = os.path.join(self.img_path, self.img_list[idx])
        clear_img_name = os.path.join(self.gt_path, self.gt_list[idx])

        blurry_img = Image.open(blurry_img_name)
        clear_img = Image.open(clear_img_name)
        #  flip
        flip = np.random.randint(0, 2)
        if flip:
            flip = np.random.randint(0, 2)
            blurry_img = blurry_img.transpose(flip)
            clear_img = clear_img.transpose(flip)
        #  rotate
        rotate = np.random.randint(0, 2)
        if rotate:
            rotate = np.random.randint(2, 5)
            blurry_img = blurry_img.transpose(rotate)
            clear_img = clear_img.transpose(rotate)
        assert blurry_img.size == clear_img.size
        crop_left = int(np.floor(np.random.uniform(0, blurry_img.size[0] - self.crop_size[0] + 1)))
        crop_top = int(np.floor(np.random.uniform(0, blurry_img.size[1] - self.crop_size[1] + 1)))
        crop_location = (crop_left, crop_top, crop_left + self.crop_size[0], crop_top + self.crop_size[1])

        img256_left, img128_left, img64_left = self.crop_resize_totensor(blurry_img, crop_location)
        label256, label128, label64 = self.crop_resize_totensor(clear_img, crop_location)
        batch = {'img256': img256_left, 'img128': img128_left,
                 'img64': img64_left, 'label256': label256, 'label128': label128,
                 'label64': label64}
        return batch


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, gt_path):
        super(type(self), self).__init__()
        self.img_path = img_path
        self.gt_path = gt_path
        self.img_list = sorted(os.listdir(img_path))
        self.gt_list = sorted(os.listdir(gt_path))
        self.to_tensor = transforms.ToTensor()

    def resize_totensor(self, img):
        img_size = img.size
        img256 = img
        img128 = img256.resize((img_size[0] // 2, img_size[1] // 2), resample=Image.BILINEAR)
        img64 = img128.resize((img_size[0] // 4, img_size[1] // 4), resample=Image.BILINEAR)
        return self.to_tensor(img256), self.to_tensor(img128), self.to_tensor(img64)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # filename processing
        blurry_img_name = os.path.join(self.img_path, self.img_list[idx])
        clear_img_name = os.path.join(self.gt_path, self.gt_list[idx])

        blurry_img = Image.open(blurry_img_name)
        h, w = blurry_img.size
        aim_h = int(np.ceil(h / 16) * 16)
        aim_w = int(np.ceil(w / 16) * 16)
        pad_h = int((aim_h - h) / 2)
        pad_w = int((aim_w - w) / 2)
        p = Image.new('RGB', (aim_h, aim_w), (0, 0, 0))
        p.paste(blurry_img, (pad_h, pad_w, h + pad_h, w + pad_w))
        blurry_img = p
        p = Image.new('RGB', (aim_h, aim_w), (0, 0, 0))
        clear_img = Image.open(clear_img_name)
        p.paste(clear_img, (pad_h, pad_w, h + pad_h, w + pad_w))
        clear_img = p

        assert blurry_img.size == clear_img.size

        img256, img128, img64 = self.resize_totensor(blurry_img)
        label256, label128, label64 = self.resize_totensor(clear_img)
        # label256 = self.to_tensor(clear_img)
        batch = {'img256': img256, 'img128': img128, 'img64': img64, 'label256': label256, 'label128': label128,
                 'label64': label64}
        return batch
