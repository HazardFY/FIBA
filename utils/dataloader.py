import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import os
import csv
import kornia.augmentation as A
import random
import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.tensorboard import SummaryWriter


class ToNumpy:
    def __call__(self, x):
        x = np.array(x)
        if len(x.shape) == 2:
            x = np.expand_dims(x, axis=2)
        return x


class ProbTransform(torch.nn.Module):
    def __init__(self, f, p=1):
        super(ProbTransform, self).__init__()
        self.f = f
        self.p = p

    def forward(self, x):  # , **kwargs):
        if random.random() < self.p:
            return self.f(x)
        else:
            return x


def get_transform(opt, train=True, pretensor_transform=False):
    transforms_list = []
    transforms_list.append(transforms.Resize((opt.input_height, opt.input_width)))
    if pretensor_transform:
        if train:
            transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))
            transforms_list.append(transforms.RandomVerticalFlip(p=0.5))
            # transforms_list.append(transforms.RandomCrop((opt.input_height, opt.input_width), padding=opt.random_crop))
            transforms_list.append(transforms.RandomRotation(opt.random_rotation))


    transforms_list.append(transforms.ToTensor())
    if opt.dataset == "cifar10":
        transforms_list.append(transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]))
    elif opt.dataset == "mnist":
        transforms_list.append(transforms.Normalize([0.5], [0.5]))
    elif opt.dataset == "gtsrb" or opt.dataset == "celeba" or opt.dataset=="ISIC2019":
        pass
    else:
        raise Exception("Invalid Dataset")
    return transforms.Compose(transforms_list)


class PostTensorTransform(torch.nn.Module):
    def __init__(self, opt):
        super(PostTensorTransform, self).__init__()
        # self.random_crop = ProbTransform(
        #     A.RandomCrop((opt.input_height, opt.input_width), padding=opt.random_crop), p=0.8
        # )
        self.random_horizontal_flip = A.RandomHorizontalFlip(p=0.5)
        self.random_Vertical_flip = A.RandomVerticalFlip(p=0.5)
        self.random_rotation = ProbTransform(A.RandomRotation(opt.random_rotation), p=0.5)
        self.random_ColorJitter = A.ColorJitter(0.1, 0.1, 0.1, 0.1, p=0.5)


    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x




class CSVDataset(data.Dataset):
    def __init__(self, root, csv_file, image_field, target_field,
                transform=None, add_extension=None,
                 limit=None, random_subset_size=None,
                 split=None):
        """

        :param root: root of dataset
        :param csv_file: csv file of whole dataset
        :param image_field: 'image'
        :param target_field: 'label
        :param transform:
        :param add_extension: 'jpg'
        :param limit:
        :param random_subset_size: int,get random subset dataset
        :param split: TXT document stored the names of images
        """
        self.root = root

        self.image_field = image_field
        self.target_field = target_field
        self.transform = transform

        self.add_extension = add_extension

        self.data = pd.read_csv(csv_file, sep=None)
        self.class_amount_dict = None

        # pdb.set_trace()
        # Split
        if split is not None:
            with open(split, 'r') as f:
                selected_images = f.read().splitlines()
            self.data = self.data[self.data[image_field].isin(selected_images)]
            self.data = self.data.reset_index()



        classes = list(self.data[self.target_field].unique())
        classes.sort()
        self.class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.classes = classes

        print('Found {} images from {} classes.'.format(len(self.data),
                                                        len(classes)))
        for class_name, idx in self.class_to_idx.items():
            n_images = dict(self.data[self.target_field].value_counts())# [class_idx: amount of class]
            self.class_amount_dict = n_images
            print("    Class '{}' ({}): {} images.".format(
                class_name, idx, n_images[class_name]))

    def __getitem__(self, index):
        path = os.path.join(self.root,
                            self.data.loc[index, self.image_field])
        if self.add_extension:
            path = path + self.add_extension

        sample = Image.open(path).convert('RGB')

        target = self.class_to_idx[self.data.loc[index, self.target_field]]
        if self.transform is not None:
            # print("transform exixts")
            sample_trans = self.transform(sample)
            # print('ok')


        return sample_trans, target

    def __len__(self):
        return len(self.data)

def get_dataloader(opt, train=True,set_ISIC2019='Train', pretensor_transform=False):
    
    transform = get_transform(opt, train, pretensor_transform)
    if opt.dataset=='ISIC2019':
        csv_path = '/media/userdisk1/yf/ISIC2019/ISIC2019_grandtruethlabels.csv'
        root_path = '/media/userdisk1/yf/ISIC2019/ISIC_2019_Training_Input/'
        if set_ISIC2019 == 'Train':
            dataset = CSVDataset(root=root_path, csv_file=csv_path, image_field='image', target_field='label',
                               transform=transform, add_extension='.jpg',
                               split='/media/userdisk1/yf/ISIC2019/txt/train'+str(opt.split_idx)+'.txt')
        elif set_ISIC2019 == 'Val':
            dataset = CSVDataset(root=root_path, csv_file=csv_path, image_field='image', target_field='label',
                                 transform=transform, add_extension='.jpg',
                                 split='/media/userdisk1/yf/ISIC2019/txt/validation' + str(opt.split_idx) + '.txt')
        elif set_ISIC2019 == 'Test':
            dataset=CSVDataset(root=root_path, csv_file=csv_path, image_field='image', target_field='label',
                              transform=transform, add_extension='.jpg',
                              split='/media/userdisk1/yf/ISIC2019/txt/test' + str(opt.split_idx) + '.txt')
        else:
            print ('Wrong set_ISIC2019',set_ISIC2019)
    else:
        raise Exception("Invalid dataset")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.bs, num_workers=opt.num_workers, shuffle=True,drop_last=True)
    return dataloader




if __name__ == "__main__":
    main()
