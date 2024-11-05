import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# class train_CustomDataset(Dataset):
#     def __init__(self, file_path, transform=None):
#         self.data = []
#         self.transform = transform

#         with open(file_path, 'r') as file:
#             lines = file.readlines()
#             for line in lines:
#                 image_path, label = line.strip().split()
#                 self.data.append((image_path, int(label)))

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         image_path, label = self.data[index]
#         image_path = './data/images_classic/' + image_path
#         image = Image.open(image_path).convert('RGB')
#         if self.transform:
#             image = self.transform(image)
#         return image, label

class train_CustomDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.data = []
        self.transform = transform

        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                image_path, label = line.strip().split()
                self.data.append({'image_path': image_path, 'label': int(label)})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        image_path, label = sample['image_path'], sample['label']

        image_path = './data/images_classic/' + image_path
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return {'data': image, 'label': label}
    
class val_CustomDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.data = []
        self.transform = transform

        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                image_path, label = line.strip().split()
                # 修改这里的解析逻辑
                image_path = image_path.split('/')[-1]  # 获取文件名
                label = int(label)
                self.data.append((image_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, label = self.data[index]
        # 修改这里的文件路径拼接逻辑
        image_path = './data/images_classic/tin/val/images/' + image_path
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        return image, label

    
class test_cifar10_Dataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.data = []
        self.transform = transform

        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                image_path, label = line.strip().split()
                image_path = './data/images_classic/' + image_path  
                label = int(label)
                self.data.append((image_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, label = self.data[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class test_cifar100_Dataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.data = []
        self.transform = transform

        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                image_path, label = line.strip().split()
                image_path = './data/images_classic/' + image_path  
                label = int(label)
                self.data.append((image_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, label = self.data[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
    

class test_tin_Dataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.data = []
        self.transform = transform

        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                image_path, label = line.strip().split()
                image_path = './data/images_classic/' + image_path  
                label = int(label)
                self.data.append((image_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, label = self.data[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label