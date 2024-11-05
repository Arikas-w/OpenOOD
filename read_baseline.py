import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class train_CustomDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.data = []
        self.transform = transform

        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                image_path, label = line.strip().split()
                self.data.append((image_path, int(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, label = self.data[index]
        image_path = './data/images_classic/' + image_path
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class val_CustomDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.data = []
        self.transform = transform

        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                image_path, label = line.strip().split()
                self.data.append((image_path, int(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, label = self.data[index]
        image_path = './data/images_classic/' + image_path
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

class test_CustomDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.data = []
        self.transform = transform

        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                image_path, label = line.strip().split()
                self.data.append((image_path, int(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, label = self.data[index]
        image_path = './data/images_classic/' + image_path
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

