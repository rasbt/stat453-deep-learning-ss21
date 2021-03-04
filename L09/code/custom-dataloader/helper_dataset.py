import os
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms


class MyDataset(Dataset):

    def __init__(self, csv_path, img_dir, transform=None):

        df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.img_names = df['File Name']
        self.y = df['Class Label']
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index]))

        if self.transform is not None:
            img = self.transform(img)

        label = self.y[index]
        return img, label

    def __len__(self):
        return self.y.shape[0]


def get_dataloaders(batch_size,
                    csv_dir='.',
                    img_dir='.',
                    num_workers=0,
                    batch_size_factor_eval=10,
                    train_transforms=None,
                    test_transforms=None):

    if train_transforms is None:
        train_transforms = transforms.ToTensor()

    if test_transforms is None:
        test_transforms = transforms.ToTensor()

    train_dataset = MyDataset(
        csv_path=os.path.join(csv_dir, 'mnist_train.csv'),
        img_dir=os.path.join(img_dir, 'mnist_train'),
        transform=train_transforms)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,  # want to shuffle the dataset
        num_workers=0)  # number processes/CPUs to use

    valid_dataset = MyDataset(
        csv_path=os.path.join(csv_dir, 'mnist_valid.csv'),
        img_dir=os.path.join(img_dir, 'mnist_valid'),
        transform=test_transforms)

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size*batch_size_factor_eval,
        shuffle=False,
        num_workers=0)

    test_dataset = MyDataset(
        csv_path=os.path.join(csv_dir, 'mnist_test.csv'),
        img_dir=os.path.join(img_dir, 'mnist_test'),
        transform=test_transforms)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size*batch_size_factor_eval,
        shuffle=False,
        num_workers=0)

    return train_loader, valid_loader, test_loader
