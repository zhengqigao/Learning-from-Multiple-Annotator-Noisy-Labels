import torch
from torch.utils.data import Dataset
import os


class MNISTDataset(Dataset):
    def __init__(self, store_data_root_path, train_val_test):
        self.store_data_root_path = store_data_root_path
        self.train_val_test = train_val_test

    def __len__(self):
        with open(self.store_data_root_path + self.train_val_test + '/len_dataset.txt', 'r') as f:
            return int(f.read().strip())

    def __getitem__(self, index):
        input_feature = torch.load(self.store_data_root_path + self.train_val_test + '/data_' + str(index) + '.pt')
        label = torch.load(self.store_data_root_path + self.train_val_test + '/label_' + str(index) + '.pt')

        return input_feature, label


def get_loader(store_data_root_path, batch_size, num_workers, shuffle):
    train_data = MNISTDataset(store_data_root_path, 'train')
    val_data = MNISTDataset(store_data_root_path, 'val')
    test_data = MNISTDataset(store_data_root_path, 'test')

    cretate_loader = lambda x: torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    # create dataloader

    return {'train': cretate_loader(train_data), 'val': cretate_loader(val_data), 'test': cretate_loader(test_data)}


if __name__ == '__main__':
    loader_dict = get_loader('/homes/zhengqi/Documents/takeda/mnist/data/MNIST_expertlabels_type1/', 64, 1, False)
    train_loader = loader_dict['train']
    wrk = []
    for epoch in range(3):
        for i, data in enumerate(train_loader):
            image, label = data[0], data[1]
            if i == 10:
                print(len(train_loader))
                break
