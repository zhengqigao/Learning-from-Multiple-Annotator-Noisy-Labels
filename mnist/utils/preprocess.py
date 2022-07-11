from torchvision import datasets
import torchvision
from torch.utils.data import Subset
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

Nmax_train = 60000
index_experts = [11, 103, 364]
NumClass = 10

class ExpertGenerator(object):
    def __init__(self, store_data_root_path, N_train, expert_type, N_expert, expert_threshold):
        assert N_train < Nmax_train

        self.store_data_root_path = store_data_root_path
        self.N_train = N_train
        self.expert_type = expert_type
        self.N_expert = N_expert

        # store train,val,test data here
        data = datasets.MNIST(root='/homes/zhengqi/Documents/takeda/mnist/data',
                              train=1,
                              transform=torchvision.transforms.Compose([
                                  torchvision.transforms.ToTensor(),
                                  torchvision.transforms.Normalize((0.1307,), (0.3081,))]),
                              download=False)
        self.train_data = Subset(data, range(self.N_train))
        self.val_data = Subset(data, range(self.N_train, Nmax_train))
        self.test_data = datasets.MNIST(root='/homes/zhengqi/Documents/takeda/mnist/data',
                                        train=0,
                                        transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.1307,), (0.3081,))]),
                                        download=False)

        # initialize parameters for generating expert labels
        if self.expert_type == 0:
            self.center_index = index_experts
            # np.random.randint(0, N_train, (N_expert,))
            # Generally, please use the np.random... line to setup center_index. Here we explictly assign three indices. Because if randomly generate the center_index, it sometimes leads to 98% accuracy when using only one experts' labels. In this case, testing different methods are meaningless. You could replace the elements in index_experts with any value as long as the corresponding test acc is small.

            self.center_experts = [self.train_data[self.center_index[i]][0] for i in range(self.N_expert)]
            print([self.train_data[self.center_index[i]][1] for i in range(self.N_expert)])
            self.expert_err_threshold = expert_threshold
        elif self.expert_type == 1:
            # the class-wise hammer-spammer does not make sense. It might lead to an expert make predictions all wrong. Here we slightly modify the original hammer-spammer synthetic method.
            self.correct_class_num = 3
            self.class_correct_matrix = np.zeros((self.N_expert, NumClass), dtype=bool)
            for i in range(self.N_expert):
                wrk = np.random.permutation(NumClass)
                self.class_correct_matrix[i, wrk[:self.correct_class_num]] = True
            print(self.class_correct_matrix)
        elif self.expert_type == 2:
            self.center_index = np.random.randint(0, N_train, (N_expert,))
            # Generally, please use the np.random... line to setup center_index. Here we explictly assign three indices. Because if randomly generate the center_index, it sometimes leads to 98% accuracy when using only one experts' labels. In this case, testing different methods are meaningless. You could replace the elements in index_experts with any value as long as the corresponding test acc is small.

            self.center_experts = [self.train_data[self.center_index[i]][0] for i in range(self.N_expert)]
            print([self.train_data[self.center_index[i]][1] for i in range(self.N_expert)])
            self.expert_err_threshold = expert_threshold
        else:
            raise ValueError("mimic_expert_type is not defined")

    def mimic_expert(self, input_feature, label):
        if self.expert_type == 0 or self.expert_type == 2:
            expert_labels = [None for _ in range(self.N_expert)]
            for i in range(self.N_expert):
                coeff = (torch.norm(self.center_experts[i] - input_feature, p=2) <= self.expert_err_threshold)
                if coeff:
                    wrk = np.array(list(set(range(0, NumClass)) - set([label])))
                    np.random.shuffle(wrk)
                    expert_labels[i] = wrk[0]
                else:
                    expert_labels[i] = label
            return [label] + expert_labels
        elif self.expert_type == 1:
            expert_labels = [None for _ in range(self.N_expert)]
            for i in range(self.N_expert):
                if not self.class_correct_matrix[i, label]:
                    wrk = np.array(list(set(range(0, NumClass)) - set([label])))
                    np.random.shuffle(wrk)
                    expert_labels[i] = wrk[0]
                else:
                    expert_labels[i] = label
            return [label] + expert_labels
        else:
            raise ValueError("expert_type is not defined")

    def preprocess(self):
        # before generating data, remove previously generated data, if any
        for ele in ['train', 'val', 'test']:
            if os.path.exists(self.store_data_root_path + ele):
                os.system("rm -rf " + self.store_data_root_path + ele)
            os.makedirs(self.store_data_root_path + ele)

        # test data doesn't have multiple expert labels
        for i, data in enumerate(self.test_data):
            cur_data_path = self.store_data_root_path + 'test/data_' + str(i) + '.pt'
            torch.save(data[0], cur_data_path)
            cur_label_path = self.store_data_root_path + 'test/label_' + str(i) + '.pt'
            torch.save([data[1]], cur_label_path)  # use list match train dataset construction
        with open(self.store_data_root_path + 'test/len_dataset.txt', 'w') as f:
            f.write(str(self.test_data.__len__()))

        # accoring to the CVPR paper, we assume that validation dataset has true labels. Thus it could be used to selecte the optimal model.
        for i, data in enumerate(self.val_data):
            cur_data_path = self.store_data_root_path + 'val/data_' + str(i) + '.pt'
            torch.save(data[0], cur_data_path)
            cur_label_path = self.store_data_root_path + 'val/label_' + str(i) + '.pt'
            torch.save([data[1]], cur_label_path)
        with open(self.store_data_root_path + 'val/len_dataset.txt', 'w') as f:
            f.write(str(self.val_data.__len__()))

        # Finally we deal with the train dataset. Here it is a little complicated, as we need to generate expert labels.

        for i, data in enumerate(self.train_data):
            cur_data_path = self.store_data_root_path + 'train/data_' + str(i) + '.pt'
            torch.save(data[0], cur_data_path)
            cur_label_path = self.store_data_root_path + 'train/label_' + str(i) + '.pt'
            syn_labels = self.mimic_expert(data[0], data[1])
            torch.save(syn_labels, cur_label_path)
        with open(self.store_data_root_path + 'train/len_dataset.txt', 'w') as f:
            f.write(str(self.train_data.__len__()))

        return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # the parameters you might need to change
    parser.add_argument('--seed', type=int, default=0, help='control random seed for better reproducibility')
    parser.add_argument('--expert_type', type=int, default=0, help='type of expert')
    parser.add_argument('--num_experts', type=int, default=3, help='number expert')
    parser.add_argument('--expert_threshold', type=float, default=30, help='default threshold equal to 30 in expert type 0')

    args = parser.parse_args()

    # control random seed
    np.random.seed(args.seed)

    mygen = ExpertGenerator(
            '/homes/zhengqi/Documents/takeda/mnist/data/MNIST_expertlabels_type' + str(args.expert_type) + '/',
            N_train=55000,
            expert_type=args.expert_type, N_expert=args.num_experts, expert_threshold=args.expert_threshold)

    mygen.preprocess()

