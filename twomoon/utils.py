import torch
import random
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch
import numpy as np
import random
from matplotlib.colors import ListedColormap

NumClass = 2


class ThreeLayerMLP(torch.nn.Module):
    def __init__(self, dim_in, dim_out=NumClass, dropout_rate=0.05):
        super(ThreeLayerMLP, self).__init__()
        self.dim_in = dim_in 
        self.dim_out = dim_out  
        self.fc1 = torch.nn.Linear(dim_in, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, self.dim_out)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        y = self.relu(self.dropout(self.fc1(x)))
        y = self.relu(self.dropout(self.fc2(y)))
        y = self.fc3(y)  # raw logits output
        return y


class ThreeLayerMLP_ours(torch.nn.Module):
    def __init__(self, dim_in, num_basis, num_experts, dim_out=NumClass, dropout_rate=0.05):
        super(ThreeLayerMLP_ours, self).__init__()
        self.num_basis = num_basis
        self.num_experts = num_experts
        self.dim_in = dim_in  
        self.dim_out = dim_out  
        self.fc1 = torch.nn.Linear(dim_in, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, self.dim_out)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout_rate)

        self.fc4 = torch.nn.Sequential(
            torch.nn.Linear(2, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, num_experts)
        )

        self.seq2_list = torch.nn.ModuleList([None for _ in range(num_experts)])
        for i in range(num_experts):
            self.seq2_list[i] =torch.nn.Sequential(
            torch.nn.Linear(2, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, num_basis)
        )

    def forward(self, x):
        y = self.relu(self.dropout(self.fc1(x)))
        y = self.relu(self.dropout(self.fc2(y)))

        # weight_expert
        weight_expert = self.fc4(x)

        # coeff
        coeff = torch.stack([self.seq2_list[i](x) for i in range(self.num_experts)], dim=1)
        
        # raw logits output 
        y_res = self.fc3(y)  
        
        return y_res, coeff, weight_expert


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_permutation_matrix(num_basis=2):
    # Return 'num_basis' permutation matrices with size (dim, dim)
    # return size: (num_basis, dim, dim)
    # Since the twomoon example is a binary classification task (K=2), 'num_basis' could be 1 or 2.
    if num_basis == 2:
        matrices = [torch.Tensor([[1, 0], [0, 1]]), torch.Tensor([[0, 1], [1, 0]])]
    else:
        matrices = [torch.Tensor([[1, 0], [0, 1]])]
    return torch.stack(matrices, dim=0)


def train_network_trivial(epochs, x_train, y1_train, y2_train, x_test, y_test, net, optimizer, use_label):
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs): # no minibatch is used. In each epoch, all training data are fed into the network for training.
        net.train()

        inputs, labels = x_train, y1_train if use_label == 0 else y2_train # 'use_label' determines using which expert's labels

        outputs = net(inputs)
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        test_loss, test_acc = cal_loss_acc(x_test, y_test, net)

        print(f' Epoch | All epochs: {epoch} | {epochs}')
        print(f' Train Loss : {loss.item():.3f}')
        print(f' Current Test Loss | Test Accuracy | {test_loss:.3f} | {test_acc:.3f}')
        print("-" * 70)

    return net, test_acc


def train_network_ours(epochs, x_train, y1_train, y2_train, x_test, y_test, net, optimizer, num_basis, num_experts,
                       hyper_parameter):
    criterion = torch.nn.KLDivLoss()

    matrices = generate_permutation_matrix(num_basis)
    for epoch in range(epochs):
        net.train()

        inputs, labels = x_train, [y1_train, y2_train]
        outputs, coeff, weight_expert = net(inputs)
        coeff = torch.softmax(coeff, dim=2) # (batch_size, num_experts, num_basis)
        weight_expert = torch.softmax(weight_expert, dim=1) # (batch_size, num_experts)

        transition_expert = (coeff.reshape(-1, num_experts, num_basis, 1, 1) * matrices).sum(dim=2) # (batch_size, num_experts, NumClass, NumClass)
        y_expert = torch.stack([F.one_hot(ele, NumClass) for ele in labels], dim=1).float() # (batch_size, num_experts, NumClass)
        y_expert_star = torch.matmul(transition_expert, y_expert.reshape(-1, num_experts, NumClass, 1)) # (batch_size, num_experts, NumClass, 1), Eq (3) in the paper
        y_train = (y_expert_star * weight_expert.reshape(-1, num_experts, 1, 1)).sum(dim=1).squeeze(-1) # (batch_size, NumClass), Eq (4) in the paper

        optimizer.zero_grad()

        # classification loss, corresponds to Eq (5) in the paper
        # [Pytorch requires: KLDivLoss needs probabilty as input, while CrossEntropyLoss needs raw logits as input]
        tmp1 = criterion(torch.nn.functional.log_softmax(outputs, dim=1), y_train) 
        
        # penalty term, note that compared to the eq (6) written in the paper, a constant term might missing (using mean or using sum).
        # Nevertheless, since we could freely set the value of hyper-parameter lambda, this is not an issue.
        # Moreover, since in different versions of Pytorch, the reduction 'mean' of KLDivLoss is performed differently. So in your running, it might be different from ours.
        diags = 1 - transition_expert.reshape(-1, NumClass, NumClass).diagonal(dim1=-2, dim2=-1)
        tmp2 = torch.sum(diags * diags) / torch.numel(diags) #  corresponds to Eq (6) in the paper
        
        # overall loss combining these three terms
        loss = tmp1 + hyper_parameter * tmp2
        
        loss.backward()
        optimizer.step()

        test_loss, test_acc = cal_loss_acc_with_aux_output(x_test, y_test, net)

        print(f' Epoch | All epochs: {epoch} | {epochs}')
        print(f' Train Loss : {loss.item():.3f}')
        print(f' Current Test Loss | Test Accuracy | {test_loss:.3f} | {test_acc:.3f}')
        print("-" * 70)

    return net, test_acc


def cal_loss_acc_with_aux_output(x, y, net):
    criterion = torch.nn.CrossEntropyLoss()  # fix to CE loss
    net.eval()
    with torch.no_grad():
        outputs, *_ = net(x)
        _, predicted = torch.max(outputs.detach(), 1)
        correct = (predicted == y).sum().item()
        loss = criterion(outputs, y) 
        val_acc = 100 * correct / y.size(0)
    return loss.item(), val_acc


def cal_loss_acc(x, y, net):
    criterion = torch.nn.CrossEntropyLoss() # fix to CE loss
    net.eval()
    with torch.no_grad():
        outputs = net(x)
        _, predicted = torch.max(outputs.detach(), 1)
        correct = (predicted == y).sum().item()
        loss = criterion(outputs, y)  
        val_acc = 100 * correct / y.size(0)
    return loss.item(), val_acc


# generate dataset
def gen_data(num_samples, train_test_ratio, random_seed, plot=False):
    datasets = make_moons(n_samples=num_samples, noise=0.05, random_state=random_seed)
    x, y = datasets
    x = StandardScaler().fit_transform(x)

    # let's mimic the two experts
    y1 = x[:, 0] >= 0
    y2 = x[:, 1] <= 0
    x_train, x_test, y_train, y_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x, y, y1, y2,
                                                                                              test_size=train_test_ratio,
                                                                                              random_state=random_seed)

    x_train = torch.FloatTensor(x_train)
    y_train = torch.LongTensor(y_train)
    y1_train = torch.LongTensor(y1_train)
    y2_train = torch.LongTensor(y2_train)

    x_test = torch.FloatTensor(x_test)
    y_test = torch.LongTensor(y_test)
    y1_test = torch.LongTensor(y1_test)
    y2_test = torch.LongTensor(y2_test)

    if plot:
        plt.figure()
        plt.scatter(x[:, 0], x[:, 1], c=y,
                    cmap=ListedColormap(['#FF0000', '#0000FF']), edgecolors='k')
        plt.yticks([-2, 0, 2], font='Times New Roman', size=24)
        plt.xticks([-2, 0, 2], font='Times New Roman', size=24)
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)
        plt.title('Golden')

        plt.figure()
        plt.scatter(x[:, 0], x[:, 1], c=y1,
                    cmap=ListedColormap(['#FF0000', '#0000FF']), edgecolors='k')
        plt.yticks([-2, 0, 2], font='Times New Roman', size=24)
        plt.xticks([-2, 0, 2], font='Times New Roman', size=24)
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)
        plt.title('expert1')

        plt.figure()
        plt.scatter(x[:, 0], x[:, 1], c=y2,
                    cmap=ListedColormap(['#FF0000', '#0000FF']), edgecolors='k')
        plt.yticks([-2, 0, 2], font='Times New Roman', size=24)
        plt.xticks([-2, 0, 2], font='Times New Roman', size=24)
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)
        plt.title('expert2')
        plt.show()

    return x_train, x_test, y_train, y_test, y1_train, y1_test, y2_train, y2_test
