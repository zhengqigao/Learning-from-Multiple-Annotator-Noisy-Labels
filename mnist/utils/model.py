import torch
import torch.nn as nn
import torch.nn.functional as F

NumClass = 10


class LeNet5(nn.Module):
    def __init__(self, num_class=NumClass, in_channel=1):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 6, kernel_size=5, padding=0, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(6, affine=False)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=0, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16, affine=False)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.linear1 = nn.Linear(4 * 4 * 16, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, num_class)

    def forward(self, x):
        out = F.relu(self.pool1(self.bn1(self.conv1(x))))
        out = F.relu(self.pool2(self.bn2(self.conv2(out))))

        out = F.relu(self.linear1(out.view(x.size(0), -1)))
        out = F.relu(self.linear2(out))
        out = self.linear3(out)
        return out  # return raw logits


class LeNet5Cvpr(nn.Module):
    def __init__(self, num_experts, num_class=NumClass, in_channel=1):
        super(LeNet5Cvpr, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 6, kernel_size=5, padding=0, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(6, affine=False)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=0, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16, affine=False)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.linear1 = nn.Linear(4 * 4 * 16, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, num_class)

        self.seq_list = nn.ParameterList([nn.Parameter(torch.eye(num_class)) for _ in range(num_experts)])

    def forward(self, x):
        out = F.relu(self.pool1(self.bn1(self.conv1(x))))
        out = F.relu(self.pool2(self.bn2(self.conv2(out))))

        out = F.relu(self.linear1(out.view(x.size(0), -1)))
        out = F.relu(self.linear2(out))
        out = self.linear3(out)
        return out, self.seq_list


class LeNet5WDN(nn.Module):
    def __init__(self, num_experts, num_class=NumClass, in_channel=1):
        super(LeNet5WDN, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 6, kernel_size=5, padding=0, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(6, affine=False)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=0, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16, affine=False)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.linear1 = nn.Linear(4 * 4 * 16, 120)
        self.linear2 = nn.Linear(120, 84)

        self.linear_list = nn.ModuleList([nn.Linear(84, num_class) for _ in range(num_experts)])

        # self.weight = nn.Parameter(torch.randn(num_experts, 1))

    def forward(self, x):
        out = F.relu(self.pool1(self.bn1(self.conv1(x))))
        out = F.relu(self.pool2(self.bn2(self.conv2(out))))

        out = F.relu(self.linear1(out.view(x.size(0), -1)))
        out = F.relu(self.linear2(out))
        out_res = [None for _ in range(len(self.linear_list))]
        for i in range(len(self.linear_list)):
            out_res[i] = self.linear_list[i](out)
        return out_res #, self.weight


class LeNet5Ours(nn.Module):
    def __init__(self, num_basis, num_experts, num_class=10, in_channel=1):
        super(LeNet5Ours, self).__init__()
        self.num_basis = num_basis
        self.num_experts = num_experts
        self.conv1 = nn.Conv2d(in_channel, 6, kernel_size=5, padding=0, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(6, affine=False)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=0, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16, affine=False)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.linear1 = nn.Linear(4 * 4 * 16, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, num_class)

        self.seq1 = nn.Sequential(
            nn.Linear(4 * 4 * 16, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_experts))

        self.seq2_list = nn.ModuleList([None for _ in range(num_experts)])
        for i in range(num_experts):
            self.seq2_list[i] = nn.Sequential(
                nn.Linear(4 * 4 * 16, 120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84, num_basis))

    def forward(self, x):
        out = F.relu(self.pool1(self.bn1(self.conv1(x))))
        out = F.relu(self.pool2(self.bn2(self.conv2(out))))

        # out_res: corresponds to the original LeNet5
        out_res = F.relu(self.linear1(out.view(x.size(0), -1)))
        out_res = F.relu(self.linear2(out_res))
        out_res = self.linear3(out_res)

        # weight_expert
        weight_expert = self.seq1(out.view(x.size(0), -1))

        # coeff
        coeff = torch.stack([self.seq2_list[i](out.view(x.size(0), -1)) for i in range(self.num_experts)], dim=1)

        return out_res, coeff, weight_expert



if __name__ == '__main__':
    device = torch.device("cuda:1")
    input = torch.rand(64, 1, 28, 28).to(device)
    net = LeNet5Ours(20, 3).to(device)
    output, coeff, weight = net(input)
    print(output.size(), coeff.size(), weight.size())
