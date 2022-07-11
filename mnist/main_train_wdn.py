import sys

sys.path.append("./utils")
from utils.helper import train_network_wdn, seed
from utils.dataloader import get_loader
from utils.model import LeNet5WDN
import torch
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # the parameters you might need to change
    parser.add_argument('--gpu', type=int, default=-1, help='gpu id')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='The learning rate of the optimizer.')
    parser.add_argument('--momentum', type=float, default=0.9, help='The momentum of optimizer.')
    parser.add_argument('--num_epochs', type=int, default=4, help='num epochs')
    parser.add_argument('--expert_type', type=int, default=0, help='type of expert')
    parser.add_argument('--num_experts', type=int, default=3, help='number expert')

    args = parser.parse_args()
    device = torch.device("cpu") if args.gpu < 0 else torch.device("cuda:" + str(args.gpu))

    # control random seed for better reproducibility
    seed(0)

    # construct data loader
    file_path = '/homes/zhengqi/Documents/takeda/mnist/data/MNIST_expertlabels_type' + str(args.expert_type) + '/'
    loader_dict = get_loader(file_path, 64, 4, True)

    # define model, loss function, optimizer
    net = LeNet5WDN(args.num_experts).to(device)
    weight = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(args.num_experts, 1))]).to(device)

    optimizer1 = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=1e-4)

    optimizer2 = torch.optim.SGD(weight.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=1e-4)

    model, acc = train_network_wdn(args.num_epochs, loader_dict, net, device, weight, optimizer1, optimizer2, save_model_path='')
    print("Acc wdn: %f" %acc)
