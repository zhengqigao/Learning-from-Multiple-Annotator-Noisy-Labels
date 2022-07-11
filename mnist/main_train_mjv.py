import sys
sys.path.append("./utils")
from utils.helper import train_network_mjv, seed
from utils.dataloader import get_loader
from utils.model import LeNet5
import torch
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # the parameters you might need to change
    parser.add_argument('--gpu', type=int, default=-1, help='gpu id')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='The learning rate of the optimizer.')
    parser.add_argument('--momentum', type=float, default=0.9, help='The momentum of optimizer.')
    parser.add_argument('--num_epochs', type=int, default=3, help='num epochs')
    parser.add_argument('--expert_type', type=int, default=0, help='type of expert')
    parser.add_argument('--num_experts', type=int, default=3, help='number expert')


    args = parser.parse_args()
    device = torch.device("cpu") if args.gpu < 0 else torch.device("cuda:" + str(args.gpu))

    # control random seed for better reproducibility
    seed(0)

    # construct data loader
    file_path = '/homes/zhengqi/Documents/takeda/mnist/data/MNIST_expertlabels_type' + str(args.expert_type) + '/'
    loader_dict = get_loader(file_path, 64, 4, True)

    # soft majority voting
    net = LeNet5().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=1e-4)

    model_soft, acc_soft = train_network_mjv(args.num_epochs, loader_dict, net, device, optimizer, 'soft', save_model_path='')

    print("Acc soft mjv: %f" %(acc_soft))





