import torch
import random
import numpy as np
from utils.model import NumClass
import torch.nn.functional as F
from copy import deepcopy
from utils.loss import EBEMLoss


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_permutation_matrix(num_basis, dim):
    # Return N=num_basis permutation matrices with size dim by dim
    # return size: (num_basis, dim, dim)
    # If num_basis is 1, it always return the identity matrix. This is very conveient for doing ablation study.
    # Actually, we use dim=1 in the Cifar-100 experiment to do the ablation.
    matrices = [torch.eye(dim) for _ in range(num_basis)]
    for i, matrix in enumerate(matrices):
        if i >= 1:
            wrk = list(range(dim))
            random.shuffle(wrk)
            matrices[i] = matrix[wrk, :]
    return torch.stack(matrices, dim=0)


def train_network_ours(epochs, loader_dict, net, device, optimizer, num_basis, num_experts, hyper_parameter,
                       save_model_path=''):
    test_acc_list, test_loss_list = [], []
    val_acc_list, val_loss_list = [], []
    model_best, best_acc_val, best_acc_test = None, 0, 0

    criterion = torch.nn.KLDivLoss()

    matrices = generate_permutation_matrix(num_basis, NumClass).to(device)
    for epoch in range(epochs):
        net.train()
        train_loss = 0.0
        for i, data in enumerate(loader_dict['train']):
            img_inputs, labels = data[0], data[1]  
            img_inputs, labels = img_inputs.to(device), [labels[i].to(device) for i in range(1, len(labels))] # expert labels start from index=1
            outputs, coeff, weight_expert = net(img_inputs)
            coeff = torch.softmax(coeff, dim=2)
            weight_expert = torch.softmax(weight_expert, dim=1)

            # Some complicated matrix manipulation happens below. 
            # As an example, if batch_size = 64, num_experts = 3, NumClass = 10, num_basis = 20
            # coeff: (64, 3, 20)
            # transition_expert: (64, 3, 10, 10)
            # y_expert: (64, 3, 10)
            # y_expert_star: (64, 3, 10, 1)
            # weight_expert: (64, 3)
            # y_train: (64, 10)

            transition_expert = (coeff.reshape(-1, num_experts, num_basis, 1, 1) * matrices).sum(dim=2)
            y_expert = torch.stack([F.one_hot(ele, NumClass) for ele in labels], dim=1).float()

            y_expert_star = torch.matmul(transition_expert, y_expert.reshape(-1, num_experts, NumClass, 1))
            y_train = (y_expert_star * weight_expert.reshape(-1, num_experts, 1, 1)).sum(dim=1).squeeze(-1)

            # begin optimization
            optimizer.zero_grad()

            # classification loss, corresponds to Eq (5) in the paper
            # [Pytorch requires: KLDivLoss needs probabilty as input, while CrossEntropyLoss needs raw logits as input]
            tmp1 = criterion(torch.nn.functional.log_softmax(outputs, dim=1), y_train)

            # penalty term, note that compared to the eq (6) written in the paper, a constant term might missing (using mean or using sum).
            # Nevertheless, since we could freely set the value of hyper-parameter lambda, this is not an issue.
            # Moreover, since in different versions of Pytorch, the reduction 'mean' of KLDivLoss is performed differently. So in your running, it might be different from ours.
            diags = 1 - transition_expert.reshape(-1, NumClass, NumClass).diagonal(dim1=-2, dim2=-1)
            tmp3 = torch.sum(diags * diags) / torch.numel(diags) #  corresponds to Eq (6) in the paper

            loss = tmp1 + hyper_parameter * tmp3
            loss.backward()
            optimizer.step()
            train_loss += loss.item() / len(loader_dict['train'])

        val_loss, val_acc = cal_loss_acc_with_aux_output(loader_dict['val'], device, net,
                                                         use_label=0) # val acc is evaluated with golden labels

        test_loss, test_acc = cal_loss_acc_with_aux_output(loader_dict['test'], device, net,
                                                           use_label=0)  # test acc is evaluated with golden labels

        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        # use validation set to select the best model
        if val_acc > best_acc_val:
            best_acc_test = test_acc
            best_acc_val = val_acc
            model_best = deepcopy(net)

        print(f' Epoch | All epochs: {epoch} | {epochs}')
        print(f' Train Loss: {train_loss:.3f}')
        print(f' Current Vali Loss | Vali Accuracy | {val_loss:.3f} | {val_acc:.3f}')
        print(f' Current Test Loss | Test Accuracy | {test_loss:.3f} | {test_acc:.3f}')
        print(f' Best Vali Acc | Test Acc  | {best_acc_val:.3f} | {best_acc_test:.3f}')
        print("-" * 70)

    if save_model_path:
        torch.save(net.state_dict(), save_model_path)

    return model_best, best_acc_test



def train_network_cvpr(epochs, loader_dict, net, device, optimizer, hyper_parameter, weight_CE, save_model_path=''):
    test_acc_list, test_loss_list = [], []
    val_acc_list, val_loss_list = [], []
    model_best, best_acc_val, best_acc_test = None, 0, 0

    criterion = torch.nn.NLLLoss()
    for epoch in range(epochs):
        net.train()
        train_loss = 0.0
        for i, data in enumerate(loader_dict['train']):
            img_inputs, labels = data[0], data[1] 
            img_inputs, labels = img_inputs.to(device), [labels[i].to(device) for i in range(1, len(labels))]  # expert labels start from index=1
            outputs, matrixA_list = net(img_inputs)
            outputs = torch.softmax(outputs, dim=1)
            optimizer.zero_grad()
            loss1, loss2 = 0, 0
            for r in range(len(labels)):
                cur_transition = torch.nn.functional.softmax(matrixA_list[r], dim=1)
                pred = torch.matmul(outputs, cur_transition)
                loss1 += weight_CE[r] * criterion(torch.log(pred), labels[r])  # loss for the r-th annotator
                loss2 += hyper_parameter * torch.trace(cur_transition)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            train_loss += loss.item() / len(loader_dict['train'])

        val_loss, val_acc = cal_loss_acc_with_aux_output(loader_dict['val'], device, net,
                                                         use_label=0) # val acc is evaluated with golden labels

        test_loss, test_acc = cal_loss_acc_with_aux_output(loader_dict['test'], device, net,
                                                           use_label=0)  # test acc is evaluated with golden labels

        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        # use validation set to select the best model
        if val_acc > best_acc_val:
            best_acc_test = test_acc
            best_acc_val = val_acc
            model_best = deepcopy(net)

        print(f' Epoch | All epochs: {epoch} | {epochs}')
        print(f' Train Loss: {train_loss:.3f}')
        print(f' Current Vali Loss | Vali Accuracy | {val_loss:.3f} | {val_acc:.3f}')
        print(f' Current Test Loss | Test Accuracy | {test_loss:.3f} | {test_acc:.3f}')
        print(f' Best Vali Acc | Test Acc  | {best_acc_val:.3f} | {best_acc_test:.3f}')
        print("-" * 70)

    if save_model_path:
        torch.save(net.state_dict(), save_model_path)

    return model_best, best_acc_test


def train_network_mjv(epochs, loader_dict, net, device, optimizer, hard_or_soft='soft', save_model_path=''):
    test_acc_list, test_loss_list = [], []
    val_acc_list, val_loss_list = [], []
    model_best, best_acc_val, best_acc_test = None, 0, 0
    for epoch in range(epochs):
        net.train()
        train_loss = 0.0
        for i, data in enumerate(loader_dict['train']):
            img_inputs, labels = data[0], data[1]  
            img_inputs, labels = img_inputs.to(device), [labels[i].to(device) for i in range(1, len(labels))] # expert labels start from index=1
            outputs = net(img_inputs)
            optimizer.zero_grad()

            if hard_or_soft == 'soft':
                y_train = torch.zeros((labels[0].size(0), NumClass)).to(device)
                for k in range(len(labels)):
                    y_train += F.one_hot(labels[k], NumClass) / len(labels)
                criterion = torch.nn.KLDivLoss()
                loss = criterion(torch.nn.functional.log_softmax(outputs, dim=1), y_train)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() / len(loader_dict['train'])

        val_loss, val_acc = cal_loss_acc(loader_dict['val'], device, net,
                                         use_label=0)

        test_loss, test_acc = cal_loss_acc(loader_dict['test'], device, net,
                                           use_label=0)

        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        # use validation set to select the best model
        if val_acc > best_acc_val:
            best_acc_test = test_acc
            best_acc_val = val_acc
            model_best = deepcopy(net)

        # please note: since in training, we are using the annotator's labels (some of them might be wrong). So in training process, it is normal to see the train loss is decreasing, but the vali and test loss increase.

        print(f' Epoch | All epochs: {epoch} | {epochs}')
        print(f' Train Loss: {train_loss:.3f}')
        print(f' Current Vali Loss | Vali Accuracy | {val_loss:.3f} | {val_acc:.3f}')
        print(f' Current Test Loss | Test Accuracy | {test_loss:.3f} | {test_acc:.3f}')
        print(f' Best Vali Acc | Test Acc  | {best_acc_val:.3f} | {best_acc_test:.3f}')
        print("-" * 70)

    if save_model_path:
        torch.save(net.state_dict(), save_model_path)

    return model_best, best_acc_test


def train_network_wdn(epochs, loader_dict, net, device, weight, optimizer, optimizer2, save_model_path=''):
    test_acc_list, test_loss_list = [], []
    val_acc_list, val_loss_list = [], []
    model_best, best_acc_val, best_acc_test = None, 0, 0

    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        net.train()
        train_loss = 0.0
        for i, data in enumerate(loader_dict['train']):
            img_inputs, labels = data[0], data[1]  
            img_inputs, labels = img_inputs.to(device), [labels[i].to(device) for i in range(1, len(labels))] # expert labels start from index=1
            output_list = net(img_inputs)
            optimizer.zero_grad()
            loss = 0
            for r in range(len(labels)):
                loss += criterion(output_list[r], labels[r])
            loss.backward()
            optimizer.step()
            train_loss += loss.item() / len(loader_dict['train'])

    # phase2: learn weight and select best model
    for epoch in range(int(epochs/8)):  # learning weight doesn't require too many epochs
        net.eval()
        train_loss = 0.0
        for i, data in enumerate(loader_dict['train']):
            img_inputs, labels = data[0], data[1]  
            img_inputs, labels = img_inputs.to(device), [labels[i].to(device) for i in range(1, len(labels))] # expert labels start from index=1
            output_list = net(img_inputs)
            outputs = 0
            y_train = torch.zeros((labels[0].size(0), NumClass)).to(device)
            for r in range(len(labels)):
                outputs += torch.softmax(output_list[r], dim=1) * weight[0][r]
                y_train += F.one_hot(labels[r], NumClass) / len(labels)
            criterion = torch.nn.KLDivLoss()
            optimizer2.zero_grad()
            loss = criterion(torch.nn.functional.log_softmax(outputs, dim=1), y_train)
            loss.backward()
            optimizer2.step()
            train_loss += loss.item() / len(loader_dict['train'])

        val_loss, val_acc = cal_loss_acc_wdn(loader_dict['val'], device, weight, net,
                                             use_label=0)

        test_loss, test_acc = cal_loss_acc_wdn(loader_dict['test'], device, weight, net,
                                               use_label=0)

        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        # use validation set to select the best model
        if val_acc > best_acc_val:
            best_acc_test = test_acc
            best_acc_val = val_acc
            model_best = deepcopy(net)

        print(f' Epoch | All epochs: {epoch} | {int(epochs/8)}')
        print(f' Train Loss: {train_loss:.3f}')
        print(f' Current Vali Loss | Vali Accuracy | {val_loss:.3f} | {val_acc:.3f}')
        print(f' Current Test Loss | Test Accuracy | {test_loss:.3f} | {test_acc:.3f}')
        print(f' Best Vali Acc | Test Acc  | {best_acc_val:.3f} | {best_acc_test:.3f}')
        print("-" * 70)

    if save_model_path:
        torch.save(net.state_dict(), save_model_path)

    return model_best, best_acc_test


def train_network_trivial(epochs, loader_dict, criterion, net, device, optimizer, use_label, save_model_path=''):
    test_acc_list, test_loss_list = [], []
    val_acc_list, val_loss_list = [], []
    model_best, best_acc_val, best_acc_test = None, 0, 0
    for epoch in range(epochs):
        net.train()
        train_loss = 0.0
        for i, data in enumerate(loader_dict['train']):
            img_inputs, labels = data[0], data[1]
            img_inputs, labels = img_inputs.to(device), labels[use_label].to(device)
            outputs = net(img_inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() / len(loader_dict['train'])

        val_loss, val_acc = cal_loss_acc(loader_dict['val'], device, net,
                                         use_label=0)

        test_loss, test_acc = cal_loss_acc(loader_dict['test'], device, net,
                                           use_label=0)

        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        # use validation set to select the best model
        if val_acc > best_acc_val:
            best_acc_test = test_acc
            best_acc_val = val_acc
            model_best = deepcopy(net)

        # please note: since in training, we are using the annotator's labels (which some of them might be wrong). So in training process, it is normal to see the train loss is decreasing, but the vali and test loss increase.

        print(f' Epoch | All epochs: {epoch} | {epochs}')
        print(f' Train Loss: {train_loss:.3f}')
        print(f' Current Vali Loss | Vali Accuracy | {val_loss:.3f} | {val_acc:.3f}')
        print(f' Current Test Loss | Test Accuracy | {test_loss:.3f} | {test_acc:.3f}')
        print(f' Best Vali Acc | Test Acc  | {best_acc_val:.3f} | {best_acc_test:.3f}')
        print("-" * 70)

    if save_model_path:
        torch.save(net.state_dict(), save_model_path)

    return model_best, best_acc_test


def train_network_mbem(epochs, loader_dict, net, device, optimizer, Niter, num_expert, save_model_path=''):
    test_acc_list, test_loss_list = [], []
    val_acc_list, val_loss_list = [], []
    model_best, best_acc_val, best_acc_test = None, 0, 0
    criterion = EBEMLoss()

    # initialize P_matrix for each epoch
    P_matrix_list = []
    for i, data in enumerate(loader_dict['train']):
        labels = data[1]  
        labels = [labels[i].to(device) for i in range(1, len(labels))] # expert labels start from index=1
        tmp = 0
        for r in range(len(labels)):
            tmp += F.one_hot(labels[r], NumClass) / len(labels)
        P_matrix_list.append(tmp)

    # goes into iteration
    for cur_t in range(Niter):
        for epoch in range(epochs + 2):  # the last epoch is used to predict
            if epoch <= epochs - 1:  # train for 0 ~ epochs-1. learn predictor function f
                train_loss = 0.
                net.train()
                for i, data in enumerate(loader_dict['train']):
                    image, labels = data[0], data[1]  # expert labels start from index=1
                    image, labels = image.to(device), [labels[i].to(device) for i in range(1, len(labels))]

                    # train classifier with the corresponding P_matrix
                    outputs = net(image)
                    optimizer.zero_grad()
                    cur_P_matrix = P_matrix_list[i].detach().clone()
                    loss = criterion(outputs, cur_P_matrix)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() / len(loader_dict['train'])
            elif epoch == epochs:  # Update qk_vec and Pi_list. this epoch doesn't train NN, just provide a chance to iterate through the dataset
                with torch.no_grad():
                    net.eval()
                    t_list = []
                    Pi_list = [torch.zeros((NumClass, NumClass)).to(device) for _ in range(num_expert)]
                    qk_vec = torch.zeros((NumClass,)).to(device)
                    for i, data in enumerate(loader_dict['train']):
                        image, labels = data[0], data[1]  # expert labels start from index=1
                        image, labels = image.to(device), [labels[i].to(device) for i in range(1, len(labels))]

                        outputs = net(image)
                        _, predicted = torch.max(outputs, 1)
                        onehot_pred = F.one_hot(predicted, NumClass)
                        t_list.append(predicted)
                        qk_vec += torch.sum(onehot_pred, dim=0)
                        for r in range(num_expert):
                            Pi_list[r] += torch.matmul(onehot_pred.T.float(), F.one_hot(labels[r], NumClass).float())
                    # Normalize Pi_list and qk_vec
                    for r in range(num_expert):
                        Pi_list[r] /= qk_vec.view(NumClass, 1).repeat(1, NumClass)
                    qk_vec /= torch.sum(qk_vec)
            else:  # Update P_matrix_list. this epoch doesn't train NN, just provide a chance to iterate through the dataset
                with torch.no_grad():

                    # Update P_matrix_list
                    for i, data in enumerate(loader_dict['train']):
                        labels = data[1]  # expert labels start from index=1
                        labels = [labels[i].to(device) for i in range(1, len(labels))]
                        batch_size = labels[0].size(0)
                        tmp = torch.ones((labels[0].size(0), NumClass)).to(device)
                        for r in range(num_expert):
                            tmp *= torch.matmul(F.one_hot(labels[r], NumClass).float(), Pi_list[r].T)
                        tmp *= qk_vec.view(1, NumClass).repeat(batch_size, 1)
                        P_matrix_list[i] = tmp / torch.sum(tmp, dim=1).view(batch_size, 1).repeat(1, NumClass)

        val_loss, val_acc = cal_loss_acc(loader_dict['val'], device, net,
                                         use_label=0)

        test_loss, test_acc = cal_loss_acc(loader_dict['test'], device, net,
                                           use_label=0)

        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        # use validation set to select the best model
        if val_acc > best_acc_val:
            best_acc_test = test_acc
            best_acc_val = val_acc
            model_best = deepcopy(net)

        # please note: since in training, we are using the annotator's labels (which some of them might be wrong). So in training process, it is normal to see the train loss is decreasing, but the vali and test loss increase.

        print(f' Iter | All Iter: {cur_t} | {Niter}')
        print(f' Train Loss: {train_loss:.3f}')
        print(f' Current Vali Loss | Vali Accuracy | {val_loss:.3f} | {val_acc:.3f}')
        print(f' Current Test Loss | Test Accuracy | {test_loss:.3f} | {test_acc:.3f}')
        print(f' Best Vali Acc | Test Acc  | {best_acc_val:.3f} | {best_acc_test:.3f}')
        print("-" * 70)

    if save_model_path:
        torch.save(net.state_dict(), save_model_path)

    return model_best, best_acc_test


def cal_loss_acc(loader, device, net, use_label):
    correct, v_loss, total = 0, 0, 0
    criterion = torch.nn.CrossEntropyLoss() # fix to CE loss
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            img_inputs, labels = data[0], data[1]
            img_inputs, labels = img_inputs.to(device), labels[use_label].to(device)
            total += labels.size(0)
            outputs = net(img_inputs)
            _, predicted = torch.max(outputs.detach(), 1)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)  
            v_loss += loss.item() / len(loader)
        val_acc = 100 * correct / total
    return v_loss, val_acc


def cal_loss_acc_wdn(loader, device, weight, net, use_label):
    correct, v_loss, total = 0, 0, 0
    criterion = torch.nn.CrossEntropyLoss() # fix to CE loss
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            img_inputs, labels = data[0], data[1]
            img_inputs, labels = img_inputs.to(device), labels[use_label].to(device)

            total += labels.size(0)
            outputs = net(img_inputs)
            pred = torch.zeros(outputs[0].size()).to(device)
            for r in range(torch.numel(weight[0])):
                pred += torch.softmax(outputs[r], dim=1) * weight[0][r]
            _, predicted = torch.max(pred.detach(), 1)
            correct += (predicted == labels).sum().item()
            loss = criterion(pred, labels)
            v_loss += loss.item() / len(loader)
        val_acc = 100 * correct / total
    return v_loss, val_acc


def cal_loss_acc_with_aux_output(loader, device, net, use_label):
    correct, v_loss, total = 0, 0, 0
    criterion = torch.nn.CrossEntropyLoss() # fix to CE loss
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            img_inputs, labels = data[0], data[1]
            img_inputs, labels = img_inputs.to(device), labels[use_label].to(device)
            total += labels.size(0)
            outputs, *_ = net(img_inputs)
            _, predicted = torch.max(outputs.detach(), 1)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)  
            v_loss += loss.item() / len(loader)
        val_acc = 100 * correct / total
    return v_loss, val_acc


if __name__ == '__main__':
    # test generate_permuation_matrix
    wrk = generate_permutation_matrix(4, 3)
    print(wrk.size())
    print(wrk[0])
    print(wrk[1])
    print(wrk[2])
