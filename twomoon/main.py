from utils import train_network_ours, train_network_trivial, generate_permutation_matrix, gen_data, ThreeLayerMLP_ours, \
    ThreeLayerMLP, seed
import torch
import numpy as np
import matplotlib.pyplot as plt


def run_exp(num_exp, hyper_lambda, plot=False):
    epochs, lr = 400, 0.05
    acc_list, acc2_list, acc3_list = [], [], []
    
    for j in range(num_exp):
        # actually we don't need 'y1_test' or 'y2_test' since testing is always done with golden labels (i.e., y_test in this case)
        x_train, x_test, y_train, y_test, y1_train, y1_test, y2_train, y2_test = gen_data(20000, 0.8, j, plot)

        # a feed forward network trained with 'use_label=0'
        net = ThreeLayerMLP(2)
        optimizer = torch.optim.SGD(net.parameters(), lr)
        model, acc = train_network_trivial(epochs, x_train, y1_train, y2_train, x_test, y_test, net, optimizer, use_label=0)

        # a feed forward network trained with 'use_label=1'
        net = ThreeLayerMLP(2)
        optimizer = torch.optim.SGD(net.parameters(), lr)
        model, acc2 = train_network_trivial(epochs, x_train, y1_train, y2_train, x_test, y_test, net, optimizer, use_label=1)

        # our method--- we fix num_basis = 2.
        net = ThreeLayerMLP_ours(2, 2, 2)
        optimizer = torch.optim.SGD(net.parameters(), lr)
        model, acc3 = train_network_ours(epochs, x_train, y1_train, y2_train, x_test, y_test, net, optimizer, 2, 2, hyper_lambda)

        print("In %d-th experiment" %j)
        print("acc of NN | use expert1: %f | use expert2: %f" % (acc, acc2))
        print("acc of NN | ours: %f" % (acc3))
        acc_list.append(acc)
        acc2_list.append(acc2)
        acc3_list.append(acc3)

        # Code snippet for some visualizations
        if plot:
            matrices = generate_permutation_matrix(2)
            outputs, coeff, weight_expert = model(x_train)
            pred = torch.softmax(outputs, dim=1)
            coeff = torch.softmax(coeff, dim=2)
            weight_expert = torch.softmax(weight_expert, dim=1)
            transition_expert = (coeff.reshape(-1, 2, 2, 1, 1) * matrices).sum(dim=2)

            # prediction map with decision boundary
            plt.figure()
            xx_grid, yy_grid = np.linspace(-2, 2, 100), np.linspace(-2,2,100)
            xx_wrk, yy_wrk = np.meshgrid(xx_grid, yy_grid)
            xx, yy = torch.from_numpy(xx_wrk).float().reshape(-1, 1), torch.from_numpy(yy_wrk).float().reshape(-1, 1)
            wrk = torch.cat([xx, yy], dim=1)
            outputs2, *_ = model(wrk)
            pred2 = torch.softmax(outputs2, dim=1)
            _, labels = torch.max(outputs2, dim=1)
            plt.contourf(xx_wrk, yy_wrk, labels.reshape(100, 100).detach().numpy(), cmap='Paired', alpha=0.5)
            plt.scatter(x_train[:, 0].detach().numpy(), x_train[:, 1].detach().numpy(),
                        c=pred[:, 0].detach().numpy(),
                        edgecolors='k') 
            plt.yticks([])
            plt.xticks([])
            ax = plt.gca()
            ax.spines['bottom'].set_linewidth(2)
            ax.spines['left'].set_linewidth(2)
            ax.spines['right'].set_linewidth(2)
            ax.spines['top'].set_linewidth(2)
            plt.title('prediction of ours')
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=24)

            # plot the weight ratio for expert1
            plt.figure()
            plt.scatter(x_train[:, 0].detach().numpy(), x_train[:, 1].detach().numpy(),
                        c=weight_expert[:, 0].detach().numpy(),
                        edgecolors='k', vmin = 0.43, vmax = 0.57)  
            plt.yticks([])
            plt.xticks([])
            ax = plt.gca()
            ax.spines['bottom'].set_linewidth(2)
            ax.spines['left'].set_linewidth(2)
            ax.spines['right'].set_linewidth(2)
            ax.spines['top'].set_linewidth(2)
            plt.title('Heatmap: weight of expert1')
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=24)

            # plot the weight ratio for expert2
            plt.figure()
            plt.scatter(x_train[:, 0].detach().numpy(), x_train[:, 1].detach().numpy(),
                        c=weight_expert[:, 1].detach().numpy(),
                        edgecolors='k',  vmin = 0.43, vmax = 0.57) 
            plt.yticks([])
            plt.xticks([])
            ax = plt.gca()
            ax.spines['bottom'].set_linewidth(2)
            ax.spines['left'].set_linewidth(2)
            ax.spines['right'].set_linewidth(2)
            ax.spines['top'].set_linewidth(2)
            plt.title('Heatmap: weight of expert2')
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=24)

            value1, value2 = np.zeros((x_train.size(0),)), np.zeros((x_train.size(0),))
            for i in range(x_train.size(0)):
                value1[i] = transition_expert[i, 0, y_train[i], y_train[i]].detach().numpy()
                value2[i] = transition_expert[i, 1, y_train[i], y_train[i]].detach().numpy()

            # print(value1.min(), value1.max(), value1.mean())
            # print(value2.min(), value2.max(), value2.mean())

            # plot the transition matrix of expert1, we only plot the diagonal matrix, represents the probability of the expert make correct prediction
            plt.figure()
            plt.scatter(x_train[:, 0].detach().numpy(), x_train[:, 1].detach().numpy(),
                        c=value1,
                        edgecolors='k',vmin =0.86, vmax = 0.95)  
            plt.title('Heatmap: prob of expert1 correct')
            plt.yticks([])
            plt.xticks([])
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=24)
            ax = plt.gca()
            ax.spines['bottom'].set_linewidth(2)
            ax.spines['left'].set_linewidth(2)
            ax.spines['right'].set_linewidth(2)
            ax.spines['top'].set_linewidth(2)

            # plot the transition matrix of expert2, we only plot the diagonal matrix, represents the probability of the expert make correct prediction
            plt.figure()
            plt.scatter(x_train[:, 0].detach().numpy(), x_train[:, 1].detach().numpy(),
                        c=value2,
                        edgecolors='k',vmin =0.86, vmax = 0.95)  
            plt.title('Heatmap: prob of expert2 correct')
            plt.yticks([])
            plt.xticks([])
            ax = plt.gca()
            ax.spines['bottom'].set_linewidth(2)
            ax.spines['left'].set_linewidth(2)
            ax.spines['right'].set_linewidth(2)
            ax.spines['top'].set_linewidth(2)
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=24)
            plt.show()

    acc_list, acc2_list, acc3_list = np.array(acc_list), np.array(acc2_list), np.array(acc3_list)

    print("-"*30)
    print("Results after averaging %d experiments:" %num_exp)
    print("acc of NN | use expert1: %f | use expert2: %f" % (acc_list.mean(), acc2_list.mean()))
    print("acc of NN | ours: %f" % (acc3_list.mean()))
    print("Among the %d experiments, #(acc of ours is the largest)= %d" %(num_exp, np.sum((acc3_list > acc_list) & (acc3_list > acc2_list))))
    return acc_list, acc2_list, acc3_list


if __name__ == '__main__':
    
    seed(0)
    
    # run_exp(1, hyper_lambda = 1.0, plot=True)

    # Note: we haven't performed replication of this twomoon experiment in our paper. When we do so in perparing the code, we find that setting lambda to 0.1 could give even better results.
    # Here for curious readers, uncomment the following line to repeat our experiments for 10 times. Mean accuracy will be reported. Run the commad: python main.py 
    run_exp(10, hyper_lambda=0.1, plot=False)


    