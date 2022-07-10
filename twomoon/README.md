## Instructiosn for TwoMoon Exmaple

We highly suggest going through the TwoMoon example. It is a bare-bone implementation covering the key ideas and code snippet of our method. Specifically, we suggest readers going through the class 'ThreeLayerMLP_ours' and the function 'train_network_ours' defined in utils.py. 

```
# For reproducing our Figs. 4 and 5 in the paper
# Figures will be plotted along writing logs into the text file.
python main.py >> ./logs/log_run_once.txt
```

**Important Notes**

Around line 154 in utils.py, careful readers will observe that compared to Eq (6) written in the paper, it is slightly different. Specifically, the penalty term is not exactly the same as the one in Eq (6), being scaled by a constant term. Nevertheless, readers should also notice that since we could freely set the value of hyper-parameter lambda, this is not an issue at all.


Moreover, the readers should be warned that in different versions of Pytorch, the reduction 'mean' of torch.nn.KLDivLoss is performed differently: KLDivLoss divides the total loss by both the batch size and the support size, or only the batch size. Since we use torch.nn.KLDivLoss to calculate the first term in Eq (6), this causes a problem: our choice of lambda might not be the optimal choice in your environments.