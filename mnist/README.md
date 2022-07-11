## Instructiosn for MNIST Exmaple


The structure of our code should (hopefully) be clear to readers. In ```$pj_dir/utils/```, we have defined the network structures in model.py, data loader in dataloder.py, loss functions in loss.py, training procedures in helper.py, and most importantly, experts' labels synthesis in preprocess.py (More details for this part, please refer to ```$pj_dir/data/data.md$```). Each main file with the name main_train_xxx.py will correspondingly call the function 'train_network_xxx' defgined in ```$pj_dir/utils/helper.py```.

Readers should be warned that in different versions of Pytorch, the reduction 'mean' of torch.nn.KLDivLoss is performed differently: KLDivLoss divides the total loss by both the batch size and the support size, or only the batch size. Since we use torch.nn.KLDivLoss to calculate the first term of our Eq (6), this causes a problem: our choice of hyper-parameter ```lambda``` might not be the optimal choice in your environment.