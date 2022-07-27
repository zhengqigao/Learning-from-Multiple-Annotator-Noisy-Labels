# Learning from Multi-AnT Noisy Labels (ECCV 2022)

This repo contains code for our paper 'Learning from Multiple Annotator Noisy Labels via Sample-wise Label Fusion' published on ECCV 2022. If you find it useful, please cite our paper. More info about the first author could be found on his homepage: https://zhengqigao.github.io/.


```
@misc{gao2022learning,
      title={Learning from Multiple Annotator Noisy Labels via Sample-wise Label Fusion}, 
      author={Zhengqi Gao and Fan-Keng Sun and Mingran Yang and Sucheng Ren and Zikai Xiong and Marc Engeler and Antonio Burazer and Linda Wildling and Luca Daniel and Duane S. Boning},
      year={2022},
      eprint={2207.11327},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

**Detailed instructions have been provided under each folder.** We suggest readers first going through the code in the TwoMoon folder. It is a bare-bone implementation covering the key ideas and code snippet of our method. After reading it, readers should be able to re-implement our method.


Note that we haven't provided code for cifar-100 and imagenet-100, because the code are nearly identical as those used in MNIST. The most notable differences are: (i) for our method, we use hyper-parameter ```num_basis=150``` in cifar-100 and imagenet-100 because they have ```K=100``` classes; (ii) We use hard majority voting (instead of soft majority voting) in cifar-100 and imagenet-100. Because when the number of classes increases, the accuracy of soft majority voting degrades. For those readers looking for an quick re-implementation of our method in 2 mins, see the following psuedo code:

```
# provided train_loader, num_experts, NumClass, num_basis, and hyper_lambda
criterion = torch.nn.KLDivLoss()
for i, data in enumerate(train_loader):
            img_inputs, labels = data[0], data[1]  
            
            # assume expert labels start from index=1
            img_inputs = img_inputs.to(device)
            labels = [labels[i].to(device) for i in range(1, len(labels))] 
            
            outputs, coeff, weight_expert = net(img_inputs)
            coeff = torch.softmax(coeff, dim=2)
            weight_expert = torch.softmax(weight_expert, dim=1)

            transition_expert = (coeff.reshape(-1, num_experts, num_basis, 1, 1) * matrices).sum(dim=2)
            y_expert = torch.stack([F.one_hot(ele, NumClass) for ele in labels], dim=1).float()
            
            # Eq (3) in the paper
            y_expert_star = torch.matmul(transition_expert, y_expert.reshape(-1, num_experts, NumClass, 1))
            
            # Eq (4) in the paper
            y_train = (y_expert_star * weight_expert.reshape(-1, num_experts, 1, 1)).sum(dim=1).squeeze(-1)

            optimizer.zero_grad()

            # classification loss, Eq (5) in the paper
            tmp1 = criterion(F.log_softmax(outputs, dim=1), y_train)

            # penalty term,  Eq (6) in the paper
            diags = 1 - transition_expert.reshape(-1, NumClass, NumClass).diagonal(dim1=-2, dim2=-1)
            tmp2 = torch.sum(diags * diags) / torch.numel(diags)
            loss = tmp1 + hyper_lambda * tmp2

            loss.backward()
            optimizer.step()
```

We provide an example of dimensions for each variables here:

```
# As an example, if batch_size = 64, num_experts = 3, NumClass = 10, num_basis = 20
# outputs: (64, 10)
# coeff: (64, 3, 20)
# weight_expert: (64, 3)
# transition_expert: (64, 3, 10, 10)
# y_expert: (64, 3, 10)
# y_expert_star: (64, 3, 10, 1)
# y_train: (64, 10)
```

## Important Notes

The environment requirement is described in ```env.yaml```. Two important things are worthy to be mentioned. First, in different versions of Pytorch, the reduction 'mean' of torch.nn.KLDivLoss is performed differently: KLDivLoss divides the total loss by both the batch size and the support size, or only the batch size. Since we use torch.nn.KLDivLoss to calculate the first term of our Eq (6), this causes a problem: our choice of hyper-parameter ```lambda``` might not be the optimal choice in your environment.

Secondly, due to the randomness of loading data, when you run preprocess.py in the MNIST example, it is possible yielding completely different results from ours. In short, it is because the ```index_experts``` variable defined in preprocess.py will correspond to different images in your and our cases due to randomness of data loading. Considering this situation, to justify our results, we have provided one set of our generated data so that you could exactly reproduce our results. 

## Acknowledgement

This research is supported by Millennium Pharmaceuticals, Inc. (a subsidiary of Takeda Pharmaceuticals).
