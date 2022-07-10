## Instructions

We have provided one set of generated MNIST data in Google drive: https://drive.google.com/drive/folders/1RTF9TJMZ-OZm1-oX5mVuslD-P1VOZqOU?usp=sharing. It corresponds to epsilon=34 in Table 1 of our paper. 

Please directly download the source data, and unzip and put it under this ./data folder. You should see a tree structure looks like:


```
data
 └─── MNIST_expertlabels_type0
        └─── train
            | data_1.pt, label_1.pt, ...
        └─── test
            | data_1.pt, label_1.pt, ...
        └─── val
            | data_1.pt, label_1.pt, ...
```

To generate data by yourself, please download the original MNIST data to this ./data folder and next run preprocess.py. Note that in preprocess.py, you actually could achieve this goal easily by taking advantage of our code: you could set ```download=True``` when calling torchvision.datasets.MNIST, and it will first download the data and then generate the experts' labels. If you want to test your own expert labeling method, please assign your way a new ```expert_type``` parameter, and update those ```if``` branches in the ```ExpertGenerator``` class of preprocess.py.

 The overall running flow is shown in the ./exp/exp_overall_flow.sh file.