## Instructions


### Using our generated data (Reproduce our results)

We have provided one set of our generated MNIST data in Google drive: https://drive.google.com/drive/folders/1RTF9TJMZ-OZm1-oX5mVuslD-P1VOZqOU?usp=sharing. It corresponds to ```epsilon=34``` in Table 1 of our paper. 

Please directly download our provided data, and unzip and put it under this ```$pj_dir/data/``` folder. You should see a tree structure looks like:


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

Then running the following commands to reproduce our results reported in Table I:

```
cd $pj_dir/exp/
./exp_reproduce.sh
```

### Using your generated data (May lead to different results)

To generate data by yourself, please download the original MNIST data to this ```$pj_dir/data``` folder and next run ```python ./utils/preprocess.py``` with the correct command line arguments. Note that in preprocess.py, you actually could set ```download=True``` when calling torchvision.datasets.MNIST, and it will first download the data and then generate the experts' labels. If you want to test your own expert labeling method, please assign your way a new ```expert_type``` parameter, and update those ```if``` branches in the ```ExpertGenerator``` class of preprocess.py. Running the following command will give you a feelign of the overall flow.

```
./exp/exp_overall_flow.sh
```

Running preprocess.py will (i) first load the original MNIST data, (ii) synthesize multiple experts' labels, (iii) and then save the data to the ```\$pj_dir/data/``` folder. However, due to the randomness of loading data in step (i), when you run this shell script, it is possible yielding completely different results from ours. **In short, it is because the ```index_experts``` variable defined in preprocess.py will correspond to different images in your and our cases due to randomness of data loading.** Considering this situation, to justify our results, we have provided one set of our generated data so that you could exactly reproduce our results. 
