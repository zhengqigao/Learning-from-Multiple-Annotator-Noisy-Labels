# Learning-from-Multiple-Annotator-Noisy-Labels

This repo contains code for our paper 'Learning from Multiple Annotator Noisy Labels via Sample-wise Label Fusion' published on ECCV 2022. If you find it useful, please cite our paper. More info about the first author could be found on his homepage: https://zhengqigao.github.io/.


```
@inproceedings{gao2022learning,
  title={Learning from Multiple Annotator Noisy Labels via Sample-wise Label Fusion},
  author={Gao, Zhengqi and Sun, Fan-Keng and Yang, Mingran and Xiong, Zikai and Engeler, Marc and Burazer, Antonio and Wilding, Linda and Daniel, Luca and S. Boning, Duane},
  booktitle={2022 European Conference on Computer Vision},
  year={2022}
}
```

**Detailed instructions have been provided under each folder.** We suggest readers first going through the code in the TwoMoon folder. It is a bare-bone implementation covering the key ideas and code snippet of our method. After reading it, readers should be able to re-implement our method.


Note that we haven't provided code for cifar-100 and imagenet-100, because the code are almost the same as those used in MNIST. The most notable differences are: (i) for our method, we use hyper-parameter ```num_basis=150``` in cifar-100 and imagenet-100 because they have ```K=100``` classes; (ii) We use hard majority voting (instead of soft majority voting) in cifar-100 and imagenet-100. Because when the number of classes increases, the accuracy of soft majority voting degrades.


## Acknowledgement

This research is supported by Millennium Pharmaceuticals, Inc. (a subsidiary of Takeda Pharmaceuticals).
