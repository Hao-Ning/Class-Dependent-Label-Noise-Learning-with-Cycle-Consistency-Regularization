# Class-Dependent Label-Noise Learning with Cycle-Consistency Regularization
## Abstract

​	In label-noise learning, estimating the transition matrix plays an important role in building statistically consistent classifier. Current state-of-the-art consistent estimator for the transition matrix has been developed under the newly proposed sufficiently scattered assumption, through incorporating the minimum volume constraint of the transition matrix $T$ into label-noise learning. To compute the volume of $T$, it heavily relies on the estimated noisy class posterior. However, the estimation error of the noisy class posterior could usually be large as deep learning methods tend to easily overfit the noisy labels. Then, directly minimizing the volume of such obtained $T$ could lead the transition matrix to be poorly estimated.  Therefore, how to reduce the side-effects of the inaccurate noisy class posterior has become the bottleneck of such method. In this paper, we creatively propose to estimate the transition matrix under the forward-backward cycle-consistency regularization, of which we have greatly reduced the dependency of estimating the transition matrix $T$ on the noisy class posterior. We show that the cycle-consistency regularization helps to minimize the volume of the transition matrix $T$ indirectly without exploiting the estimated noisy class posterior, which could further encourage the estimated transition matrix $T$ to converge to its optimal solution. Extensive experimental results consistently justify the effectiveness of the proposed method, on reducing the estimation error of the transition matrix and greatly boosting the classification performance.

## Dependencies
We implement our methods by PyTorch on NVIDIA RTX 3090 Ti. The environment is as bellow:
- [Ubuntu 20.04 Desktop](https://ubuntu.com/download)
- [PyTorch](https://PyTorch.org/), version >= 1.9.0
- [CUDA](https://developer.nvidia.com/cuda-downloads), version >= 11.1
- [Anaconda3](https://www.anaconda.com/)

## Experiments
We verify the effectiveness of the proposed method on two  synthetic noisy datasets(CIFAR-10, CIFAR-100), and two real-world noisy dataset (clothing1M and Food101N).    Here is an example: 

```bash
python main.py --dataset cifar10 --noise_rate 0.3 --lam 0.3
```

If you find this code useful in your research, please cite :

```bash
@inproceedings{ ,
  title={Class-Dependent Label-Noise Learning with Cycle-Consistency Regularization},
  author={De Cheng, Yixiong Ning, Nannan Wang, Xinbo Gao, Heng Yang, Yuxuan Du, Bo Han, Tongliang Liu},
  booktitle={NIPS},
  year={2022}
}
```
