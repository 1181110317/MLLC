# MLLC
This repository contains the implementation of the following manuscript: 
Hui Xiao, Yuting Hong, Li Dong, Diqun Yan, Jiayan Zhuang, Junjie Xiong, Dongtai Liang, Chengbin Peng. "Multi-Level Label Correction by Distilling Proximate Patterns for Semi-supervised Semantic Segmentation." IEEE Transactions on Multimedia, 2024.  
[TMM](https://ieeexplore.ieee.org/abstract/document/10462533/),  [arxiv](https://arxiv.org/abs/2404.02065).




> **Abstract.** Semi-supervised semantic segmentation relieves the reliance on large-scale labeled data by leveraging unlabeled data. Recent semi-supervised semantic segmentation approaches mainly resort to pseudo-labeling methods to exploit unlabeled data. However, unreliable pseudo-labeling can undermine the semi-supervision processes. In this paper, we propose an algorithm called Multi-Level Label Correction (MLLC), which aims to use graph neural networks to capture structural relationships in Semantic-Level Graphs (SLGs) and Class-Level Graphs (CLGs) to rectify erroneous pseudo-labels. Specifically, SLGs represent semantic affinities between pairs of pixel features, and CLGs describe classification consistencies between pairs of pixel labels. With the support of proximate pattern information from graphs, MLLC can rectify incorrectly predicted pseudo-labels and can facilitate discriminative feature representations. We design an end-to-end network to train and perform this effective label corrections mechanism. Experiments demonstrate that MLLC can significantly improve supervised baselines and outperforms state-of-the-art approaches in different scenarios on Cityscapes and PASCAL VOC 2012 datasets. Specifically, MLLC improves the supervised baseline by at least 5% and 2% with DeepLabV2 and DeepLabV3+ respectively under different partition protocols.




## Citation

```
@article{xiao2024multi,
  title={Multi-Level Label Correction by Distilling Proximate Patterns for Semi-supervised Semantic Segmentation},
  author={Xiao, Hui and Hong, Yuting and Dong, Li and Yan, Diqun and Xiong, Junjie and Zhuang, Jiayan and Liang, Dongtai and Peng, Chengbin},
  journal={IEEE Transactions on Multimedia},
  year={2024},
  publisher={IEEE}
}
```
