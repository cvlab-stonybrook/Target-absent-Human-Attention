# Target-absent-Human-Attention (ECCV2022)
Offical PyTorch implementation of the paper [Target-absent Human Attention](https://arxiv.org/abs/2207.01166) (ECCV2022)

The prediction of human gaze behavior is important for building human-computer interactive systems that can anticipate a user's attention. Computer vision models have been developed to predict the fixations made by people as they search for target objects. But what about when the image has no target? Equally important is to know how people search when they cannot find a target, and when they would stop searching. In this paper, we propose the first data-driven computational model that addresses the search-termination problem and predicts the scanpath of search fixations made by people searching for targets that do not appear in images. We model visual search as an imitation learning problem and represent the internal knowledge that the viewer acquires through fixations using a novel state representation that we call Foveated Feature Maps (FFMs). FFMs integrate a simulated foveated retina into a pretrained ConvNet that produces an in-network feature pyramid, all with minimal computational overhead. Our method integrates FFMs as the state representation in inverse reinforcement learning. Experimentally, we improve the state of the art in predicting human target-absent search behavior on the COCO-Search18 dataset

If you are using this work, please cite:
```bibtex
@InProceedings{Yang_2022_ECCV_target,
author = {Yang, Zhibo, Sounak Mondal, Seoyoung Ahn, Gregory Zelinsky, Minh Hoai, and Dimitris Samaras},
title = {Target-absent Human Attention},
booktitle = {The European conference on computer vision (ECCV)},
month = {October},
year = {2022}
}
```

## Scripts
- Train a model with
    ```
    python train.py --hparams <hparams-path> --dataset_root <dataset-path>
    ```
- Evaluation (to be updated)

    
## Data Preparation
We follow the settings in [IRL (Yang et al., CVPR2020](https://github.com/cvlab-stonybrook/Scanpath_Prediction) and use an action space of 20x32 and [COCO-Search18](https://sites.google.com/view/cocosearch/home) as the training and evaluation dataset.


```

