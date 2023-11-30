<div align="center">

# SeSDF: Self-evolved Signed Distance Field for Implicit 3D Clothed Human Reconstruction

<a href="https://yukangcao.github.io/">Yukang Cao</a>, <a href="https://www.kaihan.org/">Kai Han</a>, <a href="https://i.cs.hku.hk/~kykwong/">Kwan-Yee K. Wong</a>

[![Paper](http://img.shields.io/badge/Paper-arxiv.2306.03038-B31B1B.svg)](https://arxiv.org/abs/2304.00359)
<a href="https://yukangcao.github.io/SeSDF/"><img alt="page" src="https://img.shields.io/badge/Webpage-0054a6?logo=Google%20chrome&logoColor=white"></a>

<img src="./docs/teaser.png">

Please refer to our webpage for more visualizations.
</div>

## Abstract
We address the problem of clothed human reconstruction from a single image or uncalibrated multi-view images. Existing methods struggle with reconstructing detailed geometry of a clothed human and often require a calibrated setting for multi-view reconstruction. We propose a flexible framework which, by leveraging the parametric SMPL-X model, can take an arbitrary number of input images to reconstruct a clothed human model under an uncalibrated setting. At the core of our framework is our novel self-evolved signed distance field (SeSDF) module which allows the framework to learn to deform the signed distance field (SDF) derived from the fitted SMPL-X model, such that detailed geometry reflecting the actual clothed human can be encoded for better reconstruction. Besides, we propose a simple method for self-calibration of multi-view images via the fitted SMPL-X parameters. This lifts the requirement of tedious manual calibration and largely increases the flexibility of our method. Further, we introduce an effective occlusion-aware feature fusion strategy to account for the most useful features to reconstruct the human model. We thoroughly evaluate our framework on public benchmarks, and our framework establishes a new state-of-the-art.

<img src=""./docs/pipeline.png">


## Code
We will release the code soon... 🚧 Please stay tuned!

## Misc.
If you want to cite our work, please use the following bib entry:
```
@inproceedings{cao2023sesdf,
author    = {Yukang Cao and Kai Han and Kwan-Yee K. Wong},
title     = {SeSDF: Self-evolved Signed Distance Field for Implicit 3D Clothed Human Reconstruction},
booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year      = {2023},
}
```
