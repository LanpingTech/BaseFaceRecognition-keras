# BaseFaceRecognition-keras

## Introduction

This is a simple implementation of face recognition using keras. 

## Dataset

The dataset is from [CASIA-WebFace](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html). It contains 494414 images of 10575 identities. The images are cropped and aligned with 5 landmarks.

## Model

The model is based on [FaceNet](https://arxiv.org/pdf/1503.03832.pdf). Its backbone is [MobileNet](https://arxiv.org/abs/1704.04861). The model is trained with triplet loss.

## Usage

### Train

```bash
python train.py
```

### Predict

```bash
python predict.py
```

## Reference

[1] [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf)

[2] [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)