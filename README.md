# Sparse Gaussian Process Attention
This is an example code for the paper titled [Calibrating Transformers via Sparse Gaussian Processes (ICLR 2023)](https://arxiv.org/abs/2303.02444)

This code implememts SGPA on CIFAR10 and IMDB datasets.

To use this code: simply run train_cifar.py or train_imdb.py

The IMDB dataset can be downloaded [here](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

Dependencies:
- Python - 3.8
- Pytorch - 1.10.2
- numpy - 1.22.4
- einops - 0.4.1
- pandas - 1.4.3
- transformers - 4.18.0

ECE/MCE reported in the paper are computed according to this [script](https://colab.research.google.com/drive/1H_XlTbNvjxlAXMW5NuBDWhxF3F2Osg1F?usp=sharing#scrollTo=w1SAqFR7wPvs). Note according to this script, ECE/MCE are computed based on the differences between predicted probabilities and the labels for all classes (not just the max-prob class).

## Citing the paper (bib)
```
@inproceedings{chen2023calibrating,
  title = {Calibrating Transformers via Sparse Gaussian Processes},
  author = {Chen, Wenlong and Li, Yingzhen},
  booktitle = {International Conference on Learning Representations},
  year = {2023}
}
```
