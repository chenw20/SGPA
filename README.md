# Sparse Gaussian Process Attention
This is an example code for the paper titled [Calibrating Transformers via Sparse Gaussian Processes (ICLR 2023)](https://openreview.net/pdf?id=jPVAFXHlbL)

This code implememts SGPA on CIFAR10 and CoLA datasets.

To use this code: simply run train_cifar.py or train_cola.py

The CoLA dataset can be downloaded [here](https://nyu-mll.github.io/CoLA/)

Dependencies:
- Python - 3.8
- Pytorch - 1.10.2
- numpy - 1.22.4
- einops - 0.4.1
- allennlp - 2.9.3

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
