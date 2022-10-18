## Introduction
This is [], source code of CHAN. It is built on top of the [VSEinf](https://github.com/woodfrog/vse_infty) in PyTorch. 

![image](https://github.com/CrossmodalGroup/NAAF/blob/main/Framework%20Overview.jpg)
## Requirements and Installation
We recommended the following dependencies.

* Python 3.7
* [PyTorch](http://pytorch.org/) 1.11.0
* [Transformers](https://github.com/huggingface/transformers) (4.18.0)
* The specific required environment can be found [here]()

## Download data
Download the dataset files. We use the image feature created by SCAN, downloaded [here](https://github.com/kuanghuei/SCAN). The vocabulary required by GloVe has been placed in the 'vocab' folder of the project (for Flickr30K and MSCOCO).

## Training

```bash
sh scripts/train_region.sh
```

## Evaluation

```bash
sh scripts/eval_region.sh
```

## Reference

If you found this code useful, please cite the following paper:
```

```

