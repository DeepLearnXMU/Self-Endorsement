# Improving LLM Generations via Fine-Grained Self-Endorsement

This repository contains the code for the paper "Improving LLM Generations via Fine-Grained Self-Endorsement" (Findings of ACL 2024).

## Table of Contents

1. [Installation](#installation)
2. [Dataset](#dataset)
3. [Inference](#inference)
4. [Evaluation](#evaluation)
5. [Citation](#citation)

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Dataset

The dataset used in this paper can be downloaded from the following link:

```
https://github.com/shmsw25/FActScore
https://nlp.cs.washington.edu/triviaqa
https://github.com/openai/grade-school-math
```

## Inference

For inferencing (+generate), run the following command:

```bash
TEXT="Tell me a bio of Michael Jackson"
MODEL_PATH="/path/to/your/model"
python3 main.py $TEXT $MODEL_PATH
```

You can modify the hyperparameters in the `main.py` script as per your requirements.

## Evaluation

We follow Factscore to evaluate the factuality of generated text:

```
https://github.com/shmsw25/FActScore
```

## Citation

If you find this repository useful, please cite our paper:

```
@article{wang2024fine,
  title={Fine-Grained Self-Endorsement Improves Factuality and Reasoning},
  author={Wang, Ante and Song, Linfeng and Peng, Baolin and Tian, Ye and Jin, Lifeng and Mi, Haitao and Su, Jinsong and Yu, Dong},
  journal={arXiv preprint arXiv:2402.15631},
  year={2024}
}
```