# BERTScore
Automatic Evaluation Metric described in the paper [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org).

#### Authors:
* [Tianyi Zhang](https://scholar.google.com/citations?user=OI0HSa0AAAAJ&hl=en)*
* [Varsha Kishore]()*
* [Felix Wu](https://scholar.google.com.tw/citations?user=sNL8SSoAAAAJ&hl=en)*
* [Kilian Q. Weinberger](http://kilian.cs.cornell.edu/index.html)
* [Yoav Artzi](https://yoavartzi.com/)

*: Equal Contribution

### Overview
BERTScore leverages the pre-trained contextual embeddings from BERT and matches
words in candidate and reference sentences by cosine similarity.
It has been shown to correlate with human judgment on setence-level and
system-level evaluation.
Moreover, BERTScore computes precision, recall, and F1 measure, which can be
useful for evaluating different language generation tasks.

For an illustration, BERTScore precision can be computed as
![](https://github.com/Tiiiger/bert_score/blob/master/bert_score.png "BERTScore")

### Installation

You can easily install BERTScore by:
```sh
pip install [placeholder]
```
Install it from the source by:
```sh
git clone https://github.com/Tiiiger/bert_score
cd bert_score
pip install .
```

### Usage

#### Metric
We provide a command line interface(CLI) of BERTScore as well as a python module. 
For the CLI, you can use it as follows:
1. To evaluate English text files:

```sh
bert-score -r refs.txt -c hyps.txt --bert bert-base-uncased 
```
2. To evaluate Chinese text files:

```sh
bert-score -r refs.txt -c hyps.txt --bert bert-base-chinese
```
3. To evaluate text files in other languages:

```sh
bert-score -r refs.txt -c hyps.txt
```
See more options by `bert-score -h`.

For the python module, please refer to the docstring of `bert_score.score` in `src/bert_score.py`.
We also provide an interactive [example]().

#### Visualization
Because BERTScore measure sentence similarity by accumulating word similarites,
we can visualize it easily.

Below is an example where we visualize the pairwise cosine similarity of words
in the reference and candidate sentences.
<!-- ![]() -->

### Acknowledgement
This repo wouldn't be possible without the awesome [bert](https://github.com/google-research/bert) and [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT).
