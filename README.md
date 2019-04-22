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

You can easily install BERTScore by `pip install [placeholder]`.

### Usage

#### Metric
We provide a command line interface(CLI) of BERTScore as well as a python module. 

For the CLI, you can use it by
```
bert_score refs.txt hyps.txt
```
See more options by `man bert_score`.

For the python module, please refer to the docstring of `bert_score.score` in `src/bert_score.py`.
We also provide an interactive [example]().

#### Visualization
Because BERTScore measure sentence similarity by accumulating word similarites,
we can visualize it easily.

Below is an example where we visualize the pairwise cosine similarity of words
in the reference and candidate sentences.
<!-- ![]() -->

### Acknowledgement
This repo wouldn't be possible without the awesome [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT).
