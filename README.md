# BERTScore
Metric described in the paper [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org).

#### Authors:
* [Tianyi Zhang](https://scholar.google.com/citations?user=OI0HSa0AAAAJ&hl=en)*
* [Varsha Kishore]()*
* [Felix Wu](https://scholar.google.com.tw/citations?user=sNL8SSoAAAAJ&hl=en)*
* [Kilian Q. Weinberger](http://kilian.cs.cornell.edu/index.html)
* [Yoav Artzi](https://yoavartzi.com/)

*: Equal Contribution

### Overview
This repo contains an example implementation of the evaluation metric BERTScore, described in the paper BERTScore: Evaluating Text Generation with BERT.

BERTScore is used to evaluate similarity between candidate and corresponding reference sentences. BERTScore leverages pre-trained contextual embeddings from BERT and greedily matches similarity between words pieces in the candidate and word pieces in the refeerence to compute the final score.

### Installation


### Usage


### Acknowledgement
This repo wouldn't be possible without the awesome [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT).
