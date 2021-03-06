#!/usr/bin/env python
import os
import time
import argparse
import torch
from collections import defaultdict
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

import bert_score

VERSION=bert_score.__version__

def main():
    torch.multiprocessing.set_sharing_strategy('file_system')

    parser = argparse.ArgumentParser('Calculate BERTScore')
    parser.add_argument('--bert', default='bert-base-multilingual-cased',
                        choices=bert_score.bert_types, help='BERT model name (default: bert-base-uncased)')
    parser.add_argument('-l', '--num_layers', default=9, type=int, help='use first N layer in BERT (default: 9)')
    parser.add_argument('-b', '--batch_size', default=64, type=int, help='batch size (default: 64)')
    parser.add_argument('--no_idf', action='store_true', help='BERT Score without IDF scaling')
    parser.add_argument('-s', '--seg_level', action='store_true', help='show individual score of each pair')
    parser.add_argument('-v', '--verbose', action='store_true', help='increase output verbosity')
    parser.add_argument('-r', '--ref', required=True, type=str, help='reference file path or a string')
    parser.add_argument('-c', '--cand', required=True, type=str, help='candidate (system outputs) file path or a string')

    args = parser.parse_args()

    if os.path.isfile(args.cand) and os.path.isfile(args.ref):
        with open(args.cand) as f:
            cands = [line.strip() for line in f]

        with open(args.ref) as f:
            refs = [line.strip() for line in f]
    else:
        cands = [args.cand]
        refs = [args.ref]
        assert args.no_idf, "do not suuport idf fold for a single pair of sentences"

    assert len(cands) == len(refs)

    if args.verbose:
        print('loading BERT model...')
    tokenizer = BertTokenizer.from_pretrained(args.bert)
    model = BertModel.from_pretrained(args.bert)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # drop unused layers
    model.encoder.layer = torch.nn.ModuleList([layer for layer in model.encoder.layer[:args.num_layers]])


    if args.no_idf:
        idf_dict = defaultdict(lambda: 1.)
    else:
        if args.verbose:
            print('preparing IDF dict...')
        start = time.perf_counter()
        idf_dict = bert_score.get_idf_dict(refs, tokenizer)
        if args.verbose:
            print('done in {:.2f} seconds'.format(time.perf_counter() - start))
    if args.verbose:
        print('calculating scores...')
    start = time.perf_counter()
    all_preds = bert_score.bert_cos_score_idf(model, refs, cands, tokenizer, idf_dict, device=device,
                                                  batch_size=args.batch_size)
    avg_scores = all_preds.mean(dim=0)
    P = avg_scores[0].cpu().item()
    R = avg_scores[1].cpu().item()
    F1 = avg_scores[2].cpu().item()
    if args.verbose:
        print('done in {:.2f} seconds'.format(time.perf_counter() - start))
    msg = '{}_L{}{}_version={} BERT-P: {:.6f} BERT-R: {:.6f} BERT-F1: {:.6f}'.format(
        args.bert, args.num_layers, '_no-idf' if args.no_idf else '', VERSION, P, R, F1)
    print(msg)
    if args.seg_level:
        for p, r, f in all_preds.tolist():
            print('{:.6f}\t{:.6f}\t{:.6f}'.format(p, r, f))


if __name__ == "__main__":
    main()
