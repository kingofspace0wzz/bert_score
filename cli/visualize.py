import time
import argparse
import torch
from collections import defaultdict
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bert_score.utils import plot_example

bert_types = [
    'bert-base-uncased',
    'bert-large-uncased',
    'bert-base-cased',
    'bert-large-cased',
    'bert-base-multilingual-uncased',
    'bert-base-multilingual-cased',
]

def main():
    torch.multiprocessing.set_sharing_strategy('file_system')

    parser = argparse.ArgumentParser('Calculate BERTScore')
    parser.add_argument('--bert', default='bert-base-multilingual-cased',
                        choices=bert_types, help='BERT model name (default: bert-base-uncased)')
    parser.add_argument('-l', '--num_layers', default=9, help='use first N layer in BERT (default: 9)')
    parser.add_argument('-v', '--verbose', action='store_true', help='increase output verbosity')
    parser.add_argument('-r', '--ref', required=True, help='reference file path')
    parser.add_argument('-c', '--cand', required=True,help='candidate file path')

    args = parser.parse_args()

    with open(args.cand) as f:
        cands = [line.strip() for line in f]

    with open(args.ref) as f:
        refs = [line.strip() for line in f]

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

    idf_dict = defaultdict(lambda: 1.)
    # Assuming there is only one example to plot
    plot_example(cands[0], refs[0], model, tokenizer, idf_dict)

if __name__ == "__main__":
    main()
