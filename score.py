import time
import argparse
import torch
from collections import defaultdict
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

from bert_score.bert_idf import get_idf_dict, bert_cos_score_idf

VERSION=bert_score.__version__
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--bert', default='bert-base-multilingual-cased', choices=bert_types, help='BERT model name (default: bert-base-uncased)')
    parser.add_argument('-l', '--num_layers', default=9, help='use first N layer in BERT (default: 9)')
    parser.add_argument('-b', '--batch_size', default=64, help='batch size (default: 64)')
    parser.add_argument('-r', '--ref', help='reference file path')
    parser.add_argument('-c', '--cand', help='candidate file path')
    parser.add_argument('--no_idf', action='store_true', help='BERT Score without IDF scaling')

    args = parser.parse_args()

    with open(args.cand) as f:
        cands = [line.strip() for line in f]

    with open(args.ref) as f:
        refs = [line.strip() for line in f]

    assert len(cands) == len(refs)

    print('loading BERT model...')
    tokenizer = BertTokenizer.from_pretrained(args.bert)
    model = BertModel.from_pretrained(args.bert)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # drop layers
    # model.encoder.layer = torch.nn.ModuleList([layer for layer in model.encoder.layer[:args.num_layers]])


    if args.no_idf:
        idf_dict = defaultdict(lambda: 1.)
    else:
        print('preparing IDF dict...')
        start = time.perf_counter()
        idf_dict = get_idf_dict(refs, tokenizer)
        print('done in {:.2f} seconds'.format(time.perf_counter() - start))
    print('calculating scores...')
    start = time.perf_counter()
    all_preds_idf = bert_cos_score_idf(model, refs, cands, tokenizer, idf_dict, device=device,
                                       all_layers=True, batch_size=args.batch_size, word=False, avg=False)
    avg_scores = all_preds_idf[8].mean(dim=0)
    P = avg_scores[0].cpu().item()
    R = avg_scores[1].cpu().item()
    F1 = avg_scores[2].cpu().item()
    print('done in {:.2f} seconds'.format(time.perf_counter() - start))
    msg = '{}_L{}{}_version={} BERT-P: {:.6f} BERT-R: {:.6f} BERT-F1: {:.6f}'.format(
        args.bert, args.num_layers, '_no-idf' if args.no_idf else '', VERSION, P, R, F1)
    print(msg)


if __name__ == "__main__":
    main()
