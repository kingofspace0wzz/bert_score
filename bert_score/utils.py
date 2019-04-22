import torch
from math import log
from itertools import chain
from collections import defaultdict, Counter
from multiprocessing import Pool
from functools import partial

def padding(arr, pad_token, dtype=torch.long):
    lens = torch.LongTensor([len(a) for a in arr])
    max_len = lens.max().item()
    padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
    mask = torch.zeros(len(arr), max_len, dtype=torch.long)
    for i, a in enumerate(arr):
        padded[i, :lens[i]] = torch.tensor(a, dtype=dtype)
        mask[i, :lens[i]] = 1
    return padded, lens, mask

def bert_encode(model, x, attention_mask, all_layers=True):
    model.eval()
    x_seg = torch.zeros_like(x, dtype=torch.long)
    with torch.no_grad():
        x_encoded_layers, pooled_output = model(x, x_seg, attention_mask=attention_mask, output_all_encoded_layers=all_layers)
    return x_encoded_layers

def process(a, tokenizer=None):
    if not tokenizer is None:
        a = ["[CLS]"]+tokenizer.tokenize(a)+["[SEP]"]
        a = tokenizer.convert_tokens_to_ids(a)
    return set(a)

def get_idf_dict(arr, tokenizer, nthreads=8):
    idf_count = Counter()
    num_docs = len(arr)

    process_partial = partial(process, tokenizer=tokenizer)

    with Pool(nthreads) as p:
        idf_count.update(chain.from_iterable(p.map(process_partial, arr)))

    idf_dict = defaultdict(lambda : log((num_docs+1)/(1)))
    idf_dict.update({idx:log((num_docs+1)/(c+1)) for (idx, c) in idf_count.items()})
    return idf_dict


def collate_idf(arr, tokenize, numericalize, idf_dict,
                pad="[PAD]", device='cuda:0', bert=False):
    if bert: arr = [["[CLS]"]+tokenize(a)+["[SEP]"] for a in arr]
    else: arr = [tokenize(a) for a in arr]
    arr = [numericalize(a) for a in arr]

    idf_weights = [[idf_dict[i] for i in a] for a in arr]

    pad_token = numericalize([pad])[0]

    padded, lens, mask = padding(arr, pad_token, dtype=torch.long)
    padded_idf, _, _ = padding(idf_weights, pad_token, dtype=torch.float)

    padded = padded.to(device=device)
    mask = mask.to(device=device)
    lens = lens.to(device=device)
    return padded, padded_idf, lens, mask

def get_bert_embedding(all_sens, model, tokenizer, idf_dict,
                       batch_size=-1, word=False,
                       all_layers=True, device='cuda:0'):

    padded_sens, padded_idf, lens, mask = collate_idf(all_sens,
                                                      tokenizer.tokenize, tokenizer.convert_tokens_to_ids,
                                                      idf_dict,
                                                      device=device, bert=True)

    if batch_size == -1: batch_size = len(all_sens)

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(all_sens), batch_size):
            if not word:
                batch_embedding = bert_encode(model, padded_sens[i:i+batch_size],
                                              attention_mask=mask[i:i+batch_size],
                                              all_layers=all_layers)
                if all_layers: batch_embedding = torch.stack(batch_embedding)
                else: batch_embedding.unsqueeze(0)
            else:
                batch_embedding = model.embeddings.word_embeddings(padded_sens[i:i+batch_size]).unsqueeze(0)
            embeddings.append(batch_embedding)
            del batch_embedding

    total_embedding = torch.cat(embeddings, dim=-3)

    return total_embedding, lens, mask, padded_idf

def greedy_cos_idf(ref_embedding, ref_lens, ref_masks, ref_idf,
                   hyp_embedding, hyp_lens, hyp_masks, hyp_idf,
                   beta=1, mean_sub=False, avg=True):
    if mean_sub:
        mean_mask = (ref_lens+hyp_lens).view(1, ref_lens.size(0), 1, 1).float().to(ref_embedding.device)
        total_embedding = torch.cat([ref_embedding, hyp_embedding], dim=2)
        total_embedding.div_(mean_mask)
        m = total_embedding.sum(dim=2, keepdim=True)

        ref_embedding.sub_(m)
        hyp_embedding.sub_(m)
    
    ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
    hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))

    batch_size = ref_embedding.size(1)
    
    if not avg:
        layers = ref_embedding.size(0)
        ref_embedding = ref_embedding.view(-1, ref_embedding.size(-2), ref_embedding.size(-1))
        hyp_embedding = hyp_embedding.view(-1, hyp_embedding.size(-2), hyp_embedding.size(-1))
    else:
        layers = 1
        ref_embedding = ref_embedding[9]
        hyp_embedding = hyp_embedding[9]

    sim = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2))
    masks = torch.bmm(hyp_masks.unsqueeze(2).float(), ref_masks.unsqueeze(1).float())
    masks = masks.unsqueeze(0).expand(layers, batch_size, masks.size(1), masks.size(2))\
                              .contiguous().view_as(sim)

    masks = masks.float().to(sim.device)
    sim = sim * masks

    word_precision = sim.max(dim=2)[0]
    word_recall = sim.max(dim=1)[0]

    hyp_idf.div_(hyp_idf.sum(dim=1, keepdim=True))
    ref_idf.div_(ref_idf.sum(dim=1, keepdim=True))
    precision_scale = hyp_idf.unsqueeze(0).expand(layers, *hyp_idf.size())\
                             .contiguous().view(-1, *hyp_idf.size()[1:]).to(word_precision.device)
    recall_scale = ref_idf.unsqueeze(0).expand(layers, *ref_idf.size())\
                          .contiguous().view(-1, *ref_idf.size()[1:]).to(word_recall.device)
    P = (word_precision * precision_scale).sum(dim=1)
    R = (word_recall * recall_scale).sum(dim=1)
    
    F = (1 + beta**2) * P * R / (P * beta**2 + R)
    P = P.view(layers, batch_size)
    R = R.view(layers, batch_size)
    F = F.view(layers, batch_size)
    return P, R, F

def bert_cos_score_idf(model, refs, hyps, tokenizer, idf_dict,
                       batch_size=256, device='cuda:0',
                       all_layers=True, mean_sub=False,
                       word=False, avg=True):
    preds = []
    for batch_start in range(0, len(refs), batch_size):
        batch_refs = refs[batch_start:batch_start+batch_size]
        batch_hyps = hyps[batch_start:batch_start+batch_size]
        ref_stats = get_bert_embedding(batch_refs, model, tokenizer, idf_dict,
                                       word=word, all_layers=all_layers, device=device)
        hyp_stats = get_bert_embedding(batch_hyps, model, tokenizer, idf_dict,
                                       word=word, all_layers=all_layers, device=device)

        P, R, F1 = greedy_cos_idf(*ref_stats, *hyp_stats, mean_sub=mean_sub, avg=avg)
        preds.append(torch.stack((P, R, F1), dim=2).cpu())
    preds = torch.cat(preds, dim=1)
    return preds
