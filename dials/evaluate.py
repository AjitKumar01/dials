"""Full-catalogue ranking metrics. Seeds are excluded; exact ties go to the lower item ID.

See ARCHITECTURE.md §12. For a query with hidden items T and hit_k = 1 when the k-th ranked item is in T:

    Recall@20 = (1/|T|) sum_{k<=20} hit_k
    NDCG@20   = sum_{k<=20} hit_k / log2(k+1)  /  sum_{k<=min(|T|,20)} 1 / log2(k+1)
"""
import time
import numpy as np
import torch


def top_items(scores,seeds,k):
    """Exact top-k item IDs (boundary ties included by ascending ID), seeds excluded.

    Breaking ties by item ID makes the ranking independent of batching and hardware order.
    """
    # Seeds can never be recommended.
    values=np.asarray(scores).copy()
    values[list(seeds)]=-np.inf
    k=min(int(k),len(values)-len(set(seeds)))

    # The k-th largest score.
    threshold=np.partition(values,len(values)-k)[len(values)-k]

    # Everything strictly above it, plus as many items tied at it as needed, lowest IDs first.
    above=np.flatnonzero(values>threshold)
    tied=np.flatnonzero(values==threshold)[:k-len(above)]
    candidates=np.concatenate((above,tied))

    # Sort by score descending, then by item ID ascending.
    return candidates[np.lexsort((candidates,-values[candidates]))]


def query_metrics(scores,seeds,truth,k=20):
    """Recall@20, NDCG@20 and the top-20 list for one query."""
    truth=set(truth)
    top=top_items(scores,seeds,k)
    hits=np.isin(top,list(truth))   # hit_k for each rank

    # DCG of the ranking, and of the ideal ranking with every hidden item on top.
    dcg=float((hits/np.log2(np.arange(len(hits))+2)).sum())
    ideal=float((1/np.log2(np.arange(min(len(truth),k))+2)).sum())
    return dict(recall20=float(hits.sum())/len(truth),ndcg20=dcg/ideal,top20=top.tolist())


@torch.inference_mode()
def score_batches(model,seeds,batch=32):
    """Full-catalogue scores (batch x items) for successive batches of seed lists.

    The query-independent parts (model.cache()) are computed once and reused for every batch.
    """
    model.eval()
    cache=model.cache()
    for start in range(0,len(seeds),batch):
        values=model(seeds[start:start+batch],cache=cache).numpy()
        if not np.isfinite(values).all():
            raise RuntimeError('Nonfinite scores')
        yield values


def scores(model,seeds,batch=32):
    """Full-catalogue scores for all queries at once (queries x items)."""
    return np.concatenate(list(score_batches(model,seeds,batch)))


@torch.inference_mode()
def evaluate(model,pairs,batch=32):
    """Recall@20 and NDCG@20 per query and on average.

    Args:
        pairs: list of (seeds, hidden items).

    Returns: dict with the mean recall20 and ndcg20, the per-query records, and the time taken.
    """
    started=time.monotonic()
    records=[]
    for start,values in zip(range(0,len(pairs),batch),score_batches(model,[s for s,_ in pairs],batch)):
        records.extend(query_metrics(v,s,g) for v,(s,g) in zip(values,pairs[start:start+batch]))
    return dict(recall20=float(np.mean([r['recall20'] for r in records])),
                ndcg20=float(np.mean([r['ndcg20'] for r in records])),
                queries=records,
                seconds=time.monotonic()-started)


def subset(result,indices):
    """Mean metrics of an evaluate() result over a subset of its queries (e.g. the clean test queries)."""
    rows=[result['queries'][i] for i in indices]
    return dict(queries=len(rows),
                recall20=float(np.mean([r['recall20'] for r in rows])),
                ndcg20=float(np.mean([r['ndcg20'] for r in rows])))
