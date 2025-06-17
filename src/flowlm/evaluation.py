import torch
from typing import List, Tuple
from itertools import islice


def accuracy_buckets(
    logits: torch.Tensor, labels: torch.Tensor, attn: torch.Tensor
) -> Tuple[float, List[float]]:
    """
    Calculate accuracy metrics bucketed by masking ratio.

    Returns:
        global_acc: Overall accuracy across all masked positions
        bucket_acc: Accuracy for 4 buckets: ≤0.25, 0.25-0.5, 0.5-0.75, >0.75
    """
    with torch.no_grad():
        pred = logits.argmax(-1)
        mask = labels != -100  # only masked positions count
        correct = (pred == labels) & mask

        # Global accuracy
        tot_masked = mask.sum().item()
        tot_corr = correct.sum().item()
        global_acc = tot_corr / tot_masked if tot_masked else 0.0

        # Buckets by sample-level mask ratio
        bucket_corr = [0, 0, 0, 0]
        bucket_total = [0, 0, 0, 0]
        edges = (0.25, 0.50, 0.75, 1.01)  # last edge slightly >1

        for b in range(labels.size(0)):
            n_mask = mask[b].sum().item()
            if n_mask == 0:  # should be rare
                continue
            # denominator = real tokens (ignore pads)
            seq_len = attn[b].sum().item()
            ratio = n_mask / seq_len
            # bucket index
            for i, edge in enumerate(edges):
                if ratio <= edge:
                    bucket_total[i] += n_mask
                    bucket_corr[i] += correct[b].sum().item()
                    break

        bucket_acc = [c / t if t else 0.0 for c, t in zip(bucket_corr, bucket_total)]

    return global_acc, bucket_acc


@torch.no_grad()
def evaluate_model(
    model, val_loader, device: str = "cuda", num_batches: int = 8
) -> Tuple[float, float, List[float]]:
    """
    Evaluate model on validation data.

    Returns:
        val_loss: Average validation loss
        val_acc: Average validation accuracy
        bucket_acc: Accuracy per masking ratio bucket
    """
    model.eval()
    tot_loss, tot_acc, bucket_hits = 0.0, 0.0, [0, 0, 0, 0]

    for batch in islice(val_loader, num_batches):
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss.item()
        tot_loss += loss
        acc, bucket_acc = accuracy_buckets(
            out.logits, batch["labels"], batch["attention_mask"]
        )
        tot_acc += acc
        bucket_hits = [h + a for h, a in zip(bucket_hits, bucket_acc)]

    n = num_batches
    val_loss = tot_loss / n
    val_acc = tot_acc / n
    bucket_acc = [b / n for b in bucket_hits]

    return val_loss, val_acc, bucket_acc
