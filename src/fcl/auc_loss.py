"""Memory-efficient first-order implementation of the existing soft pairwise AUC.

For ordered off-diagonal pairs, the two orientations in the original loss
are identical after swapping i and j. Their common factor two cancels in
the denominator. q is detached exactly as in train_auc_ce.py.
"""
import torch
import torch.nn.functional as F


def _pair_block(z, q, tau):
    difference = (z[:, None, :] - z[:, :, None]) / tau
    indices = torch.arange(z.shape[1], device=z.device)
    mask = indices[:, None] != indices[None, :]
    weight = q[:, :, None] * (1-q[:, None, :]) * mask
    numerator = (weight * F.softplus(difference)).sum()
    denominator = weight.sum()
    derivative = weight * torch.sigmoid(difference)
    gradient = (derivative.sum(1)-derivative.sum(2))/tau
    return numerator, denominator, gradient


# Lazy CUDA compilation fuses elementwise/reduction kernels. CPU tests use eager.
_cuda_pair_block = torch.compile(_pair_block, dynamic=True, fullgraph=True)


class _PairwiseAUC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, posterior, tau, chunk_size, eps):
        m, classes = scores.shape
        gradient = torch.zeros_like(scores)
        numerator = scores.new_zeros(())
        denominator = scores.new_zeros(())
        for start in range(0, classes, chunk_size):
            stop = min(start + chunk_size, classes)
            z = scores[:, start:stop].t()
            q = posterior[:, start:stop].t()
            block = _cuda_pair_block if scores.is_cuda else _pair_block
            num, den, grad = block(z, q, tau)
            numerator += num
            denominator += den
            gradient[:, start:stop] = grad.t()
        # Both pair orientations contribute equally in the reference objective.
        denominator = denominator + 0.5 * eps
        gradient.div_(denominator)
        ctx.save_for_backward(gradient)
        return numerator / denominator

    @staticmethod
    def backward(ctx, upstream):
        (gradient,) = ctx.saved_tensors
        return upstream * gradient, None, None, None, None


def soft_pairwise_auc(logits, clip_probs, s, mu=0.5, tau=0.5, chunk_size=16, eps=1e-12):
    if not 0 <= mu <= 1 or not 0 < tau < float('inf') or chunk_size <= 0 or not 0 < eps < float('inf'):
        raise ValueError('Invalid AUC parameters')
    if logits.ndim != 2 or clip_probs.shape != logits.shape or s.shape != logits.shape[:1]:
        raise ValueError('Expected logits/probabilities (N,C) and selection flags (N,)')
    selected = s.eq(1)
    if int(selected.sum()) < 2:
        return logits.sum() * 0
    scores = logits[selected]
    posterior = mu * scores.softmax(-1).detach() + (1-mu) * clip_probs[selected].detach()
    return _PairwiseAUC.apply(scores, posterior, tau, int(chunk_size), eps)
