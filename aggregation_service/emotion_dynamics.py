import math


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx  = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy  = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx < 1e-6 or sy < 1e-6:
        return 0.0
    return max(-1.0, min(1.0, num / (sx * sy)))


def compute_inertia(valence_seq):
    seq = valence_seq[-12:]
    if len(seq) < 4:
        return 0.0
    r = _pearson(seq[:-1], seq[1:])
    return 0.0 if not math.isfinite(r) else r


def compute_contagion(seq_a, seq_b):
    if len(seq_a) < 3 or len(seq_b) < 3:
        return 0.0
    min_len = min(len(seq_a), len(seq_b), 12)
    a, b = seq_a[-min_len:], seq_b[-min_len:]
    r = _pearson(a[:-1], b[1:])
    return 0.0 if not math.isfinite(r) else r
