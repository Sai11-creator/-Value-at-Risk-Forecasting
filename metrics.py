import numpy as np
from sklearn.metrics import mean_pinball_loss
from scipy.stats import chi2

quantiles = [0.01, 0.05, 0.10]

def kupiec_test(hits, alpha):
    n = len(hits)
    x = int(hits.sum())
    if x == 0 or x == n:
        return np.nan, np.nan
    phat = x / n
    LR = -2 * (
        (n - x) * np.log((1 - alpha) / (1 - phat)) +
        x       * np.log(alpha / phat)
    )
    pval = 1 - chi2.cdf(LR, df=1)
    return LR, pval


def eval_preds(y_eval, preds, quantiles):
    out = {}
    for q in quantiles:
        qhat = preds[q]
        mask = ~np.isnan(qhat)
        yv = y_eval[mask]
        qv = qhat[mask]

        hits = (yv < qv).astype(int)
        cov = hits.mean()
        pin = mean_pinball_loss(yv, qv, alpha=q)
        LR, pval = kupiec_test(hits, q)

        out[q] = dict(
            actual_coverage=cov,
            pinball_loss=pin,
            kupiec_LR=LR,
            kupiec_pval=pval,
            nb_violations=int(hits.sum()),
            n_obs=int(len(hits))
        )
    return out

def composite_score(metrics_by_q, quantiles, lam=0.1):

    scores = []
    for q in quantiles:
        m = metrics_by_q[q]
        # pénalité d'écart de couverture
        scores.append(m["pinball_loss"] + lam * abs(m["actual_coverage"] - q))
    return float(np.mean(scores))
