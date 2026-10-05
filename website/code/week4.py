# Week 4 - a Gaussian HMM written from scratch (scaled forward-backward + Baum-Welch + Viterbi)
import numpy as np, matplotlib.pyplot as plt
from beijing_pm25 import load_daily, chrono_split

d = load_daily(); x = d.log_pm25.values; train, test = chrono_split(d)

def logB(x, mu, sd):                         # emission log-density; a missing day carries no information
    ll = -0.5 * ((x[:, None] - mu) / sd) ** 2 - np.log(sd * np.sqrt(2 * np.pi))
    return np.where(np.isnan(x)[:, None], 0.0, ll)

def forward_backward(lb, A, pi):
    T, K = lb.shape; B = np.exp(lb - lb.max(1, keepdims=True))
    al = np.zeros((T, K)); c = np.zeros(T)
    al[0] = pi * B[0]; c[0] = al[0].sum(); al[0] /= c[0]
    for t in range(1, T):
        al[t] = (al[t - 1] @ A) * B[t]; c[t] = al[t].sum(); al[t] /= c[t]
    be = np.ones((T, K))
    for t in range(T - 2, -1, -1):
        be[t] = A @ (B[t + 1] * be[t + 1]) / c[t + 1]
    ll = np.log(c).sum() + lb.max(1).sum()
    return al, be, ll

def fit(x, K=3, iters=60, seed=0):
    rng = np.random.default_rng(seed); v = x[~np.isnan(x)]
    mu = np.sort(rng.choice(v, K)); sd = np.full(K, v.std()); A = np.full((K, K), 1 / K); pi = np.full(K, 1 / K)
    for _ in range(iters):
        lb = logB(x, mu, sd); al, be, ll = forward_backward(lb, A, pi)
        g = al * be; g /= g.sum(1, keepdims=True)
        B = np.exp(lb - lb.max(1, keepdims=True))
        xi = al[:-1, :, None] * A[None] * (B[1:] * be[1:])[:, None, :]
        xi /= xi.sum((1, 2), keepdims=True)
        A = xi.sum(0) / xi.sum((0, 2))[:, None]; pi = g[0]
        w = g * ~np.isnan(x)[:, None]; xv = np.nan_to_num(x)[:, None]
        mu = (w * xv).sum(0) / w.sum(0); sd = np.sqrt((w * (xv - mu) ** 2).sum(0) / w.sum(0)) + 1e-3
    return mu, sd, A, pi, ll

def viterbi(lb, A, pi):
    T, K = lb.shape; dl = np.zeros((T, K)); bp = np.zeros((T, K), int); dl[0] = np.log(pi) + lb[0]
    for t in range(1, T):
        s = dl[t - 1][:, None] + np.log(A); bp[t] = s.argmax(0); dl[t] = s.max(0) + lb[t]
    z = np.zeros(T, int); z[-1] = dl[-1].argmax()
    for t in range(T - 2, -1, -1): z[t] = bp[t + 1, z[t + 1]]
    return z

mu, sd, A, pi, ll = fit(x[train], K=3)
order = np.argsort(mu); mu, sd, A, pi = mu[order], sd[order], A[np.ix_(order, order)], pi[order]
print("hidden states (daily PM2.5, ug/m3):", np.round(np.exp(mu)).astype(int), " <- clean / moderate / polluted regimes")
print("transition matrix (rows = from):\n", A.round(2))
print("expected stay (days):", np.round(1 / (1 - np.diag(A)), 1))

z = viterbi(logB(x[test], mu, sd), A, pi)
plt.figure(figsize=(10, 3.2))
plt.plot(np.exp(x[test]), "k", lw=.8, label="PM2.5")
for k, col in enumerate(["tab:green", "gold", "tab:red"]):
    plt.fill_between(range(len(z)), 0, 1, where=z == k, color=col, alpha=.25, transform=plt.gca().get_xaxis_transform())
plt.yscale("log"); plt.xlabel("day of 2014"); plt.ylabel("PM2.5 (ug/m3)")
plt.title("Viterbi path on the 2014 test year (green / yellow / red = hidden regime)")
plt.tight_layout(); plt.show()
