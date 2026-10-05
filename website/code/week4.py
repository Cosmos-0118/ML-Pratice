# Week 4 - a hidden Markov model written from scratch: wet and dry spells
import numpy as np
import matplotlib.pyplot as plt
from weather import load

d = load()
obs = np.digitize(d.precipitation, [0.001, 5.0])        # 0 = dry, 1 = light rain, 2 = heavy rain
n_train = int((d.date < "2015-01-01").sum())             # learn from 2012-2014 only

def forward(obs, A, B, pi):                              # "which regime are we in today?"
    alpha, c = np.zeros((len(obs), len(pi))), np.zeros(len(obs))
    for t in range(len(obs)):
        alpha[t] = (pi if t == 0 else alpha[t - 1] @ A) * B[:, obs[t]]
        c[t] = alpha[t].sum(); alpha[t] /= c[t]
    return alpha, c

def baum_welch(obs, K, seed, iters=150):                 # learn A, B, pi from the data alone (EM)
    rng = np.random.default_rng(seed)
    A, B, pi = rng.dirichlet(np.ones(K) * 5, K), rng.dirichlet(np.ones(3) * 5, K), np.full(K, 1 / K)
    for _ in range(iters):
        alpha, c = forward(obs, A, B, pi)
        beta = np.ones_like(alpha)
        for t in range(len(obs) - 2, -1, -1):
            beta[t] = A @ (B[:, obs[t + 1]] * beta[t + 1]) / c[t + 1]
        gamma = alpha * beta; gamma /= gamma.sum(1, keepdims=True)
        xi = alpha[:-1, :, None] * A[None] * (B[:, obs[1:]].T * beta[1:])[:, None, :]
        xi /= xi.sum((1, 2), keepdims=True)
        A = xi.sum(0) / gamma[:-1].sum(0)[:, None]
        B = np.array([[gamma[obs == m, k].sum() for m in range(3)] for k in range(K)]) / gamma.sum(0)[:, None]
        pi = gamma[0]
    return A, B, pi

def viterbi(obs, A, B, pi):                              # the single most likely sequence of regimes
    delta, back = np.log(pi) + np.log(B[:, obs[0]]), []
    for o in obs[1:]:
        s = delta[:, None] + np.log(A)
        back.append(s.argmax(0)); delta = s.max(0) + np.log(B[:, o])
    path = [delta.argmax()]
    for b in back[::-1]: path.append(b[path[-1]])
    return np.array(path[::-1])

fits = [baum_welch(obs[:n_train], 2, seed) for seed in range(5)]
A, B, pi = max(fits, key=lambda m: np.log(forward(obs[:n_train], *m)[1]).sum())   # keep the best start
wet = int(np.argmin(B[:, 0]))                            # the regime that is rarely dry
print("transition matrix (row = from):\n", A.round(2))
print("chance of dry / light / heavy rain in each regime:\n", B.round(2))
print("average spell length (days):", (1 / (1 - np.diag(A))).round(1))

path = viterbi(obs, A, B, pi)
days = np.arange(n_train, len(obs))
fig, ax = plt.subplots(figsize=(9, 3))
ax.bar(range(len(days)), d.precipitation.values[days], color="k", width=1)
ax.fill_between(range(len(days)), 0, 1, where=path[days] == wet, color="tab:blue", alpha=.25, transform=ax.get_xaxis_transform())
ax.set_xlabel("day of 2015"); ax.set_ylabel("precipitation (mm)"); ax.set_title("blue = wet spell found by the model")
plt.tight_layout(); plt.show()
