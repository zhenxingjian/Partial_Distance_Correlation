import numpy as np
import sklearn


def evaluate(latents, factors):
    assert len(latents.shape) == 2 or latents.shape[2] == 1
    latents = latents.reshape(latents.shape[0], -1)

    latents = latents.T
    factors = factors.T

    latents_discretized = _histogram_discretize(latents)
    m = discrete_mutual_info(latents_discretized, factors)

    # m is [num_latents, num_factors]
    entropy = discrete_entropy(factors)
    sorted_m = np.sort(m, axis=0)[::-1]

    scores = {
        'mig': np.mean(np.divide(sorted_m[0, :] - sorted_m[1, :], entropy[:]))
    }

    return scores


def _histogram_discretize(target, num_bins=20):
    discretized = np.zeros_like(target)

    for i in range(target.shape[0]):
        discretized[i, :] = np.digitize(target[i, :], np.histogram(target[i, :], num_bins)[1][:-1])

    return discretized


def discrete_mutual_info(mus, ys):
    num_codes = mus.shape[0]
    num_factors = ys.shape[0]

    m = np.zeros([num_codes, num_factors])
    for i in range(num_codes):
        for j in range(num_factors):
            m[i, j] = sklearn.metrics.mutual_info_score(ys[j, :], mus[i, :])

    return m


def discrete_entropy(ys):
    num_factors = ys.shape[0]
    h = np.zeros(num_factors)

    for j in range(num_factors):
        h[j] = sklearn.metrics.mutual_info_score(ys[j, :], ys[j, :])

    return h
