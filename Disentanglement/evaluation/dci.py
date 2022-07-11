from tqdm import tqdm

import numpy as np
import scipy

from sklearn import ensemble
from sklearn.model_selection import train_test_split


def evaluate(latents, factors):
    latents_train, latents_test, factors_train, factors_test = train_test_split(latents, factors, test_size=0.2, random_state=1337)

    importance_matrix, acc_train, acc_test = compute_importance_gbt(
        latents_train, latents_test,
        factors_train, factors_test
    )

    print(importance_matrix)

    scores = {
        'informativeness_train': acc_train,
        'informativeness_test': acc_test,
        'disentanglement': disentanglement(importance_matrix),
        'completeness': completeness(importance_matrix)
    }

    return scores


def compute_importance_gbt(x_train, x_test, y_train, y_test):
    latent_dim = x_train.shape[1]
    n_factors = y_train.shape[1]

    importance_matrix = np.zeros(shape=[latent_dim, n_factors], dtype=np.float64)
    acc_train = []
    acc_test = []

    pbar = tqdm(range(n_factors))
    for i in pbar:
        pbar.set_description_str('[dci] factor #{}/#{}'.format(i + 1, n_factors))

        model = ensemble.GradientBoostingClassifier()
        model.fit(x_train.reshape(x_train.shape[0], -1), y_train[:, i])

        importance_matrix[:, i] = np.sum(model.feature_importances_.reshape(latent_dim, -1), axis=-1)
        acc_train.append(np.mean(model.predict(x_train.reshape(x_train.shape[0], -1)) == y_train[:, i]))
        acc_test.append(np.mean(model.predict(x_test.reshape(x_test.shape[0], -1)) == y_test[:, i]))

    return importance_matrix, np.mean(acc_train), np.mean(acc_test)


def disentanglement_per_code(importance_matrix):
    return 1. - scipy.stats.entropy(importance_matrix.T + 1e-11, base=importance_matrix.shape[1])


def disentanglement(importance_matrix):
    per_code = disentanglement_per_code(importance_matrix)

    if importance_matrix.sum() == 0.:
        importance_matrix = np.ones_like(importance_matrix)

    code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()
    return np.sum(per_code * code_importance)


def completeness_per_factor(importance_matrix):
    return 1. - scipy.stats.entropy(importance_matrix + 1e-11, base=importance_matrix.shape[0])


def completeness(importance_matrix):
    per_factor = completeness_per_factor(importance_matrix)

    if importance_matrix.sum() == 0.:
        importance_matrix = np.ones_like(importance_matrix)

    factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
    return np.sum(per_factor * factor_importance)
