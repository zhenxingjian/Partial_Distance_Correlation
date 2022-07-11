import numpy as np

from sklearn import svm
from sklearn.model_selection import train_test_split


def evaluate(latents, factors):
    assert len(latents.shape) == 2 or latents.shape[2] == 1
    latents = latents.reshape(latents.shape[0], -1)

    latents_train, latents_test, factors_train, factors_test = train_test_split(latents, factors, test_size=0.2, random_state=1337)

    score_matrix = compute_score_matrix(
        latents_train, latents_test,
        factors_train, factors_test
    )

    scores = {
        'sap': compute_avg_diff_top_two(score_matrix)
    }

    return scores


def compute_score_matrix(x_train, x_test, y_train, y_test):
    latent_dim = x_train.shape[1]
    n_factors = y_train.shape[1]

    score_matrix = np.zeros(shape=[latent_dim, n_factors], dtype=np.float64)

    for i in range(latent_dim):
        for j in range(n_factors):
            mu_i = x_train[:, i]
            y_j = y_train[:, j]

            mu_i_test = x_test[:, i]
            y_j_test = y_test[:, j]

            # TODO: squeeze if needed

            classifier = svm.LinearSVC(C=0.01, class_weight="balanced")
            classifier.fit(mu_i[:, np.newaxis], y_j)
            pred = classifier.predict(mu_i_test[:, np.newaxis])
            score_matrix[i, j] = np.mean(pred == y_j_test)

    return score_matrix


def compute_avg_diff_top_two(matrix):
    sorted_matrix = np.sort(matrix, axis=0)
    return np.mean(sorted_matrix[-1, :] - sorted_matrix[-2, :])
