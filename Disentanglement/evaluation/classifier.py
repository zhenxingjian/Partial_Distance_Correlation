from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def logistic_regression(latents, factors):
    scaler = StandardScaler()
    latents = scaler.fit_transform(latents)

    latents_train, latents_test, factors_train, factors_test = train_test_split(latents, factors, test_size=0.2, random_state=1337)

    classifier = LogisticRegression(random_state=1337)
    classifier.fit(latents_train, factors_train)

    acc_train = classifier.score(latents_train, factors_train)
    acc_test = classifier.score(latents_test, factors_test)

    return acc_train, acc_test
