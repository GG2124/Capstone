!pip install pandas scikit-learn matplotlib -q

import pandas as pd
import numpy as np
import random
import copy
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("drebin-215-dataset-5560malware-9476-benign.csv")

df = df.replace("?", 0)

for col in df.columns:
    if col != "class":
        df[col] = pd.to_numeric(df[col])

X = df.drop("class", axis=1).values
y = LabelEncoder().fit_transform(df["class"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

def create_clients(X, y, n_clients=5):
    data = list(zip(X, y))
    random.shuffle(data)

    size = len(data) // n_clients
    clients = []

    for i in range(n_clients):
        chunk = data[i*size:(i+1)*size]
        Xc = np.array([x for x,_ in chunk])
        yc = np.array([y for _,y in chunk])
        clients.append((Xc, yc))

    return clients

def init_model():
    return SGDClassifier(loss="log_loss", max_iter=1, warm_start=True)

def evaluate(model):
    preds = model.predict(X_test)

    return [
        accuracy_score(y_test, preds),
        precision_score(y_test, preds),
        recall_score(y_test, preds),
        f1_score(y_test, preds)
    ]

def fedavg(Xt, yt, rounds=5):
    global_model = init_model()
    global_model.partial_fit(Xt[:100], yt[:100], classes=np.unique(yt))

    clients = create_clients(Xt, yt)

    for r in range(rounds):
        local_models = []

        for Xc, yc in clients:
            lm = copy.deepcopy(global_model)
            lm.partial_fit(Xc, yc)
            local_models.append(lm)

        global_model.coef_ = np.mean(
            [m.coef_ for m in local_models], axis=0
        )
        global_model.intercept_ = np.mean(
            [m.intercept_ for m in local_models], axis=0
        )

    return global_model

def fedsgd(Xt, yt, rounds=5):
    global_model = init_model()
    global_model.partial_fit(Xt[:100], yt[:100], classes=np.unique(yt))

    clients = create_clients(Xt, yt)

    for r in range(rounds):
        grads = []

        for Xc, yc in clients:
            lm = copy.deepcopy(global_model)
            lm.partial_fit(Xc, yc)
            grads.append(lm.coef_ - global_model.coef_)

        global_model.coef_ += np.mean(grads, axis=0)

    return global_model

def fedprox(Xt, yt, rounds=5, mu=0.01):
    global_model = init_model()
    global_model.partial_fit(Xt[:100], yt[:100], classes=np.unique(yt))

    clients = create_clients(Xt, yt)

    for r in range(rounds):
        local_models = []

        for Xc, yc in clients:
            lm = copy.deepcopy(global_model)
            lm.partial_fit(Xc, yc)
            lm.coef_ -= mu * (lm.coef_ - global_model.coef_)
            local_models.append(lm)

        global_model.coef_ = np.mean(
            [m.coef_ for m in local_models], axis=0
        )

    return global_model

def local_train(Xt, yt):
    model = init_model()
    model.partial_fit(Xt, yt, classes=np.unique(yt))
    return model

models = {
    "FedAvg": fedavg(X_train, y_train),
    "FedSGD": fedsgd(X_train, y_train),
    "FedProx": fedprox(X_train, y_train),
    "Local": local_train(X_train, y_train)
}

results = {}

print("\n===== RESULTS =====")

for name, model in models.items():
    scores = evaluate(model)
    results[name] = scores

    print(f"\n{name}")
    print("Accuracy :", scores[0])
    print("Precision:", scores[1])
    print("Recall   :", scores[2])
    print("F1 Score :", scores[3])

algos = list(results.keys())

accuracy_vals = [results[a][0] for a in algos]
precision_vals = [results[a][1] for a in algos]
recall_vals = [results[a][2] for a in algos]
f1_vals = [results[a][3] for a in algos]

def plot_metric(values, title, ylabel):
    plt.figure()
    plt.bar(algos, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.show()

plot_metric(accuracy_vals, "Accuracy Comparison", "Accuracy")
plot_metric(precision_vals, "Precision Comparison", "Precision")
plot_metric(recall_vals, "Recall Comparison", "Recall")
plot_metric(f1_vals, "F1 Score Comparison", "F1 Score")
