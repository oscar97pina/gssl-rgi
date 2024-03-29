"""
Extracted from https://github.com/nerdslab/bgrl
"""

import numpy as np
import torch
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder, normalize


def fit_logistic_regression(X, y, data_random_seed=1, repeat=1):
    # transfrom targets to one-hot vector
    one_hot_encoder = OneHotEncoder(categories='auto', sparse=False)

    y = one_hot_encoder.fit_transform(y.reshape(-1, 1)).astype(bool)

    # normalize x
    X = normalize(X, norm='l2')
    #X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-4)

    # set random state
    rng = np.random.RandomState(data_random_seed)  # this will ensure the dataset will be split exactly the same
                                                   # throughout training

    val_accuracies, test_accuracies = list(), list()
    for _ in range(repeat):
        # different random split after each repeat
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=rng)

        # grid search with one-vs-rest classifiers
        logreg = LogisticRegression(solver='liblinear')
        c = 2.0 ** np.arange(-10, 11)
        cv = ShuffleSplit(n_splits=5, test_size=0.5)
        clf = GridSearchCV(estimator=OneVsRestClassifier(logreg), param_grid=dict(estimator__C=c),
                           n_jobs=5, cv=cv, verbose=0)
        clf.fit(X_train, y_train)

        y_pred = clf.predict_proba(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(bool)

        test_acc = metrics.accuracy_score(y_test, y_pred)
        
        val_accuracies.append(clf.best_score_)
        test_accuracies.append(test_acc)
    return val_accuracies, test_accuracies


def fit_logistic_regression_preset_splits(X, y, train_masks, val_masks, test_mask):
    # transfrom targets to one-hot vector
    one_hot_encoder = OneHotEncoder(categories='auto', sparse=False)
    y = one_hot_encoder.fit_transform(y.reshape(-1, 1)).astype(bool)

    # normalize x
    X = normalize(X, norm='l2')

    val_accuracies, test_accuracies = list(), list()
    for split_id in range(train_masks.shape[1]):
        # get train/val/test masks
        train_mask, val_mask = train_masks[:, split_id], val_masks[:, split_id]

        # make custom cv
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # grid search with one-vs-rest classifiers
        best_test_acc, best_acc = 0, 0
        for c in 2.0 ** np.arange(-10, 11):
            clf = OneVsRestClassifier(LogisticRegression(solver='liblinear', C=c))
            clf.fit(X_train, y_train)

            y_pred = clf.predict_proba(X_val)
            y_pred = np.argmax(y_pred, axis=1)
            y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(bool)
            val_acc = metrics.accuracy_score(y_val, y_pred)
            if val_acc > best_acc:
                best_acc = val_acc
                y_pred = clf.predict_proba(X_test)
                y_pred = np.argmax(y_pred, axis=1)
                y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(bool)
                best_test_acc = metrics.accuracy_score(y_test, y_pred)

        val_accuracies.append(best_acc)
        test_accuracies.append(best_test_acc)
    return val_accuracies, test_accuracies       

def fit_torch_logistic_regression(X, y, device, seed=0):
    # batch to tensor
    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)

    # normalize
    mean, std = X.mean(0, keepdim=True), X.std(0, unbiased=False, keepdim=True)
    X = (X - mean) / std

    num_samples, num_features  = X.size()
    num_classes  = len(np.unique(y.numpy()))

    # split
    idx = np.arange(X.size(0))
    idx_train, idx_test = train_test_split(idx,       test_size=0.8, random_state=seed)
    idx_train, idx_val  = train_test_split(idx_train, test_size=0.5, random_state=seed)
    
    def train(classifier, train_data, criterion, optimizer):
        classifier.train()

        x, label = train_data
        x, label = x.to(device), label.to(device)
        for step in range(100):
            # forward
            optimizer.zero_grad()
            pred_logits = classifier(x)
            # loss and backprop
            loss = criterion(pred_logits, label)
            loss.backward()
            optimizer.step()

    def test(classifier, data):
        classifier.eval()
        x, label = data
        label = label.cpu().numpy().squeeze()

        # feed to network and classifier
        with torch.no_grad():
            pred_logits = classifier(x.to(device))
            pred_class = pred_logits.argmax(-1).cpu().numpy()

        return metrics.accuracy_score(label, pred_class)

    best_val_acc = 0
    test_acc = 0
    for weight_decay in 2.0 ** np.arange(-10, 11, 2):
        classifier = torch.nn.Linear(num_features, num_classes).to(device)
        optimizer = torch.optim.AdamW(params=classifier.parameters(), lr=0.01, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss()

        train_data = (X[idx_train,:], y[idx_train])
        val_data   = (X[idx_val,:],   y[idx_val])
        test_data  = (X[idx_test,:],  y[idx_test] )

        train(classifier, train_data, criterion, optimizer)
        val_acc = test(classifier, val_data)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = test(classifier, test_data)

    return [best_val_acc], [test_acc]

def ppi_train_linear_layer(num_classes, train_data, val_data, test_data, device):
    r"""
    Trains a linear layer on top of the representations. This function is specific to the PPI dataset,
    which has multiple labels.
    """
    def train(classifier, train_data, optimizer):
        classifier.train()

        x, label = train_data
        x, label = x.to(device), label.to(device)
        for step in range(100):
            # forward
            optimizer.zero_grad()
            pred_logits = classifier(x)
            # loss and backprop
            loss = criterion(pred_logits, label)
            loss.backward()
            optimizer.step()

    def test(classifier, data):
        classifier.eval()
        x, label = data
        label = label.cpu().numpy().squeeze()

        # feed to network and classifier
        with torch.no_grad():
            pred_logits = classifier(x.to(device))
            pred_class = (pred_logits > 0).float().cpu().numpy()

        return metrics.f1_score(label, pred_class, average='micro') if pred_class.sum() > 0 else 0

    num_feats = train_data[0].size(1)
    criterion = torch.nn.BCEWithLogitsLoss()

    # normalization
    mean, std = train_data[0].mean(0, keepdim=True), train_data[0].std(0, unbiased=False, keepdim=True)
    train_data[0] = (train_data[0] - mean) / std
    val_data[0] = (val_data[0] - mean) / std
    test_data[0] = (test_data[0] - mean) / std

    best_val_f1 = 0
    test_f1 = 0
    for weight_decay in 2.0 ** np.arange(-10, 11, 2):
        classifier = torch.nn.Linear(num_feats, num_classes).to(device)
        optimizer = torch.optim.AdamW(params=classifier.parameters(), lr=0.01, weight_decay=weight_decay)

        train(classifier, train_data, optimizer)
        val_f1 = test(classifier, val_data)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            test_f1 = test(classifier, test_data)

    return best_val_f1, test_f1