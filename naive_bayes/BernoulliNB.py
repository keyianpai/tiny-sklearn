#coding=utf-8
import numpy as np
from scipy.special import logsumexp
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB as skBernoulliNB


def onehot( y=np.array([1,2,3,2,1])):
    classes = np.unique(y) #class 有序的
    y_train = np.zeros((y.shape[0], len(classes)))
    for i, c in enumerate(classes):
        y_train[y == c, i] = 1  # 每一行是 y 的one-hot表示
    print(classes,y_train)
    return classes, y_train
onehot()


X = CountVectorizer(min_df=0.001, binary=True).fit_transform(X).toarray()
class BernoulliNB():
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def _encode(self, y):
        classes = np.unique(y)
        y_train = np.zeros((y.shape[0], len(classes)))
        for i, c in enumerate(classes):
            y_train[y == c, i] = 1 #每一行是 y 的one-hot表示
        return classes, y_train

    def fit(self, X, y):
        self.classes_, y_train = self._encode(y)
        self.feature_count_ = np.dot(y_train.T, X)   # C类*n词 一行代表是不是某一类，X一j列代表出现了没第j个词，中间m个文档正好累和消去
        self.class_count_ = y_train.sum(axis=0)
        smoothed_fc = self.feature_count_ + self.alpha #给定类别下，避免各个词频为零出现，最小
        smoothed_cc = self.class_count_ + 2 * self.alpha #某类别下特征出现与否概率，与类别个数无关，与特征个数无关，只与出现与否有关，因此加2，使得出现和不出现加起来概率为1
        self.feature_log_prob_ = (np.log(smoothed_fc) -
                                  np.log(smoothed_cc.reshape(-1, 1)))
        self.class_log_prior_ = np.log(self.class_count_) - np.log(self.class_count_.sum())
        return self

    def _joint_log_likelihood(self, X):
        return (np.dot(X, self.feature_log_prob_.T) +
                np.dot(1 - X, np.log(1 - np.exp(self.feature_log_prob_)).T) +
                self.class_log_prior_)

    def predict(self, X):
        joint_log_likelihood = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(joint_log_likelihood, axis=1)]

    def predict_proba(self, X):
        joint_log_likelihood = self._joint_log_likelihood(X)
        log_prob = joint_log_likelihood - logsumexp(joint_log_likelihood, axis=1)[:, np.newaxis]
        return np.exp(log_prob)
X = CountVectorizer(min_df=0.001).fit_transform(X).toarray()
class MultinomialNB():
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def _encode(self, y):
        classes = np.unique(y)
        y_train = np.zeros((y.shape[0], len(classes)))
        for i, c in enumerate(classes):
            y_train[y == c, i] = 1
        return classes, y_train

    def fit(self, X, y):
        self.classes_, y_train = self._encode(y)
        self.feature_count_ = np.dot(y_train.T, X)
        self.class_count_ = y_train.sum(axis=0)
        smoothed_fc = self.feature_count_ + self.alpha
        smoothed_cc = smoothed_fc.sum(axis=1)
        self.feature_log_prob_ = (np.log(smoothed_fc) -
                                  np.log(smoothed_cc.reshape(-1, 1)))
        self.class_log_prior_ = np.log(self.class_count_) - np.log(self.class_count_.sum())
        return self

    def _joint_log_likelihood(self, X):
        return np.dot(X, self.feature_log_prob_.T) + self.class_log_prior_

    def predict(self, X):
        joint_log_likelihood = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(joint_log_likelihood, axis=1)]

    def predict_proba(self, X):
        joint_log_likelihood = self._joint_log_likelihood(X)
        log_prob = joint_log_likelihood - logsumexp(joint_log_likelihood, axis=1)[:, np.newaxis]
        return np.exp(log_prob)

class GaussianNB():
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_features = X.shape[1]
        n_classes = len(self.classes_)
        self.theta_ = np.zeros((n_classes, n_features))
        self.sigma_ = np.zeros((n_classes, n_features))
        self.class_count_ = np.zeros(n_classes)
        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.theta_[i] = np.mean(X_c, axis=0)
            self.sigma_[i] = np.var(X_c, axis=0)
            self.class_count_[i] = X_c.shape[0]
        self.class_prior_ = self.class_count_ / np.sum(self.class_count_)
        return self

    def _joint_log_likelihood(self, X):
        joint_log_likelihood = np.zeros((X.shape[0], len(self.classes_)))
        for i in range(len(self.classes_)):
            p1 = np.log(self.class_prior_[i])
            p2 = -0.5 * np.log(2 * np.pi * self.sigma_[i]) - 0.5 * (X - self.theta_[i]) ** 2 / self.sigma_[i]
            joint_log_likelihood[:, i] = p1 + np.sum(p2, axis=1)
        return joint_log_likelihood

    def predict(self, X):
        joint_log_likelihood = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(joint_log_likelihood, axis=1)]

    def predict_proba(self, X):
        joint_log_likelihood = self._joint_log_likelihood(X)
        log_prob = joint_log_likelihood - logsumexp(joint_log_likelihood, axis=1)[:, np.newaxis]
        return np.exp(log_prob)