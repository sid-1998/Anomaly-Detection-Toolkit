import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


def lymphography_dataset():
    path = "./datasets/lymphography.csv"
    df = pd.read_csv(path)
    df.Label[df.Label == 3] = 0
    df.Label[df.Label == 4] = 1
    df.Label[df.Label == 1] = 1
    df.Label[df.Label == 2] = 0

    df_norm = df[df.Label == 0]
    df_anom = df[df.Label == 1]

    ds_norm = df_norm.values
    ds_anom = df_anom.values

    X_train = ds_norm[:100, :-1]
    Y_train = ds_norm[:100, -1]

    l = ds_norm.shape[0] - X_train.shape[0] 
    no_of_test_samples = l + ds_anom.shape[0]
    no_of_features = X_train.shape[1]

    X_test = np.zeros((no_of_test_samples, no_of_features))
    Y_test = np.zeros((no_of_test_samples,))

    X_test[:l, :] = ds_norm[100:, :-1]
    X_test[l:, :] = ds_anom[:,:-1]
    print(X_test.shape)
    Y_test[:l,] = ds_norm[100:, -1]
    Y_test[l:,] = ds_anom[:, -1]
    return X_train, Y_train, X_test, Y_test, ds_anom, ds_norm

def pageblocks_dataset():
    path = "./datasets/page-blocks.csv"
    df = pd.read_csv(path)
    df.label[df.label == 1] = 0
    df.label[df.label == 2] = 1
    df.label[df.label == 3] = 1
    df.label[df.label == 4] = 1
    df.label[df.label == 5] = 1

    df_norm = df[df.label == 0]
    df_anom = df[df.label == 1]

    ds_norm = df_norm.values
    ds_anom = df_anom.values

    X_train = ds_norm[:4700, :-1]
    Y_train = ds_norm[:4700, -1]

    l = ds_norm.shape[0] - X_train.shape[0] 
    no_of_test_samples = l + ds_anom.shape[0]
    no_of_features = X_train.shape[1]

    X_test = np.zeros((no_of_test_samples, no_of_features))
    Y_test = np.zeros((no_of_test_samples,))

    X_test[:l, :] = ds_norm[4700:, :-1]
    X_test[l:, :] = ds_anom[:,:-1]
    print(X_test.shape)
    Y_test[:l,] = ds_norm[4700:, -1]
    Y_test[l:,] = ds_anom[:, -1]
    return X_train, Y_train, X_test, Y_test, ds_anom, ds_norm

def postoperative_dataset():
    path = "./datasets/postop.csv"
    df = pd.read_csv(path)
    df_norm = df[df.Label == 0]
    df_anom = df[df.Label == 1]
    ds_norm = df_norm.values
    ds_anom = df_anom.values

    X_train = ds_norm[:50, :-1]
    Y_train = ds_norm[:50, -1]

    l = ds_norm.shape[0] - X_train.shape[0] 
    no_of_test_samples = l + ds_anom.shape[0]
    no_of_features = X_train.shape[1]

    X_test = np.zeros((no_of_test_samples, no_of_features))
    Y_test = np.zeros((no_of_test_samples,))

    X_test[:l, :] = ds_norm[50:, :-1]
    X_test[l:, :] = ds_anom[:,:-1]
    print(X_test.shape)
    Y_test[:l,] = ds_norm[50:, -1]
    Y_test[l:,] = ds_anom[:, -1]
    return X_train, Y_train, X_test, Y_test, ds_anom, ds_norm

def cancer_dataset():
    path = "./datasets/cancer.csv"
    df = pd.read_csv(path)
    df.Label[df.Label==1] = 0
    df.Label[df.Label==-1] = 1

    df_norm = df[df.Label == 0]
    df_anom = df[df.Label == 1]

    ds_norm = df_norm.values
    ds_anom = df_anom.values

    X_train = ds_norm[:400, :-1]
    Y_train = ds_norm[:400, -1]

    l = ds_norm.shape[0] - X_train.shape[0] 
    no_of_test_samples = l + ds_anom.shape[0]
    no_of_features = X_train.shape[1]

    X_test = np.zeros((no_of_test_samples, no_of_features))
    Y_test = np.zeros((no_of_test_samples,))

    X_test[:l, :] = ds_norm[400:, :-1]
    X_test[l:, :] = ds_anom[:,:-1]
    print(X_test.shape)
    Y_test[:l,] = ds_norm[400:, -1]
    Y_test[l:,] = ds_anom[:, -1]
    return X_train, Y_train, X_test, Y_test, ds_anom, ds_norm