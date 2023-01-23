
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os 
from PIL import Image

# training metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import seaborn as sns

# training utilities
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# feature management
from skimage import feature
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from skimage import data, exposure

random.seed(281)

# resnet features
import torch 
import torchvision.models as models
from torch import nn
from torchsummary import summary
from torchvision import transforms
import warnings

# tensorflow
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, Activation

# models
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

def plot_confusion(test_labels, test_pred, display_x_axis=True):
    cm = confusion_matrix(test_labels, test_pred)
    cm_pct = cm / cm.astype(np.float64).sum(axis=1)
    cm_pct = np.round(cm, 4) * 100

    plt.figure(figsize=(8, 5))
    ax = sns.heatmap(cm, annot=True, fmt='g', cmap='summer');
    ax.set_title(f'Confusion Matrix');

    if display_x_axis == True:

        ax.set_xlabel('Predicted')
        plt.xticks(rotation=90)
        plt.xticks(range(0, len(labels_used), 1))

        if (cm.size <= len(labels_used) ** 2):
            ax.xaxis.set_ticklabels(labels_used)

    else:

        ax.axes.get_xaxis().set_visible(False)

    ax.set_ylabel('Actual');
    plt.yticks(rotation=0)

    plt.yticks(range(0, len(labels_used), 1))
    ## For the Tick Labels, the labels should be in Alphabetical order
    if (cm.size <= len(labels_used) ** 2):
        ax.yaxis.set_ticklabels(labels_used)

    plt.show()


def display_failures(images, actual, pred):
    failed = np.nonzero(abs(actual - pred))[0]
    print(f"{len(failed)} images were misclassifed")
    ncols = 6
    if len(failed) < ncols:
        nrows = 1
        ncols = len(failed)
    else:
        nrows = int(np.ceil(len(failed) / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 4 * nrows))
    for i, ibad in enumerate(failed):
        irow = int(np.floor(i / ncols))
        icol = i % ncols
        # print(f"Rows {nrows}, Cols {ncols}")
        if nrows > 1:
            ax = axes[irow, icol]
        elif nrows == 1 and ncols == 1:
            ax = axes
        else:
            ax = axes[icol]
        ax.imshow(images[ibad])
        ax.set_title(f"Actual: {labels_used[actual[ibad]]} \nPred: {labels_used[pred[ibad]]}")
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])


def run_model_for_features(model, model_name,
                           feature_list=["Color Features", "HOG Features", "FFT Features", "LBP Features"],
                           is_GSCV=False):
    # build train/test data sets from selected features
    train_input = build_input_feature_set(train_pca, train_embeddings, feature_list)
    val_input = build_input_feature_set(val_pca, validation_embeddings, feature_list)

    # fit the model to the training data
    model.fit(train_input, train_labels)

    # test against test set
    val_pred = model.predict(val_input)
    if model_name.find('Kmeans') > 0:
        acc = 0
        f1 = 0

        # plot the confusion matrix
        plot_confusion(val_labels, val_pred, False)

    else:

        acc = accuracy_score(val_pred, val_labels)
        print(f'{model_name} validation set accuracy score: {acc * 100:9.5}')
        f1 = f1_score(val_pred, val_labels, average="weighted")
        print(f'{model_name} validation set f1 score: {f1 * 100:9.5}')

        # plot the confusion matrix
        plot_confusion(val_labels, val_pred, True)

    return (acc, f1, val_pred)
