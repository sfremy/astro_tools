import os
import matplotlib.pyplot as plt
import numpy as np
import astropy
from tensorflow.keras.models import load_model
from tensorflow.python.client import device_lib


def mk_set(postive_matrix, negative_matrix):
    
    X_train = np.vstack((postive_matrix, negative_matrix))

    y_train = np.hstack((np.ones_like(postive_matrix[:,0]), np.zeros_like(negative_matrix[:,0])))

    sidx = np.arange(X_train.shape[0])

    np.random.shuffle(sidx)

    X_train = X_train[sidx, :]

    y_train = y_train[sidx]
    
    return X_train, y_train


def nstr(s, d=2):
    
    return str(np.around(s, d))


def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def prep_matrix(protomatrix, n_sigma):
    tm1 = []

    for fold in protomatrix:
        fold_sigma =  astropy.stats.sigma_clipped_stats(fold)[2]
        folds_norm = (1 - fold) / (n_sigma*fold_sigma) 
        tm1.append(folds_norm)

    p_final = np.vstack(tm1)

    matrix = p_final.reshape((p_final.shape[0], p_final.shape[1], 1))
    
    return matrix