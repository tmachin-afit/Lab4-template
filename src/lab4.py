import datetime
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Input, Dense, Conv2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import RMSprop


def get_data_with_preprocessing(cat_indices):
    # label_names = {
    #     0: 'airplane',
    #     1: 'automobile',
    #     2: 'bird',
    #     3: 'cat',
    #     4: 'deer',
    #     5: 'dog',
    #     6: 'frog',
    #     7: 'horse',
    #     8: 'ship',
    #     9: 'truck',
    # }
    (x_train, y_train_raw), (x_test, y_test_raw) = cifar10.load_data()

    # do some pre-processsing to normalize and sort out cat class

    return (x_train, y_train_cats), (x_test, y_test_cats)


def main():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    cat_indices = [3]

    (x_train, y_train_cats), (x_test, y_test_cats) = get_data_with_preprocessing(cat_indices)

    train_model = True
    if not os.path.exists("model.h5") or train_model:
        # train your model. this should include the early models as well (but early models should not run in the finished lab)
        model = None

        # save your model
        model.save('model.h5')
    else:
        # load your model
        model = load_model('model.h5')

    # get visualization data
    visualize_test = True
    if visualize_test:
        x_visualize = x_test
        y_visualize = y_test_cats
    else:
        x_visualize = x_train
        y_visualize = y_train_cats

    # make the prediction and true classes
    y_pred = None  # add your prediction code here using x_visualize

    # get the true classes and the class names
    y_true = y_visualize
    class_names = ['not cat', 'cat']

    # make the stats and confusion matrix
    print(sklearn.metrics.classification_report(y_true, y_pred, target_names=class_names))

    confusion_matrix = sklearn.metrics.confusion_matrix(y_pred=y_pred,
                                                        y_true=y_true)

    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        this function is from https://sklearn.org/auto_examples/model_selection/plot_confusion_matrix.html

        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(confusion_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(confusion_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    plt.show()

    # comment on your recall statistic for the cat class in the final model you made

    # Retrain your model with the weighting and add a comment on the cat recall statistic.


if __name__ == "__main__":
    main()
