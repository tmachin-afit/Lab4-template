# Lab 4
We will use the machine learning workflow to solve our problem again. Thus, this lab is broken into those parts in order to solve our problem
## Problem 
We need to determine if a cat is in a picture. By cat, I mean any kind of cat. We can assume the pictures can be slimmed down to a `32x32` image without any loss of information

## Step 1 Get some data
In this lab we will be working with the slimmed down version of the CIFAR dataset called CIFAR-10. This dataset has 10 labels in it of different kinds of objects from animals (like cats) to houses, vehicles, etc. 

The dataset can be downloaded here:
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

Or you can use the build in keras dataset:
```python
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
```
Remember your problem is to only determine if the picture is a cat or not a cat.

## Step 2 Measures of Success
Since we want to know if a cat is a picture or not we are looking at a binary classification problem. Thus, we are going to use `binary_crossentropy` as a loss and `accuracy` as a metric.

Since getting a realistic baseline for an estimator that uses images (without using a CNN) is difficult, our baseline will be a simulated coin flip. The simulated coin flip we will assume gets an accuracy of 50%. Thus, we want to be better than 50% in order to have a model that is any way useful. Ideally we want the accuracy as high as possible but do not get stressed about the performance because this is lab. 

Also in this section have python print what the accuracy would be if we always guessed `not cat` for this dataset.

## Step 3 Prepare out Data
We want to remember to normalize the pixels values. We are going to keep the images as images this time and not flatten them. Thus, we will have images with shape `32 x 32 x 3`.

We also need to change the output to be cat or not cat. Here are the class labels:
```python
label_names = {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck',
    }
```

## Step 4 Determine an Evaluation Method
There is already a train/test split for this data. We will use this test split as the test data, but you will need to make your own validation data from the training data.

## Step 5 Develop a model
Develop a model that compiles given our input, output, and loss functions. Remember you will want to increase the number of filters (channels) as you get deeper in the network which should be reducing the image sizes (using some technique like valid padding CNN layers or max pooling layers). We also probably want to use a Global Average Pooling layer as our final or just before final layer.

## Step 6 Overfit Model
After you get the model to compile we need to overfit to the training data. For this we should make a couple of layers of CNNs stacked on top of each other. Remember it is a good idea to increase the number of channels while reducing the output image size. It may take a long time to train especially if you are using CPUs so do not stress if you cannot achieve a very high accuracy. However, you should be better than our baseline on the training set which would be over 50% accuracy. 

## Step 7 Regularize Model
Now that our model can do well on the training data we want to make sure we have generalized. If we have not generalized well to our validation set, add some regularization to increase your validation performance. For example, you could add dropout, or L1 or L2 kernel regularizers or data augmentation. 

## After the 7-Step Process
### Visualization
Without proper visualization/statistics it is very difficult to uncover problems in your model. Thus, we will add a model summary, confusion matrix, and statistics to show how well we can classify the cat images. 
To get a model summary use the `model.summary()` function for Keras Models. This will print out the model architecture to the console.
For the statistics use the function `sklearn.metrics.classification_report` from the scikit-learn python package. 
For the confusion matrix use the code from [here](https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html) as an example (also included in the template `lab4.py`. Run the stats and confusion matrix on the validation data at first to tune your model but for you final submission switch it to run on the test set. Remember we do not want to know any of the results of the test set until we are ready to publish (or submit the lab).

Print out the stats from the sklearn classification report and show the confusion matrix on your test dataset as your final entry. 

### Analysis and Improvement
After you get the visualization tools working, use the visualization data to comment on your recall statistic for the cat class. Recall is how often your model can actually pick out that specific class from the other classes. 

Most likely the recall statistic is very poor (you recall no cats). This is due to a class imbalance. There are many more not cats in this dataset and our model favors to say not cat and achieve high accuracy at the expense of recall for cats. To correct this imbalance we are going to weight the cat class higher. 

When calling the fit function supply this parameter (you have to fill in the `total_not_cats` and `total_cats` yourself):
```python
class_weight = {
        0: 1 / total_not_cats,
        1: 1 / total_cats,
    }
```
Where `total_cats` are the total samples that are cats in the dataset and `total_not_cats` is the total number of samples that are not cats.

Retrain your model with this new weighting and add a comment on the cat recall statistic. You may want to tweak your network to increase the cat recall. For reference my cat recall was 0.00 before the weighting and >0.50 after.  

The final entry should show the stats and visualizations for the model with the new weights, since this is our final "best" model.   
