# Okay so I've already used supervised ML to predict the what a picture is of
# I wonder if TensorFlow can do it better?
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# /// Read CSV into DataFrame, labeling columns with headers, not specifying a column (index_col) to label each row
D = pd.read_csv("/Users/douglasmckinley/PycharmProjects/DAWPS/DataSciencePlayground/fresh_juicy_data/tensorflow_vs_supervised_scikit.csv", index_col=False)  # popular LabVIEW clusters vs. ratings
print(D.head())

# /// Treat 1st column as thing to predict (dependent var), and other columns as predictors (independent variables)
colnames = D.columns.values.tolist()
yvals = D[colnames[0]]
Xvals = D[colnames[1:]]


# NOTE: THE FOLLOWING IS A FINE WAY TO GET STARTED, BUT THE ACCURACY VALUES MAY BE *OVER-OPTIMISTIC*.
# Trains a classifier and tests it on the *Training data*, which *CAN OFTEN LEAD TO OVER-FITTING*.
# DO NOT BELIEVE THAT YOUR ACCURACY will be as high, in practice, as reported here.
# THE RIGHT WAY TO DO IT IS SHOWN BELOW (WITH A TRAINING SET AND A TESTING SET)
# Returns the classifier.
def train_self_test(X, y):
    # /// Train a supervised learning model
    classifier = RandomForestClassifier()
    classifier.fit(X, y)

    # /// Compute accuracy over the entire set (NOTE: may be over-optimistic. Use cross-validation as shown below.)
    predictions = classifier.predict(X)
    print("Best-case Accuracy: %0.2f" % accuracy_score(y, predictions))
    print("Best-case F1: %0.2f" % f1_score(y, predictions, average='micro'))
    print("\nConfusion matrix\n", confusion_matrix(y, predictions))
    print("\nBreakdown by class...\n", classification_report(y, predictions))
    return classifier


# NOTE: BELOW IS THE RIGHT WAY TO DO IT, WITH A 10-FOLD CROSS VALIDATION.
# THIS USES 9/10ths OF THE DATA TO TRAIN THE MODEL, 1/10th FOR TESTING.
# IT THEN REPEATS 10 TIMES, SO THAT EVERY DATA POINT GETS TO BE A TEST POINT ONCE.
# It's quite a bit slower than just training and testing on the whole dataset.

# /// Train and test a supervised machine learning classifier, with 10-fold cross-validation, and print report
# Returns the classifier
def trainTest(X, y):
    classifier = RandomForestClassifier()
    nfolds = 10
    ttlaccuracy = np.mean(cross_val_score(classifier, X, y, scoring='accuracy', cv=nfolds))
    f1accuracy = np.mean(cross_val_score(classifier, X, y, scoring='f1_micro', cv=nfolds))
    print("Cross-validation accuracy: %0.2f" % ttlaccuracy)
    print("Cross-validation F1: %0.2f" % f1accuracy)
    return classifier


train_self_test(Xvals, yvals)
trainTest(Xvals, yvals)



# Okay let's try it with tensorflow
import tflearn
#from tflearn

# let's iterate thru the CSV and populate a test and training set of data
#for rowNum in range(0, len(labels))

#input_layer = tflearn.input_data(shape=[None, 784])


# val_acc: 0.9646 with no regularizers, no dropout, and 16 units in each of the fully connected layers
# val_acc: 0.9924 with no regularizers, no dropout, and 16 units in each of the fully connected layers
# val_acc: 0.9971 with no regularizers, no dropout, and 64 units in each of the fully connected layers

# val_acc: 0.9971 with regularizers but no dropout, 64 units each layer
#dense1 = tflearn.fully_connected(input_layer, 4, activation='tanh', weight_decay=0.001)  # regularizer='L2')
# val_acc: 0.9962 with dropout layer and regularizers, 64 units each layer
# dropout1 = tflearn.dropout(dense1, 0.8)
#dense2 = tflearn.fully_connected(dense1, 4, activation='tanh', weight_decay=0.001)  # regularizer='L2')
#softmax = tflearn.fully_connected(dense2, 10, activation='softmax')

#sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
#top_k = tflearn.metrics.Top_k(3)
#net = tflearn.regression(softmax, optimizer=sgd, metric=top_k, loss='categorical_crossentropy')
#model = tflearn.DNN(net, tensorboard_verbose=0)
#model.fit(X, Y, n_epoch=20, validation_set=(testX, testY), show_metric=True, run_id="dense_model")