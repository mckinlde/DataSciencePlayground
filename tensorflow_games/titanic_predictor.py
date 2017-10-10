# Rock on, tensorflow time

# Let's predict who survives the titanic

import numpy as np
import tflearn

# get some fresh juicy data
from tflearn.data_utils import load_csv
data, labels = load_csv('fresh_juicy_data/titanic_dataset.csv', target_column=0,
                        categorical_labels=True, n_classes=2)

# I like my data like I like my bolonga: preprocessed

#def preprocess(data, columns_to_ignore):
