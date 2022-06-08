#trained for 3 minutes, accuracy:0.54259

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datatable as dt
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
import re
import random

train = dt.fread("./train.tsv")
test = dt.fread("./test.tsv")
train = train.to_pandas()
test = test.to_pandas()


X_train = train["Phrase"][:30000]
Y_train = train["Sentiment"][:30000]
X_test = test["Phrase"]




pipeline = Pipeline([('ngram', CountVectorizer(ngram_range=(1,2),min_df=10)),
                     ('tfidf', TfidfTransformer()),
                     ('mlp', MLPClassifier(hidden_layer_sizes=(50),activation="relu",
                                           shuffle=True,batch_size=128)),
                      ])



pipeline = pipeline.fit(X_train,Y_train)


y_pred = pipeline.predict(X_test)

sub_file = pd.read_csv('sampleSubmission.csv',sep=',')
sub_file.Sentiment=y_pred
sub_file.to_csv('Submission.csv',index=False)
