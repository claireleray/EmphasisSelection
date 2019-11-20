#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 17:53:27 2019

@author: claire
"""

from read_data import create_dataset, write_results
from bert4keras.bert import build_bert_model
import gensim.downloader as api
from bert4keras.train import PiecewiseLinearLearningRate
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import *
import numpy as np

def unpad(predictions,word_ids):
    predictions_unpadded = []
    for i in range(0,len(predictions)):
        l = len(word_ids[i])
        predictions_unpadded.append(predictions[i][0:l].tolist())
    return np.array(predictions_unpadded)

filename = "data/trial_data.txt"
dict_path = "albert_base_zh/vocab.txt"
config_path = "albert_base_zh/albert_config_base.json"
results_path = "data/res/submission.txt"
ground_truth_filename = "data/ref/gold.txt"
checkpoint_path = None

word_ids,posts, X_train, X_test, y_train, y_test, maxlen = create_dataset("data/trial_data.txt",ground_truth_filename)

# embedding_model = api.load("glove-twitter-25")

# ALBERT model

model = build_bert_model(config_path, checkpoint_path, with_mlm = True, application = 'lm', albert = True, return_keras_model = True)

segments_ids_train = np.array([[1]*len(x) for x in X_train])
segments_ids_test = np.array([[1]*len(x) for x in X_test])

output = Lambda(lambda x: x[:, 0])(model.output)
output = Dense(maxlen, activation='sigmoid')(output)
model = Model(model.input, output)

model.compile(
    loss='binary_crossentropy',
    # optimizer=Adam(1e-5),  # 用足够小的学习率
    optimizer=PiecewiseLinearLearningRate(Adam(1e-4), {1000: 1, 2000: 0.1}),
    metrics=['accuracy']
)
model.summary()

model.fit(x=[X_train,segments_ids_train],y=y_train,verbose=1)

predictions = model.predict([X_test,segments_ids_test])
predictions_unpadded = unpad(predictions,word_ids)

write_results(word_ids,posts,predictions_unpadded,results_path)