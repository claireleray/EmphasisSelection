#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 17:53:27 2019

@author: claire
"""

from data_utils import create_dataset
from bert4keras.bert import build_bert_model
#from bert4keras.backend import keras, set_gelu
#from bert4keras.snippets import sequence_padding, get_all_attributes
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import numpy as np

#locals().update(get_all_attributes(keras.layers))

def data_generator(X,y,w,batch_size):
    while True:
        batch_segment_ids = []
        batch_token_ids = []
        batch_labels = []
        labels = []
        batch_weights = []
        weights = []
        for i in range(0,len(X)):
            token_ids = X[i].tolist()
            segment_ids = [0 for token in X[i]]
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            labels.append(y[i])
            weights.append(w[i])
            if len(batch_token_ids) == batch_size:
                for j in range(0,y.shape[1]):
                    batch_labels.append(np.array(labels)[:,j].tolist())
                    batch_weights.append(np.array(weights)[:,j])
                yield [np.array(batch_token_ids), np.array(batch_segment_ids)], batch_labels, batch_weights
                batch_token_ids, batch_segment_ids, batch_labels, weights, batch_weights,labels = [], [], [], [], [], []

train_filename = "data/train.txt"
print("Data path : %s" % train_filename)
model_path = "model_100_epochs_batch64_earlystopping.hd5"
print("Trained model path : %s" % model_path)
config_path = "bert_base_cased/albert_config_base.json"
print("Config path : %s" % config_path)
checkpoint_path = "bert_base_cased/bert_model.ckpt"
print("Checkpoint path : %s\n" % checkpoint_path)

# Dataset generation
print(" === Dataset generation ===\n")
X_train, y_train, weights = create_dataset(train_filename)
maxlen = len(y_train[0])

#embedding_model = api.load("glove-twitter-25")

BATCH_SIZE = 64
INIT_LR = 10e-5
NB_EPOCHS = 100

# ALBERT model
print("\n === ALBERT model configuration ===\n")
bert = build_bert_model(config_path, checkpoint_path, with_pool = True, albert = True, return_keras_model = False)

output = Dropout(rate=0.1)(bert.model.output)
output_list = [Dense(1,
               activation='sigmoid',
               kernel_initializer=bert.initializer,
               name = "Output_" + str(i+1))(output) for i in range(0,maxlen)]
model = Model(bert.model.input, output_list)

lossWeights = {}
losses = {}
for i in range(0, maxlen):
    lossWeights["Output_" + str(i+1)] = np.count_nonzero(X_train, axis = 0)
    losses["Output_" + str(i+1)] = 'binary_crossentropy'

model.compile(
    loss=losses,
    optimizer=Adam(lr = INIT_LR, decay = INIT_LR//NB_EPOCHS),
    metrics=['accuracy'],
    loss_weights=lossWeights
)
model.summary()

# ALBERT training
print("\n === ALBERT training ===\n")

nb_ones = np.count_nonzero(y_train)
nb_elements = y_train.shape[0]*y_train.shape[1]
class_weight_one = nb_elements/nb_ones
print("Class weight for 1 : %s" % class_weight_one)

model.fit_generator(data_generator(X_train,y_train,weights,BATCH_SIZE),
                    steps_per_epoch = (len(X_train)//BATCH_SIZE)+1,
                    callbacks = [EarlyStopping(monitor="loss", patience = 5, verbose = 1)],
                    epochs = NB_EPOCHS)

model.save(model_path, include_optimizer=True)