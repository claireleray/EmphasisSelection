#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 15:19:47 2019

@author: claire
"""

from data_utils import create_dataset, read_data, write_results
from keras.models import load_model
from keras.utils import get_custom_objects
import numpy as np

def unpad(predictions,word_id_lsts):
    predictions_unpadded = []
    for i in range(0,len(word_id_lsts)):
        l = len(word_id_lsts[i])
        unpadded = predictions[:,i][1:l+1]
        values = [value[0] for value in unpadded]
        predictions_unpadded.append(values)
    return predictions_unpadded

def data_creator(X):
    batch_segment_ids = []
    batch_token_ids = []
    for i in range(0,len(X)):
        token_ids = X[i].tolist()
        segment_ids = [0 for token in X[i]]
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)
    return [np.array(batch_token_ids), np.array(batch_segment_ids)]

model_path = "model_75_epochs_no_earlystopping.hd5"
print("Trained model path : %s" % model_path)
test_filename = "data/trial_data.txt"
print("Test dataset path : %s" % test_filename)
results_path = "data/res/submission.txt"
print("Results path : %s" % results_path)

# ALBERT predictions

print("\n === ALBERT predictions ===\n")

X_test, _ , _ = create_dataset(test_filename)

model = load_model(model_path, custom_objects = get_custom_objects())
model.summary()
predictions = model.predict(data_creator(X_test))
word_id_lsts, post_lsts ,  _ ,  _ , _ , _ = read_data(test_filename)
predictions_unpadded = unpad(np.array(predictions), word_id_lsts)

write_results(word_id_lsts, post_lsts, predictions_unpadded, results_path)
print("Results written")

