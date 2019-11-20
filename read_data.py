#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 09:57:09 2019

@author: claire
"""
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer
import numpy as np

def read_data(filename):
    """
    This function reads the data from .txt file.
    :param filename: reading directory
    :return: lists of word_ids, words, emphasis probabilities, POS tags
    """
    lines = read_lines(filename) + ['']
    word_id_lst, word_id_lsts =[], []
    post_lst, post_lsts = [], []
    bio_lst , bio_lsts = [], []
    freq_lst, freq_lsts = [], []
    e_freq_lst, e_freq_lsts = [], []
    pos_lst, pos_lsts =[], []
    for line in lines:
        if line:
            splitted = line.split("\t")
            word_id = splitted[0]
            words = splitted[1]
            bio= splitted[2]
            freq = splitted[3]
            e_freq = splitted[4]
            pos = splitted[5]

            word_id_lst.append(word_id)
            post_lst.append(words)
            bio_lst.append(bio)
            freq_lst.append(freq)
            e_freq_lst.append(e_freq)
            pos_lst.append(pos)

        elif post_lst:
            word_id_lsts.append(word_id_lst)
            post_lsts.append(post_lst)
            bio_lsts.append(bio_lst)
            freq_lsts.append(freq_lst)
            e_freq_lsts.append(e_freq_lst)
            pos_lsts.append(pos_lst)
            word_id_lst =[]
            post_lst =[]
            bio_lst =[]
            freq_lst =[]
            e_freq_lst =[]
            pos_lst =[]
    return word_id_lsts, post_lsts, bio_lsts, freq_lsts, e_freq_lsts, pos_lsts


def read_lines( filename):
    with open(filename, 'r') as fp:
        lines = [line.strip() for line in fp]
    return lines


def write_results(word_id_lsts, words_lsts, e_freq_lsts, write_to):
    """
    This function writes results in the format.
    :param word_id_lsts: list of word_ids
    :param words_lsts: list of words
    :param e_freq_lsts: lists of emphasis probabilities
    :param write_to: writing directory
    :return:
    """


    with open(write_to, 'w') as out:
        sentence_id=""
        # a loop on sentences:
        for i in range(len(words_lsts)):
            # a loop on words in a sentence:
            for j in range(len(words_lsts[i])):
                # writing:
                if sentence_id ==i:
                    to_write = "{}\t{}\t{}\t".format(word_id_lsts[i][j], words_lsts[i][j], e_freq_lsts[i][j])
                    out.write(to_write + "\n")
                else:
                    out.write("\n")
                    to_write = "{}\t{}\t{}\t".format(word_id_lsts[i][j], words_lsts[i][j], e_freq_lsts[i][j])
                    out.write(to_write + "\n")
                    sentence_id = i
        out.write("\n")
        out.close()
        
        
def tokenize_and_pad(sentences,y) :
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    print ("Tokenize the first sentence:")
    print (tokenized_texts[0])
    MAX_LEN = max([len(x) for x in tokenized_texts])
    # Pad our input tokens
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    
    # Pad y
    y_padded = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    
    return input_ids, y_padded, MAX_LEN

def write_ground_truth(i_test,original_filename,new_filename):
    word_ids, posts, bios, freqs, e_freqs, poss = read_data(original_filename)
    new_word_ids = np.take(word_ids,i_test)
    new_posts = np.take(posts,i_test)
    new_bios= np.take(bios,i_test)
    new_freqs = np.take(freqs,i_test)
    new_e_freqs = np.take(e_freqs,i_test)
    new_poss = np.take(poss,i_test)
    with open(new_filename,'w') as f:
        for i in range(0,len(i_test)):
            for j in range(0,len(new_word_ids[i])):
                line = [new_word_ids[i][j],new_posts[i][j],new_bios[i][j],new_freqs[i][j],new_e_freqs[i][j],new_poss[i][j]]
                f.write("\t".join(line) + "\n")
            f.write("\n")
        
def create_dataset(filename,ground_truth_filename):
    # Train and test datasets
            
    word_ids, posts, bios, freqs, e_freqs, poss = read_data(filename)

    X = posts
    y = e_freqs
    
    print(X[0])
    print(y[0])
    
    X_boundaries = ['[CLS] ' + " ".join(words) + ' [SEP]' for words in X]
    
    print(X_boundaries[0])
    
    input_ids, y_padded, maxlen = tokenize_and_pad(X_boundaries, y)
    
    indices = [i for i in range(0,len(X))]
    
    X_train, X_test, y_train, y_test, i_train, i_test = train_test_split(input_ids, y_padded, indices, test_size=0.3)
    
    word_ids_test = np.take(word_ids,i_test)
    posts_test = np.take(posts,i_test)
    
    write_ground_truth(i_test,filename,ground_truth_filename)
    
    return word_ids_test, posts_test, X_train, X_test, np.array(y_train), np.array(y_test), maxlen
    
    
    

    

