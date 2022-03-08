from collections import Counter

from official.nlp import optimization  

from nltk.tokenize import word_tokenize

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow import keras
from tensorflow.keras import backend as K 
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.layers import Input, Conv2D, ReLU, Add, BatchNormalization, AveragePooling2D, Flatten, Dense, Embedding, Bidirectional, LSTM, GRU
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

import librosa
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import pathlib
import pandas as pd
import re
import seaborn as sns
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_hub as hub
import tensorflow_text as text
import urllib.request
import zipfile

fast_speed = 1.2
slow_speed = 0.8
low_noise = 0.005
high_noise = 0.02

def aug_noise(data, noise_factor):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    return augmented_data

def aug_speed(data, speed_factor):
    augmented_data = librosa.effects.time_stretch(data, speed_factor)
    return augmented_data

def aug_pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

def augment_audio(x, sr, aug_):
    if aug_ == "low_noise":
        out_ = aug_noise(x, low_noise) 
    elif aug_ == "high_noise": 
        out_ = aug_noise(x, high_noise) 
    elif aug_ == "slower":
        out_ = aug_speed(x, slow_speed)
    elif aug_ == "faster":
        out_ = aug_speed(x, fast_speed)
    elif aug_ == "pitch":
        out_ = aug_pitch(x, sr)
    elif aug_ == "slow_low_noise":
        out_ = aug_noise(x, low_noise)
        out_ = aug_speed(out_, slow_speed)
    elif aug_ == "fast_low_noise":
        out_ = aug_noise(x, low_noise)
        out_ = aug_speed(out_, fast_speed)
    elif aug_ == "slow_high_noise":
        out_ = aug_noise(x, high_noise)
        out_ = aug_speed(out_, slow_speed)
    elif aug_ == "fast_high_noise":  
        out_ = aug_noise(x, high_noise)
        out_ = aug_speed(out_, fast_speed)
    else:
        out_ = x
    return out_

def padding(array, xx, yy):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desirex width
    :return: padded array
    """
    h = array.shape[0]
    w = array.shape[1]
    a = max((xx - h) // 2,0)
    aa = max(0,xx - a - h)
    b = max(0,(yy - w) // 2)
    bb = max(yy - b - w,0)
    return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')

def get_features(file_path, aug_, max_length=216000, mfcc=40, hop_length=256, n_fft=512):
    #load the file        
    y, sr = librosa.load(file_path, sr=20480)
    #Augument Audio
    y_aug = augment_audio(y, sr, aug_)
    #Cutting
    y_cut = y_aug[:max_length]
    data = np.array([padding(librosa.feature.melspectrogram(
                                    y_cut, 
                                    n_mels=mfcc,
                                    n_fft=n_fft, 
                                    hop_length=hop_length),
                    1,844)])[0].reshape((mfcc,844,1))   
    #Taking log and adding float 
    data = np.log(data + np.finfo(float).eps)
    return data

class Audio_Generator(keras.utils.Sequence):
    def __init__(self, audio_names, labels, aug_type, batch_size) :
        self.audio_names = audio_names
        self.labels = labels
        self.aug_type = aug_type
        self.batch_size = batch_size

    def __len__(self) :
        return (np.ceil(len(self.audio_names) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx) :
        batch_x = self.audio_names[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_aug = self.aug_type[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]

        return np.array([get_features(file_path, aug) for file_path, aug 
                          in zip(batch_x, batch_aug)]), np.array(batch_y)

def relu_bn(inputs) :
    """
    After each conv layer, a Relu activation and a Batch Normalization are 
    applied

    Parameters
    ----------
    inputs: Tensor  
      Input tensor from the previous layer

    Returns
    --------
    bn: batch normalized layer
    """
    
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)

    return bn

def residual_block(x, downsample, filters, kernel_size = 3):
    """
    This function constructs a residual block. It takes a tensor x as input and 
    passes it through 2 conv layers

    Parameters
    ----------
    x: Tensor 
      Input tensor from the previous layer (or Input if it is the
    first one)

    downsample: bool 
      When true downsampling is appplied: the stride of the 
      first Conv layer will be set to 2 and the kernel size passes from 3 to 1

    filters: int 
      Number of filters applied to the data

    kernel_size: int 
      Kernel size, default value equals to 3
    
    Returns
    --------
    out: layer of the residual block calculated
    """

    y = Conv2D(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filters,
               padding="same")(x)

    y = relu_bn(y)

    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    # With the downsaple parameter set to True:
    # the strides of the first Conv are set to 2
    # the kernel size of the conv layer on the input passes from 3 to 1
    if downsample:
        x = Conv2D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)

    # The input x is added to the output y, and then the relu activation and 
    # batch normalization are applied           
    out = Add()([x, y])
    out = relu_bn(out)
    return out

def f1_l(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def create_res_net(optimizer, input_shape, num_labels):
    """
    A function to create the ResNet. It puts together the two other functions 
    already defined
    
    Parameters
    ----------
    optimizer: str 
      The name of the optimizer we want to use to compile the model

    Returns
    ----------
    Model initalized and compiled with the optimizer
    """

    inputs = Input(shape=input_shape)

    num_filters = 64
    
    t = BatchNormalization()(inputs)
    t = Conv2D(kernel_size=3,
               strides=1,
               filters=num_filters,
               padding="same")(t)
    t = relu_bn(t)

    # The residual function is called to add the skip connections
    num_blocks_list = [2, 5, 5, 2]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t,
                               downsample=(j==0 and i!=0), 
                               filters=num_filters)

        # The number of filters applied to the residual blocks are increasing 
        # (64,128,256,512)    
        num_filters *= 2
    
    # Average pooling layer to reduce the dimension of the input by computing 
    # the average values of each region
    t = AveragePooling2D(4)(t)

    # Flatten layer 
    t = Flatten()(t)

    # Dense layer to produce the probabilities of all the classes 
    outputs = Dense(num_labels, activation='softmax')(t)

    # Initalizing the model
    model_au = Model(inputs, outputs)

    # Sparse categorical crossentropy since we do not have one-hot arrays for 
    # the probabilities
    model_au.compile(optimizer= optimizer,
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy', recall, precision, f1_l])
    return model_au

class Audio_Generator_CNNLSTM(keras.utils.Sequence):
    def __init__(self, audio_names, labels, aug_type, batch_size) :
        self.audio_names = audio_names
        self.labels = labels
        self.aug_type = aug_type
        self.batch_size = batch_size

    def __len__(self) :
        return (np.ceil(len(self.audio_names) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx) :
        batch_x = self.audio_names[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_aug = self.aug_type[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
        Y = np.array(batch_y).reshape(2,10)
        
        return np.array([get_features(file_path, aug) for file_path, aug 
                          in zip(batch_x, batch_aug)]).reshape(2, 10, 40, 844, 1), Y


def ConvLSTM_Model(frames, pixels_x, pixels_y, channels, categories):
    trailer_input  = tf.keras.layers.Input(shape=(frames, pixels_x, pixels_y, channels)
                    , name='trailer_input')
    
    first_ConvLSTM = tf.keras.layers.ConvLSTM2D(filters=20, 
                                kernel_size=(3, 3), 
                                data_format='channels_last', 
                                recurrent_activation='hard_sigmoid',
                                activation='tanh',
                                padding='same', 
                                return_sequences=True)(trailer_input)
    first_BatchNormalization = tf.keras.layers.BatchNormalization()(first_ConvLSTM)
    first_Pooling = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2), 
                                 padding='same', 
                                 data_format='channels_last')(first_BatchNormalization)
    
    second_ConvLSTM = tf.keras.layers.ConvLSTM2D(filters=10, 
                                 kernel_size=(3, 3), 
                                 data_format='channels_last',
                                 padding='same', 
                                 return_sequences=True)(first_Pooling)
    second_BatchNormalization = tf.keras.layers.BatchNormalization()(second_ConvLSTM)
    second_Pooling = tf.keras.layers.MaxPooling3D(pool_size=(1, 3, 3), 
                                  padding='same', 
                                  data_format='channels_first')(second_BatchNormalization)
    outputs = branch(second_Pooling, "cat")
    seq = Model(inputs=trailer_input, outputs=outputs, name='model_lstm')
    
    return seq

def branch(last_convlstm_layer, name):
    
    branch_ConvLSTM = tf.keras.layers.ConvLSTM2D(filters=5, kernel_size=(3, 3),
                                 data_format='channels_last',
                                 stateful = False,
                                 kernel_initializer='random_uniform',
                                 padding='same', 
                                 return_sequences=True)(last_convlstm_layer)
    branch_Pooling =tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2), 
                                  padding='same', 
                                  data_format='channels_last')(branch_ConvLSTM)
    flat_layer = tf.keras.layers.TimeDistributed(Flatten())(branch_Pooling)
    first_Dense = tf.keras.layers.TimeDistributed(Dense(512,))(flat_layer)
    second_Dense = tf.keras.layers.TimeDistributed(Dense(32,))(first_Dense)
    
    target = tf.keras.layers.TimeDistributed(Dense(num_labels, activation = "softmax"), name=name)(second_Dense)
    
    return target


########################## ALEX NET ########################       

def model_alex(opt, input_shape, num_labels):
    model_alex_a = keras.models.Sequential([
        keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=input_shape),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_labels, activation='softmax')
    ])

    model_alex_a.compile(optimizer= opt,
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
    
    return model_alex_a

########################## VGG NET ########################       

def model_vgg(opt, input_shape, num_labels):
    model_v = keras.models.Sequential([
    keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', input_shape=input_shape),
    keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
    keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
    keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
    keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
    keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
    keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dense(num_labels, activation='softmax')
            ])
    
    model_v.compile(optimizer= opt,
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
    
    return model_v

###################################### TEXT BI LSTM ###########################

def clean_text(data):
    data=re.sub(r"(#[\d\w\.]+)", '', data)
    data=re.sub(r"(@[\d\w\.]+)", '', data)
    data=word_tokenize(data)
    return data

def create_embedding_matrix(filepath,word_index,embedding_dim):
    vocab_size=17143
    embedding_matrix=np.zeros((vocab_size,embedding_dim))
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            word,*vector=line.split()
            if word in word_index:
                idx=word_index[word]
                if idx > 17142:
                    idx = 17142
                embedding_matrix[idx] = np.array(vector,dtype=np.float32)[:embedding_dim]
    return embedding_matrix

def precision_m(y_true, y_pred):
    y_pred = K.round(y_pred)
    true_positives = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    false_positives = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    predicted_positives = true_positives+false_positives
    return true_positives / (predicted_positives + K.epsilon())

def recall_m(y_true, y_pred):
    y_pred = K.round(y_pred)
    true_positives = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    false_negatives = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)
    possible_positives = true_positives+false_negatives
    return true_positives / (possible_positives + K.epsilon())

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    f1 = 2*((precision*recall)/(precision+recall+K.epsilon()))
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

def model_text(opt,vocab_size,embed_num_dims,max_seq_len,embedd_matrix, num_labels):
    embedd_layer=Embedding(vocab_size,
                           embed_num_dims,
                           input_length=max_seq_len,
                           weights=[embedd_matrix],
                           trainable=False)
    gru_output_size=128
    bidirectional=True
    model=Sequential()
    model.add(embedd_layer)
    model.add(Bidirectional(LSTM(units=gru_output_size,dropout=0.2,recurrent_dropout=0.2)))
    model.add(Dense(num_labels, activation='softmax'))
    model.compile(optimizer= opt,
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy', f1_l, recall, precision])
    return model    
        
################################ BERT MODEL ###############################
    
def get_features_tx(transcripts_batch):
    text_preprocessed = bert_preprocess_model(transcripts_batch)
    bert_results = bert_model(text_preprocessed)
    return bert_results["sequence_output"]

class TextGenerator(keras.utils.Sequence):
    def __init__(self, transcripts, labels, batch_size) :
        self.transcripts = transcripts
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self) :
        return (np.ceil(len(self.transcripts) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx) :
        batch_x = self.transcripts[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]

        return np.array(get_features_tx(batch_x)), np.array(batch_y)

def model_bert(optimizer, bert_preprocess_model, num_labels):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(bert_preprocess_model, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(bert_model, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(num_labels, activation='softmax', name='classifier')(net)
    model = Model(text_input, net)

    model.compile(optimizer= optimizer,
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy', f1_m, f1_l, precision_m, recall_m])
    return model

################################## COMBINED MODEL ################################

def model_combined(model_audio, model_txt, optimizer):
    combined = tf.keras.layers.Concatenate()([model_audio.output, model_txt.output])
    combined = tf.keras.layers.Dense(1024, activation = "relu")(combined)
    combined = tf.keras.layers.Dense(64, activation = "relu")(combined)
    outputs = tf.keras.layers.Dense(num_labels, activation = 'softmax')(combined)
    model_cb = keras.Model([model_audio.input, model_txt.input], outputs)
    return model_cb

class CombinedGenerator(keras.utils.Sequence):
    def __init__(self, audio_names, labels, aug_type, transcript, batch_size) :
        self.audio_names = audio_names
        self.labels = labels
        self.aug_type = aug_type
        self.transcript = transcript
        self.batch_size = batch_size

    def __len__(self) :
        return (np.ceil(len(self.audio_names) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx) :
        batch_x = self.audio_names[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_aug = self.aug_type[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_trans = self.transcript [idx * self.batch_size : (idx+1) * self.batch_size]

        return [np.array([get_features(file_path, aug) for file_path, aug 
                          in zip(batch_x, batch_aug)]), np.array(batch_trans)], np.array(batch_y)    