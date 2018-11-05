from keras.models import load_model
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Lambda
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn import metrics
from keras.callbacks import Callback
from keras.layers.normalization import BatchNormalization
from keras.layers import Bidirectional
from keras.layers import TimeDistributed
from numpy import cumsum
from numpy import array
from random import random
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.preprocessing import sequence
from keras import optimizers
import keras.layers as layers
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from keras.layers import Input, Embedding, LSTM, GRU, SimpleRNN, Dense, Lambda
from keras.models import Sequential, Model
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import numpy as np
import librosa

mlen = 300

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.01
set_session(tf.Session(config=config))


adam_half = optimizers.Adam(lr=0.0005)
adam_half_2 = optimizers.Adam(lr=0.0002)


# f1 score ftn.


mse_crs = load_model('model/total_s_rmse_cnn_rnnself-12-0.7714-f0.4412.hdf5')


def make_data(filename):
    data = np.zeros((1, mlen, 128))
    data_conv = np.zeros((1, mlen, 128, 1))
    data_rmse = np.zeros((1, mlen, 1))
    data_srmse = np.zeros((1, mlen, 128))
    data_srmse_conv = np.zeros((1, mlen, 128, 1))
    data_s_rmse = np.zeros((1, mlen, 129))
    data_s_rmse_conv = np.zeros((1, mlen, 129, 1))
    y, sr = librosa.load(filename)
    D = np.abs(librosa.stft(y))**2
    ss, phase = librosa.magphase(librosa.stft(y))
    rmse = librosa.feature.rmse(S=ss)
    rmse = rmse / np.max(rmse)
    rmse = np.transpose(rmse)
    S = librosa.feature.melspectrogram(S=D)
    S = np.transpose(S)
    Srmse = np.multiply(rmse, S)
    if len(S) >= mlen:
        data[0][:, :] = S[-mlen:, :]
        data_conv[0][:, :, 0] = S[-mlen:, :]
        data_rmse[0][:, 0] = rmse[-mlen:, 0]
        data_srmse[0][:, :] = Srmse[-mlen:, :]
        data_srmse_conv[0][:, :, 0] = Srmse[-mlen:, :]
        data_s_rmse[0][:, 0] = rmse[-mlen:, 0]
        data_s_rmse[0][:, 1:] = S[-mlen:, :]
        data_s_rmse_conv[0][:, 0, 0] = rmse[-mlen:, 0]
        data_s_rmse_conv[0][:, 1:, 0] = S[-mlen:, :]
    else:
        data[0][-len(S):, :] = S
        data_conv[0][-len(S):, :, 0] = S
        data_rmse[0][-len(S):, 0] = np.transpose(rmse)
        data_srmse[0][-len(S):, :] = Srmse
        data_srmse_conv[0][-len(S):, :, 0] = Srmse
        data_s_rmse[0][-len(S):, 0] = np.transpose(rmse)
        data_s_rmse[0][-len(S):, 1:] = S
        data_s_rmse_conv[0][-len(S):, 0, 0] = np.transpose(rmse)
        data_s_rmse_conv[0][-len(S):, 1:, 0] = S
    return data, data_conv, data_rmse, data_srmse, data_srmse_conv, data_s_rmse, data_s_rmse_conv


def pred_into(filename):
    data, data_conv, data_rmse, data_srmse, data_srmse_conv, data_s_rmse, data_s_rmse_conv = make_data(
        filename)
    att_source = np.zeros((1, 64))
    z = mse_crs.predict([data_s_rmse_conv, data_s_rmse, att_source])[0]
    y = int(np.argmax(z))
    if y == 0:
        print("High rise: LLH%, MLH%, HLH%")
    elif y == 1:
        print("Low rise: MMH%")
    elif y == 2:
        print("Fall rise: HLM%, MLM%")
    elif y == 3:
        print("Level: HMM%, MHH%, MHM%, MMM%")
    else:
        print("Fall: HML%, MHL%, HHL%")
    return y
