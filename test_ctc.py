import tensorflow as tf

import os
import re
import h5py
import string
import random
import shutil
import librosa
import numpy as np
from tqdm import tqdm
from scipy import signal
from scipy.io import wavfile
from scipy.fftpack import fft
import keras.callbacks
from keras import backend as K
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.models import Model, load_model
from mlcollect import HDF5DatasetWriter, HDF5DatasetGenerator
from keras.layers import Input, Dense, Activation, Bidirectional, TimeDistributed, CuDNNGRU, CuDNNLSTM, BatchNormalization, Dropout, GRU, Reshape, Lambda, Add, concatenate, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras import regularizers, initializers, optimizers

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

# L = 16000
# CHUNK = 1024
# CHANNELS = 1
# TARGET_START = 80
# TARGET_END = 400
# RECORD_SECONDS = 1
# maxValue = 2**16
# p = pyaudio.PyAudio()
# FORMAT = pyaudio.paInt16
# WAVE_OUTPUT_FILENAME = "test_audio/output.wav"
alphabet = ['a','b','c','d','e','g','h','i', 'k','l','m','n','o','p','q','r','s','t','u','v', 'x', 'y', ' ', '_', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def labels_to_text(labels):
    text = ''
    alphabet.append('_')
    for label in labels:
        index = np.argmax(label, axis=0)
        text += alphabet[index]
    return text

def vni_to_test(s):
    s = s.lower()
    s = re.sub('a65', 'ậ', s)
    s = re.sub('a81', 'ắ', s)
    s = re.sub('a82', 'ằ', s)
    s = re.sub('a83', 'ẳ', s)
    s = re.sub('a84', 'ẵ', s)
    s = re.sub('a85', 'ặ', s)
    s = re.sub('a61', 'ấ', s)
    s = re.sub('a62', 'ầ', s)
    s = re.sub('a63', 'ẩ', s)
    s = re.sub('a64', 'ẫ', s)
    s = re.sub('e61', 'ế', s)
    s = re.sub('e62', 'ề', s)
    s = re.sub('e63', 'ể', s)
    s = re.sub('e64', 'ễ', s)
    s = re.sub('e65', 'ệ', s)
    s = re.sub('o61', 'ố', s)
    s = re.sub('o62', 'ồ', s)
    s = re.sub('o63', 'ổ', s)
    s = re.sub('o64', 'ỗ', s)
    s = re.sub('o65', 'ộ', s)
    s = re.sub('o71', 'ớ', s)
    s = re.sub('o72', 'ờ', s)
    s = re.sub('o73', 'ở', s)
    s = re.sub('o74', 'ỡ', s)
    s = re.sub('o75', 'ợ', s)
    s = re.sub('u71', 'ứ', s)
    s = re.sub('u72', 'ừ', s)
    s = re.sub('u73', 'ử', s)
    s = re.sub('u74', 'ữ', s)
    s = re.sub('u75', 'ự', s)
    s = re.sub('a1',  'á', s)
    s = re.sub('a2',  'à', s)
    s = re.sub('a3',  'ả', s)
    s = re.sub('a4',  'ã', s)
    s = re.sub('a5',  'ạ', s)
    s = re.sub('a8',  'ă', s)
    s = re.sub('a6',  'â', s)
    s = re.sub('e1',  'é', s)
    s = re.sub('e2',  'è', s)
    s = re.sub('e3',  'ẻ', s)
    s = re.sub('e4',  'ẽ', s)
    s = re.sub('e5',  'ẹ', s)
    s = re.sub('e6',  'ê', s)
    s = re.sub('o1',  'ó', s)
    s = re.sub('o2',  'ò', s)
    s = re.sub('o3',  'ỏ', s)
    s = re.sub('o4',  'õ', s)
    s = re.sub('o5',  'ọ', s)
    s = re.sub('o6',  'ô', s)
    s = re.sub('o7',  'ơ', s)
    s = re.sub('i1',  'í', s)
    s = re.sub('i2',  'ì', s)
    s = re.sub('i3',  'ỉ', s)
    s = re.sub('i4',  'ĩ', s)
    s = re.sub('i5',  'ị', s)
    s = re.sub('u1',  'ú', s)
    s = re.sub('u2',  'ù', s)
    s = re.sub('u3',  'ủ', s)
    s = re.sub('u4',  'ũ', s)
    s = re.sub('u5',  'ụ', s)
    s = re.sub('u7',  'ư', s)
    s = re.sub('y1',  'ý', s)
    s = re.sub('y2',  'ỳ', s)
    s = re.sub('y3',  'ỷ', s)
    s = re.sub('y4',  'ỹ', s)
    s = re.sub('y5',  'ỵ', s)
    s = re.sub('d9',  'đ', s)
    return s

def isDuplicate(text):
    for i in range(len(text) - 2):
        if text[i] == text[i + 1]:
            return True
    return False

def ctc_to_text(ctc):
    ctc = list(ctc)
    length = len(ctc) - 1
    while isDuplicate(ctc):
        dem = length
        while dem >= 0:
            if ctc[dem] == ctc[dem - 1]:
                del ctc[dem]
                length = length - 1
            dem -= 1
    ctc = ''.join(ctc)
    text = ctc.replace('_', '')
    return text

def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=16000,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)

def pad_audio(samples, L=16000):
    if len(samples) >= L: return samples
    else: return np.pad(samples, pad_width=(L - len(samples), 0), mode='constant', constant_values=(0, 0))

def chop_audio(samples, L=16000, num=20):
    for i in range(num):
        beg = np.random.randint(0, len(samples) - L)
        yield samples[beg: beg + L]

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def get_model(img_w, img_h):
    rnn_size = 384

    input_shape = (img_w, img_h, 1)
    inp = Input(name='the_input', shape=input_shape)
    norm_inp_1 = BatchNormalization()(inp)
    img_1 = Conv2D(8, kernel_size=2, activation='relu', padding='same', kernel_initializer=keras.initializers.he_uniform(seed=None))(norm_inp_1)
    img_1 = Conv2D(8, kernel_size=2, activation='relu', padding='same', kernel_initializer=keras.initializers.he_uniform(seed=None))(img_1)
    merge_layer_1 = Add()([img_1, norm_inp_1])
    pooling_1 = MaxPooling2D(pool_size=(2, 2))(merge_layer_1)
    pooling_1 = Dropout(rate=0.1)(pooling_1)
    
    norm_inp_2 = BatchNormalization()(pooling_1)
    img_1 = Conv2D(16, kernel_size=3, activation='relu', padding='same', kernel_initializer=keras.initializers.he_uniform(seed=None))(norm_inp_2)
    img_1 = Conv2D(16, kernel_size=3, activation='relu', padding='same', kernel_initializer=keras.initializers.he_uniform(seed=None))(img_1)
    pooling_1 = Conv2D(16, kernel_size=1, activation='relu', padding='same')(pooling_1)
    merge_layer_2 = Add()([img_1, pooling_1])
    pooling_2 = MaxPooling2D(pool_size=(2, 2))(merge_layer_2)
    pooling_2 = Dropout(rate=0.1)(pooling_2)
    
    norm_inp_3 = BatchNormalization()(pooling_2)
    img_1 = Conv2D(32, kernel_size=3, activation='relu', padding='same', kernel_initializer=keras.initializers.he_uniform(seed=None))(norm_inp_3)
    img_1 = Conv2D(32, kernel_size=3, activation='relu', padding='same', kernel_initializer=keras.initializers.he_uniform(seed=None))(img_1)
    pooling_2 = Conv2D(32, kernel_size=1, activation='relu', padding='same')(pooling_2)
    merge_layer_3 = Add()([img_1, pooling_2])
    pooling_3 = MaxPooling2D(pool_size=(2, 2))(merge_layer_3)
    pooling_3 = Dropout(rate=0.1)(pooling_3)

    inner = Reshape(target_shape=(37, 10 * 32), name='reshape')(pooling_3)

    gru_1 = Bidirectional(CuDNNLSTM(rnn_size, return_sequences=True, kernel_initializer=keras.initializers.he_uniform(seed=None), name='gru1'))(inner)
    gru_1b = Bidirectional(CuDNNLSTM(rnn_size, return_sequences=True, kernel_initializer=keras.initializers.he_uniform(seed=None), name='gru1_b'))(inner)
    gru1_merged = Add()([gru_1, gru_1b])
    gru_2 = Bidirectional(CuDNNLSTM(rnn_size, return_sequences=True, kernel_initializer=keras.initializers.he_uniform(seed=None), name='gru2'))(gru1_merged)
    gru_2b = Bidirectional(CuDNNLSTM(rnn_size, return_sequences=True, kernel_initializer=keras.initializers.he_uniform(seed=None), name='gru2_b'))(gru1_merged)
    gru2_merged = concatenate([gru_2, gru_2b])
    lstm = Bidirectional(CuDNNLSTM(rnn_size , return_sequences=True, kernel_initializer=keras.initializers.he_uniform(seed=None), name='gru3'))(gru2_merged)

    inner = TimeDistributed(Dense(
        len(alphabet) + 1,
        kernel_initializer=keras.initializers.he_uniform(seed=None),
        name='dense2'
    ))(lstm)
    y_pred = Activation('softmax', name='softmax')(inner)
    model = Model(inputs=inp, outputs=y_pred)

    return model

# TEST
test_model = get_model(299, 81)
test_model.summary()
test_model.load_weights("model/model_06072019_len_30_acc_93.h5")

def predict(file_path):
    spectrograms = []
    sample_rate, samples = wavfile.read(file_path)
    print('sample_rate ', sample_rate)
    samples = pad_audio(samples, 3 * sample_rate)
    if len(samples) > 3 * sample_rate:
        n_samples = chop_audio(samples, 3 * sample_rate)
    else: n_samples = [samples]
    for samples in n_samples:
        resampled = signal.resample(samples, int(8000 / sample_rate * samples.shape[0]))
        _, _, specgram = log_specgram(resampled, sample_rate=8000)
    spectrograms.append(specgram)
    spectrograms = np.array(spectrograms)
    spectrograms = np.expand_dims(spectrograms, axis=-1)
    y_pred = test_model.predict(spectrograms)
    text = vni_to_test(ctc_to_text(labels_to_text(y_pred[0])))
    print('predict text: ', text)
    
# predict('output.wav')
for sub_folder in os.listdir('btts'):
    for file_name in os.listdir('btts/' + sub_folder): 
        if len(sub_folder) <= 30:
            # try:
            print('-----------------------------------------')
            print('file path: ', 'btts/' + sub_folder + '/' + file_name)
            print('origin text: ', vni_to_test(sub_folder))
            predict('btts/' + sub_folder + '/' + file_name)
            print('-----------------------------------------')
            # except:
            #     print('error')
