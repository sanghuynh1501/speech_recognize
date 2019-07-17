import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

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
from keras.layers import Input, Dense, Activation, Bidirectional, CuDNNGRU, CuDNNLSTM, LSTM, BatchNormalization, Dropout, TimeDistributed, GRU, Reshape, Lambda, Add, concatenate, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras import regularizers, initializers, optimizers
from keras.activations import relu

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras
TRAIN_DATA_PATH = './vn_voice_data_16k/train/'
TEST_DATA_PATH = './vn_voice_data_16k/test/'

alphabet = ['a','b','c','d','e','g','h','i', 'k','l','m','n','o','p','q','r','s','t','u','v', 'x', 'y', ' ', '_', '1', '2', '3', '4', '5', '6', '7', '8', '9']
MAX_TEXT_LEN = 50

def custom_fft(y, fs):
    T = 1.0 / fs
    N = y.shape[0]
    yf = fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    vals = 2.0/N * np.abs(yf[0:N//2])
    return xf, vals

def add_noise(data):
    noise = np.random.randn(len(data))
    data_noise = data + 0.005 * noise
    return data_noise

def shift(data, sample_rate):
    return np.roll(data, sample_rate)

def stretch(data, sample_rate, rate=1):
    input_length = sample_rate
    data = librosa.effects.time_stretch(data, rate)
    if len(data) > input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
    return data

def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
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


class TextImageGenerator(keras.callbacks.Callback):

    def __init__(self, train_data_path, test_data_path, batch_size, max_text_len=MAX_TEXT_LEN):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.max_text_len = max_text_len
        
        self.n_train = 0
        self.n_test = 0
        self.second = 3

    def get_output_size(self):
        return len(alphabet) + 1
    
    def write_data (self, number, outputPath, data_path):
        dataset = HDF5DatasetWriter(data_dims=(number, 100 * self.second - 1, 81, 1), label_dims=(number, MAX_TEXT_LEN), outputPath=outputPath, dataKey='images', bufSize=1000)
        labels = get_labels(data_path)
        count = 0
        with tqdm(total=len(labels)) as pbar:
            for label in labels:
                path = [data_path  + label + '/' + wfile for wfile in os.listdir(data_path + '/' + label)]
                for wavfile_link in path:
                    sample_rate, samples = wavfile.read(wavfile_link)
                    samples = pad_audio(samples, self.second * sample_rate)
                    
                    augmentation_data = [samples]
                    
                    # samples_noise = add_noise(samples)
                    # samples_shift = shift(samples, self.second * sample_rate)
                    # sample_stretch = stretch(samples, self.second * sample_rate)
                    
                    # augmentation_data.append(samples_noise)
                    # augmentation_data.append(samples_shift)
                    # augmentation_data.append(sample_stretch)
                    
                    for samples in augmentation_data:
                        if len(label) <= MAX_TEXT_LEN and len(label) > 5:
                            try:
                                if len(samples) <= self.second * sample_rate:
                                    # n_samples = chop_audio(samples, self.second * sample_rate)
                                # else: 
                                    # n_samples = [samples]
                                    # for samples in n_samples:
                                        # samples.astype(float)
                                        # samples = samples / (2.0**(16-1) + 1)
                                    resampled = signal.resample(samples, int(8000 / sample_rate * samples.shape[0]))
                                    _, _, specgram = log_specgram(resampled, sample_rate=8000)
                                    # print(len(specgram))
                                    if (len(specgram) == 100 * self.second - 1):
                                        specgram = np.reshape(specgram, (1, 100 * self.second - 1, 81, 1))
                                        np_label = np.array(text_to_labels(label))
                                        np_label = np.expand_dims(np_label, 0)
                                        dataset.add(specgram, np_label, [len(label)])
                                        del np_label
                                        count += 1
                                    del specgram
                            except:
                                print('error')
                    if count >= number:
                        break
                if count >= number:
                    break
                pbar.update(1)
        # print('count ', count)
        dataset.close()
        return outputPath
    
    def process_count (self, data_path):
        labels = get_labels(data_path)
        count = 0
        with tqdm(total=len(labels)) as pbar:
            for label in labels:
                path = [data_path  + label + '/' + wfile for wfile in os.listdir(data_path + '/' + label)]
                for wavfile_link in path:
                    sample_rate, samples = wavfile.read(wavfile_link)
                    samples = pad_audio(samples, self.second * sample_rate)
                    augmentation_data = [samples]
                    for samples in augmentation_data:
                        if len(label) <= MAX_TEXT_LEN and len(label) > 5:
                            try:
                                text_to_labels(label)
                                if len(samples) <= self.second * sample_rate:
                                    #     n_samples = chop_audio(samples, self.second * sample_rate)
                                    # else:
                                    # n_samples = [samples]
                                    # for samples in n_samples:
                                    # samples.astype(float)
                                    # samples = samples / (2.0**(16-1) + 1)
                                    resampled = signal.resample(samples, int(8000 / sample_rate * samples.shape[0]))
                                    _, _, specgram = log_specgram(resampled, sample_rate=8000)
                                    if (len(specgram) == self.second * 100 - 1):
                                        count += 1
                            except:
                                print('error')
                pbar.update(1)
        return count

    def build_data (self):
        print ("# train count processing")
        self.n_train = self.process_count(self.train_data_path)
        
        print ("# test count processing")
        self.n_test = self.process_count(self.test_data_path)
        
        print ("# train data processing")
        self.train_output_path = self.write_data(self.n_train, 'train_audio_data.hdf5', self.train_data_path)
        # self.train_output_path = 'train_audio_data.hdf5'
        
        print ("# test data processing")
        self.test_output_path = self.write_data(self.n_test, 'test_audio_data.hdf5', self.test_data_path)
        # self.test_output_path = 'test_audio_data.hdf5'
        
    def next_batch(self, type):
        dbPath = self.train_output_path
        if type == 'test':
            dbPath = self.test_output_path
        generator = HDF5DatasetGenerator(dbPath=dbPath, batchSize=64, binarize=False)
        while True:
            generate = generator.generator()
            for X_data, Y_data, label_length in generate:
                if 0 in label_length:
                    continue
                input_length = np.ones((X_data.shape[0], 1)) * 72
                inputs = {
                    'the_input': X_data,
                    'the_labels': Y_data,
                    'input_length': input_length,
                    'label_length': label_length,
                }
                outputs = {'ctc': np.zeros([X_data.shape[0]])}
                yield (inputs, outputs)

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

# Translation of characters to unique integer values
def text_to_labels(text):
    text = text.replace('_', ' ')
    for c in text:
        if c not in alphabet:
            print('text ', text)
    while len(text) < MAX_TEXT_LEN:
        text += '_'
    return list(map(lambda x: alphabet.index(x), text))

def get_labels(path):
    labels = os.listdir(path)
    return labels

def get_model(img_w, img_h):
        
    batch_size = 128
    rnn_size = 384
    tiger_train = TextImageGenerator(train_data_path=TRAIN_DATA_PATH, test_data_path=TEST_DATA_PATH, batch_size=batch_size)  
    tiger_train.build_data()

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
    # pooling_3 = MaxPooling2D(pool_size=(2, 2))(merge_layer_3)
    pooling_3 = Dropout(rate=0.1)(merge_layer_3)

    inner = Reshape(target_shape=(74, 20 * 32), name='reshape')(pooling_3)

    gru_1 = Bidirectional(CuDNNLSTM(rnn_size, return_sequences=True, kernel_initializer=keras.initializers.he_uniform(seed=None), name='gru1'))(inner)
    gru_1b = Bidirectional(CuDNNLSTM(rnn_size, return_sequences=True, kernel_initializer=keras.initializers.he_uniform(seed=None), name='gru1_b'))(inner)
    gru1_merged = Add()([gru_1, gru_1b])
    gru_2 = Bidirectional(CuDNNLSTM(rnn_size, return_sequences=True, kernel_initializer=keras.initializers.he_uniform(seed=None), name='gru2'))(gru1_merged)
    gru_2b = Bidirectional(CuDNNLSTM(rnn_size, return_sequences=True, kernel_initializer=keras.initializers.he_uniform(seed=None), name='gru2_b'))(gru1_merged)
    gru2_merged = concatenate([gru_2, gru_2b])
    lstm = Bidirectional(CuDNNLSTM(rnn_size , return_sequences=True, kernel_initializer=keras.initializers.he_uniform(seed=None), name='gru3'))(gru2_merged)

    inner = TimeDistributed(Dense(
        tiger_train.get_output_size(),
        kernel_initializer=keras.initializers.he_uniform(seed=None),
        name='dense2'
    ))(lstm)
    y_pred = Activation('softmax', name='softmax')(inner)
    Model(inputs=inp, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[tiger_train.max_text_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[inp, labels, input_length, label_length], outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam', metrics=['acc'])

    return model, tiger_train

# TRAIN
train_model, tiger_train = get_model(299, 81)
filepath = "model/model_30062019_len_30.h5"
# train_model.load_weights(filepath)
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

train_model.fit_generator(
    generator=tiger_train.next_batch('train'), 
    steps_per_epoch=tiger_train.n_train / 64,
    epochs=500,
    validation_data=tiger_train.next_batch('test'), 
    validation_steps=tiger_train.n_test / 64,
    callbacks=callbacks_list
)