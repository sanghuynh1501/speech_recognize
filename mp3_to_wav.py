import os
import re
import wave
import shutil
import librosa
import numpy as np
from os import path
from tqdm import tqdm
from scipy import signal
from random import randint
from scipy.io import wavfile
from pydub import AudioSegment
from scipy.io.wavfile import write
from sklearn.model_selection import train_test_split

# def add_white_noise(data): 
#     wn = np.random.randn(len(data))
#     data_wn = data + 0.005*wn
#     return data_wn

# def shift(self, data, sample_rate):
#     return np.roll(data, sample_rate)

# def stretch(data, sample_rate, rate=1):
#     input_length = sample_rate
#     data = librosa.effects.time_stretch(data, rate)
#     if len(data)>input_length:
#         data = data[:input_length]
#     else:
#         data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
#     return data
# with tqdm(total=len(os.listdir('vn_voice_data_16k/train'))) as pbar:
#     for sub_folder in os.listdir('vn_voice_data_16k/train'):
#         for file_name in os.listdir('vn_voice_data_16k/train/' + sub_folder):
#             regex = re.compile('[@_!#$%^&*()<>?/\|}{~:.,-]') 
#             if(regex.search(file_name) != None):
#                 shutil.rmtree('vn_voice_data_16k/train/' + sub_folder + '/' + file_name)

# new_sample_rate = 16000
# with tqdm(total=len(os.listdir('vn_voice_data_16k/train'))) as pbar:
#     for sub_folder in os.listdir('vn_voice_data_16k/train'):
#         for file_name in os.listdir('vn_voice_data_16k/train/' + sub_folder):
#             if '1.wav' in file_name:
#                 os.remove('vn_voice_data_16k/train/' + sub_folder + '/' + file_name)
#             else:
#                 if 'mp3' in file_name:
#                     sound = AudioSegment.from_mp3('vn_voice_data_16k/train/' + sub_folder + '/' + file_name)
#                     sound.export('vn_voice_data_16k/train/' + sub_folder + '/' + file_name.split('.')[0] + '.wav', format='wav')
#                     os.remove('vn_voice_data_16k/train/' + sub_folder + '/' + file_name)
#                 sample_rate, samples = wavfile.read('vn_voice_data_16k/train/' + sub_folder + '/' + file_name.split('.')[0] + '.wav')
#                 samples.astype(float)
#                 samples = samples / (2.0**(16-1) + 1)
#                 resampled = signal.resample(samples, int(new_sample_rate/sample_rate * samples.shape[0]))
#                 wavfile.write('vn_voice_data_16k/train/' + sub_folder + '/' + file_name.split('.')[0] + '.wav', new_sample_rate, resampled)
#         pbar.update(1)

data_files = []
labels = []
for sub_folder in os.listdir('vn_voice_data_16k/train/'):
    for file_name in os.listdir('vn_voice_data_16k/train/' + sub_folder):
        data_files.append('vn_voice_data_16k/train/' + sub_folder + '/' + file_name)
        labels.append(sub_folder)

X_train, X_test, y_train, y_test = train_test_split(data_files, labels, test_size=0.15, random_state=42)
idx = 0
for file_link in X_test:
    if not os.path.isdir('vn_voice_data_16k/test/' + y_test[idx] ):
        os.mkdir('vn_voice_data_16k/test/' + y_test[idx]) 
    shutil.move(file_link, 'vn_voice_data_16k/test/' + y_test[idx] + '/' + file_link.split('/')[len(file_link.split('/')) - 1])
    idx += 1

# data_files = []
# labels = []
# for sub_folder in os.listdir('vn_voice_data_16k/test/'):
#     print('sub_folder ', sub_folder)
#     for file_name in os.listdir('vn_voice_data_16k/test/' + sub_folder):
#         if file_name == 'caochung0.wav' or file_name == 'hatieumai0.wav' or file_name == 'lannhi0.wav' or file_name == 'leminh0.wav' or file_name == 'thudung0.wav' or file_name == 'banmai0.wav':
#             if not os.path.isdir('vn_voice_data_16k/train/' + sub_folder):
#                 os.mkdir('vn_voice_data_16k/train/' + sub_folder) 
#             shutil.move('vn_voice_data_16k/test/' + sub_folder + '/' + file_name, 'vn_voice_data_16k/train/' + sub_folder + '/' + file_name)
        # labels.append(sub_folder)

# for sub_folder in os.listdir('vn_voice_data'):
#     filenames = os.listdir('vn_voice_data/' + sub_folder)
#     test_files = []
#     test_count = int(len(filenames) * 30 / 100)
#     while len(test_files) < test_count: 
#         index = randint(0, test_count)
#         if filenames[index] not in test_files:
#             test_files.append(filenames[index])
#     for file_name in test_files:
#         if not os.path.isdir('vn_voice_data/test/' + sub_folder ):
#             os.mkdir('vn_voice_data/test/' + sub_folder) 
#         shutil.move('vn_voice_data/' + sub_folder + '/' + file_name, 'vn_voice_data/test/' + sub_folder + '/' + file_name)

# for file_name in os.listdir('test_data'):
#     sound = AudioSegment.from_mp3('test_data/' + file_name)
#     sound.export('wav_test_data/' + '/' + file_name.split('.')[0] + '.wav', format='wav')
# count = 0
# for sub_folder in os.listdir('vn_voice_data_16k/train/'):
#     sound_array = []
#     for file_name in os.listdir('vn_voice_data_16k/train/' + sub_folder):
#         if '1.wav' in file_name:
#         # print('file_name[1:] ', file_name[2:])
#         # if file_name[4:] not in sound_array:
#         #     sound_array.append(file_name[4:])
#         # else:
#         #     print('remove file_name ', file_name)
#             os.remove('vn_voice_data_16k/train/' + sub_folder + '/' + file_name)
#         # print('sound_array ', sound_array)
# # print('count ', count)