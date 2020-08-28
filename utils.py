import wave
import  requests
import  json
import os
from time import sleep
import  csv
import numpy as np
#from pydub import AudioSegment
import librosa
import  wave
import audioop
import scipy
from datetime import datetime
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from inaSpeechSegmenter import Segmenter, seg2csv

def split(file_name, out_dir):
    print('\nREMOVE MUSIC AND CUT')
    seg = Segmenter()
    segmentation = seg(file_name)
    sample_rate, raw_audio = scipy.io.wavfile.read(file_name)
    #raw_audio , sr = librosa.load(file_name, sr=16000)
    speech = []
    print(segmentation)
    count = 1
    if not os.path.exists(out_dir):
      os.mkdir(out_dir)
    list_file = []
    for s in segmentation:
      if s[0] != 'Music' and s[0] != 'NOACTIVITY':
        print(str(count),'dur of sen:',s[2]-s[1])
        speech_data = raw_audio[int(s[1]*sample_rate) - int(sample_rate/4):int(s[2]*sample_rate + int(sample_rate/4))]
        speech_data = np.array(speech_data)

        print(len(speech_data), len(speech_data)/sample_rate)
        if len(speech_data)/sample_rate < 0.5 or len(speech_data)/sample_rate > 20:
          continue
        else:
          out_filename = out_dir + '/' + file_name.split('/')[-1].replace('.wav','') + '_' + str(count) + '.wav'
          list_file.append(out_filename)
          scipy.io.wavfile.write(out_filename, sample_rate, speech_data)
          count += 1
    return list_file

def cvtToWavMono16(filename):
        #covert to wav
        #try:
        #if filename.split('.')[-1] == 'mp3':
        #    filename = mp3_to_wav(filename)
        #stereo_to_mono()
        #to16000()
        print('start convert to wav 16000 mono')
        #filename = 'myfile.wav'
        # Extract data and sampling rate from file
        #data, fs = sf.read(filename, dtype='float32')
        sig, rate = librosa.load(filename, sr=16000, mono=True)
        #print(sig, rate)
        new_filename = ''.join(filename.split('.')[:-1]) + '.wav'
        librosa.output.write_wav(new_filename, sig, sr=rate)
        print('converted to wav 16000 mono')
        return new_filename
        # except:
        #     return False
        