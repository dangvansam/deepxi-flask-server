#from deepxi.args import get_args
from deepxi.model import DeepXi
from deepxi.prelim import Prelim
import deepxi.utils as utils
import numpy as np
import os
from deepxi.utils import read_wav, save_wav
import tensorflow as tf
from sklearn import preprocessing
import librosa

def get_model_new():
    fs = 16000
    T_d = 32
    T_s = 16
    N_d = int(fs*T_d*0.001) # window duration (samples).
    N_s = int(fs*T_s*0.001) # window shift (samples).
    NFFT = int(pow(2, np.ceil(np.log2(N_d)))) # number of DFT components.
    
    print('list GPU:',tf.config.experimental.list_physical_devices())
    utils.gpu_config('1')

    deepxi = DeepXi(
        N_d=N_d,
        N_s=N_s,
        NFFT=NFFT,
        f_s=fs,
        network_type='ResNet',
        min_snr=-20,
        max_snr=40,
        snr_inter=1,
        d_model=256,
        n_blocks=40,
        n_heads=None,
        d_f=64,
        d_ff=None,
        k=3,
        max_d_rate=16,
        warmup_steps=None,
		padding="causal",
		causal=1,
		ver='resnet-1.0c',
        )
    # load weight to model
    deepxi.load_weight(model_checkpoint_path='model/resnet-1.0c/epoch-179/variables/variables', stats_path='data')
    
    def infer_file(file_path):
        
        #(wav, _) = read_wav(file_path)
        # file_name, file_extension = os.path.splitext(file_path)
        # if file_extension !=".wav":
        #     #convert here 
        #     out_file_path = file_name + "_ffmpeg.wav"
        #     if os.path.isfile(out_file_path):
        #         os.remove(out_file_path)
        #     str =u'ffmpeg -i "%s" -acodec pcm_u8 -ar 16000 -ac 1 "%s"' %(file_path,out_file_path)
        #     print (str)
        #     os.system(str)
        #     file_path= out_file_path
        print('processing file:', file_path)
        #(wav, _) = read_wav(file_path) # read wav from given file path.
        #print(type(wav[0]))
        (wav, _) = librosa.load(file_path, sr=fs, dtype=np.float32)
        print(type(wav[0]))
        print('before nomalize:', np.max(wav), np.min(wav), np.mean(wav))
        wav = preprocessing.minmax_scale(wav, (-0.6, 0.6))
        print('after nomalize:', np.max(wav), np.min(wav), np.mean(wav))
        wav = np.asarray(np.multiply(wav, 32768.0), dtype=np.int32)
        #exit()


        file_name = '.'.join(file_path.split('.')[0:-1])
        file_ext = file_path.split('.')[-1]
        #print(file_name, file_ext)
        out_file_path = file_name + '_predict.wav'
        wav_predict = deepxi.infer_custom(wav)
        print(wav_predict.shape)
        #print('wav predict:',np.max(wav_predict), np.min(wav_predict), np.mean(wav_predict))
        #wav_predict = np.asarray(np.multiply(wav_predict, 32768.0), dtype=np.int16)
        print('wav predict:',np.max(wav_predict), np.min(wav_predict), np.mean(wav_predict))
        #print(wav_predict.shape)

        save_wav(out_file_path, wav_predict, fs)
        print('process done')
        #exit()
        print('saved wav to:', out_file_path)
        return out_file_path

    return infer_file
#if __name__ == "__main__":
#khởi tạo model
#md = get_model_new()
#khử nhiễu file
#md('static/upload/21-12-18 GIAO BAN QUÝ IV GM1032 (KL).mp3')
#md('Untitled (12).wav')