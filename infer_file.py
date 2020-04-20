import os, sys 
sys.path.insert(0, 'lib')
from dev.infer import infer
from dev.sample_stats import get_stats
from dev.train import train
import dev.deepxi_net as deepxi_net
import numpy as np
import tensorflow as tf
import dev.utils as utils
import argparse

from dev.utils import read_wav
from tqdm import tqdm
import dev.gain as gain
import dev.utils as utils
import dev.xi as xi
import numpy as np
import os
import scipy.io as spio
import librosa
import pickle
from scipy.io.wavfile import write as wav_write
np.set_printoptions(threshold=1e6)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def str2bool(s): return s.lower() in ("yes", "true", "t", "1")

def create_args():
    with open('data/stats.p', 'rb') as f:
        stats = pickle.load(f)

    parser = argparse.ArgumentParser()
    ## OPTIONS (GENERAL)
    parser.add_argument('--gpu', default='0', type=str, help='GPU selection')
    parser.add_argument('--ver', type=str, help='Model version')
    parser.add_argument('--epoch', type=int, help='Epoch to use/retrain from')
    parser.add_argument('--train', default=False, type=str2bool, help='Training flag')
    parser.add_argument('--infer', default=True, type=str2bool, help='Inference flag')
    parser.add_argument('--verbose', default=False, type=str2bool, help='Verbose')
    parser.add_argument('--model', default='ResNet', type=str, help='Model type')

    ## OPTIONS (TRAIN)
    parser.add_argument('--cont', default=False, type=str2bool, help='Continue testing from last epoch')
    parser.add_argument('--mbatch_size', default=10, type=int, help='Mini-batch size')
    parser.add_argument('--sample_size', default=1000, type=int, help='Sample size')
    parser.add_argument('--max_epochs', default=250, type=int, help='Maximum number of epochs')
    parser.add_argument('--grad_clip', default=True, type=str2bool, help='Gradient clipping')

    parser.add_argument('--out_type', default='y', type=str, help='Output type for testing')

    ## GAIN FUNCTION
    parser.add_argument('--gain', default='mmse-lsa', type=str, help='Gain function for testing')

    ## PATHS
    parser.add_argument('--model_path', default='model/3f/epoch-175', type=str, help='Model save path')
    parser.add_argument('--set_path', default='set', type=str, help='Path to datasets')
    parser.add_argument('--data_path', default='data', type=str, help='Save data path')
    parser.add_argument('--test_x_path', default='set/test_noisy_speech', type=str, help='Path to the noisy speech test set')
    parser.add_argument('--in_filepath', default='test.wav', type=str, help='Output path')
    parser.add_argument('--out_filepath', default='out.wav', type=str, help='Output path')

    ## FEATURES
    parser.add_argument('--min_snr', default=-10, type=int, help='Minimum trained SNR level')
    parser.add_argument('--max_snr', default=20, type=int, help='Maximum trained SNR level')
    parser.add_argument('--f_s', default=16000, type=int, help='Sampling frequency (Hz)')
    parser.add_argument('--T_w', default=32, type=int, help='Window length (ms)')
    parser.add_argument('--T_s', default=16, type=int, help='Window shift (ms)')
    parser.add_argument('--nconst', default=32768.0, type=float, help='Normalisation constant (see feat.addnoisepad())')
    parser.add_argument('--N_w', default=int(16000*32*0.001), type=int, help='window length (samples)')
    parser.add_argument('--N_s', default=int(16000*16*0.001), type=int, help='window shift (samples)')
    parser.add_argument('--NFFT', default=int(pow(2, np.ceil(np.log2(int(16000*32*0.001))))), type=float, help='number of DFT components')
    parser.add_argument('--stats', default=stats)

    ## NETWORK PARAMETERS
    parser.add_argument('--d_in', default=257, type=int, help='Input dimensionality')
    parser.add_argument('--d_out', default=257, type=int, help='Ouput dimensionality')
    parser.add_argument('--d_model', default=256, type=int, help='Model dimensions')
    parser.add_argument('--n_blocks', default=40, type=int, help='Number of blocks')
    parser.add_argument('--d_f', default=64, type=int, help='Number of filters')
    parser.add_argument('--k_size', default=3, type=int, help='Kernel size')
    parser.add_argument('--max_d_rate', default=16, type=int, help='Maximum dilation rate')
    parser.add_argument('--norm_type', default='FrameLayerNorm', type=str, help='Normalisation type')
    parser.add_argument('--net_height', default=[4], type=list, help='RDL block height')

    args = parser.parse_args()
    return args

def build_restore_model(model_path, args, config):
    ## MAKE DEEP XI NNET
    print('Start: Build and Restore model!')
    sess = tf.Session(config=config)
    net = deepxi_net.deepxi_net(args)
    net.saver.restore(sess, args.model_path)
    print('Done: Build and Restore model!')
    return net, sess

# def infer(filename ,net, sess):
#     print('Start infer file: {}'.format(filename))
#     #(wav, _) = read_wav(args.in_filepath) # read wav from given file path.
#     (wav, _) = librosa.load(filename, 16000, mono=True) # read wav from given file path.
#     wav = np.asarray(np.multiply(wav, 32768.0), dtype=np.int16)

#     print(max(wav), min(wav), np.mean(wav))
#     print(wav.shape)

#     input_feat = sess.run(net.infer_feat, feed_dict={net.s_ph: [wav], net.s_len_ph: [len(wav)]}) # sample of training set.
#     xi_bar_hat = sess.run(
#                         net.infer_output, feed_dict={net.input_ph: input_feat[0], 
#                         net.nframes_ph: input_feat[1], net.training_ph: False}) # output of network.
#     xi_hat = xi.xi_hat(xi_bar_hat, args.stats['mu_hat'], args.stats['sigma_hat'])

#     #file_name = filename.split('/')[-1].split('.')

#     y_MAG = np.multiply(input_feat[0], gain.gfunc(xi_hat, xi_hat+1, gtype=args.gain))
#     y = np.squeeze(sess.run(net.y, feed_dict={net.y_MAG_ph: y_MAG, 
#                                             net.x_PHA_ph: input_feat[2], net.nframes_ph: input_feat[1], net.training_ph: False})) # output of network.
#     if np.isnan(y).any(): ValueError('NaN values found in enhanced speech.')
#     if np.isinf(y).any(): ValueError('Inf values found in enhanced speech.')

#     y = np.asarray(np.multiply(y, 32768.0), dtype=np.int16)
#     out_filepath = filename.replace('.'+filename.split('.')[-1], '_pred.wav')
#     wav_write(out_filepath, args.f_s, y)
#     print('Infer out file: {} done'.format(out_filepath))
#     return out_filepath

def get_model():
    args = create_args()

    ## GPU CONFIGURATION
    config = tf.ConfigProto()
    config.allow_soft_placement=True
    config.gpu_options.allow_growth=True
    config.log_device_placement=False

    net, sess = build_restore_model(args.model_path, args, config)

    def infer(filename):
        print('Start infer file: {}'.format(filename))
        #(wav, _) = read_wav(args.in_filepath) # read wav from given file path.
        (wav, _) = librosa.load(filename, 16000, mono=True) # read wav from given file path.
        wav = np.asarray(np.multiply(wav, 32768.0), dtype=np.int16)

        print(max(wav), min(wav), np.mean(wav))
        print(wav.shape)

        input_feat = sess.run(net.infer_feat, feed_dict={net.s_ph: [wav], net.s_len_ph: [len(wav)]}) # sample of training set.
        xi_bar_hat = sess.run(
                            net.infer_output, feed_dict={net.input_ph: input_feat[0], 
                            net.nframes_ph: input_feat[1], net.training_ph: False}) # output of network.
        xi_hat = xi.xi_hat(xi_bar_hat, args.stats['mu_hat'], args.stats['sigma_hat'])

        #file_name = filename.split('/')[-1].split('.')

        y_MAG = np.multiply(input_feat[0], gain.gfunc(xi_hat, xi_hat+1, gtype=args.gain))
        y = np.squeeze(sess.run(net.y, feed_dict={net.y_MAG_ph: y_MAG, 
                                                net.x_PHA_ph: input_feat[2], net.nframes_ph: input_feat[1], net.training_ph: False})) # output of network.
        if np.isnan(y).any(): ValueError('NaN values found in enhanced speech.')
        if np.isinf(y).any(): ValueError('Inf values found in enhanced speech.')

        y = np.asarray(np.multiply(y, 32768.0), dtype=np.int16)
        out_filepath = filename.replace('.'+filename.split('.')[-1], '_pred.wav')
        wav_write(out_filepath, args.f_s, y)
        print('Infer out file: {} done'.format(out_filepath))
        return out_filepath
    return infer

if __name__ == '__main__':

    infer = get_model()
    infer('Toàn cảnh phòng chống dịch COVID-19 ngày 18-4-2020 - VTV24.mp3')
    #infer('set/test_noisy_speech/198853.wav')

    



    