from os import path
from pydub import AudioSegment
import soundfile as sf
import librosa
import numpy as np
# files                                                                         
src = "Tap4_baLan_23m47_23m54_tucgian_nhieu.mp3"
dst = "test.wav"

# convert wav to mp3                                                            
#sound = AudioSegment.from_mp3(src)
#sound.export(dst, format="wav")

s, r = librosa.load(src, None, dtype='int16')
print(len(s),r)
print(np.mean(s))