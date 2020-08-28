import librosa
print(librosa.__version__)
import scipy
filename = 'music.mp3'
#sr, data = scipy.io.wavfile.read(filename)
data2, sr2 = librosa.load(filename,None)

#print(sr,sr2)
print(sr2,data2)