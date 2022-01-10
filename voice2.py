import os
from os.path import isfile, join
import os, sklearn.cluster
import subprocess
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
from pyAudioAnalysis.audioBasicIO import read_audio_file, stereo_to_mono
from pyAudioAnalysis.MidTermFeatures import mid_feature_extraction as mT
from pyAudioAnalysis.audioSegmentation import labels_to_segments
from pyAudioAnalysis.audioTrainTest import normalize_features
import scipy.io.wavfile as wavfile
import IPython
import webbrowser


"""
#大量改檔名

path='儲存的路徑' #這就是欲進行檔名更改的檔案路徑，路徑的斜線是為/，要留意下！
files=os.listdir(path)
print('files') #印出讀取到的檔名稱，用來確認自己是不是真的有讀到

n=0 #設定初始值
for i in files: #因為資料夾裡面的檔案都要重新更換名稱
	oldname=path+files[n] #指出檔案現在的路徑名稱，[n]表示第n個檔案
	newname=path+str(n+1)+'_mp3.mp3' #在本案例中的命名規則為：年份+ - + 次序，最後一個.wav表示該檔案的型別
	os.rename(oldname,newname)
	print(oldname+'>>>'+newname) #印出原名與更名後的新名，可以進一步的確認每個檔案的新舊對應
	n=n+1 #當有不止一個檔案的時候，依次對每一個檔案進行上面的流程，直到更換完畢就會結束


#因為檔名一樣就自動讀取,之後檔名不一樣可以統一改檔名
#多個轉檔
count = 0
path = 'C:\\Users\\user\\voiceprocessing\\internship\\VOA_audio\\mp3'
dirlist = os.listdir(path)
for i in dirlist:
    Completepath = join(path,i)
    if isfile(Completepath):
        print('檔案:',i,'路徑:',Completepath)
        #計算有幾個檔案
        count = count + 1
        # for mp3 files --> .wav files(轉成單聲道,頻率為16000Hz 是否要一樣頻率?)
        input = f'ffmpeg -i C:\\Users\\User\\voiceprocessing\\internship\\VOA_audio\\mp3\\{count}_mp3.mp3 -ar 16000 -ac 1 -acodec pcm_s16le C:\\Users\\User\\voiceprocessing\\internship\\VOA_audio\\wav\\{count}_wav.wav'
        subprocess.call(input, shell = True)
"""


#讀取單個指定wav檔
vid = 2
speakers = 3

print("現在的檔案為: ",vid)

input_file = f"internship\\VOA_audio\\wav\\{vid}_wav.wav" 
rate, data = read_audio_file(input_file)
#查看取樣頻率(只要大於8000Hz就是無損音質)
print("Sample rate: {} Hz".format(rate))
# 查看取樣資料的類型
print("Data type: {}".format(data.dtype))
# 繪製前 1024 點資料的波形圖
plt.figure(figsize=(15, 5))
plt.plot(data[0:1024])
plt.show()
#經過傅立葉轉換之後，繪製頻譜圖（spectrum）
# 傅立葉轉換(頻率分析)
from scipy.fftpack import fft
dataFFT = fft(data[0:1024])
dataFFTAbs = abs(dataFFT[1:512])

# 繪製頻譜圖
plt.figure(figsize=(15, 5))
plt.plot(dataFFTAbs, 'r')
plt.show()
#將全部資料的波形圖與時頻譜圖（spectrogram）繪製在一起
# 產生時間資料
time = np.arange(0, len(data)) / rate


plt.figure(figsize=(15, 5))
#Time Domain(Amplitude vs Time)
#Frequency Domain(Amplitude vs Frequency)
# 繪製波形圖
plotA = plt.subplot(211)
plotA.plot(time, data)
plotA.set_ylabel("Amplitude")
plotA.set_xlim(0, len(data) / rate)

# 繪製時頻譜圖
plotB = plt.subplot(212)
plotB.specgram(data, NFFT=1024, Fs=rate, noverlap=900)
plotB.set_ylabel("Frequency")
plotB.set_xlabel("Time")

plt.show()
#fs(rate): 採樣頻率,x(data): 輸入的音頻(數字化的)
# read signal and get normalized segment feature statistics:
fs, x = read_audio_file(input_file)
#st_win, st_step: 短期時長的窗口及步長
mt_size, mt_step, st_win = 2, 0.1, 0.05
#特徵抽取(Mid-term feature extraction)
[mt_feats, st_feats, _] = mT(x, fs, mt_size * fs, mt_step * fs, round(fs * st_win), round(fs * st_win * 0.5))
(mt_feats_norm, MEAN, STD) = normalize_features([mt_feats.T])
mt_feats_norm = mt_feats_norm[0].T

# perform clustering
n_clusters = speakers
x_clusters = [np.zeros((fs, )) for i in range(n_clusters)]
k_means = sklearn.cluster.KMeans(n_clusters=n_clusters)
k_means.fit(mt_feats_norm.T)
cls = k_means.labels_

# create segments and classes
segs, c = labels_to_segments(cls, mt_step)  # convert flags to segment limits
#pyAudioAnalysis切割時間成txt檔
#make segs and tags ready for mach_segs\\{vid}.txt file
seg2txt = []
for i in np.arange(0,len(segs)):
    seg2txt.append(str(segs[i][0])+","+ str(segs[i][1])+","+str(c[i]))
with open(f'internship\\VOA_audio\\mach_segs\\{vid}.txt', 'w') as f:
    for line in seg2txt:
        f.write(line)
        f.write('\n')

webbrowser.open(f'internship\\VOA_audio\\mach_segs\\{vid}.txt')

#畫切割圖
man_data = np.genfromtxt(f'internship\\VOA_audio\\clean_segs\\{vid}.txt',delimiter=',')   # array([[1,2,0],[2,4,1],[4,9,0],...])
mach_data = np.genfromtxt(f'internship\\VOA_audio\\mach_segs\\{vid}.txt',delimiter=',')
colors = {0:'red',1:'purple',2:'orange',3:'blue',4:'yellow',5:'pink',6:'green',7:'black'}

man_segs= []
man_tags = []
mach_segs=[]
mach_tags=[]
for i in (man_data):
    man_segs.append([i[0],i[1]])
    man_tags.append(i[2])
for i in (mach_data):
    mach_segs.append([i[0],i[1]])
    mach_tags.append(i[2])

man_colors = []
mach_colors=[]
for i in man_tags:
    if i in colors:
        man_colors.append(colors[i])
for i in mach_tags:
    if i in colors:
        mach_colors.append(colors[i])
man_line = [[2,2] for i in man_segs]
mach_line = [[1,1] for i in mach_segs]

figure(figsize=(15,3), dpi=70)
import matplotlib.font_manager as fm
font = fm.FontProperties(fname='c:\\windows\\fonts\\mingliu.ttc',size='xx-large')

for i,j,z in zip(man_segs,man_line,man_colors):
    plt.plot(i,j,color=z,linewidth=20,solid_capstyle='butt')
for i,j,z in zip(mach_segs,mach_line,mach_colors):
    plt.plot(i,j,color=z,linewidth=20,solid_capstyle='butt')

plt.ylim(0,3)
plt.xlim(0,mach_segs[-1][1])
plt.xlabel('Time (seconds)')
plt.yticks([0,1,2,3],["","Machine Diarization\n自動分段","Manual Diarization\n手工分段",""],fontproperties=font)
plt.show()