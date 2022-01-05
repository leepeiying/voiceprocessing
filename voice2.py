import os
from os.path import isfile, join
import subprocess
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import numpy as np
from pyAudioAnalysis.audioBasicIO import read_audio_file, stereo_to_mono

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
        # for mp3 files --> .wav files(轉成單聲道,頻率為16000Hz 是否一樣頻率?)
        input = f'ffmpeg -i C:\\Users\\User\\voiceprocessing\\internship\\VOA_audio\\mp3\\{count}_mp3.mp3 -ar 16000 -ac 1 -acodec pcm_s16le C:\\Users\\User\\voiceprocessing\\internship\\VOA_audio\\wav\\{count}_wav.wav'
        subprocess.call(input, shell = True)
"""


#讀取單個指定wav檔
vid = 2
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
# 傅立葉轉換
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