import os
from os.path import isfile, join
import subprocess


#大量改檔名
"""
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
"""

#因為檔名一樣就自動讀取,之後檔名不一樣可以統一改檔名
count = 0
path = 'C:\\Users\\user\\voiceprocessing\\internship\\VOA_audio\\mp3'
dirlist = os.listdir(path)
for i in dirlist:
    Completepath = join(path,i)
    if isfile(Completepath):
        print('檔案:',i,'路徑:',Completepath)
        #計算有幾個檔案
        count = count + 1
        # for mp3 files --> .wav files
        input = f'ffmpeg -i C:\\Users\\User\\voiceprocessing\\internship\\VOA_audio\\mp3\\{count}_mp3.mp3 -ar 16000 -ac 1 -acodec pcm_s16le C:\\Users\\User\\voiceprocessing\\internship\\VOA_audio\\wav\\{count}_wav.wav'
        subprocess.call(input, shell = True)



