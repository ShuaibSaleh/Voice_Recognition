import os
import uuid
from flask import Flask, flash, request, redirect,url_for,render_template,jsonify 
import wave
from scipy.io.wavfile import write
import librosa as lr 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.svm import SVC 
import pickle
import matplotlib.pyplot as plt 
import python_speech_features as mfcc
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import librosa.display as im
from sklearn import preprocessing
import joblib


speaker = ""
app = Flask(__name__)


UPLOAD_FOLDER = 'static/file/'
app.secret_key = "cairocoders-ednalan"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#-----------------------------------------------------------Functions------------------------------------------------------------------#

#function that calculate the delta of mfcc features 
def calculate_delta(array):
   
    rows, cols = array.shape
    deltas = np.zeros((rows,20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i-j < 0:
                first =0
            else:
                first = i-j
            if i+j > rows-1:
                second = rows-1
            else:
                second = i+j 
            index.append((second,first))
            j+=1
        deltas[i] = ( array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]])) ) / 10
    return deltas

# function that extract the mfcc feature 
def extract_features(audio,rate):
    global combined  
    mfcc_feature = mfcc.mfcc(audio, rate, 0.025, 0.01, 20, nfft = 2205, appendEnergy = True)    
    mfcc_feature = preprocessing.scale(mfcc_feature)
    delta = calculate_delta(mfcc_feature)
    combined = np.hstack((mfcc_feature, delta)) 
    return combined


# function that get the id of the word 
def comparison (word_id):
    if word_id== 0:
            word_recognation="Close_the_door"
    elif word_id== 1:
            word_recognation="Open_the_door"
    elif word_id== 2:
            word_recognation="Close_window"
    elif word_id == 3:
            word_recognation="Open_book"
    
    return word_recognation


#--------------------------------------------------Main page-----------------------------------------------------------------# 

@app.route('/', methods=['POST','GET'])
def main():
    if request.method == 'POST':    
        file = request.files['audio_data']

        if file.filename == '':  
            flash('No selected file')
            return redirect(request.url)
        file_name = "recordeAudio" + ".wav"
        full_file_name = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
        file.save(full_file_name)

        # load the record and get the audio and sample rate
        features=[]
        audio,sr_freq = lr.load(full_file_name)
        S = np.abs(lr.stft(audio))

        # load the model of the persons 
        gmm_files = [ i + '.joblib' for i in ['Doha', 'shuaib', 'rabea', 'Rahma']]
        models    = [joblib.load(fname) for fname in gmm_files]
        x= extract_features(audio, sr_freq)

        # load the model of the words 
        gmm_files_word = [ i + '.joblib' for i in ['Close_the_door','Open_the_door','Close_window','Open_book']]
        models_word    = [joblib.load(fname) for fname in gmm_files_word]
        x_word= extract_features(audio, sr_freq)

        # loop on the models of the persons to get the max score of the person to detect who 
        log_likelihood = np.zeros(len(models)) 
        for j in range(len(models)):
                gmm = models[j] 
                scores = np.array(gmm.score(x))
                log_likelihood[j] = scores.sum()

        winner = np.argmax(log_likelihood)
        print(log_likelihood)



        # loop on the models of the words to get the max score of the word 
        log_likelihood_word = np.zeros(len(models_word)) 
        for j in range(len(models_word)):
                gmm = models_word[j] 
                scores = np.array(gmm.score(x))
                log_likelihood_word[j] = scores.sum()


        winner_word = np.argmax(log_likelihood_word)
        print(log_likelihood_word)
    
        # Flag to detect the other person that aren't in the group 
        flag=False
        flagLst=log_likelihood-max(log_likelihood)
        for i in range(len(flagLst)):
            if  flagLst[i]==0:
                continue
            if abs(flagLst[i])<0.5:
                flag=True
        
        print(flagLst)
        print(winner)
        if flag:
            winner=4
        
        print(winner)

        # check the id of the persons 
        if winner== 3:
            speaker="Hello it's me Rahma"
            word_rec=comparison(winner_word)
        elif winner== 0:
            speaker="Hello it's me Doha"
            word_rec=comparison(winner_word)
        elif winner== 2:
            speaker="Hello it's me Rabea"
            word_rec=comparison(winner_word)
        elif winner== 1:
            speaker="Hello it's me Shuaib"
            word_rec=comparison(winner_word)
        elif winner == 4:
            speaker=" "
            word_rec="You aren't in the group se we can't open the door "


        
        
        print(speaker)
        

        return f'{speaker +"         "+ word_rec}'  
        

    else:
        return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True,port=21000000)
