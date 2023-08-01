import os
from flask import Flask, flash, request, redirect,url_for,render_template,jsonify 
from scipy.io.wavfile import write
import librosa as lr 
import numpy as np 
import joblib
import function as func
speaker = " "
app = Flask(__name__)


UPLOAD_FOLDER = 'static/file/'
app.secret_key = "cairocoders-ednalan"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
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
        x= func.extract_features(audio, sr_freq)

        # load the model of the words 
        gmm_files_word = [ i + '.joblib' for i in ['Close_the_door','Open_the_door','Close_window','Open_book']]
        models_word    = [joblib.load(fname) for fname in gmm_files_word]
        x_word= func.extract_features(audio, sr_freq)

        # loop on the models of the persons to get the max score of the person to detect who 
        log_likelihood = np.zeros(len(models)) 
        for j in range(len(models)):
                gmm = models[j] 
                scores = np.array(gmm.score(x))
                log_likelihood[j] = scores.sum()

        winner = np.argmax(log_likelihood)
        print(winner)
        # print(log_likelihood)
        # loop on the models of the words to get the max score of the word 
        log_likelihood_word = np.zeros(len(models_word)) 
        for j in range(len(models_word)):
                gmm = models_word[j] 
                scores = np.array(gmm.score(x))
                log_likelihood_word[j] = scores.sum()


        winner_word = np.argmax(log_likelihood_word)
        # Flag to detect the other person that aren't in the group 
        flag=False
        flagLst=log_likelihood-max(log_likelihood)
        for i in range(len(flagLst)):
            if  flagLst[i]==0:
                continue
            if abs(flagLst[i])<0.5:
                flag=True
        
        if flag:
            winner=4
        img = func.histpgram(winner,log_likelihood,winner_word,log_likelihood_word)
        image=func.spectrogram(audio,sr_freq)
        drwa_sig=func.signal_draw(audio,sr_freq)
        # check the id of the persons 
        if winner== 3:
            speaker="Hello Rahma"
            word_rec=func.comparison(winner_word)
        elif winner== 0:
            speaker="Hello Doha"
            word_rec=func.comparison(winner_word)
        elif winner== 2:
            speaker="Hello Rabea"
            word_rec=func.comparison(winner_word)
        elif winner== 1:
            speaker="Hello Shuaib"
            word_rec=func.comparison(winner_word)
        elif winner == 4:
            speaker=" "
            word_rec="Others can't open"
        return f'{speaker +"         "+ word_rec}'  
        

    else:
        return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True,port = 1000)