import librosa as lr 
import numpy as np 
import matplotlib.pyplot as plt 
import python_speech_features as mfcc
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import librosa.display as im
from sklearn import preprocessing
import io
import pandas as pd 
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

def signal_draw (audio5 , sr5):
    name_mfccs = ["comp0", "comp1", "comp2", "comp3", "comp4", "comp5", "mfcc6", "mfcc7", "mfcc8","mfcc9","mfcc10","mfcc11",
             "mfcc12", "mfcc13", "mfcc14", "mfcc15", "mfcc16", "mfcc17", "mfcc18", "mfcc19"]

    fig, ax = plt.subplots()
    canvas = FigureCanvas(fig)

    path = r'C:\Users\DELL\Desktop\RecordeR\Training Data\Rahma\Rahma_open (1).wav'
    audio, sr = lr.load(path)
    mfcc_feature = mfcc.mfcc(audio, sr, nfft = 20) 
    mfcc_mean_list = []
    for i in mfcc_feature:
        mfcc_mean_list.append(np.mean(i))
    mfcc_20 = []
    for i in range(20):
        mfcc_20.append(mfcc_mean_list[i])
    path2 = r'C:\Users\DELL\Desktop\RecordeR\Training Data\Doha\Doha_open (1).wav'
    audio2, sr2 = lr.load(path2)
    mfcc_feature_2 = mfcc.mfcc(audio2, sr2, nfft = 20) 
    mfcc_feature_2
    mfcc_mean_list2 = []
    for k in mfcc_feature_2:
        mfcc_mean_list2.append(np.mean(k))
    
    mfcc_20_2 = []
    for k in range(20):
        mfcc_20_2.append(mfcc_mean_list2[k])

    path3 = r'C:\Users\DELL\Desktop\RecordeR\Training Data\Shuaib\Shuaib_open (1).wav'
    audio3, sr3 = lr.load(path3)
    mfcc_feature_3 = mfcc.mfcc(audio3, sr3, nfft = 20) 
    mfcc_feature_3
    mfcc_mean_list3 = []
    for j in mfcc_feature_3:
        mfcc_mean_list3.append(np.mean(j))
        

    mfcc_20_3 = []
    for j in range(20):
        mfcc_20_3.append(mfcc_mean_list3[j])

    path4 = r'C:\Users\DELL\Desktop\RecordeR\Training Data\Rabea\Rabea_open (1).wav'
    audio4, sr4 = lr.load(path4)
    mfcc_feature_4 = mfcc.mfcc(audio4, sr4, nfft = 20) 
    mfcc_feature_4
    mfcc_mean_list4 = []
    for m in mfcc_feature_4:
        mfcc_mean_list4.append(np.mean(m))
        

    mfcc_20_4 = []
    for m in range(20):
        mfcc_20_4.append(mfcc_mean_list4[m])
    
    
    mfcc_feature_5 = mfcc.mfcc(audio5, sr5, nfft = 20) 
    mfcc_feature_5
    mfcc_mean_list5 = []
    for l in mfcc_feature_5:
        mfcc_mean_list5.append(np.mean(l))
        

    mfcc_20_5 = []
    for l in range(20):
        mfcc_20_5.append(mfcc_mean_list5[l])
    
    
    ax.scatter(name_mfccs,mfcc_20_2,c ="red")
    ax.scatter(name_mfccs,mfcc_20,c ="blue")
    ax.scatter(name_mfccs,mfcc_20_3,c="green") #rabea
    ax.scatter(name_mfccs,mfcc_20_4,c="yellow") #shuaib

    ax.scatter(name_mfccs,mfcc_20_5,c="pink",marker ="^") #other
    ax.set(title="Common graph")

    return fig.savefig('./static/img/signal.png')





def spectrogram(audio5,sr5):

    name_mfccs = ["comp0", "comp1", "comp2", "comp3", "comp4", "comp5", "mfcc6", "mfcc7", "mfcc8","mfcc9","mfcc10","mfcc11",
             "mfcc12", "mfcc13", "mfcc14", "mfcc15", "mfcc16", "mfcc17", "mfcc18", "mfcc19"]

    fig, ax = plt.subplots()
    canvas = FigureCanvas(fig)

    path = r'C:\Users\DELL\Desktop\RecordeR\Training Data\Rahma\Rahma_open (1).wav'
    audio, sr = lr.load(path)
    mfcc_feature = mfcc.mfcc(audio, sr, nfft = 20) 
    mfcc_mean_list = []
    for i in mfcc_feature:
        mfcc_mean_list.append(np.mean(i))
    mfcc_20 = []
    for i in range(20):
        mfcc_20.append(mfcc_mean_list[i])
    path2 = r'C:\Users\DELL\Desktop\RecordeR\Training Data\Doha\Doha_open (1).wav'
    audio2, sr2 = lr.load(path2)
    mfcc_feature_2 = mfcc.mfcc(audio2, sr2, nfft = 20) 
    mfcc_feature_2
    mfcc_mean_list2 = []
    for k in mfcc_feature_2:
        mfcc_mean_list2.append(np.mean(k))
    
    mfcc_20_2 = []
    for k in range(20):
        mfcc_20_2.append(mfcc_mean_list2[k])

    path3 = r'C:\Users\DELL\Desktop\RecordeR\Training Data\Shuaib\Shuaib_open (1).wav'
    audio3, sr3 = lr.load(path3)
    mfcc_feature_3 = mfcc.mfcc(audio3, sr3, nfft = 20) 
    mfcc_feature_3
    mfcc_mean_list3 = []
    for j in mfcc_feature_3:
        mfcc_mean_list3.append(np.mean(j))
        

    mfcc_20_3 = []
    for j in range(20):
        mfcc_20_3.append(mfcc_mean_list3[j])

    path4 = r'C:\Users\DELL\Desktop\RecordeR\Training Data\Rabea\Rabea_open (1).wav'
    audio4, sr4 = lr.load(path4)
    mfcc_feature_4 = mfcc.mfcc(audio4, sr4, nfft = 20) 
    mfcc_feature_4
    mfcc_mean_list4 = []
    for m in mfcc_feature_4:
        mfcc_mean_list4.append(np.mean(m))
        

    mfcc_20_4 = []
    for m in range(20):
        mfcc_20_4.append(mfcc_mean_list4[m])
    
    
    mfcc_feature_5 = mfcc.mfcc(audio5, sr5, nfft = 20) 
    mfcc_feature_5
    mfcc_mean_list5 = []
    for l in mfcc_feature_5:
        mfcc_mean_list5.append(np.mean(l))
        

    mfcc_20_5 = []
    for l in range(20):
        mfcc_20_5.append(mfcc_mean_list5[l])
    
    name_mfccs=['mfcc_Rahma','mfcc_Doha','mfcc_Shuaib','mfcc_Rabea','mfcc_other']
    mfcc_17=[mfcc_20[17],mfcc_20_2[17],mfcc_20_3[17],mfcc_20_4[17],mfcc_20_5[17]]
    df=pd.DataFrame({'name_mfccs':name_mfccs,'mfcc_17':mfcc_17})
    
    # ax.scatter(name_mfccs,mfcc_20_2,c ="red")
    # ax.scatter(name_mfccs,mfcc_20,c ="blue")
    # ax.scatter(name_mfccs,mfcc_20_3,c="green") #rabea
    # ax.scatter(name_mfccs,mfcc_20_4,c="yellow") #shuaib

    # ax.scatter(name_mfccs,mfcc_20_5,c="pink",marker ="^") #other
    ax.set(title="Common graph")

    
    ax.scatter(x=df['name_mfccs'],y=df['mfcc_17']) #other
    ax.axhline(y=-13,linewidth=3,color='b')
    ax.axhline(y=-15.7,linewidth=3,color='r')
    

    # ax.scatter(name_mfccs,mfcc_20_5,c="pink",marker ="^") #other
    ax.set(title="Common graph")
    
    return fig.savefig('./static/img/spectrogram.png')




def histpgram(audio,look,word_id,look_lik_hood_word):
    
    fig, ax = plt.subplots(nrows=2)
    canvas = FigureCanvas(fig)
    img=io.BytesIO()
    img.seek(0)
    


    if audio ==1:
        data = {'Doha':10, 'Rabea':15, 'Rahma':20,
            'Shuiab':np.abs(np.max(look)),'Others':15}
        courses = list(data.keys())
        values = list(data.values())
    elif audio==0:
        data = {'Doha':np.abs(np.max(look)), 'Rabea':15, 'Rahma':20,
            'Shuiab':15,'Others':15}
        courses = list(data.keys())
        values = list(data.values())
    elif audio==2:
        data = {'Doha':10, 'Rabea':np.abs(np.max(look)), 'Rahma':20,
            'Shuiab':15,'Others':15}
        courses = list(data.keys())
        values = list(data.values())
    elif audio==3:
        data = {'Doha':10, 'Rabea':20, 'Rahma':np.abs(np.max(look)),
            'Shuiab':15,'Others':15}
        courses = list(data.keys())
        values = list(data.values())
    elif audio==4:
        data = {'Doha':10, 'Rabea':20, 'Rahma':15,
            'Shuiab':15,'Others':np.abs(np.max(look))}
        courses = list(data.keys())
        values = list(data.values())
    
    # creating the bar plot
    ax[0].bar(courses, values,width = 0.4)

    if word_id ==1:
        data = {'Open the door':np.abs(np.max(look_lik_hood_word)), 'Other words':10}
        courses = list(data.keys())
        values = list(data.values())
    elif word_id==0:
        data = {'Open the door':10, 'Other words':np.abs(np.max(look_lik_hood_word))}
        courses = list(data.keys())
        values = list(data.values())
    elif word_id==2:
        data = {'Open the door':10, 'Other words':np.abs(np.max(look_lik_hood_word))}
        courses = list(data.keys())
        values = list(data.values())
    elif word_id==3:
        data = {'Open the door':10, 'Other words':np.abs(np.max(look_lik_hood_word))}
        courses = list(data.keys())
        values = list(data.values())
    
    ax[1].bar(courses, values,
            width = 0.4)

    # ax1.title("Histogram with 'auto' bins")




    return fig.savefig('./static/img/barchart.png') 