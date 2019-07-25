#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 11:40:44 2019

@author: greert
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:22:52 2019

@author: greert
"""
import pickle
import numpy as np
import pylab
import matplotlib.pyplot as plt
from tsne import *
from skmultilearn.adapt import MLkNN
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import re
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.metrics import hamming_loss
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from scipy import sparse
from sklearn.naive_bayes import GaussianNB
from skmultilearn.ensemble import RakelD
from sklearn.metrics import f1_score
from pull_in_annotations import *
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import tree


###TODO: Get the vectors and visualizations for the pop, rock, latin, country songs!


def h_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    http://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        #print('\nset_true: {0}'.format(set_true))
        #print('set_pred: {0}'.format(set_pred))
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        #print('tmp_a: {0}'.format(tmp_a))
        acc_list.append(tmp_a)
    return np.mean(acc_list)

    
def clean_text(word):
    word = re.sub('[!@#$\'.,?\-;\"’]', '', word)
    word = word.strip()
    word = word.lower()
    return word

def create_vector_dict(raw_text):
    raw_text = raw_text[1:]
    my_keys = [x.split(' ')[0] for x in raw_text]
    my_values = [x.split(' ')[1:-1] for x in raw_text]
    for i in range(len(my_values)):
        my_values[i] = [float(x) for x in my_values[i]]
    my_dict = {}
    for i in range(len(my_keys)):
        my_dict[my_keys[i]] = my_values[i][:]
    return my_dict

def create_vector_names(raw_text):
    raw_text = raw_text[1:]
    my_keys = [x.split(' ')[0] for x in raw_text]
    return my_keys

def create_vectors(raw_text):
    raw_text = raw_text[1:]
    my_values = [x.split(' ')[1:-1] for x in raw_text]
    for i in range(len(my_values)):
        my_values[i] = [float(x) for x in my_values[i]]
    return my_values

def find_embeddings_chords(text_list,my_dict):        
    temp_text_list = text_list.split(' ')
    my_len = len(temp_text_list)+0.0
    temp_embs = np.zeros((1,200))
    temp_text_list = [x for x in temp_text_list if x not in  ('','SONG','OVER','Vdim','VIIdim','w3dim')]
    for i in range(len(temp_text_list)):   
        temp_embs += my_dict[temp_text_list[i]]
    return temp_embs/my_len

#TODO: Save anns as pickles!
def find_embeddings_lyrics(text_list,my_dict):
    temp_text = text_list.replace(' \'', '')
    temp_text = text_list.replace(' n\'t ', 'nt ')
    #Clean up 51st entry
    temp_text = text_list.replace('ohhhhh-aah', 'oh')
    #temp_text = text_list
    temp_text_list = temp_text.split(' ')
    my_len = len(temp_text_list)+0.0
    temp_embs = np.zeros((1,200))        
    j_minus = -1
    #temp_text_list = [x.replace(' \'', '') for x in temp_text_list]
    for j in range(len(temp_text_list)):        
        #print(j)            
        try:
            temp_embs += my_dict[clean_text(temp_text_list[j])]
        except:
            if clean_text(temp_text_list[j]):
                print(clean_text(temp_text_list[j]))
                j_minus = j_minus+1       
    my_len = j-j_minus
    return temp_embs/my_len

def find_BOW(chords, lyrics):
    full_BOW = [str(m)+' ' +str(n) for m,n in zip(chords,lyrics)]
    vectorizer = CountVectorizer()
    my_BOW = vectorizer.fit_transform(full_BOW)
    return my_BOW

#Uncomment from here to ***

with open('multivec-master/my_models/lyrics_sg.txt', 'r') as myFile:
    mono_lyrics = myFile.readlines()
with open('multivec-master/my_models/chords_sg.txt', 'r') as myFile:
    mono_chords = myFile.readlines()
with open('multivec-master/my_models/lyrics_2_chords_chords_sg.txt', 'r') as myFile:
    bi_lyrics_chords = myFile.readlines()
with open('multivec-master/my_models/lyrics_2_chords_lyrics_sg.txt', 'r') as myFile:
    bi_lyrics_lyrics = myFile.readlines()   
with open('multivec-master/my_models/chords_2_lyrics_chords_sg.txt', 'r') as myFile:
    bi_chords_chords = myFile.readlines()
with open('multivec-master/my_models/chords_2_lyrics_lyrics_sg.txt', 'r') as myFile:
    bi_chords_lyrics = myFile.readlines()

#TODO: We already unseparated these, but we use the separated songs here so the
#Methods above will work



#MONO
chord_dict = create_vector_dict(mono_chords)
lyrics_dict = create_vector_dict(mono_lyrics)
chord_names = create_vector_names(mono_chords)
chord_vectors = create_vectors(mono_chords)
lyric_names = create_vector_names(mono_lyrics)
lyric_vectors = create_vectors(mono_lyrics)

#BI
chord_dict_bi = create_vector_dict(bi_chords_chords)
lyrics_dict_bi = create_vector_dict(bi_lyrics_lyrics)
chord_names_bi = create_vector_names(bi_chords_chords)
chord_vectors_bi = create_vectors(bi_chords_chords)
lyric_names_bi = create_vector_names(bi_lyrics_lyrics)
lyric_vectors_bi = create_vectors(bi_lyrics_lyrics)

#BI_2
chord_dict_bi_2 = create_vector_dict(bi_lyrics_chords)
lyrics_dict_bi_2 = create_vector_dict(bi_chords_lyrics)
chord_names_bi_2 = create_vector_names(bi_lyrics_chords)
chord_vectors_bi_2 = create_vectors(bi_lyrics_chords)
lyric_names_bi_2 = create_vector_names(bi_chords_lyrics)
lyric_vectors_bi_2 = create_vectors(bi_chords_lyrics)


#Find the features for the chords and lyrics
all_chords = []
all_lyrics = []

grand_embs_chords = []
grand_embs_lyrics = []
grand_embs_chords_bi = []
grand_embs_lyrics_bi = []
grand_embs_chords_bi_2 = []
grand_embs_lyrics_bi_2 = []
my_labels = []
count = 1

exist_chords = pickle.load(open( "./data/exist_chords.p", "rb"))
exist_lyrics = pickle.load(open( "./data/exist_lyrics.p", "rb"))
exist_anns = pickle.load(open( "./data/exist_anns.p", "rb"))


for i in range(len(exist_chords)):
    #TODO: We are using lines instead of song endings now!
    #TODO Use extend here!!
    #Chords - MONO
    clean_chords = exist_chords[i]
    chord_embs = find_embeddings_chords(clean_chords,chord_dict)
    
    
    #Lyrics - MONO
    clean_lyrics = exist_lyrics[i]
    lyric_embs = find_embeddings_lyrics(clean_lyrics,lyrics_dict)
    
    
    #Chords - BI
    chord_embs_bi = find_embeddings_chords(clean_chords,chord_dict_bi)
    
    #Lyrics - BI
    lyric_embs_bi = find_embeddings_lyrics(clean_lyrics,lyrics_dict_bi)
    
    #Chords - BI
    chord_embs_bi_2 = find_embeddings_chords(clean_chords,chord_dict_bi_2)
    
    #Lyrics - BI
    lyric_embs_bi_2 = find_embeddings_lyrics(clean_lyrics,lyrics_dict_bi_2)
    grand_embs_chords.append(chord_embs)
    grand_embs_lyrics.append(lyric_embs)
    grand_embs_chords_bi.append(chord_embs_bi)
    grand_embs_lyrics_bi.append(lyric_embs_bi)
    grand_embs_chords_bi_2.append(chord_embs_bi_2)
    grand_embs_lyrics_bi_2.append(lyric_embs_bi_2)
    

    all_chords.append(clean_chords)
    all_lyrics.append(clean_lyrics)

#BOW 
#TODO: this isn't right
    
all_BOW = find_BOW(all_chords,all_lyrics)




BOW_right = sparse.lil_matrix(sparse.csr_matrix(all_BOW)[:,:])

print("Total len")
print(len(exist_anns))
print(len(chord_embs))
print(len(lyric_embs))
print(len(chord_embs_bi))
print(len(lyric_embs_bi))

Xc = np.asarray([np.array(xi) for xi in grand_embs_chords]).reshape(len(grand_embs_chords),grand_embs_chords[0].shape[1])
Xl = np.asarray([np.array(xi) for xi in grand_embs_lyrics]).reshape(len(grand_embs_chords),grand_embs_chords[0].shape[1])
Xc_bi = np.asarray([np.array(xi) for xi in grand_embs_chords_bi]).reshape(len(grand_embs_chords),grand_embs_chords[0].shape[1])
Xl_bi = np.asarray([np.array(xi) for xi in grand_embs_lyrics_bi]).reshape(len(grand_embs_chords),grand_embs_chords[0].shape[1])
Xc_bi_2 = np.asarray([np.array(xi) for xi in grand_embs_chords_bi_2]).reshape(len(grand_embs_chords),grand_embs_chords[0].shape[1])
Xl_bi_2 = np.asarray([np.array(xi) for xi in grand_embs_lyrics_bi_2]).reshape(len(grand_embs_chords),grand_embs_chords[0].shape[1])

BOW_right_2 = np.asarray([np.array(xi) for xi in BOW_right])


# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(exist_anns)
#print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
y_hot = onehot_encoder.fit_transform(integer_encoded)

#If ovr
y_hot = integer_encoded
#print(y_hot)


#... to here ****


good_j = []
corr_i = []
my_accs = []
my_f1s = []
#Reduce the dimension to 50
dims = 50
max_acc = .553


for i in range(8):
    #Chords only
    if i == 0:
        X = Xc
    #Lyrics only
    elif i == 1:
        X = Xl
    #Chords enriched by lyrics, chords are source
    elif i == 2:
        X = Xc_bi
    #Lyrics enriched by chords, lyrics are source
    elif i == 3:
        X = Xl_bi
    #Chords enriched by lyrics, lyrics are source
    elif i == 4:
        X = Xc_bi_2
    #Lyrics enriched by chords, chords are source
    elif i == 5:
        X = Xl_bi_2
    #Chords enriched by lyrics, chords and lyrics are source
    elif i == 6:
        X = (Xc_bi+Xc_bi_2)/2.0
    #Lyrics enriched by chords, chords and lyrics are source
    elif i == 7:
        X = (Xl_bi+Xl_bi_2)/2.0
    #All of the learned embeddings, averaged
    else:
        X = (Xl_bi+Xl_bi_2+Xc_bi+Xc_bi_2)/4.0

    pca = PCA(n_components=dims)
    pca.fit(X)
    X = pca.transform(X)
    svd = TruncatedSVD(dims)
    Xpca = svd.fit_transform(BOW_right)
    #For BOW 
    #X_shuf, y_hot_shuf = shuffle(Xpca, y_hot, random_state = 91)
    #For anything else:
    X_shuf, y_hot_shuf = shuffle(X, y_hot,random_state = 92)

        
        
    #TODO: Decision tree or logreg!
        
    #Logistic Regression
    classifier = LogisticRegression(random_state = 92, multi_class='ovr', solver='lbfgs', C = .25)#, class_weight = 'balanced')
    #kNN
    #classifier = tree.DecisionTreeClassifier()
    kfold = KFold(n_splits=5, random_state = 92)
    #X_train, X_test, y_train, y_test = train_test_split(X, y_hot, test_size=0.20, random_state=42)
    
    
    kf = KFold(n_splits=5, random_state = 92)
    kf.get_n_splits(X_shuf)
    
    accs = []
    f1s = []
    for train_index, test_index in kf.split(X_shuf):
        
        #LogReg
        classifier = LogisticRegression(random_state=92, multi_class='ovr', solver='lbfgs', C = .25)#, class_weight = 'balanced')
        #kNN
        #classifier = tree.DecisionTreeClassifier()
        
        X_train, X_test = X_shuf[train_index], X_shuf[test_index]
        y_train, y_test = y_hot_shuf[train_index], y_hot_shuf[test_index]
        
        #set baselines
        #y_frequent = [[0,1,0,0,0]]*len(y_test)
        #y_hamming = [[0,1,1,1,0]]*len(y_test)
        #y_average = [y_train.sum(0)/len(y_train)]*len(y_test)
        
        #train logreg
        #classifier.fit(X_train, y_train.ravel())
        classifier.fit(X_train, y_train)
        #predict
        y_pred = classifier.predict(X_test)
        
        #print("RMSE:")
        #print(RMSE)
        #Logreg
        #acc = 0
        #for jj in range(len(y_test)):
        #    if (y_test[jj][0] == y_pred[jj]).all():
        #        acc = acc+1
        #Knn
        acc = 0
        for jj in range(len(y_test)):
            if (y_test[jj] == y_pred[jj]).all():
                acc = acc+1
        
        #print("Accuracy:")
        accuracy = (acc+0.0)/(len(y_test))
        accs.append(accuracy)
        

            
        
        f1s.append((f1_score(y_test, y_pred,average='weighted')))
        #print(accuracy)
    
    if (sum(accs)/5.0) > max_acc:
        max_acc = (sum(accs)/5.0)
        good_j.append(j)
        corr_i.append(i)
        my_accs.append((sum(accs)/5.0))
        my_f1s.append(sum(f1s)/5.0)


    print("i ",str(i))
#        print("best j")
#        print(best_j)
#        print("best kk")
#        print(best_kk)
    print("Accuracy")
    print(sum(accs)/5.0)
    print("f1")
    print(sum(f1s)/5.0)
    
        
    
    
    
#set baselines
#y_frequent = [0,1,0]
#y_average = [y_hot.sum(0)/len(y_hot)]


#Frequent
acc = 0
for j in range(len(y_hot)):
    if (y_hot[j][0] == 1):
        acc = acc+1
print("Accuracy frequent:")
print((acc+0.0)/(len(y_hot)))



print("Frequent f-score")
print(f1_score(y_hot_shuf, np.asarray([1.0]*len(y_hot_shuf)).reshape(828,1),average='weighted'))

