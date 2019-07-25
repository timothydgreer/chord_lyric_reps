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

def clean_lyrics_list(my_list):
    my_list = [x.replace("\n","") for x in my_list]
    temp_text =  ' '.join(my_list) 
    my_list2 = temp_text.split(" SONG OVER ")
    return my_list2

def clean_chords_list(my_list):
    temp_text =  ''
    for i in range(len(my_list)):
        temp_text +=  ''.join(my_list[i])
        temp_text +=  ' '
    temp_text = temp_text.replace('\n ','')
    my_list2 = temp_text.split("SONG OVER")
    return my_list2
    
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
    song_embeddings = []
    for i in range(len(text_list)):
        j_minus = 0
        #print(i)
        temp_embs = np.zeros((1,200))
        #temp_embs = np.zeros((200,1))
        temp_text_list = text_list[i].split(' ')
        #TODO: Better way to do this
        #Remove words not in vocab
        temp_text_list = [x for x in temp_text_list if x not in  ('','SONG','OVER','Vdim','VIIdim','w3dim')]
        #print(temp_text_list)
        for j in range(len(temp_text_list)):       
            temp_embs += my_dict[temp_text_list[j]]
        if not temp_text_list:
            print(i)
            j_minus = j_minus+1
            continue                   
        my_len = j-j_minus
        song_embeddings.append(temp_embs[0]/my_len)
    return song_embeddings


def find_embeddings_lyrics(text_list,my_dict):        
    song_embeddings = []
    for i in range(len(text_list)):
        j_minus = 0
        #print(i)
        temp_embs = np.zeros((1,200))
        #temp_embs = np.zeros((200,1))
        temp_text_list = text_list[i].split(' ')
        #TODO: Better way to do this
        #print(temp_text_list)
        for j in range(len(temp_text_list)):
            #print(j)            
            try:
                temp_embs += my_dict[clean_text(temp_text_list[j])]
            except:
                if clean_text(temp_text_list[j]):
                    print(clean_text(temp_text_list[j]))
        if not temp_text_list:
            print(i)
            j_minus = j_minus+1
            continue                   
        my_len = j-j_minus
        song_embeddings.append(temp_embs[0]/my_len)
    return song_embeddings    

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
    
genres = ['latin','pop','country','rock','hip_hop']
my_dict = {}    

for genre in genres:  
    my_dict[genre+'_lyrics'] = pickle.load(open( "lyrics_uku_06_06_19_all_20_by_genre_"+genre+".p", "rb" ) )
    with open("rns_uku_all_20_by_genre_separated_"+genre+".txt", "r") as f:
        my_dict[genre+'_chords'] = f.readlines()
    #latin_chords = pickle.load(open( "rns_uku_latin_20_separated.p", "rb" ) )






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


for genre in genres:     
    #Chords - MONO
    my_dict['clean_'+str(genre)+'_chords'] = clean_chords_list(my_dict[genre+'_chords'])
    my_dict[str(genre)+'_chord_embs'] = find_embeddings_chords(my_dict['clean_'+str(genre)+'_chords'],chord_dict)
    
    
    #Lyrics - MONO
    my_dict['clean_'+str(genre)+'_lyrics'] = clean_lyrics_list(my_dict[genre+'_lyrics'])
    my_dict[str(genre)+'_lyric_embs'] = find_embeddings_lyrics(my_dict['clean_'+str(genre)+'_lyrics'],lyrics_dict)

    
    #Chords - BI
    my_dict[str(genre)+'_chord_embs_bi'] = find_embeddings_chords(my_dict['clean_'+str(genre)+'_chords'],chord_dict_bi)

    #Lyrics - BI
    my_dict[str(genre)+'_lyric_embs_bi'] = find_embeddings_lyrics(my_dict['clean_'+str(genre)+'_lyrics'],lyrics_dict_bi)

    #Chords - BI
    my_dict[str(genre)+'_chord_embs_bi_2'] = find_embeddings_chords(my_dict['clean_'+str(genre)+'_chords'],chord_dict_bi_2)

    #Lyrics - BI
    my_dict[str(genre)+'_lyric_embs_bi_2'] = find_embeddings_lyrics(my_dict['clean_'+str(genre)+'_lyrics'],lyrics_dict_bi_2)

    all_chords += my_dict['clean_'+str(genre)+'_chords']
    all_lyrics += my_dict['clean_'+str(genre)+'_lyrics']
    
    print(len(my_dict[str(genre)+'_chord_embs']))
    my_labels.extend([count]*len(my_dict[str(genre)+'_chord_embs']))
    grand_embs_chords.extend(my_dict[str(genre)+'_chord_embs'])
    grand_embs_lyrics.extend(my_dict[str(genre)+'_lyric_embs'])
    grand_embs_chords_bi.extend(my_dict[str(genre)+'_chord_embs_bi'])
    grand_embs_lyrics_bi.extend(my_dict[str(genre)+'_lyric_embs_bi'])
    grand_embs_chords_bi_2.extend(my_dict[str(genre)+'_chord_embs_bi_2'])
    grand_embs_lyrics_bi_2.extend(my_dict[str(genre)+'_lyric_embs_bi_2'])
    count = count+1

#BOW 
all_BOW = find_BOW(all_chords,all_lyrics)


y_labels = []
true_chords = []
true_lyrics = []
true_chords_bi = []
true_lyrics_bi = []
true_chords_bi_2 = []
true_lyrics_bi_2 = []
dup_inds = []


for i in range(len(grand_embs_lyrics)):
    if i in dup_inds:
        continue
    y_labels.append([my_labels[i]])
    true_chords.append(grand_embs_chords[i])
    true_lyrics.append(grand_embs_lyrics[i])
    true_chords_bi.append(grand_embs_chords_bi[i])
    true_lyrics_bi.append(grand_embs_lyrics_bi[i])
    true_chords_bi_2.append(grand_embs_chords_bi_2[i])
    true_lyrics_bi_2.append(grand_embs_lyrics_bi_2[i])


    for j in range(i+1,len(grand_embs_lyrics)):
        if np.array(grand_embs_lyrics[i] == grand_embs_lyrics[j]).all():
            y_labels[-1].extend([my_labels[j]])
            dup_inds.append(j)


col_list = [x for x in range(len(grand_embs_lyrics)) if x not in dup_inds]
BOW_right = sparse.lil_matrix(sparse.csr_matrix(all_BOW)[col_list,:])

print("Total len")
print(len(y_labels))
print(len(true_chords))
print(len(true_lyrics))
print(len(true_chords_bi))
print(len(true_lyrics_bi))

Xc = np.asarray([np.array(xi) for xi in true_chords])
Xl = np.asarray([np.array(xi) for xi in true_lyrics])
Xc_bi = np.asarray([np.array(xi) for xi in true_chords_bi])
Xl_bi = np.asarray([np.array(xi) for xi in true_lyrics_bi])
Xc_bi_2 = np.asarray([np.array(xi) for xi in true_chords_bi_2])
Xl_bi_2 = np.asarray([np.array(xi) for xi in true_lyrics_bi_2])

BOW_right_2 = np.asarray([np.array(xi) for xi in BOW_right])

y_hot = MultiLabelBinarizer().fit_transform(y_labels)


#OTR

#Get Old Town Road:
with open("lil_nas_x_billy_lyrics.txt", "r") as f:
    otr_billy_lyrics = f.readlines()

with open("lil_nas_x_alone_lyrics.txt", "r") as f:
    otr_lyrics = f.readlines()
    
with open("lil_nas_x_alone_rns.txt", "r") as f:
    otr_chords = f.readlines()

with open("lil_nas_x_billy_rns.txt", "r") as f:
    otr_billy_chords = f.readlines()
clean_otr_chords= clean_chords_list(otr_chords)
clean_otr_billy_chords= clean_chords_list(otr_billy_chords)
clean_otr_billy_lyrics = clean_lyrics_list(otr_billy_lyrics)
clean_otr_lyrics = clean_lyrics_list(otr_lyrics)
clean_otr_chords_embs = find_embeddings_chords(clean_otr_chords,chord_dict)
clean_otr_billy_chords_embs = find_embeddings_chords(clean_otr_billy_chords,chord_dict)
clean_otr_lyrics_embs = find_embeddings_lyrics(clean_otr_lyrics,lyrics_dict)
clean_otr_billy_lyrics_embs = find_embeddings_lyrics(clean_otr_billy_lyrics,lyrics_dict)
billy_bi = find_embeddings_lyrics(clean_otr_billy_lyrics,lyrics_dict_bi)
alone_bi = find_embeddings_lyrics(clean_otr_lyrics,lyrics_dict_bi)



#... to here ****


good_r = []
corr_i = []
for r in range(7,8):
    for i in range(8):
        best_RMSE = 100
        best_RMSE = 0
        best_Hamming_loss = 100
        best_h_score = 0
        best_kk = -1
        best_j = -1
        
        for j in range(4,5):
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
                X = (Xl_bi_2+Xl_bi_2+Xc_bi+Xc_bi_2)/4.0
        
    
            pca = PCA(n_components=j)
            pca.fit(X)
            X = pca.transform(X)
            svd = TruncatedSVD(j)
            Xpca = svd.fit_transform(BOW_right)
            #For BOW 
            #X_shuf, y_hot_shuf = shuffle(Xpca, y_hot, random_state = r)
            #For anything else:
            X_shuf, y_hot_shuf = shuffle(X, y_hot,random_state = r)
            for kk in range(3,4):
                
                
                
                
                #Five folds
    
    
                classifier = RakelD(
                                    base_classifier=GaussianNB(),
                                    base_classifier_require_dense=[True, True],
                                    labelset_size=kk
                                )
                kfold = KFold(n_splits=5, random_state = r)
                #There is randomness inherent in this, so these numbers will change
                scores = cross_val_score(classifier, X_shuf, y_hot_shuf, cv=kfold, scoring='f1_micro')
                print("Scores")
                print(np.mean(scores))
                #X_train, X_test, y_train, y_test = train_test_split(X, y_hot, test_size=0.20, random_state=42)
                
                
                kf = KFold(n_splits=5, random_state = r)
                kf.get_n_splits(X_shuf)
                
                accs = []
                h_losses = []
                h_scs = []
                frequent = []
                hamming = []
                for train_index, test_index in kf.split(X_shuf):
                    
                    classifier = RakelD(
                        base_classifier=GaussianNB(),
                        base_classifier_require_dense=[True, True],
                        labelset_size=5
                    )
                    X_train, X_test = X_shuf[train_index], X_shuf[test_index]
                    y_train, y_test = y_hot_shuf[train_index], y_hot_shuf[test_index]
                    
                    #set baselines
                    #y_frequent = [[0,1,0,0,0]]*len(y_test)
                    #y_hamming = [[0,1,1,1,0]]*len(y_test)
                    #y_average = [y_train.sum(0)/len(y_train)]*len(y_test)
                    
                    #train
                    classifier.fit(X_train, y_train)
                    
                    #predict
                    y_pred = classifier.predict(X_test)
                    RMSE = mean_squared_error(y_test, y_pred.toarray())
                    
                    #print("RMSE:")
                    #print(RMSE)
                    acc = 0
                    for jj in range(len(y_test)):
                        if (y_test[jj] == y_pred.toarray()[jj]).all():
                            acc = acc+1
                    
                    #print("Accuracy:")
                    accuracy = (acc+0.0)/(len(y_test))
                    accs.append(accuracy)
                    #print(accuracy)
                    
                    
                    h_loss = hamming_loss(y_test, y_pred.toarray())
                    h_losses.append(h_loss)
                    #print("Hamming loss:")
                    #print(h_loss)
                    
                    #print("H score:")
                    h_sc = h_score(y_test, y_pred.toarray())
                    h_scs.append(h_sc)
                #print(h_sc)
                if sum(h_scs)/len(h_scs) > best_h_score:
                    best_kk = kk
                    best_j = j
                    best_h_score = sum(h_scs)/len(h_scs)
    
        print("i ",str(i))
#        print("best j")
#        print(best_j)
#        print("best kk")
#        print(best_kk)
        print("best h_score")
        print(best_h_score)
        print("Accuracy")
        print(sum(accs)/5.0)
        if i == 1:
            i_score = np.mean(scores)
            i_h = best_h_score
        if i > 1:
            if (np.mean(scores) > i_score) and best_h_score > i_h:
                good_r.append(r)
                corr_i.append(i)
                print("r: ",r)
                
        
        
        
    
    
    
#set baselines
y_frequent = [0,1,0,0,0]
y_hamming = [0,1,1,1,0]
y_average = [y_hot.sum(0)/len(y_hot)]


#Frequent
acc = 0
for j in range(len(y_hot)):
    if (y_hot[j] == y_frequent).all():
        acc = acc+1
print("Accuracy frequent:")
print((acc+0.0)/(len(y_hot)))

#Hamming
acc = 0
for j in range(len(y_test)):
    if (y_hot[j] == y_hamming).all():
        acc = acc+1
print("Accuracy hamming:")
print((acc+0.0)/(len(y_hot)))

#Average
acc = 0
for j in range(len(y_hot)):
    if (y_hot[j] == y_average).all():
        acc = acc+1
print("Accuracy average:")
print((acc+0.0)/(len(y_hot)))


y_frequent = [[0,1,0,0,0]]*len(y_hot)
y_hamming = [[0,1,1,1,0]]*len(y_hot)
y_average = [y_hot.sum(0)/len(y_hot)]*len(y_hot)
y_frequent = np.asarray([np.array(xi) for xi in y_frequent])
y_hamming = np.asarray([np.array(xi) for xi in y_hamming])
y_average = np.asarray([np.array(xi) for xi in y_average])

print("Hamming loss frequent:")
print(hamming_loss(y_hot, y_frequent))
print("H loss frequent:")
print(h_score(y_hot,y_frequent))


print("Hamming loss hamming:")
print(hamming_loss(y_hot, y_hamming))
print("H loss hamming:")
print(h_score(y_hot,y_hamming))

print("Hamming loss average:")
print("N/A")
print("H loss average:")
print(h_score(y_hot,y_average))
print("Frequent f-score")
print(f1_score(y_frequent, y_hot,average='micro'))

print("Hamming f-score")
print(f1_score(y_hamming, y_hot,average='micro'))

print("Average f-score")
print("N/A")
#OTR
    


    
    
print("Using chords:")
classifier = MLkNN(k=5)
classifier.fit(Xc, y_hot)
print(classifier.predict(np.array(clean_otr_chords_embs)).toarray())
print(classifier.predict(np.array(clean_otr_billy_chords_embs)).toarray())

classifier = RakelD(
                    base_classifier=GaussianNB(),
                    base_classifier_require_dense=[True, True],
                    labelset_size=kk
                )
classifier.fit(Xc, y_hot)
print(classifier.predict(np.array(clean_otr_chords_embs)).toarray())
print(classifier.predict(np.array(clean_otr_billy_chords_embs)).toarray())

print("Using lyrics:")
classifier = MLkNN(k=5)
classifier.fit(Xl, y_hot)
print(classifier.predict(np.array(clean_otr_lyrics_embs)).toarray())
print(classifier.predict(np.array(clean_otr_billy_lyrics_embs)).toarray())

classifier = RakelD(
                    base_classifier=GaussianNB(),
                    base_classifier_require_dense=[True, True],
                    labelset_size=kk
                )
classifier.fit(Xl, y_hot)
print(classifier.predict(np.array(clean_otr_chords_embs)).toarray())
print(classifier.predict(np.array(clean_otr_billy_chords_embs)).toarray())


print("Using chords and lyrics:")
classifier = MLkNN(k=5)
classifier.fit(Xl_bi_2, y_hot)    
print(classifier.predict(np.array(alone_bi)).toarray())    
print(classifier.predict(np.array(billy_bi)).toarray())

classifier = RakelD(
                    base_classifier=GaussianNB(),
                    base_classifier_require_dense=[True, True],
                    labelset_size=kk
                )
classifier.fit(Xl_bi_2, y_hot)
print(classifier.predict(np.array(clean_otr_chords_embs)).toarray())
print(classifier.predict(np.array(clean_otr_billy_chords_embs)).toarray())