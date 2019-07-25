#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 17:38:08 2019

@author: greert
"""

import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np

def generate_MER_data():
    df = pd.read_excel('Grand_Annotations.xlsx', sheetname='Sheet1')
    
    print("Column headings:")
    print(df.columns)
    
    lyrics = df['Lyric Line']
    chords = df['Chord Line']
    
    ta = df['Total Anns']
    final_lyrics = []
    final_chords = []
    final_anns = []
    
    for i in range(len(ta)):
        if not np.isnan(ta[i]):
            final_lyrics.append(lyrics[i])
            final_chords.append(chords[i])
            final_anns.append(ta[i])
    
    print(len(final_anns))
    
    return(final_chords, final_lyrics, final_anns)
    

def find_lines(final_chords, final_lyrics, chord_corpus, lyrics_corpus):
    new_chords = []
    return(new_chords)

if __name__ == '__main__':
    (final_chords, final_lyrics, final_anns) = generate_MER_data()
    with open('./data/rns_uku_all_20_by_genre_unseparated.txt','r') as myFile:
        my1 = myFile.readlines()
    with open('./data/lyrics_uku_all_20_by_genre_unseparated.txt','r') as myFile:
        my2 = myFile.readlines()
    my1 = [x.replace(' \n','') for x in my1]
    my2 = [x.replace('\n','') for x in my2]
    counter = 0
    counter2 = 0
    exist_chords = []
    exist_lyrics = []
    exist_anns = []
    for i in range(len(final_lyrics)):
        if final_lyrics[i] in my2:
            indices = [ind for ind, x in enumerate(my2) if x == final_lyrics[i]]
            for index in indices:                
                temp_rns = my1[index]
                if len(temp_rns.split(' ')) == len(final_chords[i].split(' ')):
                    counter = counter+1
                    exist_chords.append(temp_rns)
                    exist_lyrics.append(final_lyrics[i])
                    exist_anns.append(final_anns[i])
                    break
                else:
                    #print("Mismatch")
                    #print(final_chords[i])
                    #print(temp_rns)
                    print(len(indices))
                    counter2 = counter2+1
        else:
            pass#print(final_lyrics[i])
    print(counter)