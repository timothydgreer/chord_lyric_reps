#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Jul 16 11:31:04 2019

@author: greert
"""
import pickle
import numpy as np
import pandas as pd
from os import path
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


def CountFrequency(my_list): 
  
    # Creating an empty dictionary  
    freq = {} 
    for item in my_list: 
        if (item in freq): 
            freq[item] += 1
        else: 
            freq[item] = 1
    
    return freq


song_links = pickle.load(open( "song_links_uku_06_06_19_20_by_genre.p", "rb"))

artists = []
for i in range(len(song_links)):
    url = str(song_links[i])
    url_split = url.split('/')
    artist = '_'.join(url_split[-3].split('-'))
    artists.append(artist)

#You can also put this artist list into wordclouds.com to make your wordcloud    
#print(artists)

print(len(list(set(artists))))

#Will need to download wordcloud
wordcloud = WordCloud(font_path='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',width = 1600, height = 800, 
                background_color ='black', 
                min_font_size = 5, font_step = 1, max_words = 500).generate_from_frequencies(CountFrequency(artists)) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8),facecolor = None) 
plt.imshow(wordcloud,interpolation="bilinear") 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 