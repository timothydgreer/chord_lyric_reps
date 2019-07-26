#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 15:50:00 2019

@author: greert
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 13:20:49 2017

@author: greert
"""
import urllib.request, urllib.error, urllib.parse
from bs4 import BeautifulSoup
import pickle
from string import ascii_lowercase
import enchant


#Download pyenchant to use this method: https://github.com/rfk/pyenchant
"""
Test
d=enchant.Dict("en_US")
print(d.check("Hello")) #should print True
print(d.check("Helo")) #should print False
"""
def checkEnglish(title):
    ENGLISH_THRESHOLD = 0.3 #if 30% or more of the words are in English, consider english
    dict = enchant.Dict("en_US")
    titleWords = title.split()
    totalWords = len(titleWords)+0.0
    if totalWords == 0.0:
        return True
    englishWords = 0.0
    for word in titleWords:
        if (dict.check(word)): #if this word is in the English dictionary
            englishWords+=1.0
    return (englishWords/totalWords >= ENGLISH_THRESHOLD)

#urllib2 object can open URLs for us
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]

#NOTE: Comment from this until the the *** flag if you already have the links!
#
#
#
'''
#OVERVIEW: on Ukutabs, the website is formatted with different levels of hierarchy
#Goes like this: 1) Each letter of alphabet 2) Each artist whose name starts with that letter
# 3) Each song by that artist
# So to get every song we first cycle through every letter, then every artist in that letter, then their songs
genre_links = []
#access every artist by going alphabetically
#example URL: https://ukutabs.com/artist/d/

genres = ["Pop", "rock", "indie", "alternative", "female-vocalists", 
          "singer-songwriter", "british", "acoustic", "folk", "seen-live", 
          "indie-pop", "electronic", "indie-rock", "alternative-rock", 
          "pop-rock", "pop-punk", "classic-rock", "emo", "soul", "country", 
          "rnb", "Hip-Hop", "male-vocalists", "dance", "80s", "rap", 
          "Canadian", "punk", "powerpop", "britpop", "60s", "american", 
          "christian", "australian", "X-factor", "hard-rock", "piano", 
          "oldies", "Disney", "jazz", "punk-rock", "youtube", "blues", 
          "christian-rock", "irish", "70s", "easy-listening", "indie-folk", 
          "boyband", "one-direction", "experimental", "electronica", 
          "female-vocalist", "Soundtrack", "folk-rock", "worship", 
          "modern-country", "funk", "90s", "metal", "House", "reggae", 
          "post-hardcore", "hardcore", "guilty-pleasure", "Grime", "chillout", 
          "american-idol", "screamo", "bluegrass", "new-wave", "chill", "r&b", 
          "progressive-rock", "soft-rock", "swedish", "comedy", "grunge", 
          "musical", "emocore", "Alt-country", "mellow", "ska", "psychedelic", 
          "Scottish", "Lo-Fi", "swing", "nu-metal", "piano-rock", "cute", 
          "Acoustic-Rock", "country-pop", "Rock-and-Roll", "rhythm-and-blues", 
          "female", "Selena-Gomez", "New-Zealand", "Broadway", "americana"]

for g in genres:
    url = "https://ukutabs.com/genre/"+g+"/"
    print(url)
    response = opener.open(url)
    page = response.read()
    soup = BeautifulSoup(page, 'lxml') #BeautifulSoup allows us to scan and process HTML
    
    #access every link in the HTML--we're going to find the links to
    # individual artist pages from the list of artists
    #example URL: https://ukutabs.com/artist/d/daft-punk/
    for link in soup.find_all('a'):
        try:
            #out of all the links on the page, find only the ones that are artist pages
            # (i.e. not the links to the HOME page, advertisement links, etc.)
            if link.get('href')[:22] in ['https://ukutabs.com/'+str(x)+'/' for x in ascii_lowercase]:
                #print link.get('href')
                temp = link.get('href')
                genre_links.append(temp)
        except:
            print('No Link Class or link is smaller than 29 letters')
            continue
print((len(genre_links)))

genre_links = sorted(list(set(genre_links)))

print((len(genre_links)))


song_links = []
artist_links = []
#access every artist by going alphabetically
#example URL: https://ukutabs.com/artist/d/


for c in ascii_lowercase:
    url = "https://ukutabs.com/"+str(c)+"/"
    if url == 'https://ukutabs.com/s/':
        continue
    response = opener.open(url)
    page = response.read()
    soup = BeautifulSoup(page, 'lxml') #BeautifulSoup allows us to scan and process HTML
    
    #access every link in the HTML--we're going to find the links to
    # individual artist pages from the list of artists
    #example URL: https://ukutabs.com/artist/d/daft-punk/
    for link in soup.find_all('a'):
        try:
            #out of all the links on the page, find only the ones that are artist pages
            # (i.e. not the links to the HOME page, advertisement links, etc.)
            if link.get('href')[:22] == 'https://ukutabs.com/'+str(c)+'/':
                #print link.get('href')
                temp = link.get('href')
                artist_links.append(temp)
        except:
            print('No Link Class or link is smaller than 29 letters')
            continue
print((len(artist_links)))

#now for each artist, open all their individual songs
for i in range(len(artist_links)):
    url = str(artist_links[i])
    response = opener.open(url)
    page = response.read()
    soup = BeautifulSoup(page, 'lxml')
    for link in soup.find_all('a'): #find all links on artist page
        try:
            #remember song links drop the 'artist' from the URL
            #example artist URL: https://ukutabs.com/artist/d/daft-punk/
            #example song URL: https://ukutabs.com/d/daft-punk/get-lucky-feat-pharrel-williams/
            if link.get('href')[:len(artist_links[i])] == artist_links[i]:
                #print link.get('href')
                song_links.append(link.get('href'))
        except:
            print('No Link Class or link is smaller than 29 letters')
            continue
            
#Remove riffraff and duplicate links and sort
song_links_2 = list(set(song_links+genre_links))
song_links_3 = sorted(song_links+genre_links)
song_links_4 = [x for x in song_links_3 if x.count('/') == 6]


#Dump
pickle.dump(song_links_4,open( "data/song_links.p", "wb"))


#
#
#
#***This is the endpoint of the comment block if you already have the links
'''
#now all of our song URLs are stored

#genre should be '' if analyzing all song links, and '_country' for country, etc.
genre = '_latin'
song_links = pickle.load(open( "data/song_links"+genre+".p", "rb"))
print((len(list(set(song_links)))))
song_links = sorted(list(set(song_links)))

b = 0 #b keeps track of how many songs we've successfully transcribed to text

non_eng = []
val_links = []
for i in range(len(song_links)):
    url = str(song_links[i])
    url_split = url.split('/')
    song_title = ' '.join(url_split[-2].split('-'))
    artist = ' '.join(url_split[-3].split('-'))
    try:
        response = opener.open(url)
    except:
        print((url + ' cannot be opened'))
        continue
    page = response.read()
    soup = BeautifulSoup(page, 'lxml')

    #print soup.prettify()
    pre = soup.find_all('pre') #the chord+lyric data we're looking for is in a <pre> HTML tag
    tempy = ''
    for link in soup.find_all('pre'):
            try:
                tempy = tempy+link.text
            except:
                print('No Pre')
                continue
    #this finds all anchors (<a>) in all <span>s in all <td>s in each <pre>
    #^This implementation is site specific to the way UkuTabs organizes the content
    anchors = [a for a in (td.find_all('span') for td in soup.find_all('pre')) if a]
    for link in soup.find_all('pre'):
        try:
            if not checkEnglish(link.text):
                print((str(song_links[i])))
                non_eng.append(song_links[i])
                print("Skipping")
                continue
            val_links.append(link.text+'\n'+'||SONG_ENDING_MARKER|| '+ artist + ' - ' + song_title + "\n")
            b = b+1
        except:
            continue
with open('data/non-english-songs'+genre+'.txt','w') as myFile:
    myFile.writelines(non_eng)

with open('data/chords_and_lyrics'+genre+'.txt','w') as myFile:
    myFile.writelines(val_links)

#5014 songs for .3 on 04/01/18
#5304 songs for .3 on 06/19/19
#5474 songs for .3 on 07/09/19 using genre as well
print(('Number of songs ' + str(b)))

