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

#NOTE: Comment or uncomment from this until the next flag if you already have the links!
#
#
#

#OVERVIEW: on Ukutabs, the website is formatted with different levels of hierarchy
#Goes like this: 1) Each letter of alphabet 2) Each artist whose name starts with that letter
# 3) Each song by that artist
# So to get every song we first cycle through every letter, then every artist in that letter, then their songs
song_links = []
artist_links = []
no_artist_links = []
#access every artist by going alphabetically
#example URL: https://ukutabs.com/artist/d/


for c in ascii_lowercase:
    url = "https://ukutabs.com/artist/"+str(c)+"/"
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
            if link.get('href')[:29] == 'https://ukutabs.com/artist/'+str(c)+'/':
                #print link.get('href')
                temp = link.get('href')
                artist_links.append(temp)
                # we'll need this next one for later b/c the URL formatting for individual songs is different...
                no_artist_links.append(str(temp[:20])+str(c)+str(temp[28:]))
        except:
            print('No Link Class or link is smaller than 29 letters')
            continue

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
            if link.get('href')[:len(no_artist_links[i])] == no_artist_links[i]:
                #print link.get('href')
                song_links.append(link.get('href'))
        except:
            print('No Link Class or link is smaller than 29 letters')
            continue
pickle.dump(song_links,open( "song_links_uku.p", "wb"))

#
#
#
#This is the endpoint of the comment block if you already have the links

#now all of our song URLs are stored
song_links = pickle.load(open( "song_links_uku.p", "rb"))
print((len(song_links)))
b = 0 #b keeps track of how many songs we've successfully transcribed to text
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
                with open('non-english-songs_using_lyrics.txt','a') as myFile:
                        myFile.write(song_links[i]+'\n')
                print("Skipping")
                continue
            with open('chords_and_lyrics_uku_pipes_english_only_using_lyrics.txt','a') as myFile:
                myFile.write(link.text+'\n'+'||SONGMARKER|| '+ artist + ' - ' + song_title + '\n')
            b = b+1
        except:
            continue
#5014 songs for .3 on 04/01
#10936 songs for .3 on 06/07
print(('Number of songs ' + str(b)))

