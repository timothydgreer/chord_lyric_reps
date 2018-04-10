# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 13:20:49 2017

@author: greert
"""
#Shows that scraping chords and lyrics is easy
#Data cleaning is harder
#We don't use Heartwood Guitar because it's not as easy to clean as UkuTabs
import urllib2
from bs4 import BeautifulSoup
opener = urllib2.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
url = "http://www.heartwoodguitar.com/chords/"
response = opener.open(url)
page = response.read()
soup = BeautifulSoup(page, 'lxml')
chord_links = []
for link in soup.find_all('a'):
    chord_links.append(link.get('href'))
chord_links = [x for x in chord_links if x is not None]
chord_short = []
#Open up the chords and scrape the text
for i in xrange(len(chord_links)):
    if 'http://www.heartwoodguitar.com/chords/' in chord_links[i]:
        chord_short.append(chord_links[i])
#Number of songs
print len(chord_short)

for i in xrange(len(chord_short)):
    url = str(chord_short[i])
    #Print what we're scraping
    print url
    response = opener.open(url)
    page = response.read()
    soup = BeautifulSoup(page, 'lxml')
    new_soup = soup.find('div', {'id': 'chord-content'})
    try:
        temp = u''.join(new_soup.text).encode('utf-8').strip()
        with open('heartwood_chords_and_lyrics_full.txt','wb') as myFile:
            myFile.write(temp)
    except:
        #Print that we can't get the unicode
        print "NoneType"
                

