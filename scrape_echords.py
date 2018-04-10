# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 13:20:49 2017

@author: greert
"""
#Scrape echords
#This is a proof of concept. Data here is not clean or easy to scrape
#There are lots of songs, but the quality of the arrangements is not good
#Better to use ukutabs
#We don't save because the resultant file is huge.
import urllib2
from bs4 import BeautifulSoup
from string import ascii_lowercase
opener = urllib2.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
for c in ascii_lowercase:
    url = "http://www.chordie.com/browsesong.php/"+str(c)+".php?start=0&end=50000000&filter=chordsonly"
    response = opener.open(url)
    page = response.read()
    soup = BeautifulSoup(page, 'lxml')
    chord_links = []
    print soup
    for link in soup.find_all('a'):
        try:
            print link
            if link['class'][0] == 'songname':
                print link.get('href')
                chord_links.append('http://www.chordie.com/'+link.get('href'))
        except:
            print 'No Link Class'
            continue                
    