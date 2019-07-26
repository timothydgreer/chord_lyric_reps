#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 16:05:40 2019

@author: greert
"""

import json
import pickle
import string

    
with open('data/ranks_per_week_20_years.json') as json_file2:
    data = json.load(json_file2)

#TODO: Probably better to make a dictionary with this
    
#Get the country songs and artists from Billboards
country_songs = []
country_artists = []
for i in data['country'].values():
    for j in range(len(i)):
        country_songs.append(i[j][0])
        country_artists.append(i[j][1])
        
latin_songs = []
latin_artists = []
for i in data['latin'].values():
    for j in range(len(i)):
        latin_songs.append(i[j][0])
        latin_artists.append(i[j][1])
        
pop_songs = []
pop_artists = []
for i in data['pop'].values():
    for j in range(len(i)):
        pop_songs.append(i[j][0])
        pop_artists.append(i[j][1])
        
hip_hop_songs = []
hip_hop_artists = []
for i in data['r-b-hip-hop'].values():
    for j in range(len(i)):
        hip_hop_songs.append(i[j][0])
        hip_hop_artists.append(i[j][1])
        

rock_songs = []
rock_artists = []
for i in data['rock'].values():
    for j in range(len(i)):
        rock_songs.append(i[j][0])
        rock_artists.append(i[j][1])



#Remove duplicates but keep the artists lined up with songs
country_songs_2 = []
country_artists_2 = []
for i in country_songs:
    if i not in country_songs_2:
        ind = country_songs.index(i)
        country_songs_2.append(i)
        country_artists_2.append(country_artists[ind])

latin_songs_2 = []
latin_artists_2 = []
for i in latin_songs:
    if i not in latin_songs_2:
        ind = latin_songs.index(i)
        latin_songs_2.append(i)
        latin_artists_2.append(latin_artists[ind])
    
pop_songs_2 = []
pop_artists_2 = []
for i in pop_songs:
    if i not in pop_songs_2:
        ind = pop_songs.index(i)
        pop_songs_2.append(i)
        pop_artists_2.append(pop_artists[ind])
    
hip_hop_songs_2 = []
hip_hop_artists_2 = []
for i in hip_hop_songs:
    if i not in hip_hop_songs_2:
        ind = hip_hop_songs.index(i)
        hip_hop_songs_2.append(i)
        hip_hop_artists_2.append(hip_hop_artists[ind])

rock_songs_2 = []
rock_artists_2 = []
for i in rock_songs:
    if i not in rock_songs_2:
        ind = rock_songs.index(i)
        rock_songs_2.append(i)
        rock_artists_2.append(rock_artists[ind])


#Sort so that searching later is faster. Keep artist and song lined up
country_songs_2, country_artists_2 = (list(x) for x in zip(*sorted(zip(country_songs_2, country_artists_2))))
latin_songs_2, latin_artists_2 = (list(x) for x in zip(*sorted(zip(latin_songs_2, latin_artists_2))))
pop_songs_2, pop_artists_2 = (list(x) for x in zip(*sorted(zip(pop_songs_2, pop_artists_2))))
hip_hop_songs_2, hip_hop_artists_2 = (list(x) for x in zip(*sorted(zip(hip_hop_songs_2, hip_hop_artists_2))))
rock_songs_2, rock_artists_2 = (list(x) for x in zip(*sorted(zip(rock_songs_2, rock_artists_2))))

song_links = pickle.load(open('data/song_links.p','rb'))

#Use the urls of the links to match up the billboards to the songs
song_links_titles = []
song_links_artists = []
for i in song_links:
    temp = i.split('/')
    song_links_artists.append(' '.join(temp[4].split('-')))
    song_links_titles.append(' '.join(temp[5].split('-')))


#Unnecessary, but you can sort the same way as above
#sorted_song_links_titles, sorted_song_links_artists = (list(x) for x in zip(*sorted(zip(song_links_titles, song_links_artists))))


#Go through songs and see if there's a match
print('Country...')
count = 0
country_inds = []
for i in country_songs_2:
    #Lowercase to match the url
    k = i.lower()
    #remove punctuation to match the url
    k = k.translate(str.maketrans('', '', string.punctuation))
    #If the song title is a match
    if k in song_links_titles:
        #Find the index of the song in the billboards and ukutabs
        ind = country_songs_2.index(i)
        ind2 = song_links_titles.index(k)
        #Now compare and see if the artist in ukutabs is in the billboards artist of the same song
        if country_artists_2[ind].lower() in song_links_artists[ind2]:#True:
            count = count+1
            #Add the link if there's a match!
            country_inds.append(song_links_titles.index(k))
print(count)

print('Latin...')
count = 0
latin_inds = []
for i in latin_songs_2:
    k = i.lower()
    k = k.translate(str.maketrans('', '', string.punctuation))
    if k in song_links_titles:
        ind = latin_songs_2.index(i)
        ind2 = song_links_titles.index(k)
        #If you want to compare titles and artists, uncomment below
        #print("Song in Links: ",k)
        #print("Artist in Links: ",song_links_artists[ind2])
        #print("Song in Billboard: ",latin_songs_2[ind])
        #print("Artist in Billboards: ",latin_artists_2[ind].lower())
        #print("")
        if latin_artists_2[ind].lower() in song_links_artists[ind2]:#True:#
            count = count+1
            latin_inds.append(song_links_titles.index(k))
print(count)
print('Pop...')
count = 0
pop_inds = []
for i in pop_songs_2:
    k = i.lower()
    k = k.translate(str.maketrans('', '', string.punctuation))
    if k in song_links_titles:
        ind = pop_songs_2.index(i)
        ind2 = song_links_titles.index(k)
        if pop_artists_2[ind].lower() in song_links_artists[ind2]:#True:#
            count = count+1
            pop_inds.append(song_links_titles.index(k))
        #print(i)
print(count)

print('Hip Hop...')
count = 0
hip_hop_inds = []
for i in hip_hop_songs_2:
    k = i.lower()
    k = k.translate(str.maketrans('', '', string.punctuation))
    if k in song_links_titles:
        ind = hip_hop_songs_2.index(i)
        ind2 = song_links_titles.index(k)
        if hip_hop_artists_2[ind].lower() in song_links_artists[ind2]:#True:#
            count = count+1
            hip_hop_inds.append(song_links_titles.index(k))
        #print(i)
print(count)

print('Rock...')
count = 0
rock_inds = []
for i in rock_songs_2:
    k = i.lower()
    k = k.translate(str.maketrans('', '', string.punctuation))
    if k in song_links_titles:
        ind = rock_songs_2.index(i)
        ind2 = song_links_titles.index(k)
        if rock_artists_2[ind].lower() in song_links_artists[ind2]:#True:#
            count = count+1
            rock_inds.append(song_links_titles.index(k))
        #print(i)
print(count)

#Add in the links
country_links = []
for i in range(len(country_inds)):
    country_links.append(song_links[country_inds[i]])
    
latin_links = []
for i in range(len(latin_inds)):
    latin_links.append(song_links[latin_inds[i]])
    
pop_links = []
for i in range(len(pop_inds)):
    pop_links.append(song_links[pop_inds[i]])
    
hip_hop_links = []
for i in range(len(hip_hop_inds)):
    hip_hop_links.append(song_links[hip_hop_inds[i]])
    
rock_links = []
for i in range(len(rock_inds)):
    rock_links.append(song_links[rock_inds[i]])
    
#See if there's overlap using as an example:
#len(set(pop_links).intersection(set(country_links)))

pickle.dump(country_links,open('data/song_links_country.p','wb'))
pickle.dump(latin_links,open('data/song_links_latin.p','wb'))
pickle.dump(pop_links,open('data/song_links_pop.p','wb'))
pickle.dump(hip_hop_links,open('data/song_links_hip_hop.p','wb'))
pickle.dump(rock_links,open('data/song_links_rock.p','wb'))