# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 18:29:01 2017

@author: greert
"""
import re
import caster
import pickle
import urllib2
from string import ascii_lowercase
import nltk

#TODO: Why are there only 51 lines??
#Read in Chords and Lyrics in order to analyze them
with open('chords_and_lyrics_uku_pipes_english_only_using_lyrics.txt', 'rb') as myFile:
    temp_text = myFile.readlines()
print type(temp_text[4]) #make sure we have the right type (string)
print temp_text[4]

#read in chord casting table
with open('chord_casting_UTF-8.txt', 'r') as f:
    inputs=[]
    outputs=[]
    exploLine=[]
    for line in f:
        exploLine = line.split("\t")
        inputs.append(exploLine[0].replace('\ufeff','').strip()) #clean up the special chars
        outputs.append(exploLine[1].replace('\n', '').strip()) #clean up special chars
    castingTable = list(zip(inputs, outputs)) #convert casting table to a list of tuples
    castingTable = castingTable[1:] #Unicode error if you don't remove the duplicate first element
    print("Opened Chord Casting Table")
print(castingTable) #test to make sure we've read in the table correctly

#ADDITION FOR CHECKER: KEEP TRACK OF ANY CHORDS THAT AREN'T IN CASTING TABLE
uncastedChords = []

#define regex rules
re_natural = r'[A-G]'
re_modifier = r'#*b*'
re_note = (re_natural + re_modifier) #This is one of the twelve root notes A, A#, B, C, etc.
re_chord = (r'(maj|min|dim|aug|add|sus|m)')
re_interval = (r'([1-9]|1[0-3])')
re_slash = '/'
re_optional = r'('+re_chord+'|'+re_interval+'|'+re_slash+'|'+re_note+')' #any of these extra bits are optional
returnablePattern = re_note + re_optional + r'*'
chordPattern = (caster.nm('chordRet', returnablePattern) + r'\s')
#find matches
big_list_of_chords = []
big_list_of_lyrics = []
chordflag = False
lyricflag = False
count = 0
multiple = 1
#Each "temp section" is a block: i.e. a single verse, chorus, bridge, etc.
temp_section_chords = [[]] #outer array is each line in the block, and each line can have multiple chords (inner array)
temp_section_lyrics = ['']
h = 0
keep = 0
keep2 = 0
#for each line in the fulltext... note that each line is either chords or lyrics, no combo
for u in xrange(len(temp_text)):
    fullText = ' ' + temp_text[u] #fullText is the current line to analyze
    fullText = fullText.replace('ADD','add')
    fullText = fullText.replace('BFF','bff')
    fullText = fullText.replace('CD','cd')
    fullText = fullText.replace('FB','fb')
    fullText = fullText.replace('AC','ac')
    fullText = fullText.replace('GG','G')
    fullText = fullText.replace('EmD','Em D')
    fullText = fullText.replace('EmG','Em G')
    fullText = fullText.replace('AA','A')
    fullText = fullText.replace('BB','B')
    fullText = fullText.replace('Dsus4/D','D')
    fullText = fullText.replace('A/Bb/A','A')
    fullText = fullText.replace('A/Bb','F#7')
    fullText = fullText.replace('Bbb7','F#7')
    fullText = fullText.replace('AB','ab')
    fullText = fullText.replace('Eb3','Eb')
    fullText = fullText.replace('AA','aa')
    fullText = fullText.replace('EE','ee')
    fullText = fullText.replace('CE','ce')
    fullText = fullText.replace('BAm','B am')
    fullText = fullText.replace('CA','ca')
    fullText = fullText.replace('ADA','ada')
    fullText = fullText.replace('GE','ge')
    if "|-" in fullText or "CAPO ON" in fullText:
        continue #skip this line... nothing to analyze here
    #if there is a multiplier (e.g. a chorus that repeats twice, CHORUSx2), we want to weight it that many times as much
    if "Intro" in fullText or"Interlude" in fullText or "Chorus" in fullText or "Outro" in fullText or "Bridge" in fullText or "Verse" in fullText or "Solo" in fullText or "|-" in fullText or "CAPO ON" in fullText or "RIFF " in fullText or "Tuning:" in fullText: 
        if "x2" in fullText:
            multiple = 2
        elif "x3" in fullText:
            multiple = 3
        elif "x4" in fullText:
            multiple = 4
        elif "x5" in fullText:
            multiple = 5
        elif "x6" in fullText:
            multiple = 6
        elif "x7" in fullText:
            multiple = 7
        elif "x8" in fullText:
            multiple = 7
        elif "x9" in fullText:
            multiple = 8
        elif "x10" in fullText:
            multiple = 9
        elif "x11" in fullText:
            multiple = 10
        elif "x12" in fullText:
            multiple = 11
        else:
            multiple = 1
        continue
    #a SONGMARKER marks the end of a song and the beginning of a new song
    if "||SONGMARKER||" in fullText:
        if len(temp_section_chords) != len(temp_section_lyrics):
            h = h+1
            if len(temp_section_chords)-1 != len(temp_section_lyrics):
                keep = keep+1
            temp_section_chords = temp_section_chords[1:]
            del big_list_of_chords[-len(temp_section_chords)]
        for i in xrange(1,multiple):
            big_list_of_chords.extend(temp_section_chords)
            big_list_of_lyrics.extend(temp_section_lyrics)    
        chordflag = False
        lyricflag = False
        multiple = 1
        temp_section_chords = [[]]
        temp_section_lyrics = ['']
        #Uncomment if you want to split by song.
        #big_list_of_chords.append("SONG OVER")
        #big_list_of_lyrics.append("SONG OVER")
        continue
    #if this line is a line break, our current temp section is over. Add to big list and reset.
    if fullText.isspace():
        #get rid of leading space if there is one
        if temp_section_chords[0] == []: 
            temp_section_chords = temp_section_chords[1:]
        #get rid of leading space if there is one
        if temp_section_lyrics[0] == '':
            temp_section_lyrics = temp_section_lyrics[1:]
        #make sure we have 1:1 ratio of chord and lyric lines
        if len(temp_section_chords) != len(temp_section_lyrics):
            #this variable is for debugging--tracks number of mismatched chord/lyric lines
            h = h+1
            if len(temp_section_chords)-1 != len(temp_section_lyrics):
                #also a tracking variable
                keep = keep+1
            if len(temp_section_chords) == 0:
                print "SHO)T"
            del big_list_of_chords[-len(temp_section_chords)]
            temp_section_chords = temp_section_chords[1:]
        #add our temp section multiple times if it has a multiplier
        for i in xrange(1,multiple):
            if len(temp_section_chords) != len(temp_section_lyrics):
                keep2 = keep2+1
                print len(temp_section_chords)-len(temp_section_lyrics)-1
            big_list_of_chords.extend(temp_section_chords)
            big_list_of_lyrics.extend(temp_section_lyrics)
        #We're DONE with this temp section, reset and get ready for the next one
        multiple = 1
        temp_section_chords = [[]]
        temp_section_lyrics = ['']        
        chordflag = False
        lyricflag = False
        continue
    #add our chords (identified using regex) to origChords list
    matchIter = re.finditer(chordPattern, fullText)
    origChords = []
    for elem in matchIter:
        s = elem.start()
        e = elem.end()
        #print('Found "%s" in the text from %d to %d ("%s")' % \
        #      (elem.re.pattern, s, e, elem.group('chordRet') ))
        #check for special case "Am I"
        #Removed because it would just recast Am as Am
        if elem.group('chordRet') != 'Am' or e >= len(fullText) or fullText[e] != 'I':
            origChords.append(elem.group('chordRet'))
        #We should append A minor still?        

        
    #now cast our chords to simplify them
        
    #make a new list with the cast of each chord, using the table, and count the amount of each chord
    castedChords = []
    #have to check the sharps first so it doesn't switch 'C#' into 'w0#'
    noteNums = [('C#',1),('D#',3),('F#',6),('G#',8),('A#',10),('C',0),('D',2),('E',4),('F',5),('G',7),('A',9),('B',11)]
    
    for origChord in origChords:
        castedFlag = False #ADDITIONAL CODE FOR CHECKER VERSION
        #example of process: E/C# -> w4/w1 -> w0/w9 (store shift as 4) -> C/A -> Am -> w9m -> w1m -> C#m
        #convert origChord to numeral version
        origChord = caster.makeFlatsSharps(origChord)
        tempChord = caster.letterChordToNumChord(origChord,noteNums)
        #shift numeral chord to C numeral version
        if tempChord[1]=='1' and len(tempChord)>2 and \
            (tempChord[2]=='0' or tempChord[2]=='1'):
            rootNum = int(tempChord[1:3])
        else:
            rootNum = int(tempChord[1])
        shift = rootNum
        tempChord = caster.shiftNumChord(tempChord, -shift)
        #convert C numeral version to C letter version
        tempChord = caster.numChordToLetterChord(tempChord)
        #cast C letter version using table
        for chord in castingTable:
            if tempChord == chord[0]:
                tempChord = chord[1]
                castedFlag = True;
                break
        #convert casted letter chord to casted numeral chord
        tempChord = caster.letterChordToNumChord(tempChord,noteNums)
        #shift casted numeral chord back
        tempChord = caster.shiftNumChord(tempChord, shift)
        #convert casted numeral chord to casted letter chord
        tempChord = caster.numChordToLetterChord(tempChord)
        #print('Converted '+origChord+' to '+tempChord)
        #with open('conversions.txt','a') as myFile:
        #    myFile.write('Converted '+origChord+' to '+tempChord+'\n')
        castedChords.append(tempChord)
        #ADDITIONAL CODE FOR CHECKER
        if(not castedFlag):
            print fullText
            uncastedChords.append(origChord)
    #Now let's figure out whether our current line is chords or lyrics, and process it appropriately
    #this means our current line is chords
    if castedChords:
        #if our last line was also chords, just combine the two lines
        if chordflag == True:
            lyricflag = False
            big_list_of_chords[-1].extend(castedChords)
            temp_section_chords[-1].extend(castedChords)
            chordflag = True
        else:
            lyricflag = False
            chordflag = True
            big_list_of_chords.append(castedChords)
            temp_section_chords.append(castedChords)
    else:
        #if our last line was lyrics, combine the lines
        if chordflag == False and lyricflag == True:
            big_list_of_lyrics[-1] = big_list_of_lyrics[-1]+fullText
            temp_section_lyrics[-1] = temp_section_lyrics[-1]+fullText
        if chordflag == True and lyricflag == False:
            chordflag = False
            lyricflag = True
            big_list_of_lyrics.append(fullText)
            temp_section_lyrics.append(fullText)

#We've done it!  Now let's do some final post-processing, printing overall stats and putting our data in files
print len(big_list_of_chords)
print len(big_list_of_lyrics)
print len(temp_text)


pickle.dump( big_list_of_lyrics, open( "lyrics_uku.p", "wb" ) )
pickle.dump( big_list_of_chords, open( "chords_uku.p", "wb" ) )
for i in xrange(len(big_list_of_chords)):
    big_list_of_chords[i] = ' '.join(big_list_of_chords[i])+'\n'
for i in xrange(len(big_list_of_lyrics)):
    big_list_of_lyrics[i] = ' '.join(nltk.word_tokenize(big_list_of_lyrics[i].replace('\n',' ').lower()))+'\n'

k = 0
kk = 0
list_of_nulls = []
for i in xrange(len(big_list_of_lyrics)):
    if big_list_of_lyrics[i].isspace():
        list_of_nulls.append(i)
for i in xrange(len(list_of_nulls)):
    del big_list_of_chords[list_of_nulls[i]-i]
    del big_list_of_lyrics[list_of_nulls[i]-i]

with open('lyrics_uku.txt','wb') as myFile:
    myFile.writelines(big_list_of_lyrics)
with open('chords_uku.txt','wb') as myFile:
    myFile.writelines(big_list_of_chords)
print h
print keep
print keep2
print k
print kk

#ADDITIONAL CODE FOR CHECKER:
if(len(uncastedChords)==0):
    print("Every chord was found in the casting table! Good job.")
else:
    print ("Chords which were not found in the casting table: ")
    for chord in uncastedChords:
        print (chord)
print len(uncastedChords) 
