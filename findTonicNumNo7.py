"""
Ben Ma
Python 3.x

Contains the utility function findTonicNumNo7.
"""

import copy #for deep copy

def findTonicNumNo7(songChords, keyTable): #songChords is a list, keyTable is a list of lists
    # edit songChords to change 7ths to just major
    songChordsNo7 = copy.deepcopy(songChords)
    for i in range(0, len(songChordsNo7)):
        songChordsNo7[i] = songChordsNo7[i].replace("7", "")

    maxKey = 0 #0 thru 11 for C thru B
    maxScore = 0
    for i in range(0,len(keyTable)): #go thru each of the 12 keys--example for key of C: C Dm Em F G G7 Am Bdim
        curScore = 0
        key = keyTable[i]
        for chord in songChordsNo7:
            for j in range(0,len(key)): #go thru each note in the major scale of the key
                note = key[j]
                if chord==note:
                    if (j == 1 or j == 2 or j == 7):
                        curScore+=0.9 #tiebreaker: the ii, iii, and vii are weighted less
                    else:
                        curScore+=1 #if it's a match, add 1 to the "score" of the current key
                    break
        if curScore>maxScore:
            maxScore=curScore
            maxKey = i
    return maxKey #return key with most matches for the chords in the song