#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 15:37:27 2018

@author: greert
"""

import csv
tim = []
ben = []
#different parms if karan is in
#karan = []

with open("raw_annotations_3_27.csv") as myFile:
    spamreader = csv.reader(myFile)
    for row in spamreader:
        tim.append(int(row[0]))
        ben.append(int(row[1]))
        #karan.append(int(row[2]))

pe = 1/3.0
po = 0
count = 0
for i in xrange(len(tim)):
    if tim[i] == ben[i]:
        count = count+1
po = count/1000.0
kappa = (po-pe)/(1-pe)
print(kappa)
#.538 for tim and ben

countw = 0
#Weight by chance
w = 8000/9.0
for i in xrange(len(tim)):
    if abs(tim[i] - ben[i]) == 2:
        countw = countw+2
    if abs(tim[i] - ben[i]) == 1:
        countw = countw+1
    else:
        pass
kappaw = 1.0 - (countw/w)
print(kappaw)
#.6445 for tim and ben


#Possibilities for tim and ben: both on -1 1
#both on 0 1
#both on 1 1
#one on -1, one on 1 2
#one on 0, one on 1 2
#one on 0, one on -1 2