#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 14:34:52 2020

@author: greert
"""

import csv
ann1 = []
ann2 = []
ann3 = []

with open("raw_annotations_3_27.csv") as myFile:
    spamreader = csv.reader(myFile)
    for row in spamreader:
        ann1.append(int(row[0]))
        ann2.append(int(row[1]))
        ann3.append(int(row[2]))

pe = 1/9.0
po = 0
count = 0
for i in range(len(ann1)):
    if ann1[i] == ann2[i] and ann1[i] == ann3[i]:
        count = count+1
po = count/1000.0
kappa = (po-pe)/(1-pe)
print(kappa)

countw = 0
#Weight by chance
w = 24000/9.0
for i in range(len(ann1)):
    mysum = 0
    mysum += abs(ann1[i] - ann2[i])
    mysum += abs(ann2[i] - ann3[i])
    mysum += abs(ann1[i] - ann3[i])
    countw = countw+mysum
    
kappaw = 1.0 - (countw/w)
print(kappaw)



#Possibilities: both on -1 1
#both on 0 1
#both on 1 1
#one on -1, one on 1 2
#one on 0, one on 1 2
#one on 0, one on -1 2