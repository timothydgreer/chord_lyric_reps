#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 16:28:44 2019

@author: greert
"""

# Collapse a song with SONG OVER markers

def collapse(input_file,output_file):
    #Example input and output files:
    #input_file = "rns_uku_all_20_separated.txt"
    #output_file = "rns_uku_all_20_unseparated.txt"
    
    with open(input_file, 'r') as f:
        list_of_inputs = f.readlines()
        for i in range(len(list_of_inputs)):
            #Remove the song ending markers
            list_of_inputs[i] = list_of_inputs[i].replace('song over','')
            list_of_inputs[i] = list_of_inputs[i].replace('SONG OVER','')
    
    #Print to file
    with open(output_file, 'w') as out:
        for i in range(len(list_of_inputs)):
            if list_of_inputs[i].isspace():
                continue
            else:
                out.write(list_of_inputs[i])
