#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 13:21:42 2019

@author: greert
"""

import requests
from bs4 import BeautifulSoup
import json

from datetime import timedelta, datetime
from time import strftime
from os import path

billboard_url = 'https://www.billboard.com/charts/%s-songs'
billboard_categories = ['country', 'pop', 'latin', 'r-b-hip-hop','rock']
today_str_fmt = '%d-%02d-%02d'
number_of_years = 20
WEEKS_PER_YEAR = 52

ranking_class_name = 'chart-list-item__rank'
title_class_name = 'chart-list-item__title-text'
artist_class_name = 'chart-list-item__artist'

if __name__ == '__main__':
    responses = {}
    ranks_per_week = {}
    today = datetime.today()
    today -= timedelta(days=(today.weekday()+2))
    print('Today\'s date: %s'%(today_str_fmt % (today.year, today.month, today.day)))

    stopping_date = today - timedelta(weeks=(number_of_years * WEEKS_PER_YEAR))
    print('Grabbing data until:', (today_str_fmt % (stopping_date.year, stopping_date.month, stopping_date.day)))
    for c in billboard_categories:
        ranks_per_week[c] = {}
        date = today
        while date > stopping_date:
            date_str = (today_str_fmt % (date.year, date.month, date.day))
            weekly_url = path.join(billboard_url%(c), date_str)
            print('Grabbing url: %s' % weekly_url)
            try:
                responses[weekly_url] = requests.get(weekly_url)
                print('Response Code', responses[weekly_url].status_code)
                soup = BeautifulSoup(responses[weekly_url].content, 'lxml')
                print('Soup Compiled')

                artist_divs = soup.find_all('div', {'class': artist_class_name})
                artist_names = [div.text.strip() for div in artist_divs]
                title_spans = soup.find_all('span', {'class': title_class_name})
                title_names = [span.text.strip() for span in title_spans]

                ranks_per_week[c][date_str] = []
                for i in range(len(artist_names)):
                    ranks_per_week[c][date_str].append((title_names[i], artist_names[i]))
            except Exception as e:
                print('Error grabbing:', weekly_url)

            date -= timedelta(weeks=1)

json.dump(ranks_per_week, open('data/ranks_per_week_'+str(number_of_years)+'_years.json', 'w'))