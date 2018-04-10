# chordword2vec
We put two other examples of scraping chords and lyrics: scrape_heartwood_guitar.py and scrape_echords.py. For this project, we use UkuTabs because it is easy to scrape, has high quality, accurate arrangements, and is easy to clean. 
First run scrape_uku.py to get the data.
This will give you non-english-songs_using_lyrics.txt and chords_and_lyrics_uku_pipes_english_only_using_lyrics.txt. We mostly want chords_and_lyrics_uku_pipes_english_only_using_lyrics.txt. Note that because we are appending to this file, you will want to delete this file every time you scrape.
You might have to install pyEnchant for the checkEnglish method.
There should be about 5,000 songs (as of April 1, 2018)
Then run analyze_uku.py to get the chords and lyrics. They will be saved in lyrics_uku.p and chords_uku.txt and chords_uku.p and chords_uku.txt 
If you want to convert these chords into their roman numerals (which blocks for key), run convert_to_rns.py
If you want to find the number of cadences (which we define as a 2-gram of chords), run count_cadences.py
Then, we can run interannotator_agreement.py to get Cohen's kappa measure using the raw_annotations_3_27.csv file.
