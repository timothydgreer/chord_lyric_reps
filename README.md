# Learning Shared Vector Representations of Lyrics and Chords in Music
For this project, we use UkuTabs because it is easy to scrape, has high quality, accurate arrangements, and is easy to clean. 
First run scrape_uku_2.py to get the data. If you want the music tags, you can run scrape_uku_tagger.py. This gets information such as artist bio, genre(s), and band origin.
The scraping script will give you non-english-songs_using_lyrics.txt and chords_and_lyrics_uku_pipes_english_only_using_lyrics.txt. We mostly want chords_and_lyrics_uku_pipes_english_only_using_lyrics.txt.
You will have to install pyEnchant (https://github.com/rfk/pyenchant) if you want to use the checkEnglish method.
There should be about 5,500 songs (as of July 1, 2019)
Then run analyze_uku.py to get the chords and lyrics files. They will be saved in lyrics_uku_separated.p and chords_uku_separated.txt and chords_uku_separated.p and chords_uku_separated.txt. convert_to_rns.py and collapse_song.py are also used here, which convert the chords to a predicted key and remove the song ending markers, respectively. These modules are pretty straightforward.
After that, you can use a simple package like multivec (https://github.com/eske/multivec) to take in these files: lyrics_uku_unseparated and rns_uku_unseparated.txt. These will give you embeddings.
Lastly, you can run a genre classification task using find_embeddings.py. This uses the outputs from scrape_billboard.py and pull_in_jsons.py to scrape the billboard charts and determine which songs are in the Ukutabs dataset. Files are supplied to you in this github repo as well, so it should run out of the box.



# Other scripts:
If you want to find the number of cadences (which we define as a 2-gram of chords), run count_cadences.py
You can run interannotator_agreement.py to get Cohen's kappa measure using the raw_annotations_3_27.csv file.
