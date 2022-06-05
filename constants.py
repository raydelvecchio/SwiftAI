import os

# GENIUS_CLIENT_ID = os.environ['GENIUS_CLIENT_ID']
#
# GENIUS_CLIENT_SECRET = os.environ['GENIUS_CLIENT_SECRET']
#
# GENIUS_ACCESS_TOKEN = os.environ['GENIUS_ACCESS_TOKEN']

ALBUMS = ['Taylor Swift', 'Fearless', 'Speak Now', 'Red', '1989', 'Reputation', 'Lover', 'Folklore', 'Evermore']

DATA_LOCATION = 'data_dump/all_lyrics.txt'

SAVE_LOCATION = 'saved_model'

TEMPERATURES = [i / 10 for i in range(5, 10)]

VALIDATION_SZ = 0.03
