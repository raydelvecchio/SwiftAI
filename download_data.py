import lyricsgenius
from constants import *
import json
import sqlite3


# def download_lyrics():
#     """
#     Downloads all lyrics from all albums into JSON files from GENIUS API. Should be called only once.
#     Can also download all lyrics from every Taylor Swift song, but I opted for just the albums.
#     """
#     genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN)
#     album_data = list()
#     for album in ALBUMS:
#         album_data.append(genius.search_album(album, 'Taylor Swift'))
#     for jsn in album_data:
#         jsn.save_lyrics()


def txt_dump():
    """
    Writes all lyrics into one large text file.
    """
    filenames = [f'data_dump/Lyrics_{i.replace(" ", "")}.json' for i in ALBUMS]
    with open('data_dump/all_lyrics.txt', 'w') as txt:
        for name in filenames:
            file = open(name)
            data = json.load(file)
            for song in data['tracks']:
                song = song['song']
                lyrics = song['lyrics']
                txt.write(lyrics)
            file.close()


def sql_dump():
    """
    Writes all lyrics into one SQL database to call.
    """
    conn = sqlite3.connect('data_dump/lyrics_db.db')
    curr = conn.cursor()
    curr.execute("""DROP TABLE IF EXISTS lyrics""")
    curr.execute("""CREATE TABLE IF NOT EXISTS lyrics(verses TEXT)""")
    filenames = [f'data_dump/Lyrics_{i.replace(" ", "")}.json' for i in ALBUMS]
    for name in filenames:
        file = open(name)
        data = json.load(file)
        for song in data['tracks']:
            song = song['song']
            lyrics = song['lyrics']
            curr.execute("""INSERT INTO lyrics values (?)""", (lyrics,))
    conn.commit()


if __name__ == "__main__":
    sql_dump()
