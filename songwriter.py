from model import SwiftAI
from constants import DATA_LOCATION, SAVE_LOCATION


def write_songs(length=176):
    """
    Loads the trained model that we have saved and then writes 9 new songs (1 for each temperature). Returns
    a list of all songs created to cycle through them.
    """
    taylor_swift = SwiftAI(DATA_LOCATION)
    model = taylor_swift.load_saved_model(SAVE_LOCATION)
    songs = taylor_swift.write_songs(model, song_length=length)
    return songs


if __name__ == "__main__":
    write_songs()
