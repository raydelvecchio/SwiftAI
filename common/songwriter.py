from common.swiftai import SwiftAI
from common.constants import DATA_LOCATION, SAVE_LOCATION


def write_songs(swift_ai, preset_seed, song_length):
    """
    Loads the trained model that we have saved and then writes 9 new songs (1 for each temperature). Returns
    a list of all songs created to cycle through them.
    :param song_length: length of the song duh
    :param seed: seed phrase to write songs with
    :param swift_ai: SwiftAI object
    """
    model = swift_ai.load_saved_model(SAVE_LOCATION)
    songs = swift_ai.generate_songs(model, preset_seed, song_length=song_length)
    return songs


if __name__ == "__main__":
    taylor_swift = SwiftAI(DATA_LOCATION)
    write_songs(taylor_swift, 'i knew you were trouble')
