import random
from keras.layers import Dense, Activation, LSTM, Bidirectional, Embedding
from keras.models import Sequential, load_model
import numpy as np
from process_data import preprocess
from constants import DATA_LOCATION, SAVE_LOCATION, TEMPERATURES


class SwiftAI:
    def __init__(self, data_file, pred_len=5):
        """
        INIT FUNCTION. Define all hyperparameters here, including prediction length!
        """
        # preprocesses data in the init function since we need all of this even to generate new songs.
        # because of this, every time we change the preprocess function (different cutoffs, pre_len, etc) we'll
        # have to re-train and save the model to match.
        self.train_I, self.test_I, self.train_L, self.test_L, self.dictionary = preprocess(data_file, pred_len)
        self.total_sequences = len(self.train_I) + len(self.test_I)
        self.num_unique_words = len(self.dictionary)
        self.pred_length = pred_len
        self.batch_sz = 64
        self.embedding_out_sz = 128
        self.lstm_hidden_dim = 256
        self.epochs = 20
        self.model = self.create_model()

    def create_model(self):
        """
        Builds our model that we'll use to generate new songs. All layers and compilation defined here.
        """
        nn = Sequential()
        nn.add(Embedding(input_dim=self.num_unique_words, output_dim=self.embedding_out_sz))
        nn.add(Bidirectional(LSTM(self.lstm_hidden_dim)))
        nn.add(Dense(self.num_unique_words))
        nn.add(Activation('softmax'))
        nn.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return nn

    def train(self):
        """
        Trains our Taylor Swift model on the dataset!
        """
        self.model.fit(self.generator_func(self.train_I, self.train_L),
                       steps_per_epoch=self.total_sequences / self.batch_sz + 1,
                       epochs=self.epochs, validation_data=self.generator_func(self.test_I, self.train_L),
                       validation_steps=int(len(self.train_L) / self.batch_sz) + 1)

    def generator_func(self, X, Y):
        place = 0
        while True:
            x = np.zeros((self.batch_sz, self.pred_length), dtype=np.int32)
            y = np.zeros(self.batch_sz, dtype=np.int32)
            for i in range(self.batch_sz):
                for j, word in enumerate(X[place % len(X)]):
                    x[i, j] = self.dictionary[word]
                y[i] = self.dictionary[Y[place % len(X)]]
                place += 1
            yield x, y

    def save_model(self, location):
        """
        Saves our model locally to avoid training every time we need to call it. Ideally save only after training.
        """
        print("Saving trained model...")
        self.model.save(location)

    @staticmethod
    def load_saved_model(location):
        """
        Loads the saved Keras model from the input location, then returns it for use in generating new songs.
        """
        return load_model(location)

    def sample_word(self, preds, temperature=1.0):
        """
        From a predicted distribution output by a model for a given seed phrase, this method will find the most
        likely next word (via argmax) and return it.
        :param preds: predictions from our model (outputs) for a given seed phrase
        :param temperature: in machine learning, temperature is a number applied to softmax which represents the
        confidence of the model. By applying low temperatures, we're telling softmax to behave confidently, and err
        on the side of caution. In effect, this results in more repeat words being placed, because our weights
        more confidently correlate them with one another. However, with high temperature, we tell softmax to behave
        less confidently. Higher temperatures flatten the "spiky" distribution across words depending on how high it is.
        This is good for us, because it will give a more equal distribution over all possible words and
        result in more randomness, which would make better songs. Simply check the outputs of write_songs to see
        this principle in action (lower temperature songs are more boring, higher temperature songs are better).
        """
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probs = np.random.multinomial(1, preds, 1)
        sample_idx = np.argmax(probs)
        rev_dict = {v: k for k, v in self.dictionary.items()}
        return rev_dict[sample_idx]

    def generate_songs(self, model, preset_seed, song_length=176, filename='outputs.txt'):
        """
        Writes new songs by "Taylor Swift"! Passes all inputs through the model until a song of the specified length
        is written. After the first seed, the input to the model becomes the seeds last four words plus the previous
        output. This ensures a cohesive song is written sequentially based on previous words.
        :param song_length: the length of the song duh
        :param model: a trained model must be passed into this method; this model will be called to predict next words
        :param filename: output filename to write our songs to locally for viewing.
        :param preset_seed: seed input by the user to generate songs for. This the raw text, and words may or may
        not be included in the corpus. This will be the very first thing displayed for aesthetic purposes.
        :return: returns all songs generated by the method in list form
        """
        seed = self.mask_seed(preset_seed)  # generates MODEL SEED by masking out all of the words not in the corpus
        print(f'Writing with masked seed: {seed}')
        save_seed = seed
        song_list = list()
        with open(filename, 'w') as txt:
            txt.write(f'[META TEXT] Generating songs with seed {preset_seed}\n\n')
            for temp in TEMPERATURES:
                seed = save_seed
                song = ""
                for word in preset_seed.split(' '):
                    song += word + " "
                txt.write(f'[META TEXT] Song with diversity {temp}:\n')
                for i in range(song_length):
                    inp = np.zeros((1, self.pred_length))
                    for i, word in enumerate(seed):
                        inp[0, i] = self.dictionary[word]
                    preds = model.predict(inp, verbose=0)[0]
                    next_word = self.sample_word(preds, temp)
                    song += next_word + " "
                    song = self.post_process_song(song)
                    seed = seed[1:] + [next_word]
                song_list.append(song)
                txt.write(song)
                txt.write('\n\n')
            txt.write('=' * 100 + '\n\n')
        return song_list

    @staticmethod
    def post_process_song(song):
        """
        Post processes any given peice of text to the desired format. In this case, we feed it our generated songs
        to remove spaces and such.
        """
        song = song.replace('\n\n\n', '\n\n')
        song = song.replace('\n\n', '\n')
        song = song.replace('\n ', '\n')
        return song

    def mask_seed(self, seed):
        """
        Returns a valid seed compatible with our model containing only words that are found in the corpus. This is the
        "behind the scenes" seed: where the preset seed will be displayed to the user, this seed, composed by replacing
        all invalid words with random valid words, will be used to generate new words in a song.
        """
        seed = seed.split(' ')
        valid_words = self.dictionary.keys()
        for i, word in enumerate(seed):
            if word not in valid_words:
                seed[i] = random.choice(list(valid_words))  # replaces invalid words with random valid word
        return seed


if __name__ == "__main__":
    swift = SwiftAI(DATA_LOCATION, pred_len=5)
    swift.train()
    swift.save_model(SAVE_LOCATION)
