from keras.layers import Dense, Activation, LSTM, Bidirectional, Embedding
from keras.models import Sequential, load_model
import numpy as np
from process_data import preprocess
from constants import DATA_LOCATION, SAVE_LOCATION


class SwiftAI:
    def __init__(self, data_file, pred_len=5):
        # preprocesses data in the init function since we need all of this even to generate new songs.
        # because of this, every time we change the preprocess function (different cutoffs, pre_len, etc) we'll
        # have to re-train and save the model to match.
        self.train_I, self.test_I, self.train_L, self.test_L, self.dictionary = preprocess(data_file, pred_len)
        self.total_sequences = len(self.train_I) + len(self.test_I)
        self.num_unique_words = len(self.dictionary)
        self.pred_length = pred_len
        self.batch_sz = 64
        self.embedding_out_sz = 256
        self.lstm_hidden_dim = 512
        self.epochs = 15
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
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probs = np.random.multinomial(1, preds, 1)
        sample_idx = np.argmax(probs)
        rev_dict = {v: k for k, v in self.dictionary.items()}
        return rev_dict[sample_idx]

    def write_songs(self, model, song_length=176, filename='songs.txt', preset_seed=None):
        if preset_seed is not None:
            seed = preset_seed.split(' ')
        else:
            seed = self.make_custom_seed()
        save_seed = seed
        song_list = list()
        print("\nWriting songs...")
        with open(filename, 'w') as txt:
            txt.write(f'[META TEXT] Generating songs with seed {save_seed}\n\n')
            for temp in [i / 10 for i in range(1, 10)]:
                seed = save_seed
                song = ""
                txt.write(f'[META TEXT] Song with diversity {temp}:\n')
                for word in seed:
                    song += word + " "
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
        return song

    def make_random_seed(self):
        """
        Finds random seed in training inputs to start generating a song with.
        """
        seed_idx = np.random.randint(len(self.train_I))
        seed = self.train_I[seed_idx]
        return seed

    def make_custom_seed(self):
        """
        Takes input from the user to start a seed phrase. Must include words found in the corpus, and the seed phrase
        must be equal to the prediction length we used to train the model.
        """
        all_words = self.dictionary.keys()
        invalid = True
        while invalid:
            bad_word = False
            seed = input(f'Enter {self.pred_length}-word seed phrase: ').lower()
            seed = seed.split(' ')
            count = 0
            for word in seed:
                if word in all_words:
                    count += 1
                else:
                    print(f'Word \"{word}\" invalid.')
                    bad_word = True
            if count == self.pred_length:
                return seed
            else:
                if not bad_word:
                    print(f'Please input exactly {self.pred_length} words separated by spaces.')


if __name__ == "__main__":
    swift = SwiftAI(DATA_LOCATION, pred_len=5)
    swift.train()
    swift.save_model(SAVE_LOCATION)
