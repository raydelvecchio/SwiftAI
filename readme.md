# TaylorNN
Neural network for generating new Taylor Swift songs!! Deployed at [swiftai.herokuapp.com](https://swiftai.herokuapp.com). NN model originally loosely based on [this article](https://www.activestate.com/blog/how-to-build-a-lyrics-generator-with-python-recurrent-neural-networks/).

# Tools used
* python3.9
* nltk
* heroku
* tensorflow-cpu
* streamlit

# Improvements:
* Remove word "lyrics" from dataset; this is just metadata
* Generalize to all artists; we can train a model to write like any artist as long as we provide their album names!