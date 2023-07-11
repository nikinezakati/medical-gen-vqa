

import random
import numpy as np
import matplotlib

from src.word2vec import dataset
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
from .dataset import Dataset
from .word2vec import *
from .sgd import *
import os
import sys
assert sys.version_info[0] == 3
assert sys.version_info[1] >= 5
random.seed(314)
from src.constants import WORD2VEC_SUFFIX

class Word2VecModel:
    def __init__(self, C, dimVectors, data_url, class_name):
        self.dimVectors = dimVectors
        # Context size
        self.C = C
        self.dataset = Dataset(data_url)
        self.class_name = class_name
        self.wordVectors = []
        self.data_ready = False
    def train(self, iters):
        tokens = self.dataset.tokens
        nWords = len(tokens)
        self.wordVectors = np.concatenate(
            ((np.random.rand(nWords, self.dimVectors) - 0.5) /
            self.dimVectors, np.zeros((nWords, self.dimVectors))),
            axis=0)
        self.wordVectors = sgd(
        lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, self.dataset, self.C,
            negSamplingLossAndGradient),
        self.wordVectors, 0.3, iters, self.class_name , None, False, PRINT_EVERY=10)
        self.data_ready = True
    def get_word_vector(self, word):
        tokens = self.dataset.tokens
        idx = tokens[word]
        if not self.data_ready:
            return None
        return self.wordVectors[idx]
    def load_saved_params(self, base_dir):
        params_file = os.path.join(base_dir, f"{self.class_name}.{WORD2VEC_SUFFIX}.npy")
        state_file = os.path.join(base_dir, f"state.{self.class_name}.{WORD2VEC_SUFFIX}.pickle")
        params = np.load(params_file)
        with open(state_file, "rb") as f:
            state = pickle.load(f)
        self.wordVectors = params
        self.dimVectors = self.wordVectors.shape[1]
        random.setstate(state)
        self.data_ready = True
    def save(self):
        save_params(self.class_name, self.wordVectors)