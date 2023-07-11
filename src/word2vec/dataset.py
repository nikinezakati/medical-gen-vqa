import json
import random
import numpy as np

class Dataset:
    def __init__(self, json_path, tablesize=10000000):
        with open(json_path) as f:
            self.__data_dict = json.load(f)
        
        self.__data_list = list(self.__data_dict.values())
        self.__ids = self.__data_dict.keys()
        self.__tokens_setting()
        self.tablesize = tablesize
        
    def __tokens_setting(self):
        tokens = {}
        tokenfreq = {}
        revtokens = []
        idx = 0
        for tweet in self.__data_list:
            for token in tweet:
                if token not in tokens:
                    tokens[token] = idx
                    idx += 1
                    revtokens += token
                    tokenfreq[token] = 1
                else:
                    tokenfreq[token] += 1
        tokens['<UNK>'] = idx
        revtokens += ['<UNK>']
        tokenfreq['<UNK>'] = 1
        self.tokens = tokens
        self._revtokens = revtokens
        self._tokenfreq = tokenfreq
    
    def get_by_id(self, id):
        return self.__data[id]
    
    def __getitem__(self, start, stop, step):
        index = start
        if stop == None:
            end = start + 1
        else:
            end = stop
        if step == None:
            stride = 1
        else:
            stride = step
        return self.__data_list[index:end:stride]
    
    def getRandomContext(self, C=5):
        data = self.__data_list
        dataIdx = random.randint(0, len(data) - 1)
        tweet = data[dataIdx]
        wordID = random.randint(0, len(tweet) - 1)

        context = tweet[max(0, wordID - C):wordID]
        if wordID+1 < len(tweet):
            context += tweet[wordID+1:min(len(tweet), wordID + C + 1)]

        centerword = tweet[wordID]
        context = [w for w in context if w != centerword]

        if len(context) > 0:
            return centerword, context
        else:
            return self.getRandomContext(C)
    
    def sampleTable(self):
        if hasattr(self, '_sampleTable') and self._sampleTable is not None:
            return self._sampleTable

        nTokens = len(self.tokens)
        samplingFreq = np.zeros((nTokens,))
        i = 0
        for w in range(nTokens):
            w = self._revtokens[i]
            if w in self._tokenfreq:
                freq = 1.0 * self._tokenfreq[w]
                freq = freq ** 0.75
            else:
                freq = 0.0
            samplingFreq[i] = freq
            i += 1

        samplingFreq /= np.sum(samplingFreq)
        samplingFreq = np.cumsum(samplingFreq) * self.tablesize

        self._sampleTable = [0] * self.tablesize

        j = 0
        for i in range(self.tablesize):
            while i > samplingFreq[j]:
                j += 1
            self._sampleTable[i] = j
        return self._sampleTable
    
    def sampleTokenIdx(self):
        return self.sampleTable()[random.randint(0, self.tablesize - 1)]