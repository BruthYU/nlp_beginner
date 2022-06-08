import numpy as np
import pandas as pd
import os
import datatable as dt
import keras.preprocessing.sequence as sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence

class SimpleTokenizer:
    def __init__(self,filename="../data/train.tsv"):
        self.tokenizer = Tokenizer(lower=True,num_words=15220,oov_token='<oov>',char_level=False,filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ ')
        train = dt.fread(filename)
        train = train.to_pandas()["Phrase"]
        self.tokenizer.fit_on_texts(train)
        self.num_words = self.tokenizer.num_words
    def encode(self,text):
        res = self.tokenizer.texts_to_sequences([text])
        return res


        #res=self.tokenizer.texts_to_sequences(text)
        #return sequence.pad_sequences(res,maxlen=300)

if __name__ =='__main__':
    tokenizer = SimpleTokenizer()
    print(tokenizer.tokenizer.texts_to_sequences([]))



