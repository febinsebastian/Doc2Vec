# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 16:53:58 2020

@author: ElayanithF
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from collections import  Counter
plt.style.use('ggplot')
stop=set(stopwords.words('german'))
import re
import os
import random
from nltk.tokenize import word_tokenize
import gensim
import string
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense,SpatialDropout1D
from keras.initializers import Constant
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

data = pd.read_csv('datasetfull.csv')
pid = list(data['asin'])

train_images_path = 'img/'

CATEGORIES = ['Electronics', 'Sports & Outdoors',
       'Cell Phones & Accessories', 'Automotive',
       'Tools & Home Improvement', 'Health & Personal Care', 'Beauty',
        'Grocery & Gourmet Food', 'Office Products',
       'Arts, Crafts & Sewing', 'Pet Supplies', 'Patio, Lawn & Garden',
       'Clothing, Shoes & Jewelry', 'Baby',
       'Musical Instruments', 'Industrial & Scientific', 'Baby Products',
       'Appliances', 'All Beauty', 'All Electronics']

NUM_CLASSES = len(CATEGORIES)

from nltk.corpus import stopwords
stop_english=set(stopwords.words('english'))

def get_token(description):
        tokens = set(gensim.utils.tokenize(description))
        tokens = [i for i in tokens if(len(i) > 2)]
        tokens = [s for s in tokens if s not in stop_english]
        return tokens
    
    
corpus = []
labels = []
files = os.listdir(train_images_path)
random.shuffle(files)
for img in files:
    img_id = img.split('.')[0]

    index = data[data['asin']==img_id].index.values[0]
    row = data.iloc[index , :]
    family = row[3]
    if family in CATEGORIES:
        description = row[2]
        token_list = get_token(str(description))
        corpus.append(token_list)
        class_num = CATEGORIES.index(family)
        labels.append(class_num)


labels = pd.get_dummies(labels).values


print(corpus[32])

from gensim.models.wrappers import FastText

model_german = FastText.load_fasttext_format('cc.en.300.bin')

MAX_LEN=15
tokenizer_obj=Tokenizer()
tokenizer_obj.fit_on_texts(corpus)
sequences=tokenizer_obj.texts_to_sequences(corpus)

tweet_pad=pad_sequences(sequences,maxlen=MAX_LEN,truncating='post',padding='post')

word_index=tokenizer_obj.word_index

num_words=len(word_index)+1
embedding_matrix=np.zeros((num_words,300))

known_words = ["" for x in range(num_words)]
unknown_words = []
for word,i in tqdm(word_index.items()):
    if i > num_words:
        continue
    
    if word in model_german:
        known_words[i] = word
        embedding_matrix[i] = model_german.wv[word]
    else:
        unknown_words.append(word)

len(unknown_words)
model_german.wv['Holz']
for i in unknown_words:
    print(i)

print(model_german.most_similar('hähnchenschnitte'))

model=Sequential()

embedding=Embedding(num_words,300,embeddings_initializer=Constant(embedding_matrix),
                   input_length=MAX_LEN,trainable=False)

model.add(embedding)
model.add(SpatialDropout1D(0.2))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(NUM_CLASSES, activation='sigmoid', name='out_dense') )


optimzer=Adam(learning_rate=1e-5)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
df.shape[0]

train=tweet_pad[:data.shape[0]]
test=tweet_pad[data.shape[0]:]
    
X_train,X_test,y_train,y_test=train_test_split(train,labels,test_size=0.25)

import tensorflow as tf

# Place tensors on the CPU
with tf.device('/CPU:0'):
    history=model.fit(X_train,y_train,batch_size=64,epochs=50,validation_data=(X_test, y_test),verbose=True)
    
y_pred = model.predict(X_test)

import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
 
plt.show()

import io
out_v = io.open('FastText/vec50k.tsv', 'w', encoding='utf-8')

for i, vec in enumerate(embedding_matrix[:50000]):
    out_v.write('\t'.join([str(x) for x in vec]) + "\n")
out_v.close()

out_m = io.open('FastText/meta50k.tsv', 'w', encoding='utf-8')
for i,vec in enumerate(embedding_matrix[:50000]):
    if str(known_words[i]) == '':
        label = '0'
    else:
        label = str(known_words[i])
    out_m.write(label + "\n")
out_m.close()

classes = np.argmax(y_pred, axis = 1)
y_test = np.argmax(y_test, axis = 1)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,classes)
print(cm)

e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)

print(model_german.similarity("Jubiläumflasche", "Jubilaumflasche"))

print(model_german.similarity("Jubiläumflasche", "Jubilaumflasche"))

#Image Model

from tensorflow.keras.layers import Input
from keras.models import Sequential
from keras.layers import Activation, Convolution2D, Dropout, Conv2D, Dense,Concatenate
import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet

IMG_SIZE = (80,80)
INPUT_SHAPE = IMG_SIZE + (1,)

input_tensor = Input(shape=INPUT_SHAPE)
model = MobileNet(input_tensor=input_tensor, alpha=1.0,
                    include_top=False, weights=None)
#output = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', input_shape=(1,1,2))(model.output)
output = tf.keras.layers.Reshape((11520,))(model.output)
output = tf.keras.layers.Dense(2, activation='softmax')(output)