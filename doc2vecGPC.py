# -*- coding: utf-8 -*-
"""
Created on Tue May  5 08:48:17 2020

@author: ElayanithF
"""

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import io
import os
import pandas as pd
import numpy as np
import gensim
from nltk.corpus import stopwords
import nltk
import matplotlib as plt
nltk.download('stopwords')
stop_german=set(stopwords.words('german'))
stop_english=set(stopwords.words('english'))


brand = pd.read_csv('brandList.csv')
data = pd.read_csv('ti_description_final.csv')

brands = list(brand["BRAND_NAME"])

GPC_FAMILY_NAME = list(data["GPC_FAMILY_NAME"])


indexNames = data[data['GPC_BRICK_NAME']=='Temporary Classification'].index.values

print(indexNames)
descriptions = list(data['TI_DESCRIPTION'])

brandToken = []
for i, brnd in enumerate(brands):
    brandToken.append(str(brnd).lower())
    
    
# Delete these row indexes from dataFrame
data.drop(indexNames, inplace=True)
brandToken[0]
data.to_csv('ti_description_final_clean.csv')
train_data = data.loc[:, ['DESCRIPTION_SHORT','GPC_BRICK_NAME']]
train_data

def read_corpus(tokens_only=False):
    #for i, row in data.iterrows():
    for i, row in enumerate(descriptions):
        #sentence = str(row[0])+' '+str(row[1])+' '+str(row[2])+' '+str(row[3])+' '+str(row[4])
        sentence = str(row)
        tokens = list(gensim.utils.tokenize(sentence.lower()))
        tokens = [i for i in tokens if(len(i) > 1)]
        tokens = [s for s in tokens if s not in stop_german]
        tokens = [s for s in tokens if s not in stop_english]
        
        #Remove brand
        #tokens = [s for s in tokens if s not in brandToken]
        #tags = list(gensim.utils.tokenize(str(row[1])))
        #tags = [i for i in tags if(len(i) > 1)]
        if tokens_only:
            yield tokens
        else:
            yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

train_corpus = list(read_corpus())

print(len(GPC_FAMILY_NAME))
print(data.iloc[18064])

model = gensim.models.doc2vec.Doc2Vec(dm=1, vector_size=50,  window=2, epochs=100)

model.build_vocab(train_corpus)

model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

model.save('model/model')

print(len(model.docvecs.vectors_docs))
for i, vec in enumerate(model.docvecs.vectors_docs):
    print(i)
    if i == 100:
        break


from gensim.models.doc2vec import Doc2Vec
model = Doc2Vec.load('model/model')

out_v = io.open('model/vec.tsv', 'w', encoding='utf-8')

for i, vec in enumerate(model.docvecs.vectors_docs):
    out_v.write('\t'.join([str(x) for x in vec]) + "\n")
out_v.close()

out_m = io.open('model/meta.tsv', 'w', encoding='utf-8')
for i in range(len(model.docvecs.vectors_docs)):
    #label = str(data.iloc[i][3])
    label = str(GPC_FAMILY_NAME[i])
    out_m.write(label + "\n")
out_m.close()

data.iloc[0][4]
#list_id  = list(data["GPC_BRICK_NAME"])

print(str(data.iloc[0][1])+str(data.iloc[0][2])+str(data.iloc[0][3]))
print(data.iloc[0][2])
print(data.iloc[0][3])


len(list_id)
print(len(model.docvecs.vectors_docs))

def get_similar_products(description):
    #tokens = gensim.utils.simple_preprocess(description)
    tokens = list(gensim.utils.tokenize(description))
    tokens = [i for i in tokens if(len(i) > 1)]
    inferred_vector = model.infer_vector(tokens)
    sims = model.docvecs.most_similar(positive=[inferred_vector], topn=10)
    for index in range(10):
        result = {
                #"REGULATED_PRODUCT_NAME":data.iloc[sims[index][0]][0],
                "GPC_BRICK_NAME":data.iloc[sims[index][0]][1],
                "GPC_CLASSIFICATION_NAME":data.iloc[sims[index][0]][2],
                "GPC_FAMILY_NAME":data.iloc[sims[index][0]][3],
                "GPC_SEGMENT_NAME":data.iloc[sims[index][0]][4]
        }
        print(result)
        print('########################################################')
              
get_similar_products('Rasiergel, sensitiv.')

model.most_similar('beer')

model.most_similar('weißwein')

model.most_similar('olivenöl')

model.most_similar('schweine')

model.most_similar('chocolate')

model.most_similar('kaffee')

model.most_similar('vodka')

model.wv.most_similar(positive=["schokolade"])

tsnescatterplot(model, 'apfel', [i[0] for i in model.wv.most_similar(negative=["apfel"])])

tsnescatterplot(model, 'schokolade', [i[0] for i in model.wv.most_similar(positive=["schokolade"])])

tsnescatterplot(model, "schokolade", [t[0] for t in model.wv.most_similar(positive=["schokolade"], topn=20)][10:])

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
 
import seaborn as sns
sns.set_style("darkgrid")

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def tsnescatterplot(model, word, list_names):
    """ Plot in seaborn the results from the t-SNE dimensionality reduction algorithm of the vectors of a query word,
    its list of most similar words, and a list of words.
    """
    arrays = np.empty((0, 50), dtype='f')
    word_labels = [word]
    color_list  = ['red']

    # adds the vector of the query word
    arrays = np.append(arrays, model.wv.__getitem__([word]), axis=0)
    
    # gets list of most similar words
    close_words = model.wv.most_similar([word])
    
    # adds the vector for each of the closest words to the array
    for wrd_score in close_words:
        wrd_vector = model.wv.__getitem__([wrd_score[0]])
        word_labels.append(wrd_score[0])
        color_list.append('blue')
        arrays = np.append(arrays, wrd_vector, axis=0)
    
    # adds the vector for each of the words from list_names to the array
    for wrd in list_names:
        wrd_vector = model.wv.__getitem__([wrd])
        word_labels.append(wrd)
        color_list.append('green')
        arrays = np.append(arrays, wrd_vector, axis=0)
        
    # Reduces the dimensionality from 300 to 50 dimensions with PCA
    reduc = PCA(n_components=10).fit_transform(arrays)
    
    # Finds t-SNE coordinates for 2 dimensions
    np.set_printoptions(suppress=True)
    
    Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduc)
    
    # Sets everything up to plot
    df = pd.DataFrame({'x': [x for x in Y[:, 0]],
                       'y': [y for y in Y[:, 1]],
                       'words': word_labels,
                       'color': color_list})
    
    fig, _ = plt.subplots()
    fig.set_size_inches(9, 9)
    
    # Basic plot
    p1 = sns.regplot(data=df,
                     x="x",
                     y="y",
                     fit_reg=False,
                     marker="o",
                     scatter_kws={'s': 40,
                                  'facecolors': df['color']
                                 }
                    )
    
    # Adds annotations one by one with a loop
    for line in range(0, df.shape[0]):
         p1.text(df["x"][line],
                 df['y'][line],
                 '  ' + df["words"][line].title(),
                 horizontalalignment='left',
                 verticalalignment='bottom', size='medium',
                 color=df['color'][line],
                 weight='normal'
                ).set_size(15)

    
    plt.xlim(Y[:, 0].min()-50, Y[:, 0].max()+50)
    plt.ylim(Y[:, 1].min()-50, Y[:, 1].max()+50)
            
    plt.title('t-SNE visualization for {}'.format(word.title()))
