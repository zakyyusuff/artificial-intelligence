# -- coding: utf-8 --
"""
Created on Sun Apr 17 13:52:21 2022

@author: Dellavianti
"""

#%% Import modul
import os
import gensim, logging
import re
import random
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
#%% NOMOR 1
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#SUMBER https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300
eni = gensim.models.KeyedVectors.load_word2vec_format(r"C:\Users\Dellavianti\chapter 5\GoogleNews-vectors-negative300.bin", binary=True, limit=500000)

#%% Percobaan 1
print(eni['love'])

#%%  Percobaan 2
print(eni['faith'])

#%% Percobaan 3
print(eni['fall'])

#%% Percobaan 4
print(eni['sick'])

#%% Percobaan 5
print(eni['clear'])

#%% Percobaan 6
print(eni['shine'])

#%% Percobaan 7
print(eni['bag'])

#%% Percobaan 8
print(eni['car'])

#%% Percobaan 9
print(eni['wash'])

#%% Percobaan 10
print(eni['motor'])

#%% Percobaan 11
print(eni['cycle'])

#%% Percobaan perbandingan 1
print(eni.similarity('wash', 'clear'))

#%% Percobaan perbandingan 2
print(eni.similarity('bag', 'love'))

#%% Percobaan perbandingan 3
print(eni.similarity('motor', 'car'))

#%% Percobaan perbandingan 4
print(eni.similarity('sick', 'faith'))

#%% Percobaan perbandingan 5
print(eni.similarity('cycle', 'shine'))


#%% NOMOR 2
input_string = "Dellavianti, 1194070, D4, Teknik Informatika, Politeknik Pos Indonesia"
print("Biodata: " + input_string)
re_string = re.findall(r'\w+', input_string)
print("List katanya adalah: " + str(re_string))

#%% Percobaan 1
input_matrix = [['Della', 'vianti'], ['1194070', 'D4'], ['Teknik', 'Informatika'], ['Politeknik Pos', 'Indonesia']]
result = ""
for s in input_matrix:
    result += random.choice(s) + " "
    print(result)

#%% Percobaan 2 - 
def extract_words(sent):
    sent = sent.lower()
    sent = re.sub(r'<[^>]+>', ' ', sent) # Mengahpus tag HTML
    sent = re.sub(r'(\W)\'(\W)', ' ', sent) # Menghapus petik satu
    sent = re.sub(r'\W', ' ', sent) # Mengahpus tanda baca
    sent = re.sub(r'\s+', ' ', sent) # Menghapus spasi yang berurutan
    return sent.split()

#%% NOMOR 3
#%% Percobaan 1
# Contoh Dokumen
doc_ku = ['eni suka berenang', 
          'eni suka berlari', 
          'eni suka makan', 
          'eni suka mendengarkan musik', 
          'eni suka tidur']
token_doc = ['sering']
print(doc_ku, token_doc)

#%% Percobaan 2
tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(token_doc)]
print(tagged_data)

# Mengtrain Doc2Vec model
model = Doc2Vec(tagged_data, vector_size=20, window=2, min_count=1, workers=4, epochs=100)

# Menyimpan trained Doc2Vec model
model.save('test_doc2vec.model')

# Meload Doc2Vec model
model = Doc2Vec.load('test_doc2vec.model')

# Menampilkan model
print(model.wv)

#%% NOMOR 4 - SUMBER https://ai.stanford.edu/~amaas/data/sentiment/
unsup_sentences = []

for dirname in ['train/pos', 'train/neg', 'train/unsup', 'test/pos', 'test/neg']:
    for fname in sorted(os.listdir('aclImdb/' + dirname)):
        if fname[-4:] == '.txt':
            with open('aclImdb/' + dirname + '/' + fname, encoding='UTF-8') as f:
                sent = f.read()
                words = (sent)
                print(unsup_sentences.append(TaggedDocument(words, [dirname + '/' + fname])))

#%% NOMOR 5
mute = (unsup_sentences) # Mengacak data
print(mute)
# model.delete_temporary_training_data(keep_inference=True) # Membersihkan data

#%% NOMOR 6
model.save('eni.d2v') # Menyimpan data

# model.delete_temporary_training_data(keep_inference=True) # Membersihkan temporary data

#%% NOMOR 7
print(model.infer_vector(extract_words('Dellavianti')))

#%% NOMOR 8
cs = cosine_similarity(
    [model.infer_vector(extract_words('Dellavianti'))], 
    [model.infer_vector(extract_words('Karena sekarang saat nya matkul Kecerdasan Buatan'))])
print(cs)

#%% NOMOR 9
eni = datasets.load_diabetes()
X = eni.data[:150]
y = eni.target[:150]
lasso = linear_model.Lasso()
print(cross_val_score(lasso, X, y, cv=3))


#%%