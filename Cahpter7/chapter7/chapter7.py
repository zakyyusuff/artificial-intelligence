import pandas as pd
from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold
#%%
dataset = pd.concat([pd.read_csv(r'/home/kali/tugas_prog/pak rolly/Youtube01-Psy.csv')])
                   
dataset = dataset.sample(frac=1)
#%%
kfold = StratifiedKFold(n_splits=5)
splits = kfold.split(dataset, dataset['CLASS'])

#%%
for train, test in splits:
    print("-----------SPLITS-----------")
    print(test)
#%%
def train_and_test(train_idx, test_idx):
    train_content = dataset['CONTENT'].iloc[train_idx]
    test_content = dataset['CONTENT'].iloc[test_idx]
    
    tokenizer = Tokenizer(num_words=2000)
    
    tokenizer.fit_on_texts(train_content)
    
    dataset_train_inputs = tokenizer.texts_to_matrix(train_content, mode='tfidf')
    dataset_test_inputs = tokenizer.texts_to_matrix(test_content, mode='tfidf')
    
    dataset_train_inputs = dataset_train_inputs/np.amax(np.absolute(dataset_train_inputs))
    dataset_test_inputs = dataset_test_inputs/np.amax(np.absolute(dataset_test_inputs))
    
    dataset_train_inputs = dataset_train_inputs - np.mean(dataset_train_inputs)
    dataset_test_inputs = dataset_test_inputs - np.mean(dataset_test_inputs)
    
    dataset_train_outputs = np_utils.to_categorical(dataset['CLASS'].iloc[train_idx])
    dataset_test_outputs = np_utils.to_categorical(dataset['CLASS'].iloc[test_idx])
    
    model = Sequential()
    model.add(Dense(512, input_shape=(2000, )))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    
    model.compile(loss="categorical_crossentropy", optimizer='adamax', metrics=['accuracy'])
    model.fit(dataset_train_inputs, dataset_train_outputs, epochs=10, batch_size=16)
    scores = model.evaluate(dataset_test_inputs, dataset_test_outputs)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    return scores

#%%
kfold = StratifiedKFold(n_splits=5)
splits = kfold.split(dataset, dataset['CLASS'])
cvscores = []
for train_idx, test_idx, in splits:
    scores = train_and_test(train_idx, test_idx)
    cvscores.append(scores[1]*100)
#%%
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))