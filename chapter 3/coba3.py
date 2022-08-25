# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 19:52:09 2022

@author: Dellavianti
"""

#%%
# SOAL NOMOR 1
import pandas as pd # Melakukan import library pandas menjadi pd

hobi = {"Nama Buah" : ['Apel', 'Durian', 'Stroberi', 'Mangga']} # membuat variabel yang bernama hobi
x = pd.DataFrame(hobi) # membuat variable x yang akan membuat dataframe dari library pandas yaitu hobi
print ('Dellavianti suka makan Buah : ' + x) # print hasil dari x

#%%
#SOAL NOMOR 2 
import numpy as np # melakukan import library numpy menjadi nama lain np

matrix_x = np.eye(15) # membuat sebuah matrix numpy dg menggunakan fungsi eye
matrix_x # mendeklarasikan matrix_x yang telah dibuat

print (matrix_x) # menampilkan matrix_x yang telah dibuat dengan berbentuk 15x15

#%%
#SOAL NOMOR 3
import matplotlib.pyplot as plt # melakukan import library numpy menjadi nama lain np
plt.plot([1,6,4,5,0,3,6]) # masukan nilai pada plot
plt.xlabel('Dellavianti') # menambhkan label x
plt.ylabel('1194070') # menambhkan label pada y
plt.show() # menampilkan grafik plot

#%%
#SOAL NOMOR 4
import pandas as pd # melakukan import pandas dg nama pd

# some lines have too many fields (?), so skip bad lines
imgatt = pd.read_csv(r"C:\Users\Dellavianti\chapter 3\image_attribute_labels.txt",
                     sep='\s+', header=None, error_bad_lines=False, warn_bad_lines=False,
                     usecols=[0,1,2], names=['imgid', 'attid', 'present']) # membuat vaiable imgatt untuk membaca csv dari dataset

#%%
imgatt.head() # menampilkan data yang telah dibaca

#%%

imgatt.shape # Menampilkan jumlah seluruh data

#%%

imgatt2 = imgatt.pivot(index='imgid', columns='attid', values='present') # MEMBUAT SEBUAH VARIABLE BARU DARI FUNGSI IMGATT DENGAN MENGGANTI INDEK MENJADI KOLOM DAN KOLOM MENJADI INDEX

#%%

imgatt2.head() # menampilkan data yang sudah dibaca 5 teratas

#%%

imgatt2.shape # menampilkan jumlah seluruh data

#%%
imglabels = pd.read_csv(r"C:\Users\Dellavianti\chapter 3\image_class_labels.txt", 
                        sep=' ', header=None, names=['imgid', 'label']) # baca data csv dg ketentuan yg ada

imglabels = imglabels.set_index('imgid') # variable imglabels di set index menjadi imgid

#%%

imglabels.head() # membaca data yang dimasukkan ke imglabels

#%%


imglabels.shape # menampilkan seluruh jumlah data dan kolomnya


#%%

df = imgatt2.join(imglabels) # variable df dimasukkan fungsi join dari data imgatt2 ke variable imglables
df = df.sample(frac=1) # variable df sebagai sample dengan ketentuan frac=1

#%%

df_att = df.iloc[:, :312] # membuat kolom dengan ketentuan 312
df_label = df.iloc[:, 312:] 

#%%

df_att.head() # menampilkan hanya data teratas

#%%


df_label.head() # menampilkan data yg sudah dibaca namun hanya data teratas


#%%
 
df_train_att = df_att[:8000] # data akan dibagi dari 8000 row pertama menjadi data training dan sisanya adalah data testing
df_train_label = df_label[:8000] # data akan dibagi dari 8000 row pertama menjadi data training dan sisanya adalah data testing
df_test_att = df_att[8000:] #  berbalik dengan sebelumnya, data akan dibagi mulai dari 8000 row pertama menjadi data testing dan sisanya menjadi data traning
df_test_label = df_label[8000:]

df_train_label = df_train_label['label'] # menambahkan label
df_test_label = df_test_label['label'] 

#%%

from sklearn.ensemble import RandomForestClassifier # import fungsi random forest
clf = RandomForestClassifier(max_features=50, random_state=0, n_estimators=100) #clf sebagai variable untuk klasifikasi random forest

#%%

clf.fit(df_train_att, df_train_label) # vriable clf untuk fit yaitu menjadi data training

#%%

print(clf.predict(df_train_att.head())) # print clf yang sudah prediksi dar training tetapi anya menampilkan data paling atas

#%%

print(clf.predict(df_test_att.head()))

#%%

clf.score(df_test_att, df_test_label) # memunculkan clf sebagai testing yang sudah di training 


#%%
#SOAL NOMOR 5

from sklearn.metrics import confusion_matrix # menginportkan matrix
pred_labels = clf.predict(df_test_att) # membuat variable pred label dari data testing
cm = confusion_matrix(df_test_label, pred_labels) # cd sebagai variable dari data label


#%%


cm # memunculkan data label bentuk array


#%%
# from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
import matplotlib.pyplot as plt # menginportkan library matplotlib sebagai plt
import itertools # mengmportkan library itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues): # membuat fungsi dengan ketentuan data yang ada pada cm
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")#jika normlisasi sebgai ketentuan yang ada maka print akn menampilkan normalized
    else:
        print('Confusion matrix, without normalization')#jika tidak maka akan enampilkan else

    print(cm)#menampilkan cm

    plt.imshow(cm, interpolation='nearest', cmap=cmap)#plt sebagai pungsi untuk membuat plot
    plt.title(title)#membuat tittle pada plot
    #plt.colorbar()
    tick_marks = np.arange(len(classes))#membuat maks pada plot
    plt.xticks(tick_marks, classes, rotation=90)#membuat tick padaa x
    plt.yticks(tick_marks, classes)#membuat tick padaa y


    fmt = '.2f' if normalize else 'd' #sebagai normalisasi
    thresh = cm.max() / 2. # mengambil data maksimum pada cm kemudian dibagi menjadi 2

    #mengatur layout pada plot dan menambahkan nama label pada sumbu x dan y 
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


#%%
# membaca csv dengan ketentuan nama bird nya
birds = pd.read_csv(r"C:\Users\Dellavianti\chapter 3;\classes.txt",
                    sep='\s+', header=None, usecols=[1], names=['birdname'])
birds = birds['birdname']
birds

#%%
import numpy as np
np.set_printoptions(precision=2) # membuat variable set_precision 2
plt.figure(figsize=(60,60), dpi=300) # sbg figure dg ketentuan size 60 dan dpi 300
plot_confusion_matrix(cm, classes=birds, normalize=True) # data cm dan class bird dibuat menjadi plot
plt.show()


#%%
#SOAL NOMOR 6

from sklearn import tree
clftree = tree.DecisionTreeClassifier() # variable untuk decision tree
clftree.fit(df_train_att, df_train_label) # selanjutnya kan melakukan data training dan testing
clftree.score(df_test_att, df_test_label)


#%%

from sklearn import svm
clfsvm = svm.SVC() # variable untuk mengatur fungsi svc 
clfsvm.fit(df_train_att, df_train_label)
clfsvm.score(df_test_att, df_test_label)

#%%

from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, df_train_att, df_train_label, cv=5) # membuat variable score sbg variable untuk prediksi dr data training
# show average score and +/- two standard deviations away (covering 95% of scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)) # menampilkan datascore dg ketentuan akurasi


#%%
#SOAL NOMOR 7
#membuat suatu varible prediksi dari data training 
scorestree = cross_val_score(clftree, df_train_att, df_train_label, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scorestree.mean(), scorestree.std() * 2))


#%%


scoressvm = cross_val_score(clfsvm, df_train_att, df_train_label, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scoressvm.mean(), scoressvm.std() * 2))


#%%
#SOAL NOMOR 8

max_features_opts = range(5, 50, 5) # variblenya akan membuat range
n_estimators_opts = range(10, 200, 20)
rf_params = np.empty((len(max_features_opts)*len(n_estimators_opts),4), float) # variablenya untuk menjumlahkan 
i = 0
for max_features in max_features_opts:
    for n_estimators in n_estimators_opts: # untuk melakukan perulangan
        clf = RandomForestClassifier(max_features=max_features, n_estimators=n_estimators)
        scores = cross_val_score(clf, df_train_att, df_train_label, cv=5) # score sebagai variable training
        rf_params[i,0] = max_features
        rf_params[i,1] = n_estimators
        rf_params[i,2] = scores.mean()
        rf_params[i,3] = scores.std() * 2
        i += 1
        print("Max features: %d, num estimators: %d, accuracy: %0.2f (+/- %0.2f)" %
              (max_features, n_estimators, scores.mean(), scores.std() * 2))


#%%

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
fig = plt.figure() # menghasilkan plot sebagai figure
fig.clf() # figure akan diambil dari clf
ax = fig.gca(projection='3d') # akan dijadikan projection 3D
x = rf_params[:,0]
y = rf_params[:,1]
z = rf_params[:,2]
ax.scatter(x, y, z)
ax.set_zlim(0.2, 0.5) # set berupa zlim
ax.set_xlabel('Max features')
ax.set_ylabel('Num estimators')
ax.set_zlabel('Avg accuracy')
plt.show()
