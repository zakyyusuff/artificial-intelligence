#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 10:46:27 2022

@author: yunita
"""

# coding: utf-8

# In[41]
# SOAL NOMOR 1
import pandas as pd #melakukan Import library pandas menjadi nama lain yaitu pd

kegiatan = {"Nama kegiatan" : ['Belanja', 'Memasak', 'Menyanyi', 'Berkuda']} #mmebuat variable yang bernama kegiatan dan mengisi dataframe aplikasi
x = pd.DataFrame(kegiatan) # membuat variable x yang akan mmebuat dataframe dari library pandas yang akan memanggil variable aplikasi
print ('Nur ikhsani sedang:' + x) #print hasil dari x

# In[42]
#SOAL NOMOR 2
import numpy as np # melakukan import library numpy menjadi nama lain yaitu np

matrix_x = np.eye(10) # membuat sebuah matrix pake numpy dengan menggunakan fungsi eye
matrix_x #mendeklarasikan matrix_x yang telah dibuat

print (matrix_x) # menampilkan matrix_x yang telah dibat dengan berbentuk 10x10

# In[43]
#SOAL NOMOR 3
import matplotlib.pyplot as mp # melakukan import library numpy menjadi nama lain yaitu mp
mp.plot([1,1,8,7,0,9,9]) #memasukkan nilai pada plot
mp.xlabel('Nur Ikhsani Suwandy Futri') # menambahkan label pada x
mp.ylabel('1194029')# menambahkan label pada y
mp.show() # menampilkan grafik plot
# In[44]:
#SOAL NOMOR 4


import pandas as pd #MELAKUKAN IMPORT PANDA DENGAN NAMA LAIN YAITU PD

# some lines have too many fields (?), so skip bad lines
imgatt = pd.read_csv("image_attribute_labels.txt",
                     sep='\s+', header=None, error_bad_lines=False, warn_bad_lines=False,
                     usecols=[0,1,2], names=['imgid', 'attid', 'present']) #MEMBUAT VARIABLE DENGAN IMGATT UNTUK MEMBACA CSV DARI DATA SET

# description from dataset README:
# 
# The set of attribute labels as perceived by MTurkers for each image
# is contained in the file attributes/image_attribute_labels.txt, with
# each line corresponding to one image/attribute/worker triplet:
#
# <image_id> <attribute_id> <is_present> <certainty_id> <time>
#
# where <image_id>, <attribute_id>, <certainty_id> correspond to the IDs
# in images.txt, attributes/attributes.txt, and attributes/certainties.txt
# respectively.  <is_present> is 0 or 1 (1 denotes that the attribute is
# present).  <time> denotes the time spent by the MTurker in seconds.


# In[45]:


imgatt.head() #MENAMPILKAN DATA YANG TELAH DIBACA


# In[46]:


imgatt.shape#MENAMPILKAN JUMLAH SELURUH DATA 


# In[47]:


# need to reorganize imgatt to have one row per imgid, and 312 columns (one column per attribute),
# with 1/0 in each cell representing if that imgid has that attribute or not

imgatt2 = imgatt.pivot(index='imgid', columns='attid', values='present')#MEMBUAT SEBUAH VARIABLE BARU DARI FUNGSI IMGATT DENGAN MENGGANTI INDEK MENJADI KOLOM DAN KOLOM MENJADI INDEX


# In[48]:


imgatt2.head()#MENAMPILKAN DATA YANG SUDAH DIBACA DENGAN 5 DATA TERATAS.


# In[49]:


imgatt2.shape# MENAMPILKAN JUMLAH SELURUH DATA 


# In[50]:


# now we need to load the image true classes

imglabels = pd.read_csv("image_class_labels.txt", 
                        sep=' ', header=None, names=['imgid', 'label'])#BACA DATA CSV DENGAN KETENTUAN YANG ADA

imglabels = imglabels.set_index('imgid')#VARIABLE IMGLABELS SEBAGAI SET INDEX IMGID




# In[51]:


imglabels.head()#MEMBACA DATA YANG DIMASUKKAN KE VARIABLE IMGLABELS


# In[52]:


imglabels.shape# MENAMPILKAN JUMLAH DATA SELURUH DATA SERTA KOLOMNYA


# In[53]:


# now we need to attach the labels to the attribute data set,
# and shuffle; then we'll separate a test set from a training set

df = imgatt2.join(imglabels)#VARIABLE DF DIMASUKKAN FUNGSI JOIN DARI DATA IMGATT2 KE VARIABLE IMGLABELS
df = df.sample(frac=1)#VARIABLE DF SEBAGAI SAMPLE DENGAN KETENTUAN FRAC=1


# In[54]:


df_att = df.iloc[:, :312]#MEMBUAT KOLOM DENGAN KETENTUAN 312
df_label = df.iloc[:, 312:]#MEMBUAT KOLOM DENGAN KETENTUAN 312


# In[55]:


df_att.head()# MENAMPILKAN DATA YANG SUDAH DIBACA TADI NAMUN HANYA DATA YANG TERATAS.


# In[56]:


df_label.head()#MENAMPILKAN DATA YANG SUDAH DIBACA TADI NAMUN HANYA DA


# In[57]:


df_train_att = df_att[:8000]#DATA AKAN DIBAGI DARI 8000 ROW PERTAMA MENJADI DATA TRANINING DAN SISINYA ADALAH DATA TESTING
df_train_label = df_label[:8000]#DATA AKAN DIBAGI DARI 8000 ROW PERTAMA MENJADI DATA TRAINING DAN SISANYA ADALAH DATA TESTING 
df_test_att = df_att[8000:]#BERBALIK DARI SEBELUMNYA DTA AKAN DIBAGI MULAI DARI 8000 ROW PERTAMA MENJADI DATA TRAINING DAN SISINYA MENJADI DATA TESTING
df_test_label = df_label[8000:]#BERBALIK DARI SEBELUMNYA DTA AKAN DIBAGI 

df_train_label = df_train_label['label']#MENAMBAHKAN LABEL
df_test_label = df_test_label['label']#NEMANBAHKAN LABEL


# In[58]:


from sklearn.ensemble import RandomForestClassifier #imort fungsi random forest
clf = RandomForestClassifier(max_features=50, random_state=0, n_estimators=100)#clf sebagai variable untuk klasifikasi random forest


# In[59]:


clf.fit(df_train_att, df_train_label)#vriable clf untuk fit yaitu menjadi data training


# In[60]:


print(clf.predict(df_train_att.head()))#print clf yang sudah prediksi dar training tetapi anya menampilkan data paling atas
# In[60]:
print(clf.predict(df_test_att.head()))

# In[61]:


clf.score(df_test_att, df_test_label)#memunculkan clf sebagai testing yang sudah di training 


# In[62]:
#SOAL NOMOR 5


from sklearn.metrics import confusion_matrix#menginportkan matrix
pred_labels = clf.predict(df_test_att)#membuat variable pred label dari data testing
cm = confusion_matrix(df_test_label, pred_labels)#cd sebagai variable dari data label


# In[63]:


cm#memunculkan data label bentuk array


# In[64]:


# from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
import matplotlib.pyplot as plt #menginportkan library matplotlib sebagai plt
import itertools #mengmportkan library itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):#membuat fungsi dengan ketentuan data yang ada pada cm
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


    fmt = '.2f' if normalize else 'd' #
    thresh = cm.max() / 2.

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[65]:


birds = pd.read_csv("classes.txt",
                    sep='\s+', header=None, usecols=[1], names=['birdname'])
birds = birds['birdname']
birds


# In[66]:


import numpy as np
np.set_printoptions(precision=2)
plt.figure(figsize=(60,60), dpi=300)
plot_confusion_matrix(cm, classes=birds, normalize=True)
plt.show()


# In[67]:
#SOAL NOMOR 6


from sklearn import tree
clftree = tree.DecisionTreeClassifier()
clftree.fit(df_train_att, df_train_label)
clftree.score(df_test_att, df_test_label)


# In[68]:


from sklearn import svm
clfsvm = svm.SVC()
clfsvm.fit(df_train_att, df_train_label)
clfsvm.score(df_test_att, df_test_label)


# In[69]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, df_train_att, df_train_label, cv=5)
# show average score and +/- two standard deviations away (covering 95% of scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[70]:
#SOAL NOMOR 7


scorestree = cross_val_score(clftree, df_train_att, df_train_label, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scorestree.mean(), scorestree.std() * 2))


# In[71]:


scoressvm = cross_val_score(clfsvm, df_train_att, df_train_label, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scoressvm.mean(), scoressvm.std() * 2))


# In[72]:
#SOAL NOMOR 8

max_features_opts = range(5, 50, 5)
n_estimators_opts = range(10, 200, 20)
rf_params = np.empty((len(max_features_opts)*len(n_estimators_opts),4), float)
i = 0
for max_features in max_features_opts:
    for n_estimators in n_estimators_opts:
        clf = RandomForestClassifier(max_features=max_features, n_estimators=n_estimators)
        scores = cross_val_score(clf, df_train_att, df_train_label, cv=5)
        rf_params[i,0] = max_features
        rf_params[i,1] = n_estimators
        rf_params[i,2] = scores.mean()
        rf_params[i,3] = scores.std() * 2
        i += 1
        print("Max features: %d, num estimators: %d, accuracy: %0.2f (+/- %0.2f)" %
              (max_features, n_estimators, scores.mean(), scores.std() * 2))


# In[90]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
fig = plt.figure()
fig.clf()
ax = fig.gca(projection='3d')
x = rf_params[:,0]
y = rf_params[:,1]
z = rf_params[:,2]
ax.scatter(x, y, z)
ax.set_zlim(0.2, 0.5)
ax.set_xlabel('Max features')
ax.set_ylabel('Num estimators')
ax.set_zlabel('Avg accuracy')
plt.show()

