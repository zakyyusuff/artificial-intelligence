import sys
import argparse
import numpy as np
import theano as th
import theano.tensor as T
import lasagne
import lasagne.layers as LL
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import scipy.io

from theano.sandbox.rng_mrg import MRG_RandomStreams

#%%

# Argument settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=7)
parser.add_argument('--seed_data', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=128)
args = parser.parse_args()
print(args)

#Xavier initialization of weights in the network,
#you can alternatively use Lasagne.init.GlorotNormal as well

#%%
def xavier_init(size_data):
    in_dim = size_data[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return np.random.normal(scale=xavier_stddev, size=size_data).astype("float32")


#%%
# fixed random seeds
rng = np.random.RandomState(args.seed)
theano_rng = MRG_RandomStreams(rng.randint(2 ** 18))
lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 18)))

#%%
##### Theano variables fortraining the networks
input_var=T.matrix('input_var')
noise_var=T.matrix('noise')
input_labels=T.matrix('labels')


#%%%
Gen_input=T.matrix('gen_input')
Dis_input=T.matrix('dis_input')
#%%%
######loading the dataset ######

data = np.load('mnist.npz')
trainx = np.concatenate([data['x_train'], data['x_valid']], axis=0)
trainy = np.concatenate([data['y_train'], data['y_valid']], axis=0)

#%%%

Digit_Generate=2


#%%
nr_batches_train = int(trainx.shape[0]/args.batch_size)
classes=10

#%%
y_onehot = np.zeros((trainy.shape[0],classes)).astype('float32')
for i, label in enumerate(trainy):
    y_onehot[i,trainy[i]] = 1.0


#%%%
y_dim=y_onehot.shape[1]
print("Shape",y_dim)

#%%
sample_y=np.copy(y_onehot)

#%%%
y = trainy.astype(np.int)

#%%%
Digit_Labels=(trainy[(y == Digit_Generate)][:16])

#%%%

Digit_onehot = np.zeros((Digit_Labels.shape[0],classes)).astype('float32')
for i, label in enumerate(Digit_Labels):
    Digit_onehot[i,Digit_Labels[i]] = 1.0

#%%%




