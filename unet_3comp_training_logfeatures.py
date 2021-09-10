#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 11:27:07 2020

Train a CNN to pick P and S wave arrivals with log features

@author: amt
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import h5py
from scipy import signal
import gnss_unet_tools
import argparse

# # parse arguments
# parser = argparse.ArgumentParser()
# parser.add_argument("-subset", "--subset", help="train on a subset or no?", type=int)
# parser.add_argument("-pors", "--pors", help="train P or S network", type=int)
# parser.add_argument("-train", "--train", help="do you want to train?", type=int)
# parser.add_argument("-drop", "--drop", help="want to add a drop layer to the network", type=int)
# parser.add_argument("-plots", "--plots", help="want plots", type=int)
# parser.add_argument("-resume", "--resume", help="want to resume training?", type=int)
# parser.add_argument("-large", "--large", help="what size do you want the network to be?", type=float)
# parser.add_argument("-epochs", "--epochs", help="how many epochs", type=int)
# parser.add_argument("-std", "--std", help="standard deviation of target", type=float)
# parser.add_argument("-sr", "--sr", help="sample rate in hz", type=int)
# args = parser.parse_args()

# subset=args.subset #True # train on a subset or the full monty?
# ponly=args.pors #True # 1 - P+Noise, 2 - S+noise
# train=args.train #True # do you want to train?
# drop=args.drop #True # drop?
# plots=args.plots #False # do you want to make some plots?
# resume=args.resume #False # resume training
# large=args.large # large unet
# epos=args.epochs # how many epocs?
# std=args.std # how long do you want the gaussian STD to be?
# sr=args.sr

train=1 #True # do you want to train?
drop=1 #True # drop?
plots=1 #False # do you want to make some plots?
resume=0 #False # resume training
large=0.5 # large unet
epos=50# how many epocs?
std=3 # how long do you want the gaussian STD to be?
sr=1

epsilon=1e-6

print("train "+str(train))
print("drop "+str(drop))
print("plots "+str(plots))
print("resume "+str(resume))
print("large "+str(large))
print("epos "+str(epos))
print("std "+str(std))
print("sr "+str(sr))

# LOAD THE DATA
print("LOADING DATA")
n_data = h5py.File('100k_noise.hdf5', 'r')
x_data = h5py.File('100k_clean_data_REDO.hdf5', 'r')
model_save_file="gnssunet_3comp_logfeat_250000_pn_eps_"+str(epos)+"_sr_"+str(sr)+"_std_"+str(std)+".tf" 
x_data=x_data['100k_REDO'][:,:]
n_data=n_data['100k_noise'][:,:]
# TODO: delete this line
x_data[:,:128]=x_data[:,256:256+128]=x_data[:,512:512+128]=0
        
if large:
    fac=large
    model_save_file="large_"+str(fac)+"_"+model_save_file

if drop:
    model_save_file="drop_"+model_save_file

# MAKE TRAINING AND TESTING DATA
print("MAKE TRAINING AND TESTING DATA")
np.random.seed(0)
siginds=np.arange(x_data.shape[0])
noiseinds=np.arange(n_data.shape[0])
np.random.shuffle(siginds)
np.random.shuffle(noiseinds)
sig_train_inds=np.sort(siginds[:int(0.9*len(siginds))])
noise_train_inds=np.sort(noiseinds[:int(0.9*len(noiseinds))])
sig_test_inds=np.sort(siginds[int(0.9*len(siginds)):])
noise_test_inds=np.sort(noiseinds[int(0.9*len(noiseinds)):])

# plot the data
if plots:
    # # plot ps to check
    # plt.figure()
    # for ii in range(2000):
    #     plt.plot(x_data[ii,120:136]/np.max(np.abs(x_data[ii,120:136]))) #/np.max(np.abs(x_data[ii,:]))+ii)

    plt.figure()
    for ii in range(2000):
        plt.plot(x_data[ii,120:136]) #/np.max(np.abs(x_data[ii,:]))+ii)
    plt.ylim((-0.01,0.01))
        
    # # plot noise to check
    # plt.figure()
    # for ii in range(20):
    #     plt.plot(n_data[ii,:]/np.max(np.abs(n_data[ii,:]))+ii)

# generate batch data
print("FIRST PASS WITH DATA GENERATOR")
my_data=gnss_unet_tools.my_3comp_data_generator(32,x_data,n_data,sig_train_inds,noise_train_inds,sr,std,valid=True)
normdata,target,origdata=next(my_data)

# PLOT GENERATOR RESULTS
if plots:
    for ind in range(20):
        fig, ax = plt.subplots(nrows=2,ncols=3,sharex=True,figsize=(20,10))
        t=1/sr*np.arange(normdata.shape[1])
        for kk in range(3):
            ax[0,kk].set_xlabel('Time (s)')
            ax[0,kk].set_ylabel('Amplitude', color='tab:red')
            ax[0,kk].plot(t, origdata[ind,:,kk], color='tab:red', label='data')
            ax[0,kk].tick_params(axis='y')
            ax[0,kk].legend(loc="lower right")
            ax1 = ax[0,kk].twinx()  # instantiate a second axes that shares the same x-axis
            ax1.set_ylabel('Prediction', color='black')  # we already handled the x-label with ax1
            ax1.plot(t, target[ind,:], color='black', linestyle='--', label='target')
            ax1.legend(loc="upper right")
            ax[1,kk].plot(t, normdata[ind,:,kk*2], color='tab:green', label='ln(data amp)')
            ax[1,kk].plot(t, normdata[ind,:,kk*2+1], color='tab:blue', label='data sign')
            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            ax[1,kk].legend(loc="lower right")
        # plt.show()

# BUILD THE MODEL
print("BUILD THE MODEL")
if drop:
    model=gnss_unet_tools.make_large_unet_drop(fac,sr,ncomps=3)    
else:
    model=gnss_unet_tools.make_large_unet(fac,sr,ncomps=3)  
        
# ADD SOME CHECKPOINTS
print("ADDING CHECKPOINTS")
checkpoint_filepath = './checks/'+model_save_file+'_{epoch:04d}.ckpt'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath, save_weights_only=True, verbose=1,
    monitor='val_acc', mode='max', save_best_only=True)

# TRAIN THE MODEL
print("TRAINING!!!")
if train:
    batch_size=32
    if resume:
        print('Resuming training results from '+model_save_file)
        model.load_weights(checkpoint_filepath)
    else:
        print('Training model and saving results to '+model_save_file)
        
    csv_logger = tf.keras.callbacks.CSVLogger(model_save_file+".csv", append=True)
    # unet_tools.my_3comp_data_generator(32,x_data,n_data,sig_train_inds,noise_train_inds,sr,std)
    history=model.fit_generator(gnss_unet_tools.my_3comp_data_generator(batch_size,x_data,n_data,sig_train_inds,noise_train_inds,sr,std),
                        steps_per_epoch=(len(sig_train_inds)+len(noise_train_inds))//batch_size,
                        validation_data=gnss_unet_tools.my_3comp_data_generator(batch_size,x_data,n_data,sig_test_inds,noise_test_inds,sr,std),
                        validation_steps=(len(sig_test_inds)+len(noise_test_inds))//batch_size,
                        epochs=epos, callbacks=[model_checkpoint_callback,csv_logger])
    model.save_weights("./"+model_save_file)
else:
    print('Loading training results from '+model_save_file)
    model.load_weights("./"+model_save_file)
    
# plots=1
# # plot the results
# if plots:
#     # training stats
#     training_stats = np.genfromtxt("./"+model_save_file+'.csv', delimiter=',',skip_header=1)
#     f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
#     ax1.plot(training_stats[:,0],training_stats[:,1])
#     ax1.plot(training_stats[:,0],training_stats[:,3])
#     ax1.legend(('acc','val_acc'))
#     ax2.plot(training_stats[:,0],training_stats[:,2])
#     ax2.plot(training_stats[:,0],training_stats[:,4])
#     ax2.legend(('loss','val_loss'))
#     ax2.set_xlabel('Epoch')
#     ax1.set_title(model_save_file)

# # See how things went
# my_test_data=gnss_unet_tools.my_3comp_data_generator(50,x_data,n_data,sig_test_inds,noise_test_inds,sr,std, valid=True)
# normdata,target,origdata=next(my_test_data)
# test_predictions=model.predict(normdata)
# if plots:
#     for ind in range(20):
#         fig, ax = plt.subplots(nrows=2,ncols=3,sharex=True,figsize=(20,10))
#         t=1/sr*np.arange(normdata.shape[1])
#         for kk in range(3):
#             ax[0,kk].set_xlabel('Time (s)')
#             ax[0,kk].set_ylabel('Amplitude', color='tab:red')
#             ax[0,kk].plot(t, origdata[ind,:,kk], color='tab:red', label='data')
#             ax[0,kk].tick_params(axis='y')
#             ax[0,kk].legend(loc="lower right")
#             ax1 = ax[0,kk].twinx()  # instantiate a second axes that shares the same x-axis
#             ax1.set_ylabel('Prediction', color='black')  # we already handled the x-label with ax1
#             ax1.plot(t, target[ind,:], color='black', linestyle='--', label='target')
#             ax1.plot(t, test_predictions[ind,:], color='purple', linestyle='--', label='prediction')
#             ax1.legend(loc="upper right")
#             ax[1,kk].plot(t, normdata[ind,:,kk*2], color='tab:green', label='ln(data amp)')
#             ax[1,kk].plot(t, normdata[ind,:,kk*2+1], color='tab:blue', label='data sign')
#             fig.tight_layout()  # otherwise the right y-label is slightly clipped
#             ax[1,kk].legend(loc="lower right")