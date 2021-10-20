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
# parser.add_argument('-subset', '--subset', help = 'train on a subset or no?', type = int)
# parser.add_argument('-pors', '--pors', help = 'train P or S network', type = int)
# parser.add_argument('-train', '--train', help = 'do you want to train?', type = int)
# parser.add_argument('-drop', '--drop', help = 'want to add a drop layer to the network', type = int)
# parser.add_argument('-plots', '--plots', help = 'want plots', type = int)
# parser.add_argument('-resume', '--resume', help = 'want to resume training?', type = int)
# parser.add_argument('-large', '--large', help = 'what size do you want the network to be?', type = float)
# parser.add_argument('-epochs', '--epochs', help = 'how many epochs', type = int)
# parser.add_argument('-std', '--std', help = 'standard deviation of target', type = float)
# parser.add_argument('-sr', '--sr', help = 'sample rate in hz', type = int)
# args = parser.parse_args()

# subset = args.subset #True # train on a subset or the full monty?
# ponly = args.pors #True # 1 - P+Noise, 2 - S+noise
# train = args.train #True # do you want to train?
# drop = args.drop #True # drop?
# plots = args.plots #False # do you want to make some plots?
# resume = args.resume #False # resume training
# large = args.large # large unet
# epos = args.epochs # how many epocs?
# std = args.std # how long do you want the gaussian STD to be?
# sr = args.sr

# 0 = False, 1 = True

train = 0 # do you want to train?
drop = 1 # drop?
plots = 1 # do you want to make some plots?
resume = 0 # resume training
large = 0.5 # large unet
epos = 50 # how many epocs?
std = 3 # how long do you want the gaussian STD to be?
sr = 1

epsilon = 1e-6

print('train ' + str(train))
print('drop ' + str(drop))
print('plots ' + str(plots))
print('resume ' + str(resume))
print('large ' + str(large))
print('epos ' + str(epos))
print('std ' + str(std))
print('sr ' + str(sr))

##### -------------------- LOAD THE DATA -------------------- #####

print('LOADING DATA')

# Signals of earthquakes
x_data = h5py.File('100k_clean_data_REDO.hdf5', 'r')
x_data = x_data['100k_REDO'][:,:]

# Noise data
n_data = h5py.File('100k_noise.hdf5', 'r')
n_data = n_data['100k_noise'][:,:]

# Metadata with information about earthquakes in x_data
meta_data = np.load('100k_clean_info_REDO.npy')

# Array of NaNs to use to match added noise in concatenation later
nan_array = np.empty((len(x_data), 3))
nan_array[:] = np.NaN

model_save_file = 'gnssunet_3comp_logfeat_250000_pn_eps_' + str(epos) + '_sr_' + str(sr) + '_std_' + str(std) + '.tf' 
        
if large:
    fac = large
    model_save_file = 'large_' + str(fac) + '_' + model_save_file

if drop:
    model_save_file = 'drop_' + model_save_file

##### -------------------- MAKE TRAINING AND TESTING DATA -------------------- #####

print('MAKE TRAINING AND TESTING DATA')

np.random.seed(0)

## Training

# Shuffling indices of earthquake data and grabbing 90%, then putting them back in order
siginds = np.arange(x_data.shape[0]) # numbers between 0 and 100,043
np.random.shuffle(siginds) # randomly shuffles the numbers between 0 and 100,043
sig_train_inds = np.sort(siginds[:int(0.9*len(siginds))]) # grabs the front 90% of the numbers, then sorts them back into order

# Shuffling indices of noise data and grabbing 90%, then putting them back in order
noiseinds = np.arange(n_data.shape[0])
np.random.shuffle(noiseinds)
noise_train_inds = np.sort(noiseinds[:int(0.9*len(noiseinds))])

## Testing
sig_test_inds = np.sort(siginds[int(0.9*len(siginds)):]) # grabs the back 10% (90% through the end) and sorts 
noise_test_inds = np.sort(noiseinds[int(0.9*len(noiseinds)):])

# Some plots to check what we've loaded
# if plots:
    
#     # Plot the data (no noise, just earthquakes)
#     plt.figure()   
#     for ii in range(20): # plot 20 of them
#         plt.plot(x_data[ii,:]) #/np.max(np.abs(x_data[ii,:]))+ii)
#     plt.show()
        
#     # Plot noise to check
#     plt.figure()
#     for ii in range(20):
#         plt.plot(n_data[ii,:]/np.max(np.abs(n_data[ii,:]))+ii) # normalized?
#     plt.show()

# ##### -------------------- FIRST GENERATOR TEST: generate batch data -------------------- #####

print('FIRST PASS WITH DATA GENERATOR')

my_data = gnss_unet_tools.my_3comp_data_generator(32, x_data, n_data, meta_data, nan_array, sig_train_inds, noise_train_inds, sr, std, valid = True) # Valid = True to get original data back
batch_out, target, origdata, metadata = next(my_data) # batch_out and origdata are the same with GNSS implementation
# Shapes:
    # batch_out: (batch_size, 128, 3) # N, E, Z
    # target: (5000, 128)
    # origdata: (5000, 128, 3) # N, E, Z
    # metadata: (5000, 3) Rupt name, station name, magnitude

## Plot generator results

# if plots:
    
#     for ind in range(3): # Number of samples to look at 
        
#         fig = plt.subplots(nrows = 1, ncols = 3, figsize = (18,4))
#         t = 1/sr * np.arange(batch_out.shape[1])
        
#         ax1 = plt.subplot(131)
#         ax1.plot(t, origdata[ind,:,0], label = 'N original data', color = 'C0')
#         ax2 = ax1.twinx()
#         ax2.plot(t, target[ind,:], color = 'black', linestyle = '--', label = 'Target')
#         ax1.legend(loc = 'upper right')
#         ax2.legend(loc = 'lower right')
        
#         ax3 = plt.subplot(132)
#         ax3.plot(t, origdata[ind,:,1], label = 'E original data', color = 'C1')
#         ax4 = ax3.twinx()
#         ax4.plot(t, target[ind,:], color = 'black', linestyle = '--', label = 'Target')
#         ax3.legend(loc = 'upper right')
#         ax4.legend(loc = 'lower right')
        
#         ax5 = plt.subplot(133)
#         ax5.plot(t, origdata[ind,:,2], label = 'Z original data', color = 'C2')
#         ax6 = ax5.twinx()
#         ax6.plot(t, target[ind,:], color = 'black', linestyle = '--', label = 'Target')
#         ax5.legend(loc = 'upper right')
#         ax6.legend(loc = 'lower right')
        
#     plt.savefig('Plot_generator_pass.png', format = 'PNG')
#     plt.close()
    

# ##### -------------------- BUILD THE MODEL -------------------- #####

print('BUILD THE MODEL')

if drop:
    model = gnss_unet_tools.make_large_unet_drop(fac, sr, ncomps = 3)    
    
else:
    model = gnss_unet_tools.make_large_unet(fac, sr, ncomps = 3)  
        
# ##### -------------------- ADD SOME CHECKPOINTS -------------------- #####

print('ADDING CHECKPOINTS')

checkpoint_filepath = './checks/' + model_save_file + '_{epoch:04d}.ckpt'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_filepath, save_weights_only = True, verbose = 1,
    monitor = 'val_acc', mode = 'max', save_best_only = True) # rename val_loss to validation_loss, or val_acc to val_accuracy

# ##### -------------------- TRAIN THE MODEL -------------------- #####

print('TRAINING!!!')

if train:
    
    batch_size = 32
    
    if resume:
        print('Resuming training results from ' + model_save_file)
        model.load_weights(checkpoint_filepath)
        
    else:
        print('Training model and saving results to ' + model_save_file)
        
    csv_logger = tf.keras.callbacks.CSVLogger(model_save_file + '.csv', append = True)
    # unet_tools.my_3comp_data_generator(32,x_data,n_data,sig_train_inds,noise_train_inds,sr,std)
    
    history = model.fit_generator(gnss_unet_tools.my_3comp_data_generator(batch_size, x_data, n_data, meta_data, nan_array, sig_train_inds, noise_train_inds, sr, std), # Valid = False for training; implied
                        steps_per_epoch = (len(sig_train_inds) + len(noise_train_inds))//batch_size,
                        validation_data = gnss_unet_tools.my_3comp_data_generator(batch_size, x_data, n_data, meta_data, nan_array, sig_test_inds, noise_test_inds, sr, std),
                        validation_steps = (len(sig_test_inds) + len(noise_test_inds))//batch_size,
                        epochs = epos, callbacks = [model_checkpoint_callback, csv_logger])
    
    model.save_weights('./' + model_save_file)
    
else:
    print('Loading training results from ' + model_save_file)
    model.load_weights('./' + model_save_file)
    
plots = 1

# plot the results
if plots:
    
    # training stats
    training_stats = np.genfromtxt('./' + model_save_file + '.csv', delimiter = ',', skip_header = 1)
    f, (ax1, ax2) = plt.subplots(2, 1, sharex = True)
    ax1.plot(training_stats[:,0], training_stats[:,1])
    ax1.plot(training_stats[:,0], training_stats[:,3])
    ax1.legend(('acc','val_acc'))
    ax2.plot(training_stats[:,0], training_stats[:,2])
    ax2.plot(training_stats[:,0], training_stats[:,4])
    ax2.legend(('loss','val_loss'))
    ax2.set_xlabel('Epoch')
    ax1.set_title(model_save_file)
    # plt.savefig('3_training_stats.png', format = 'PNG')
    # plt.close()

# number of events to test with
num_test = 5000

# See how things went
my_test_data = gnss_unet_tools.my_3comp_data_generator(num_test, x_data, n_data, meta_data, nan_array, sig_test_inds, noise_test_inds, sr, std, valid = True)
batch_out, target, origdata, metadata = next(my_test_data)
test_predictions = model.predict(batch_out)

if plots:
    
    for ind in range(20):
        
        fig, ax = plt.subplots(nrows = 2,ncols = 3,sharex = True,figsize = (20,10))
        t = 1/sr*np.arange(batch_out.shape[1])
        
        for kk in range(3):
            
            ax[0,kk].set_xlabel('Time (s)')
            ax[0,kk].set_ylabel('Amplitude', color = 'tab:red')
            ax[0,kk].plot(t, origdata[ind,:,kk], color = 'tab:red', label = 'data')
            ax[0,kk].tick_params(axis = 'y')
            ax[0,kk].legend(loc = 'lower right')
            ax1 = ax[0,kk].twinx()  # instantiate a second axes that shares the same x-axis
            ax1.set_ylabel('Prediction', color = 'black')  # we already handled the x-label with ax1
            ax1.plot(t, target[ind,:], color = 'black', linestyle = '--', label = 'target')
            ax1.plot(t, test_predictions[ind,:], color = 'purple', linestyle = '--', label = 'prediction')
            ax1.legend(loc = 'upper right')
            ax[1,kk].plot(t, batch_out[ind,:,kk*2], color = 'tab:green', label = 'ln(data amp)')
            ax[1,kk].plot(t, batch_out[ind,:,kk*2+1], color = 'tab:blue', label = 'data sign')
            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            ax[1,kk].legend(loc = 'lower right')
            
        # plt.savefig('4_testing_' + str(ind) + '.png', format = 'PNG')
        # plt.close()
        
# ##### -------------------- CLASSIFICATION TESTS -------------------- #####

# # Decision threshold evaluation

# thresholds = np.arange(0.01, 1, 0.01)

# # threshold = 0.01
# # print(origdata.shape)
# # # print(origdata[0,:,0])
# # print(target.shape)
# # print(target[0])
# # print(test_predictions.shape)
# # print(test_predictions[0])

# # use np.where to see whether anywhere in test_predictions is > threshold
# # if there is a value that's >, the 'result' of the array is 1. If not 0
# # then compare these 1s and 0s to the target array value for PAR

# accuracies = []
# precisions = []
# recalls = []
# F1s = []

# for threshold in thresholds:
    
#     # print('Threshold: ' + str(threshold))
    
#     # Convert the predictions arrays to single ones and zeroes
    
#     pred_binary = np.zeros(len(test_predictions))
#     iterate = np.arange(0,num_test,1)
    
#     for k in iterate:
#         i = np.where(test_predictions[k] >= threshold)[0]
#         # print(i)
#         # print(len(i))
#         if len(i) == 0:
#             # picks.append(0)
#             pred_binary[k] = 0
#         elif len(i) > 0:
#             # picks.append(1)
#             pred_binary[k] = 1
    
#     # print('Predictions: ')
#     # print(pred_binary)
    
#     # Convert the target arrays to single ones and zeroes
    
#     targ_binary = np.zeros(len(target))
#     iterate = np.arange(0,num_test,1)
    
#     for k in iterate:
#         i = np.where(target[k] >= threshold)[0]
#         # print(i)
#         # print(len(i))
#         if len(i) == 0:
#             # picks.append(0)
#             targ_binary[k] = 0
#         elif len(i) > 0:
#             # picks.append(1)
#             targ_binary[k] = 1
    
#     # print('Targets: ')
#     # print(targ_binary)
    
#     # Calculating the accuracy, precision, recall, and F1
    
#     num_preds = num_test # total number of predictions. Amanda did 50
#     correct_preds = []
#     wrong_preds = []
#     true_pos = []
#     true_neg = []
#     false_pos = []
#     false_neg = []
    
#     for i in iterate:
#         pred = pred_binary[i]
#         targ = targ_binary[i]
        
#         # print(pred)
#         # print(targ)
        
#         if pred == targ: # add one to list of correct predictions if matching
#             # print('Correct!')
#             correct_preds.append(1)
            
#             if pred == 1 and targ == 1:
#                 # print('True pos: ')
#                 true_pos.append(1)
#             elif pred == 0 and targ == 0:
#                 true_neg.append(1)
            
#         elif pred != targ: # add ones to list of incorrect predictions if not matching
#             # print('Incorrect!')
#             wrong_preds.append(1)
            
#             if pred == 1 and targ == 0:
#                 false_pos.append(1)
#             elif pred == 0 and targ == 1:
#                 false_neg.append(1)
    
#     # print('Correct preds')
#     # print(correct_preds)
#     # print('Wrong preds')
#     # print(wrong_preds)
#     # print('True pos')
#     # print(true_pos)
#     # print('True neg')
#     # print(true_neg)
#     # print('False pos')
#     # print(false_pos)
#     # print('False neg')
#     # print(false_neg)
    
#     num_correct_preds = len(correct_preds)
#     num_wrong_preds = len(wrong_preds)
#     num_true_pos = len(true_pos)
#     num_true_neg = len(true_neg)
#     num_false_pos = len(false_pos)
#     num_false_neg = len(false_neg)
    
#     # print('Correct preds: ' + str(num_correct_preds))
#     # print('Wrong preds: ' + str(num_wrong_preds))
#     # print('True pos: ' + str(num_true_pos))
#     # print('True neg: ' + str(num_true_neg))
#     # print('False pos: ' + str(num_false_pos))
#     # print('False neg: ' + str(num_false_neg))
    
#     accuracy = num_correct_preds / num_preds
#     # print(accuracy)
    
#     if num_true_pos == 0  and num_false_pos == 0:
#         precision = 0
#     else:
#         precision = num_true_pos / (num_true_pos + num_false_pos)
#     # print(precision)
    
#     if num_true_pos == 0 and num_false_neg == 0:
#         recall = 0
#     else:
#         recall = num_true_pos / (num_true_pos + num_false_neg)
#     # print(recall)
    
#     if precision + recall == 0:
#         F1 = 0
#     else:
#         F1 = 2 * ((precision * recall) / (precision + recall))
    
#     accuracies.append(accuracy)
#     precisions.append(precision)
#     recalls.append(recall)
#     F1s.append(F1)

# # print('Accuracies')
# # print(accuracies)
# # print('Precisions')
# # print(precisions)
# # print('Recalls')
# # print(recalls)
# # print('F1s')
# # print(F1s)

# # plt.figure()
# # plt.scatter(thresholds,accuracies)
# # plt.xlabel('Threshold')
# # plt.ylabel('Accuracy')
# # plt.title('Accuracy')
# # plt.savefig('accuracies_' + str(num_test) + '.png',format='PNG')
# # plt.close()

# # plt.figure()
# # plt.scatter(thresholds,precisions)
# # plt.xlabel('Threshold')
# # plt.ylabel('Precision')
# # plt.title('Precision')
# # plt.savefig('precisions_' + str(num_test) + '.png',format='PNG')
# # plt.close()

# # plt.figure()
# # plt.scatter(thresholds,recalls)
# # plt.xlabel('Threshold')
# # plt.ylabel('Recall')
# # plt.title('Recall')
# # plt.savefig('recalls_' + str(num_test) + '.png',format='PNG')
# # plt.close()

# # plt.figure()
# # plt.scatter(thresholds,F1s)
# # plt.xlabel('Threshold')
# # plt.ylabel('F1')
# # plt.title('F1')
# # plt.savefig('F1s_' + str(num_test) + '.png',format='PNG')
# # plt.close()

# ##### -------------------- GAUSSIAN PEAK POSITION TEST -------------------- #####

# # print(target[4])
# # print(test_predictions[4])

# thresholds = np.array([0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99])

# # threshold = 0.2

# iterate = np.arange(0,num_test,1)
# s = 0

# for threshold in thresholds:

#     pred_binary = np.zeros(len(test_predictions))
#     iterate = np.arange(0,num_test,1)
    
#     for k in iterate:
#         i = np.where(test_predictions[k] >= threshold)[0]
#         # print(i)
#         # print(len(i))
#         if len(i) == 0:
#             # picks.append(0)
#             pred_binary[k] = 0
#         elif len(i) > 0:
#             # picks.append(1)
#             pred_binary[k] = 1
#             # signals.append(k) # add the index of this sample to the list
    
#     # print('Predictions: ')
#     # print(pred_binary)
    
#     # Convert the target arrays to single ones and zeroes
    
#     targ_binary = np.zeros(len(target))
#     iterate = np.arange(0,num_test,1)
    
#     for k in iterate:
#         i = np.where(target[k] >= threshold)[0]
#         # print(i)
#         # print(len(i))
#         if len(i) == 0:
#             # picks.append(0)
#             targ_binary[k] = 0
#         elif len(i) > 0:
#             # picks.append(1)
#             targ_binary[k] = 1
    
#     # print('Targets: ')
#     # print(targ_binary)
    
#     signals = []
    
#     for i in iterate:
#         pred = pred_binary[i]
#         targ = targ_binary[i]
        
#         # print(pred)
#         # print(targ)
        
#         if pred == 1 and targ == 1: # True positive, there was a signal and it found it
#             signals.append(i) # Grab index from list of events that are correct and have a pick
#         else:
#             pass
    
#     print(signals)
    
#     samples_off_list = []
    
#     for index in signals:
        
#         # Find the peak and then the index where that peak is and compare 
        
#         print('----------------------')
#         print('Signal number: ' + str(index))
        
#         target_max_idx = np.argmax(target[index])
#         print('Target: ' + str(target_max_idx))
        
#         pred_max_idx = np.argmax(test_predictions[index])
#         print('Prediction: ' + str(pred_max_idx))
        
#         samples_off = np.abs(pred_max_idx - target_max_idx)
#         print('Samples off: ' + str(samples_off))
#         samples_off_list.append(samples_off)
        
#     print(samples_off_list)
    
#     # plt.figure()
#     # plt.hist(samples_off_list,bins=128,range=(0,128))
#     # plt.xlim(0,128)
#     # plt.xlabel('Samples off')
#     # plt.title('Prediction vs. target Gaussian peak position. Threshold = ' + str(threshold))
#     # plt.savefig('histogram_thr_' + str(threshold) + '_' + str(num_test) + '.png',format='PNG') 
#     # plt.close() 
    
#     # plt.figure()
#     # plt.boxplot(samples_off_list)
#     # # plt.xlim(0,128)
#     # # plt.xlabel('Samples off')
#     # plt.title('Prediction vs. target Gaussian peak position. Threshold = ' + str(threshold))
#     # plt.savefig('boxplot_thr_' + str(threshold) + '_' + str(num_test) + '.png',format='PNG') 
#     # plt.close() 
    
# #     if s == 0:
# #         samples_off_array_1 = np.asarray(samples_off_list)
# #         s += 1
# #     else:
# #         samples_off_array_2 = np.asarray(samples_off_list)
# #         all_samples_off_array = np.r_[samples_off_array_1,samples_off_array_2]
# #         samples_off_array_1 = all_samples_off_array
        
# # print(all_samples_off_array)
# # print(len(all_samples_off_array))
    
#     # plt.figure()
#     # plt.plot(test_predictions[34])
#     # plt.plot(target[34])
#     # plt.savefig('bad_pick.png',format='PNG')
#     # plt.close()
        
# print(test_predictions.shape)
# print(target.shape)
# print(origdata.shape)        
    
                
                
            
            
            
            
            
            
            
            
            
            
            
            