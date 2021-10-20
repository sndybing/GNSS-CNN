#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 13:08:55 2020

Unet models

@author: amt
"""

import tensorflow as tf
import numpy as np
from scipy import signal

def make_large_unet(fac, sr, ncomps = 3, winsize = 128):
    # BUILD THE MODEL
    # These models start with an input
    if ncomps == 1:
        input_layer = tf.keras.layers.Input(shape = (winsize,2)) # 1 Channel seismic data
    elif ncomps == 3:
        # input_layer = tf.keras.layers.Input(shape = (winsize,6)) # 1 Channel seismic data
        input_layer = tf.keras.layers.Input(shape = (winsize,3)) # 3 channel GNSS data
    
    # First block
    level1 = tf.keras.layers.Conv1D(int(fac*32),21,activation = 'relu',padding = 'same')(input_layer) # N filters, Filter Size, Stride, padding
    network = tf.keras.layers.MaxPooling1D()(level1) #32
    
    # Second Block
    level2 = tf.keras.layers.Conv1D(int(fac*64),15,activation = 'relu',padding = 'same')(network)
    network = tf.keras.layers.MaxPooling1D()(level2) #16
    #network = tf.keras.layers.ZeroPadding1D((0,1))(network)
    
    #Next Block
    level3 = tf.keras.layers.Conv1D(int(fac*128),11,activation = 'relu',padding = 'same')(network)
    #network = tf.keras.layers.BatchNormalization()(level3)
    network = tf.keras.layers.MaxPooling1D()(level3) #8
    
    #Base of Network
    network = tf.keras.layers.Flatten()(network)
    base_level = tf.keras.layers.Dense(16,activation = 'relu')(network)
    
    #network = tf.keras.layers.BatchNormalization()(base_level)
    network = tf.keras.layers.Reshape((16,1))(base_level)
    
    #Upsample and add skip connections
    network = tf.keras.layers.Conv1D(int(fac*128),11,activation = 'relu',padding = 'same')(network) # N filters, Filter Size, Stride, padding
    network = tf.keras.layers.UpSampling1D()(network)
    
    level3 = tf.keras.layers.Concatenate()([network,level3]) # N filters, Filter Size, Stride, padding
    # level3 = tf.keras.layers.Lambda( lambda x: x[:,:-1,:])(level3)
    
    #Upsample and add skip connections
    network = tf.keras.layers.Conv1D(int(fac*64),15,activation = 'relu',padding = 'same')(level3) # N filters, Filter Size, Stride, padding
    network = tf.keras.layers.UpSampling1D()(network)
    level2 = tf.keras.layers.Concatenate()([network,level2]) # N filters, Filter Size, Stride, padding
    
    #Upsample and add skip connections
    network = tf.keras.layers.Conv1D(int(fac*32),21,activation = 'relu',padding = 'same')(level2) # N filters, Filter Size, Stride, padding
    network = tf.keras.layers.UpSampling1D()(network)
    level1 = tf.keras.layers.Concatenate()([network,level1]) # N filters, Filter Size, Stride, padding
    
    #End of network
    network = tf.keras.layers.Conv1D(1,21,activation = 'sigmoid',padding = 'same')(level1) # N filters, Filter Size, Stride, padding
    output = tf.keras.layers.Flatten()(network) # N filters, Filter Size, Stride, padding
    
    model = tf.keras.models.Model(input_layer,output)
    opt  =  tf.keras.optimizers.Adam(lr = 0.0001)
    model.compile(loss = 'binary_crossentropy',optimizer = opt,metrics = ['accuracy'])
    model.summary()
    
    return model

def make_large_unet_drop(fac,sr,ncomps = 3,winsize = 128):
    
    if ncomps == 1:
        input_layer = tf.keras.layers.Input(shape = (winsize,2)) # 1 Channel seismic data
    elif ncomps == 3:
        # input_layer = tf.keras.layers.Input(shape = (winsize,6)) # 1 Channel seismic data
        input_layer = tf.keras.layers.Input(shape = (winsize,3)) # 3 channel GNSS data
    
    # First block
    level1 = tf.keras.layers.Conv1D(int(fac*32),21,activation = 'relu',padding = 'same')(input_layer) # N filters, Filter Size, Stride, padding
    network = tf.keras.layers.MaxPooling1D()(level1) #32
    
    # Second Block
    level2 = tf.keras.layers.Conv1D(int(fac*64),15,activation = 'relu',padding = 'same')(network)
    network = tf.keras.layers.MaxPooling1D()(level2) #16
    #network = tf.keras.layers.ZeroPadding1D((0,1))(network)
    
    #Next Block
    level3 = tf.keras.layers.Conv1D(int(fac*128),11,activation = 'relu',padding = 'same')(network)
    #network = tf.keras.layers.BatchNormalization()(level3)
    network = tf.keras.layers.MaxPooling1D()(level3) #8
    
    #Base of Network
    network = tf.keras.layers.Flatten()(network)
    base_level = tf.keras.layers.Dense(16,activation = 'relu')(network)
    
    #network = tf.keras.layers.BatchNormalization()(base_level)
    network = tf.keras.layers.Reshape((16,1))(base_level)
    
    #Upsample and add skip connections
    network = tf.keras.layers.Conv1D(int(fac*128),11,activation = 'relu',padding = 'same')(network) # N filters, Filter Size, Stride, padding
    network = tf.keras.layers.UpSampling1D()(network)
    
    level3 = tf.keras.layers.Concatenate()([network,level3]) # N filters, Filter Size, Stride, padding
    #level3 = tf.keras.layers.Lambda( lambda x: x[:,:-1,:])(level3)
    
    #Upsample and add skip connections
    network = tf.keras.layers.Conv1D(int(fac*64),15,activation = 'relu',padding = 'same')(level3) # N filters, Filter Size, Stride, padding
    network = tf.keras.layers.UpSampling1D()(network)
    level2 = tf.keras.layers.Concatenate()([network,level2]) # N filters, Filter Size, Stride, padding
    
    #Upsample and add skip connections
    network = tf.keras.layers.Conv1D(int(fac*32),21,activation = 'relu',padding = 'same')(level2) # N filters, Filter Size, Stride, padding
    network = tf.keras.layers.UpSampling1D()(network)
    level1 = tf.keras.layers.Concatenate()([network,level1]) # N filters, Filter Size, Stride, padding
    
    #End of network
    network = tf.keras.layers.Dropout(.2)(level1)
    network = tf.keras.layers.Conv1D(1,21,activation = 'sigmoid',padding = 'same')(level1) # N filters, Filter Size, Stride, padding
    output = tf.keras.layers.Flatten()(network) # N filters, Filter Size, Stride, padding
    
    model = tf.keras.models.Model(input_layer,output)
    opt  =  tf.keras.optimizers.Adam(lr = 0.0001)
    model.compile(loss = 'binary_crossentropy',optimizer = opt,metrics = ['accuracy'])
    
    tf.keras.utils.plot_model(model, to_file = 'gnss_cnn_picker.png', show_shapes = True, show_layer_names = True)
    
    model.summary()
    
    return model

def my_3comp_data_generator(batch_size, x_data, n_data, meta_data, nan_array, sig_inds, noise_inds, sr, std, valid = False, nlen = 256, winsize = 128):
   
    epsilon = 1e-6
    
    while True:
        
        ### ----- Defining the batch information ----- ###
        
        start_of_data_batch = np.random.choice(len(sig_inds) - batch_size//2) # Randomly selecting a starting index for the earthquake data batch (which is half the length of the full batch size)
        start_of_noise_batch1 = np.random.choice(len(noise_inds) - batch_size//2) # Randomly selecting a starting index for the FIRST noise batch (to be added to earthquake data)
        start_of_noise_batch2 = np.random.choice(len(noise_inds) - batch_size//2) # Randomly selecting a starting index for the SECOND noise batch (noise-only to be concatenated) (the other half of the full batch)
        
        ### ----- Getting indices of data from the batch info ----- ###
        
        datainds = sig_inds[start_of_data_batch:start_of_data_batch + batch_size//2] # Getting range of indicies from earthquake data
        noiseinds1 = noise_inds[start_of_noise_batch1:start_of_noise_batch1 + batch_size//2] # Getting range of indicies from first noise batch
        noiseinds2 = noise_inds[start_of_noise_batch2:start_of_noise_batch2 + batch_size//2] # Getting range of indicies from second noise batch
        
        ### ----- Making the full batches of data and targets ----- ###
        
        # DATA: Grab earthquake batch, add data and first noise batch to make noisy data
        comp1 = np.concatenate((x_data[datainds, :nlen] + n_data[noiseinds1, :nlen], n_data[noiseinds2, :nlen])) # Concatenating noisy data and just noise to make full batch      
        comp2 = np.concatenate((x_data[datainds, nlen:2*nlen] + n_data[noiseinds1, nlen:2*nlen], n_data[noiseinds2, nlen:2*nlen]))
        comp3 = np.concatenate((x_data[datainds, 2*nlen:] + n_data[noiseinds1, 2*nlen:], n_data[noiseinds2, 2*nlen:]))
        
        # METADATA: Using the same indices as with the earthquakes, make an array of the metadata that contains the correct info about each earthquake. Concatenating NaNs as if it's noise to make batch length match
        metacomp = np.concatenate((meta_data[datainds, :], nan_array[noiseinds2, :]))
        
        # TARGETS
        target = np.concatenate((np.ones_like(datainds), np.zeros_like(noiseinds2))) # Making the target data vector for the full batch - ones and zeros for signal versus noise
        batch_target = np.zeros((batch_size, nlen)) # Making structure to hold target functions

        ### ----- Second shuffle ----- ###
        
        inds = np.arange(batch_size) # New indices the size of the batch
        np.random.shuffle(inds) # Shuffle the indices 
        
        # Apply shuffle 
        
        comp1 = comp1[inds, :] # Shuffle the noisy data (for training) with the new inds array 
        comp2 = comp2[inds, :]
        comp3 = comp3[inds, :]
        
        metacomp = metacomp[inds, :] # Shuffle the metadata array the SAME WAY to keep track of earthquake info
        
        target = target[inds] # Shuffle the target array so the ones and zeros still match
        
        ### ----- Making the picks into Gaussians ----- ###
        
        # Adds a Gaussian when the batch_target is one, indicating an earthquake
        for ii, targ in enumerate(target):

            if targ == 0:
                batch_target[ii, :] = np.zeros((1, nlen)) # batch_target was all zeroes before. If the target (whether it's signal or not) is zero, leave batch_target as zero
                # print(batch_target)
                
            elif targ == 1:
                batch_target[ii, :] = signal.gaussian(nlen, std = int(std*sr)) # If the target is one, add a Gaussian to the batch target centered in the middle to match the pick location
        
        ### ----- Shifting the pick location so it's not always centered ----- ###
        
        # Making an array of possible offsets - I have nlen s of data and want to have winsize s windows in which the arrival can occur anywhere
        time_offset = np.random.uniform(0, winsize, size = batch_size) # Random numbers for offset between 0 and 128 seconds, decimals okay

        # Initialize arrays to hold shifted data and targets
        new_batch = np.zeros((batch_size, int(winsize*sr), 3)) # 3D array - 5000 samples of 3 components, each 128 seconds long
        new_batch_target = np.zeros((batch_size, int(winsize*sr))) # 2D array - 5000 samples each 128 seconds long for the target (only need to know for one component) 
            
        # This loop shifts data and targets and stores results
        for ii, offset in enumerate(time_offset):
            
            # New timespan of data
            bin_offset = int(offset*sr) # Takes offset time and makes it an integer times the sampling rate (which is just 1 Hz here)
            start_bin = bin_offset # Start of new 128 second data
            end_bin = start_bin+int(winsize*sr) # End of new 128 second data

            # Grabbing out the new batch of data using the shifted timespans
            new_batch[ii, :, 0] = comp1[ii, start_bin:end_bin] # New N component - row in counter, all 128 columns, 0 for first component. Grabs 128s section from comp1
            new_batch[ii, :, 1] = comp2[ii, start_bin:end_bin]
            new_batch[ii, :, 2] = comp3[ii, start_bin:end_bin]
            
            new_batch_target[ii, :] = batch_target[ii, start_bin:end_bin] # New target
    
        ### ----- Creating batch_out ----- ###
        
        # FOR SYDNEY'S GNSS PROJECT
        
        batch_out = new_batch
    
        # # FOR SEISMIC DATA: does feature log
        
        # new_batch_sign = np.sign(new_batch) # Gets the signs of the numbers in new_batch
        # new_batch_val = np.log(np.abs(new_batch) + epsilon) # Takes the absolute value of the data in the new_batch, adds 1e-6, then takes the log
        
        # batch_out = []
        
        # print(new_batch_sign.shape)
        # print(new_batch_val.shape)
        
        # for ii in range(new_batch_target.shape[0]): # Counting through the rows
            
        #     stack = np.hstack([new_batch_val[ii,:,0].reshape(-1,1), new_batch_sign[ii,:,0].reshape(-1,1), # Stacks the component columns back side by side 
        #                        new_batch_val[ii,:,1].reshape(-1,1), new_batch_sign[ii,:,1].reshape(-1,1),
        #                        new_batch_val[ii,:,2].reshape(-1,1), new_batch_sign[ii,:,2].reshape(-1,1)]) 
            
        #     # print(stack.shape) # Stack is 128 rows by 6 columns. 128 rows for each of the samples, 6 numbers for the sign and val of each component
            
        #     batch_out.append(stack)
        
        # batch_out = np.array(batch_out)
        
        if valid: # If valid = True, we are testing and we want the metadata and original data for plotting/analysis
            yield(batch_out, new_batch_target, new_batch, metacomp)
            
        else: # If valid = False, we are training and only want to give the generator the batch_out and the targets
            yield(batch_out, new_batch_target)
            
def main():
    make_large_unet(1,1,ncomps = 3)

if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    