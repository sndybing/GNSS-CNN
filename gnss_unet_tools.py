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

def make_large_unet(fac, sr, ncomps=3, winsize=128):
    # BUILD THE MODEL
    # These models start with an input
    if ncomps==1:
        input_layer=tf.keras.layers.Input(shape=(winsize,2)) # 1 Channel seismic data
    elif ncomps==3:
        input_layer=tf.keras.layers.Input(shape=(winsize,6)) # 1 Channel seismic data
    
    # First block
    level1=tf.keras.layers.Conv1D(int(fac*32),21,activation='relu',padding='same')(input_layer) # N filters, Filter Size, Stride, padding
    network=tf.keras.layers.MaxPooling1D()(level1) #32
    
    # Second Block
    level2=tf.keras.layers.Conv1D(int(fac*64),15,activation='relu',padding='same')(network)
    network=tf.keras.layers.MaxPooling1D()(level2) #16
    #network=tf.keras.layers.ZeroPadding1D((0,1))(network)
    
    #Next Block
    level3=tf.keras.layers.Conv1D(int(fac*128),11,activation='relu',padding='same')(network)
    #network=tf.keras.layers.BatchNormalization()(level3)
    network=tf.keras.layers.MaxPooling1D()(level3) #8
    
    #Base of Network
    network=tf.keras.layers.Flatten()(network)
    base_level=tf.keras.layers.Dense(16,activation='relu')(network)
    
    #network=tf.keras.layers.BatchNormalization()(base_level)
    network=tf.keras.layers.Reshape((16,1))(base_level)
    
    #Upsample and add skip connections
    network=tf.keras.layers.Conv1D(int(fac*128),11,activation='relu',padding='same')(network) # N filters, Filter Size, Stride, padding
    network=tf.keras.layers.UpSampling1D()(network)
    
    level3=tf.keras.layers.Concatenate()([network,level3]) # N filters, Filter Size, Stride, padding
    # level3=tf.keras.layers.Lambda( lambda x: x[:,:-1,:])(level3)
    
    #Upsample and add skip connections
    network=tf.keras.layers.Conv1D(int(fac*64),15,activation='relu',padding='same')(level3) # N filters, Filter Size, Stride, padding
    network=tf.keras.layers.UpSampling1D()(network)
    level2=tf.keras.layers.Concatenate()([network,level2]) # N filters, Filter Size, Stride, padding
    
    #Upsample and add skip connections
    network=tf.keras.layers.Conv1D(int(fac*32),21,activation='relu',padding='same')(level2) # N filters, Filter Size, Stride, padding
    network=tf.keras.layers.UpSampling1D()(network)
    level1=tf.keras.layers.Concatenate()([network,level1]) # N filters, Filter Size, Stride, padding
    
    #End of network
    network=tf.keras.layers.Conv1D(1,21,activation='sigmoid',padding='same')(level1) # N filters, Filter Size, Stride, padding
    output=tf.keras.layers.Flatten()(network) # N filters, Filter Size, Stride, padding
    
    model=tf.keras.models.Model(input_layer,output)
    opt = tf.keras.optimizers.Adam(lr=0.0001)
    model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
    model.summary()
    
    return model

def make_large_unet_drop(fac,sr,ncomps=3,winsize=128):
    
    if ncomps==1:
        input_layer=tf.keras.layers.Input(shape=(winsize,2)) # 1 Channel seismic data
    elif ncomps==3:
        input_layer=tf.keras.layers.Input(shape=(winsize,6)) # 1 Channel seismic data
    
    # First block
    level1=tf.keras.layers.Conv1D(int(fac*32),21,activation='relu',padding='same')(input_layer) # N filters, Filter Size, Stride, padding
    network=tf.keras.layers.MaxPooling1D()(level1) #32
    
    # Second Block
    level2=tf.keras.layers.Conv1D(int(fac*64),15,activation='relu',padding='same')(network)
    network=tf.keras.layers.MaxPooling1D()(level2) #16
    #network=tf.keras.layers.ZeroPadding1D((0,1))(network)
    
    #Next Block
    level3=tf.keras.layers.Conv1D(int(fac*128),11,activation='relu',padding='same')(network)
    #network=tf.keras.layers.BatchNormalization()(level3)
    network=tf.keras.layers.MaxPooling1D()(level3) #8
    
    #Base of Network
    network=tf.keras.layers.Flatten()(network)
    base_level=tf.keras.layers.Dense(16,activation='relu')(network)
    
    #network=tf.keras.layers.BatchNormalization()(base_level)
    network=tf.keras.layers.Reshape((16,1))(base_level)
    
    #Upsample and add skip connections
    network=tf.keras.layers.Conv1D(int(fac*128),11,activation='relu',padding='same')(network) # N filters, Filter Size, Stride, padding
    network=tf.keras.layers.UpSampling1D()(network)
    
    level3=tf.keras.layers.Concatenate()([network,level3]) # N filters, Filter Size, Stride, padding
    #level3=tf.keras.layers.Lambda( lambda x: x[:,:-1,:])(level3)
    
    #Upsample and add skip connections
    network=tf.keras.layers.Conv1D(int(fac*64),15,activation='relu',padding='same')(level3) # N filters, Filter Size, Stride, padding
    network=tf.keras.layers.UpSampling1D()(network)
    level2=tf.keras.layers.Concatenate()([network,level2]) # N filters, Filter Size, Stride, padding
    
    #Upsample and add skip connections
    network=tf.keras.layers.Conv1D(int(fac*32),21,activation='relu',padding='same')(level2) # N filters, Filter Size, Stride, padding
    network=tf.keras.layers.UpSampling1D()(network)
    level1=tf.keras.layers.Concatenate()([network,level1]) # N filters, Filter Size, Stride, padding
    
    #End of network
    network=tf.keras.layers.Dropout(.2)(level1)
    network=tf.keras.layers.Conv1D(1,21,activation='sigmoid',padding='same')(level1) # N filters, Filter Size, Stride, padding
    output=tf.keras.layers.Flatten()(network) # N filters, Filter Size, Stride, padding
    
    model=tf.keras.models.Model(input_layer,output)
    opt = tf.keras.optimizers.Adam(lr=0.0001)
    model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
    
    tf.keras.utils.plot_model(model, to_file='gnss_cnn_picker.png', show_shapes=True, show_layer_names=True)
    
    model.summary()
    
    return model

def my_3comp_data_generator(batch_size,x_data,n_data,sig_inds,noise_inds,sr,std,valid=False,nlen=256,winsize=128):
    epsilon=1e-6
    while True:
        # randomly select a starting index for the data batch
        start_of_data_batch=np.random.choice(len(sig_inds)-batch_size//2)
        # randomly select a starting index for the noise batch
        start_of_noise_batch1=np.random.choice(len(noise_inds)-batch_size//2)
        # randomly select a starting index for the noise batch
        start_of_noise_batch2=np.random.choice(len(noise_inds)-batch_size//2)
        # get range of indicies from data
        datainds=sig_inds[start_of_data_batch:start_of_data_batch+batch_size//2]
        # get range of indicies from noise
        noiseinds1=noise_inds[start_of_noise_batch1:start_of_noise_batch1+batch_size//2]
        # get range of indicies from noise
        noiseinds2=noise_inds[start_of_noise_batch2:start_of_noise_batch2+batch_size//2]
        # grab batch, add data and noise to make noisy data and keep some noise separate for the zero target batch
        comp1=np.concatenate((x_data[datainds,:nlen]+n_data[noiseinds1,:nlen],n_data[noiseinds2,:nlen]))
        comp2=np.concatenate((x_data[datainds,nlen:2*nlen]+n_data[noiseinds1,nlen:2*nlen],n_data[noiseinds2,nlen:2*nlen]))
        comp3=np.concatenate((x_data[datainds,2*nlen:]+n_data[noiseinds1,2*nlen:],n_data[noiseinds2,2*nlen:]))
        # make target data vector for batch
        target=np.concatenate((np.ones_like(datainds),np.zeros_like(noiseinds2)))
        # make structure to hold target functions
        batch_target=np.zeros((batch_size,nlen))
        # shuffle things (not sure if this is needed)
        inds=np.arange(batch_size)
        np.random.shuffle(inds)
        comp1=comp1[inds,:]
        comp2=comp2[inds,:]
        comp3=comp3[inds,:]
        target=target[inds]
        # this just makes a nonzero value where the pick is
        for ii, targ in enumerate(target):
            #print(ii,targ)
            if targ==0:
                batch_target[ii,:]=np.zeros((1,nlen))
            elif targ==1:
                batch_target[ii,:]=signal.gaussian(nlen,std=int(std*sr))
        # I have nlen s of data and want to have winsize s windows in which the arrival can occur anywhere
        time_offset=np.random.uniform(0,winsize,size=batch_size)
        # initialize arrays to hold shifted data
        new_batch=np.zeros((batch_size,int(winsize*sr),3))
        new_batch_target=np.zeros((batch_size,int(winsize*sr)))        
        # this loop shifts data and targets and stores results
        for ii,offset in enumerate(time_offset):
            bin_offset=int(offset*sr) #HZ sampling Frequency
            start_bin=bin_offset 
            end_bin=start_bin+int(winsize*sr) 
            new_batch[ii,:,0]=comp1[ii,start_bin:end_bin]
            new_batch[ii,:,1]=comp2[ii,start_bin:end_bin]
            new_batch[ii,:,2]=comp3[ii,start_bin:end_bin]
            new_batch_target[ii,:]=batch_target[ii,start_bin:end_bin]
        # does feature log
        new_batch_sign=np.sign(new_batch)
        new_batch_val=np.log(np.abs(new_batch)+epsilon)
        batch_out=[]
        for ii in range(new_batch_target.shape[0]):
            batch_out.append(np.hstack( [new_batch_val[ii,:,0].reshape(-1,1), new_batch_sign[ii,:,0].reshape(-1,1), 
                                          new_batch_val[ii,:,1].reshape(-1,1), new_batch_sign[ii,:,1].reshape(-1,1),
                                          new_batch_val[ii,:,2].reshape(-1,1), new_batch_sign[ii,:,2].reshape(-1,1)] ) )
        batch_out=np.array(batch_out)
        if valid:
            yield(batch_out,new_batch_target,new_batch)
        else:
            yield(batch_out,new_batch_target)
            
def main():
    make_large_unet(1,1,ncomps=3)

if __name__ == "__main__":
    main()