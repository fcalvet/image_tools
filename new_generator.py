# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 10:25:26 2018

@authors: fcalvet, fdubost
under GPLv3
"""

import numpy as np #to manipulate the arrays
import keras #to use the Sequence class

from pdb import set_trace as bp

import matplotlib
matplotlib.use('Agg')


from matplotlib import pyplot as plt #to create figures examples
import os #to save figures

from image_augmentation import random_transform, deform_grid, deform_pixel
from math import ceil, floor

class DataGenerator(keras.utils.Sequence):
    """
    Generates data for Keras
    Based on keras.utils.Sequence for efficient and safe multiprocessing
    idea from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
    Needs on initialization:
        list_IDs: a list of ID which will be supplied to the ReadFunction to obtain
        params: a dictionnary of parameters, explanation supplied in the README
        batch_size: number of IDs used to generate a batch
        shuffle: if set to True, the list will be shuffled every time an epoch ends
        plotgenerator: number of times a part of a batch will be saved to disk as examples
    """
    def __init__(self, list_IDs, GTisArray, params, batch_size=1, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.params = params
        self.on_epoch_end()
        self.GTisArray = GTisArray

    def __len__(self):
        'Denotes the number of batches per epoch'
        nb = int(np.floor(len(self.list_IDs) / self.batch_size))
        if nb==0:
            raise ValueError('Batch size too large, number of batches per epoch is zero')
        return nb

    def __getitem__(self, index):
        """
        Generate one batch of data by:
            generating a list of indexes which corresponds to a list of ID, 
            use prepare_batch to prepare the batch
        """
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        #read and augment data
        X,Y = self.prepare_batch(list_IDs_temp)
        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
            
    def prepare_batch(self, list_IDs):
        """
        Prepare a bacth of data:
            creating a list of images and masks after having preprocessed (and possibly augmented them)
            saving a few examples to disk if required
        """
        X = np.zeros([len(list_IDs)]+self.params["shape"])
        Y = np.zeros([len(list_IDs)]+self.params["shape"])
        # Generate data
        for i,ID in enumerate(list_IDs):
            #read data
            x = self.params['reader_X'](ID)
            y = self.params['reader_Y'](ID)
            #augment data
            if self.GTisArray:
                X[i],Y[i] = self.imaugment(x,y) 
            else:
                X[i],_ = self.imaugment(x)
                Y[i] = y
        
        #add a channel axis for tensorflow
        X = np.expand_dims(X, len(X.shape))
        if self.GTisArray:
            Y = np.expand_dims(Y, len(Y.shape))

        return X,Y
    
    def imaugment(self, x, y=None):
        """
        Preprocess the tuple (image,mask) and then apply if selected:
            augmentation techniques adapted from Keras ImageDataGenerator
            elastic deformation
        """
        if y is not None and x.shape != y.shape:
            raise ValueError("image and mask should have the same size")
        #print("preprocessed", X)
        if self.params["augmentation"][0] == True:
            #print("augmented")
            x, y = random_transform(x, y, **self.params["random_deform"])
            #print("augmented",X)
        if self.params["augmentation"][1] == True:
            x, y = deform_pixel(x,y, **self.params["e_deform_p"])
        if self.params["augmentation"][2] == True:
            x, y = deform_grid(x,y, **self.params["e_deform_g"])
            #print("deformed!")
        return x,y
        
    def plot_list_images(self,images,name_plot):
        nbr_samples = len(images)
        
        for i in range(nbr_samples):
            plt.subplot(floor(np.sqrt(nbr_samples)),ceil(np.sqrt(nbr_samples)),i+1)
        
            #if the images are 3D, take the middle slice
            if len(images.shape) == 4:
                middle_slice = images[i,:,:,int(images.shape[2]/2)]
            elif len(images.shape) == 3:
                middle_slice = images[i]
            else:
                raise ValueError('images should be either 2D or 3D, images.shape is ' + str(images.shape))
            
            #plot and save
            plt.imshow(middle_slice, cmap='gray', interpolation='none', vmin=0, vmax=1)
            plt.axis('off')   
            plt.savefig(os.path.join(self.params["savefolder"], name_plot+'.png')) 
        
    def plot_samples_IDs(self, IDs):
        X,Y = self.prepare_batch(IDs)
        self.plot_samples_array(np.squeeze(X),np.squeeze(Y))
    
    def plot_samples_array(self,X,Y=None):          
        #define arrays
        X_augmented = np.zeros(X.shape)
        if Y is not None:
            Y_shape = Y.shape
        else:
            Y_shape = len(X)
        Y_augmented = np.zeros(Y_shape)
        
        #augment images - iterate over images
        for i in range(len(X)):
            if Y is not None:
                X_augmented[i], Y_augmented[i] = self.imaugment(X[i], Y[i])
            else:
                X_augmented[i], _ = self.imaugment(X[i], None)
                
        #plot X_augmented
        X_augmented = np.squeeze(X_augmented)
        self.plot_list_images(X_augmented,'samples_data_generator_X')
        
        #plot Y_augmented
        if Y is not None:
            Y_augmented = np.squeeze(Y_augmented)
            self.plot_list_images(Y_augmented,'samples_data_generator_Y')
        