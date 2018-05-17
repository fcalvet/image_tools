# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 10:25:26 2018

@author: fcalvet
under GPLv3
"""

import numpy as np #to manipulate the arrays
import keras #to use the Sequence class

import matplotlib
matplotlib.use('Agg')#to solve the demo ?....


from matplotlib import pyplot as plt #to create figures examples
import os #to save figures

from image_augmentation import random_transform, deform_grid, deform_pixel

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
    def __init__(self, list_IDs, params, batch_size=1, shuffle=True, plotgenerator=0):
        'Initialization'
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.plotgenerator = plotgenerator
        self.params = params
        self.on_epoch_end()

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
        #print(index)
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X,Y = self.prepare_batch(list_IDs_temp)
        #print(list_IDs_temp)
        #print("generated a batch!")
        #print(X.shape,Y.shape)
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
        X = list()#X = np.empty((self.batch_size, *self.dim_in))
        Y = list()
        ReadFunction = self.params['ReadFunction']
        pre_process = self.params['PreProcessing']
        # Generate data
        for ID in list_IDs:
            x = ReadFunction(ID,im_mask="im")
            y = ReadFunction(ID,im_mask="mask")
            #print(ID)
            #print(np.min(x),np.max(x),np.min(y),np.max(y))
            if self.params["to_slice"] == False:
                #print("original", X)
                x = pre_process(x, self.params["shape"])
                y = pre_process(y, self.params["shape"], mask=True)
                x,y = self.imaugment(x,y) 
                #print(np.min(x),np.max(x),np.min(y),np.max(y))  
                print(x.shape)
                X.append(x)
                Y.append(y)
                
            elif self.params["to_slice"] == True and len(x.shape) == 3:
                x,y = self.imaugment(x,y)
                slices_x, slices_y = self.slicer(x,y)
                for x,y in zip(slices_x, slices_y):
                    x = pre_process(x, self.params["shape"])
                    y = pre_process(y, self.params["shape"], mask=True)
                    #print(len(x),x)
                    X.append(x)
                    Y.append(y)
            else:
                raise RuntimeError("asked to slice but the input is not a 3D volume")
        #FIXME: add batch normalization in a way ?
            
        #print(len(X),len(X[0].shape))
        X=np.asarray(X)
        Y=np.asarray(Y)
        #print(X.shape,Y.shape)
        self.save_images(X,Y,list_IDs)
        X = np.expand_dims(X, len(X.shape)) #add a channel axis for tensorflow
        Y = np.expand_dims(Y, len(Y.shape))
        return X,Y
    
    def slicer(self, x, y):
        """
        Slices the 3D volume into slices of thickness corresponding to the last dimension
        """
        sz=self.params["shape"][:-1] #removing the channel dimension
        slices=int(np.floor(x.shape[-1]/sz[-1]))
        slices_x=[]
        slices_y=[]
        for i in range(slices):
            slices_x.append(x[...,i*sz[-1]:(i+1)*sz[-1]])
            slices_y.append(y[...,i*sz[-1]:(i+1)*sz[-1]])
        return slices_x, slices_y
    
    def imaugment(self, X, Y):
        """
        Preprocess the tuple (image,mask) and then apply if selected:
            augmentation techniques adapted from Keras ImageDataGenerator
            elastic deformation
        """
        if X.shape != Y.shape:
            raise ValueError("image and mask should have the same size")
        #print("preprocessed", X)
        if self.params["augmentation"][0] == True:
            #print("augmented")
            X, Y = random_transform(X, Y, **self.params["random_deform"])
            #print("augmented",X)
        if self.params["augmentation"][1] == True:
            X, Y = deform_pixel(X,Y, **self.params["e_deform"])
        if self.params["augmentation"][2] == True:
            X, Y = deform_grid(X,Y, **self.params["e_deform"])
            #print("deformed!")
        return X,Y
    
    def save_images(self, X,Y, list_IDs):
        """
        Save a png to disk (params["savefolder"]) to illustrate the data been generated
        """
        if self.plotgenerator > 0 and len(X[0].shape) == 2:
            '''
            Save augmented images for 2D (will save 10 slices from different patients)
            '''
            nbr_samples = len(X)
            # print("Saving image batch...")
            plt.figure(figsize=(6,11),dpi=200)
            #print(X[1])
            for i in range(min(nbr_samples,10)):
                im=X[i]
                #print(i,im)
                ax = plt.subplot(5, 2, i+1)#fig, ax = subplots(figsize=(18, 2))
                plt.imshow(np.squeeze(im), cmap='gray')#, vmin=0, vmax=1)
                plt.axis('off')
                pltname = list_IDs[i][-27:]
                fz = 5  # Works best after saving
                ax.set_title(pltname, fontsize=fz)
            plt.savefig(os.path.join(self.params["savefolder"], str(self.params["dataset"])+str(self.params["augmentation"])+'generator_' + str(self.plotgenerator) +'_im.png'))
            
            plt.figure(figsize=(6,11),dpi=200)
            # print("Saving mask batch...")
            #print(Y[1])
            for i in range(min(nbr_samples,10)):
                im=Y[i]
                ax = plt.subplot(5, 2, i+1)
                plt.imshow(np.squeeze(im), cmap='gray')#, vmin=0, vmax=1)
                plt.axis('off')
                pltname = list_IDs[i][-27:]
                fz = 5  # Works best after saving
                ax.set_title(pltname, fontsize=fz)
            plt.savefig(os.path.join(self.params["savefolder"], str(self.params["dataset"])+str(self.params["augmentation"])+'generator_' +  str(self.plotgenerator) +'_mask.png'))
            
            
        if self.plotgenerator > 0 and len(X[0].shape) == 3:
            '''
            Save augmented images for 3D (will save 10 slices from a single volume)
            '''
            Xto_print = X[0]
            Yto_print = Y[0]
            print(Xto_print.shape)
            steps = np.linspace(0, Xto_print.shape[2]-1, num=10, dtype=np.int)
            # print("Saving image batch...")
            plt.figure(figsize=(6,11),dpi=200)
            plt.suptitle(list_IDs[0], fontsize=5)
            #print(X[1])
            for i in range(10):
                #print(Xto_print.shape)
                im=Xto_print[:,:,steps[i]]
                #print(i,im)
                ax = plt.subplot(5, 2, i+1)#fig, ax = subplots(figsize=(18, 2))
                plt.imshow(np.squeeze(im), cmap='gray')#, vmin=0, vmax=1)
                plt.axis('off')
                pltname = "slice "+str(steps[i])
                fz = 5  # Works best after saving
                ax.set_title(pltname, fontsize=fz)
            plt.savefig(os.path.join(self.params["savefolder"], str(self.params["dataset"])+str(self.params["augmentation"])+'generator_' + str(self.plotgenerator) +'_im.png'))
            
            plt.figure(figsize=(6,11),dpi=200)
            plt.suptitle(list_IDs[0], fontsize=5)
            # print("Saving mask batch...")
            #print(Y[1])
            for i in range(10):
                im=Yto_print[:,:,steps[i]]
                #print(i,im)
                ax = plt.subplot(5, 2, i+1)#fig, ax = subplots(figsize=(18, 2))
                plt.imshow(np.squeeze(im), cmap='gray')#, vmin=0, vmax=1)
                plt.axis('off')
                pltname = "slice "+str(steps[i])
                fz = 5  # Works best after saving
                ax.set_title(pltname, fontsize=fz)
            plt.savefig(os.path.join(self.params["savefolder"], str(self.params["dataset"])+str(self.params["augmentation"])+'generator_' +  str(self.plotgenerator) +'_mask.png'))
        self.plotgenerator-=1