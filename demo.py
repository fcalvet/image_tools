#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 13:32:10 2018

@authors: fcalvet, fdubost
under GPLv3

Warning: this is for demo purpose only !
"""
import os
import pydicom
import SimpleITK as sitk
import numpy as np
from new_generator import DataGenerator

def Read3Dircad(ID, isGT=False):
    '''
    Function to read files for the 3Dircadb project. Only for liver
    segmentations.
    '''
    if not isGT:
        p=ID
    else:
        p,im=os.path.split(ID)
        p,g=os.path.split(p)
        p=os.path.join(p,'masks/',im)
        
    im = pydicom.read_file(p)
    im = im.pixel_array
    
    return im

def preprocess_step1_LITS(image, mask=False):
    """
    Preprocesses the image (3d or 2d) by performing the following :
    2- Set pixels with hounsfield value great than 1200, to zero.
    3- Clip all hounsfield values to the range [-100, 400]
    4- Normalize values to [0, 1]
    Parameters
    ----------
    image: an input itk image.

   Returns
    -------
    normalized_image: the normalized itk image.
    """
    # image = image[0]
    image = sitk.GetImageFromArray(image)

    # Steps from Christ et al. 2016
    img_slc = sitk.GetArrayFromImage(image)
    if not mask:
        # Threshold Hounsfield units
        img_slc[img_slc > 1200] = 0
        img_slc = np.clip(img_slc, -100, 400)

    # Normalize image
    img_slc = normalize_image(img_slc)

    return np.asarray(img_slc)
    
def normalize_image(img):
    """ 
    Normalize image values to [0,1]
    takes care of always returning floats
    """
    min_, max_ = float(np.min(img)), float(np.max(img))
    if (max_ - min_) != 0:
        return (img - min_) / (max_ - min_)
    else:
        #print("an empty image")
        return 1.*img
   
def reader_wrapper(isGT):
    def read_and_preprocess(ID, mask=False):
        image = Read3Dircad(ID, isGT)
        return preprocess_step1_LITS(image, mask)
    return read_and_preprocess
    

if __name__ == '__main__':    
    ## Parameters
    # creating a single dictionnrary for parameters

    GTisArray = True
    
    params = {}
    params["augmentation"] = [1,1,1]
    params["shape"] = [512, 512]
    batch_size = 15
    
    # Standard data augmentation
    params["random_deform"] = dict()
    params["random_deform"]['width_shift_range'] = 0.1
    params["random_deform"]['height_shift_range'] = 0.1
    params["random_deform"]['rotation_range_alpha'] = 20

    # Add elastic deformations
    params["e_deform_g"] = dict()
    params["e_deform_g"]["points"] = 3
    params["e_deform_g"]["sigma"] = 10
    params["e_deform_p"] = dict()
    params["e_deform_p"]["alpha"] = 10
    params["e_deform_p"]["sigma"] = 3

    #readers
    params['reader_X'] = reader_wrapper(False) 
    params['reader_Y'] = reader_wrapper(True)
    
    # define saveFolder
    params["savefolder"] = 'demo/results/'
    
    # IDs
    IDs = ["demo/images/image_29","demo/images/image_30","demo/images/image_36","demo/images/image_49","demo/images/image_50","demo/images/image_68","demo/images/image_69","demo/images/image_77","demo/images/image_92","demo/images/image_105"]
    print(IDs) 
    
    print("using params: ", params)
    
    #prepare generator
    validation_generator = DataGenerator(IDs,GTisArray,params, batch_size=5)
    validation_generator.plot_samples_IDs(IDs)
    