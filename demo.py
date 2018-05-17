#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 13:32:10 2018

@author: fcalvet
under GPLv3

Warning: this is for demo purpose only !
"""
import os
import pydicom
import SimpleITK as sitk
import numpy as np
from new_generator import DataGenerator

def Read3Dircad(ID, im_mask="im"):
    '''
    Function to read files for the 3Dircadb project. Only for liver
    segmentations.
    '''
    if im_mask=="im":
        p=ID
    elif im_mask=="mask":
        p,im=os.path.split(ID)
        p,g=os.path.split(p)
        p=os.path.join(p,'masks/',im)
    #print("reading:",im_mask," at ", p)
    im = pydicom.read_file(p)
    im = im.pixel_array
    return im

def preprocess_step1_LITS(image, sz, mask=False):
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

def normalize_image(img, mask=True):
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


if __name__ == '__main__':    
    ## Parameters
    # creating a single dictionnrary for parameters
    
    params = {}
    params["dataset"] = "3Dircad"
    params["augmentation"] = [1,0,1]
    augmentparams = dict()
    params["random_deform"] = dict()
    params["e_deform"] = dict()
    params["to_slice"] = False
    params["shape"] = [512, 512,1]
    batch_size = 15
    # Standard data augmentation
    params["random_deform"]['width_shift_range'] = 0.1
    params["random_deform"]['height_shift_range'] = 0.1
    params["random_deform"]['rotation_range_alpha'] = 20

    # Add elastic deformations
    params["e_deform"]["points"] = 3
    #params["e_deform"]["alpha"] = 5#0.05*shape[0]
    params["e_deform"]["sigma"] = 10#(5, 10)

    params['ReadFunction'] = Read3Dircad
    params['PreProcessing'] = preprocess_step1_LITS
    # define saveFolder
    params["savefolder"] = 'demo/results/'
    # Datasets
    liste_id = ["demo/images/image_29","demo/images/image_30","demo/images/image_36","demo/images/image_49","demo/images/image_50","demo/images/image_68","demo/images/image_69","demo/images/image_77","demo/images/image_92","demo/images/image_105"]
    print(liste_id)    
    print("using params: ", params)
    validation_generator = DataGenerator(liste_id, params, plotgenerator=1, batch_size=5)
    X_valid, Y_valid = validation_generator.prepare_batch(liste_id) 