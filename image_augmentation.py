#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 15:36:21 2018

@author: fcalvet
under GPLv3

This file contains 2 augmentation "techniques":
    - elastic deformation with 2 different methods: deform_pixel on a pixel wide basis and deform_grid on a grid basis.
    - random_transform, which provides most of the keras image augmentation techniques.
    
These 3 functions take as input X (the image), Y (an optionnal mask), and a list of keyed parameters.
They also work both on 2D and 3D images.
Elastic deformation is quite slow for 3D images, one could try to tune the order of the splines used for the different interpolations.
"""
import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage import map_coordinates, gaussian_filter


def deform_pixel(X, Y = None, alpha=15, sigma=3):
    """
    Elastic deformation of 2D or 3D images on a pixelwise basis
    X: image
    Y: segmentation of the image
    alpha = scaling factor the deformation
    sigma = smooting factor
    inspired by: https://gist.github.com/fmder/e28813c1e8721830ff9c which inspired imgaug through https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a
    based on [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    First a random displacement field (sampled from a gaussian distribution) is created, 
    it's then convolved with a gaussian standard deviation, σ determines the field : very small if σ is large,  
        like a completely random field if σ is small, 
        looks like elastic deformation with σ the elastic coefficent for values in between.
    Then the field is added to an array of coordinates, which is then mapped to the original image.
    """
    shape = X.shape
    dx = gaussian_filter(np.random.randn(*shape), sigma, mode="constant", cval=0) * alpha #originally with random_state.rand * 2 - 1
    dy = gaussian_filter(np.random.randn(*shape), sigma, mode="constant", cval=0) * alpha
    if len(shape)==2:
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = x+dx, y+dy
    
    elif len(shape)==3:
        dz = gaussian_filter(np.random.randn(*shape), sigma, mode="constant", cval=0) * alpha
        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
        indices = x+dx, y+dy, z+dz
        
    else:
        raise ValueError("can't deform because the image is not either 2D or 3D")
        
    if Y is None:
        return map_coordinates(X, indices, order=3).reshape(shape)
    else:
        return map_coordinates(X, indices, order=3).reshape(shape), map_coordinates(Y, indices, order=0).reshape(shape)
    

def deform_grid (X, Y=None, sigma=25, points=3):
    """
    Elastic deformation of 2D or 3D images on a gridwise basis
    X: image
    Y: segmentation of the image
    sigma = standard deviation of the normal distribution
    points = number of points of the each side of the square grid
    Elastic deformation approach found in
        Ronneberger, Fischer, and Brox, "U-Net: Convolutional Networks for Biomedical
        Image Segmentation" also used in Çiçek et al., "3D U-Net: Learning Dense Volumetric
        Segmentation from Sparse Annotation"
    based on a coarsed displacement grid interpolated to generate displacement for every pixel
    deemed to represent more realistic, biologically explainable deformation of the image
    for each dimension, a value for the displacement is generated on each point of the grid
    then interpolated to give an array of displacement values, which is then added to the corresponding array of coordinates
    the resulting (list of) array of coordinates is mapped to the original image to give the final image
    """
    shape = X.shape
    if len(shape)==2:
        coordinates = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij') #creates the grid of coordinates of the points of the image (an ndim array per dimension)
        xi=np.meshgrid(np.linspace(0,points-1,shape[0]), np.linspace(0,points-1,shape[1]), indexing='ij') #creates the grid of coordinates of the points of the image in the "deformation grid" frame of reference
        grid = [points, points]
        
    elif len(shape)==3:
        coordinates = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij') #creates the grid of coordinates of the points of the image (an ndim array per dimension)
        xi = np.meshgrid(np.linspace(0,points-1,shape[0]), np.linspace(0,points-1,shape[1]), np.linspace(0,points-1,shape[2]), indexing='ij') #creates the grid of coordinates of the points of the image in the "deformation grid" frame of reference
        grid = [points, points, points]
        
    else:
        raise ValueError("can't deform because the image is not either 2D or 3D")

    for i in range(len(shape)): #creates the deformation along each dimension and then add it to the coordinates
        yi=np.random.randn(*grid)*sigma #creating the displacement at the control points
        y = map_coordinates(yi, xi, order=3).reshape(shape)
        #print(y.shape,coordinates[i].shape) #y and coordinates[i] should be of the same shape otherwise the same displacement is applied to every ?row? of points ?
        coordinates[i]=np.add(coordinates[i],y) #adding the displacement
    
    if Y is None:
        return map_coordinates(X, coordinates, order=3).reshape(shape)
    else:
        return map_coordinates(X, coordinates, order=3).reshape(shape), map_coordinates(Y, coordinates, order=0).reshape(shape)


#The folllowing is adapted by fdubost and fcalvet from https://github.com/keras-team/keras/blob/master/keras/preprocessing/image.py under MIT license

#def random_channel_shift(x, intensity, channel_index=0):
#    x = np.rollaxis(x, channel_index, 0)
#    min_x, max_x = np.min(x), np.max(x)
#    channel_images = [np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x, max_x)
#                      for x_channel in x]
#    x = np.stack(channel_images, axis=0)
#    x = np.rollaxis(x, 0, channel_index+1)
#    return x

def apply_transform_3d(x, transform_matrix, channel_index=0, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:3, :3]
    final_offset = transform_matrix[:3, 3]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                      final_offset, order=0, mode=fill_mode, cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x

def apply_transform_2d(x,
                    transform_matrix,
                    channel_axis=0,
                    fill_mode='nearest',
                    cval=0.):
    """Applies the image transformation specified by a matrix.
    # Arguments
        x: 2D numpy array, single image.
        transform_matrix: Numpy array specifying the geometric transformation.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        The transformed version of the input.
    """
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=1,
        mode=fill_mode,
        cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x

def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def transform_matrix_offset_center_3d(matrix, x, y, z):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    o_z = float(z) / 2 + 0.5
    offset_matrix = np.array([[1, 0, 0, o_x], [0, 1, 0, o_y], [0, 0, 1, o_z], [0, 0, 0, 1]])
    reset_matrix =  np.array([[1, 0, 0, -o_x], [0, 1, 0, -o_y], [0, 0, 1, -o_z], [0, 0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def transform_matrix_offset_center_2d(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def random_transform(x, y=None,
                     rotation_range_alpha = 0,
                     rotation_range_beta = 0,
                     rotation_range_gamma = 0,
                     height_shift_range = 0,
                     width_shift_range = 0,
                     depth_shift_range = 0,
                     zoom_range = [1, 1],
                     horizontal_flip = False,
                     vertical_flip = False,
                     z_flip = False
                     ):
    '''Random image tranformation of 2D or 3D images
    x: image
    y: segmentation of the image
    # Arguments
        rotation_range: degrees (0 to 180).
        width_shift_range: fraction of total width.
        height_shift_range: fraction of total height.
        shear_range: shear intensity (shear angle in radians).
        zoom_range: amount of zoom. A sequence of two zoom will be randomly picked
            in the range [a, b].
        channel_shift_range: shift range for each channels.
        horizontal_flip: whether to randomly flip images horizontally.
        vertical_flip: whether to randomly flip images vertically.
        z_flip: whether to randomly flip images alogn the z axis.
    '''
    # x is a single image, so it doesn't have image number at index 0
    print(x.shape)
        
    # use composition of homographies to generate final transform that needs to be applied
    if rotation_range_alpha:
        alpha = np.pi / 180 * np.random.uniform(-rotation_range_alpha, rotation_range_alpha)
    else:
        alpha = 0
        
    if rotation_range_beta:
        beta = np.pi / 180 * np.random.uniform(-rotation_range_beta, rotation_range_beta)
    else:
        beta = 0
            
    if rotation_range_gamma:
        gamma = np.pi / 180 * np.random.uniform(-rotation_range_gamma, rotation_range_gamma)
    else:
        gamma = 0
        
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    
    cb = np.cos(beta)
    sb = np.sin(beta)
    
    cg = np.cos(gamma)
    sg = np.sin(gamma)
    
    img_row_index = 0
    img_col_index = 1
    if len(x.shape) == 2:
        img_z_index = None
        img_channel_index = 2
    elif len(x.shape) == 3:
        img_z_index = 2
        img_channel_index = 3
    else:
        raise ValueError("can't augment because the image is not either 2D or 3D")
        
        
    if height_shift_range:
        tx = np.random.uniform(-height_shift_range, height_shift_range) * x.shape[img_row_index]
    else:
        tx = 0

    if width_shift_range:
        ty = np.random.uniform(-width_shift_range, width_shift_range) * x.shape[img_col_index]
    else:
        ty = 0
        
    if depth_shift_range:
        tz = np.random.uniform(-depth_shift_range, depth_shift_range) * x.shape[img_z_index]
    else:
        tz = 0
        
    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy, zz = 1, 1, 1
    else:
        zx, zy, zz = np.random.uniform(zoom_range[0], zoom_range[1], 3)
    
    if len(x.shape) == 2:
        rotation_matrix = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                                    [np.sin(alpha), np.cos(alpha), 0],
                                    [0, 0, 1]])
        
        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])
        
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        transform_matrix = np.dot(np.dot(rotation_matrix, translation_matrix), zoom_matrix)
        h, w = x.shape[img_row_index], x.shape[img_col_index]
        transform_matrix = transform_matrix_offset_center_2d(transform_matrix, h, w)
        
        apply_transform_gd = apply_transform_2d
        
    else:
        rotation_matrix = np.array([[cb*cg, -cb*sg, sb, 0],
                                [ca*sg+sa*sb*cg, ca*cg-sa*sb*sg, -sa*cb, 0],
                                [sa*sg-ca*sb*cg, sa*cg+ca*sb*sg, ca*cb, 0],
                                [0, 0, 0, 1]])
            
        translation_matrix = np.array([[1, 0, 0, tx],
                                       [0, 1, 0, ty],
                                       [0, 0, 1, tz],
                                       [0, 0, 0, 1]])
        
        zoom_matrix = np.array([[zx, 0, 0, 0],
                                [0, zy, 0, 0],
                                [0, 0, zz, 0],
                                [0, 0, 0, 1]])

        transform_matrix = np.dot(np.dot(rotation_matrix, translation_matrix), zoom_matrix)
        h, w, d = x.shape[img_row_index], x.shape[img_col_index], x.shape[img_z_index]
        transform_matrix = transform_matrix_offset_center_3d(transform_matrix, h, w, d)
        
        apply_transform_gd = apply_transform_3d
#        if self.shear_range:
#            shear = np.random.uniform(-self.shear_range, self.shear_range)
#        else:
#            shear = 0
#        shear_matrix = np.array([[1, -np.sin(shear), 0],
#                                 [0, np.cos(shear), 0],
#                                 [0, 0, 1]])
#        transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)   
    
    if y is None:
        x = np.expand_dims(x, len(x.shape))
        x = apply_transform_gd(x, transform_matrix, img_channel_index)
#        if channel_shift_range != 0:
#            x = random_channel_shift(x, channel_shift_range, img_channel_index)
    
        if horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_index)
    
        if vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_index)
                
        if z_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_z_index)
                
        x = np.squeeze(x)
        return x
    
    else:
        x = np.expand_dims(x, len(x.shape))
        y = np.expand_dims(y, len(y.shape))
        x = apply_transform_gd(x, transform_matrix, img_channel_index)
        y = apply_transform_gd(x, transform_matrix, img_channel_index)
#        if channel_shift_range != 0:
#            x = random_channel_shift(x, channel_shift_range, img_channel_index)
#            y = random_channel_shift(y, channel_shift_range, img_channel_index)
    
        if horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_index)
                y = flip_axis(y, img_row_index)
    
        if vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_index)
                y = flip_axis(y, img_row_index)
                
        if z_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_z_index)
                y = flip_axis(y, img_z_index)
        
        x = np.squeeze(x)
        y = np.squeeze(y)
        return x, y