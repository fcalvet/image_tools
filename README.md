This repository contains a generator for Keras (using a channel last configuration, eg with TensorFlow backend), capable of handling 2D and 3D arrays, and augmenting the data using affine transforms and elastics deformations while being based on keras.utils.Sequence for efficient and safe multiprocessing.


The requirements for the generator and augmentation are the following: numpy, matplotlib, scipy, and keras (only for the generator part)

The demo requires on top: SimpleITK and pydicom

### Demo images
A few examples are present in the demo folder extracted from the 3D-IRCADb 01 dataset (https://www.ircad.fr/research/3d-ircadb-01/)

#### One with deform_grid
    'augmentation': [1, 0, 1], 'random_deform': {'rotation_range_alpha': 20, 'width_shift_range': 0.1, 'height_shift_range': 0.1}, 'e_deform': {'points': 3, 'sigma': 10}
![CT scans](demo/results/3Dircad[1%2C 0%2C 1]generator_1_im.png)
![masks of the scans](demo/results/3Dircad[1%2C 0%2C 1]generator_1_mask.png)

#### One with deform_pixel
    'augmentation': [1, 1, 0], 'random_deform': {'height_shift_range': 0.1, 'width_shift_range': 0.1, 'rotation_range_alpha': 20}, 'e_deform': {'sigma': 3, 'alpha': 10} 
![CT scans](demo/results/3Dircad[1%2C 1%2C 0]generator_1_im.png)
![masks of the scans](demo/results/3Dircad[1%2C 1%2C 0]generator_1_mask.png)


### Files
+ new_generator.py contains the generator class (it's built for generating images and their corresponding segmentation mask, but can be adapted to only work on the images without mask)
+ image_augmentation.py contains the image augmentation functions (which can act on either image, or image,mask )

+ demo.py contains a demo script loading dicom files and deforming them
+ demo contains demo images and output from the demo script

### the params dictionnary
is used to supply the generator with a lot of useful informations regarding the data to be generated. The following list explains the parameters useful for data generation:
+ augmentation: a tuple of booleans values selecting the augmentations techniques to apply in the following order: random_transform, deform_grid, deform_pixel
+ random_deform: a dictionnary containing the parameters to be passed to random_transform
+ e_deform: a dictionnary containing the parameters to be passed to the chosen elastic deformation function
+ shape: the input shape of the network, ie the shape of the images including the channel
+ ReadFunction: a function to read the images, which given an ID and if it should query the corresponding image or mask, returns a npy array
+ PreProcessing: a preprocessing function, which takes as input the image, the size of the desired output image and, optionnally if the image is a mask or not (in order to apply different preprocessing) and return the preprocessed image
+ to_slice: if set to True, slices 3D volumes into slices of thickness corresponding to last dimension of shape (of course excluding the channel) /!\ Experimental behaviour !
+ savefolder: the path of the folder to which the examples are going to be saved
+ dataset: the name of the dataset used

### the augmentation techniques

The file contains 2 augmentation "techniques":
    - elastic deformation with 2 different methods: deform_pixel on a pixel wide basis and deform_grid on a grid basis.
    - random_transform, which provides most of the keras image augmentation techniques.
    
These 3 functions take as input X (the image), Y (an optionnal mask), and some keyed parameters.
They also work both on 2D and 3D images.
They depend on numpy and scipy.ndimage
Elastic deformation is quite slow for 3D images, one could try to tune the order of the splines used for the different interpolations.

#### random_transform

The following parameters can be supplied in the random_transform dictionnary:
+ rotation_range_alpha = angle in degrees (0 to 180), produces a range in which to uniformly pick the rotation.
+ rotation_range_beta = ...
+ rotation_range_gamma = ...
+ width_shift_range: fraction of total width, produces a range in which to uniformly pick the shift.
+ height_shift_range: fraction of total height, produces a range in which to uniformly pick the shift.
+ depth_shift_range: fraction of total depth, produces a range in which to uniformly pick the shift.
+ zoom_range = factor of zoom. A zoom factor per axis will be randomly picked in the range [a, b].
+ horizontal_flip: boolean, whether to randomly flip images horizontally.
+ vertical_flip: boolean, whether to randomly flip images vertically.
+ z_flip: boolean, whether to randomly flip images along the z axis.

#### e_deform
for grid based deformation:
+ sigma = standard deviation of the normal distribution
+ points = number of points of the each side of the square grid

for pixel based deformation:
+ alpha = scaling factor the deformation
+ sigma = smooting factor
