# image_tools
This repository contains a generator for Keras (using a channel last configuration, eg with TensorFlow backend), capable of handling 2D and 3D arrays, and augmenting the data using affine transforms and elastics deformations while being based on keras.utils.Sequence for efficient and safe multiprocessing

## the params dictionnary
is used to supply the generator with a lot of useful informations regarding the data to be generated. The following list explains the parameters useful for data generation:
+ augmentation: a tuple of booleans value selecting the augmentations techniques to apply in the following order: random_transform, deform_grid, deform_pixel
+ dataset: the name of the dataset used
+ random_deform: a dictionnary containing the parameters to be passed to random_transform
+ e_deform: a dictionnary containing the parameters to be passed to the chosen elastic deformation function
+ shape: the input shape of the network, ie the shape of the images including the channel
+ ReadFunction: a function to read the images, which given an ID returns a npy array
+ PreProcessing: a preprocessing function
+ to_slice: if set to True, slices 3D volumes into slices of thickness corresponding to last dimension of shape (of course excluding the channel)
+ savefolder: the path of the folder to which the examples are going to be saved

