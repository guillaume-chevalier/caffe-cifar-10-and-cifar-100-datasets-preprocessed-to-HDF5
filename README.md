# Caffe cifar-10 and cifar-100 datasets preprocessed to HDF5 (can be opened in PyCaffe with h5py)
Both deep learning datasets can be imported in python directly with h5py (HDF5 format) or converted with a script to then be imported.

***NOTE: The training set has been shuffled and put in a big single batch, it was originally splitted into 5 files and not shuffled. The test set has not been shuffled and remained intact.***


## How to use

### Method A: Direct import in caffe
You can simply download and unzip the zip file for the desired dataset and ignore the rest of this repository.

### Method B: Download and convert manually with the Python script
You can run my python script for the desired dataset if you already got caffe and pycaffe installed. This way, you will not need to download the whole repository, but only the desired python script. 

As an example, for the cifar-10 dataset, you can run the following commands in a terminal once in a folder where you want to download the datasets:
>wget https://raw.githubusercontent.com/guillaume-chevalier/Caffe-cifar-10-and-cifar-100-datasets-preprocessed-to-HDF5/master/download-and-convert-cifar-10.py

>python download-and-convert-cifar-10.py

#### Not shuffling datasets
If you do not want the datasets to be shuffled, my scripts can be a good starting point for the conversion of the cifar 10 and 100 datasets to the HDF5 caffe format. You may refer to this interesting tutorial to understand better how to do the conversion: https://github.com/BVLC/caffe/blob/master/examples/02-brewing-logreg.ipynb


## Datasets shape info

### cifar-10
Each element are of shape `3*32*32`:
`print(X.shape)` --> `(50000, 3, 32, 32)`

From the Caffe documentation:
"*The conventional blob dimensions for batches of image data are **number N x channel K x height H x width W**.*"

### cifar-100
[TODO]

## Link to original datasets
https://www.cs.toronto.edu/~kriz/cifar.html
