# Caffe cifar-10 and cifar-100 datasets preprocessed to HDF5 (can be opened in PyCaffe with h5py)
Both deep learning datasets can be imported in python directly with h5py (HDF5 format) once downloaded and converted by the script.

***NOTE: For each dataset (separately), the training set has been shuffled and put in a single batch, it was originally split into 5 files and not shuffled. Test sets has also been shuffled, but remained separated from the training sets.***


## Usage

If you got the requirements or got caffe installed, you can simply do this, with the cifar-10 dataset as an example:

>wget https://raw.githubusercontent.com/guillaume-chevalier/Caffe-cifar-10-and-cifar-100-datasets-preprocessed-to-HDF5/master/download-and-convert-cifar-10.py

>python download-and-convert-cifar-10.py

### Not shuffling datasets
If you do not want the datasets to be shuffled nor merged, my scripts can be a good starting point for the complete conversion of the cifar 10 and 100 datasets to the HDF5 caffe format. You may refer to this interesting tutorial to understand better how to do the conversion: https://github.com/BVLC/caffe/blob/master/examples/02-brewing-logreg.ipynb


## Datasets shape info

### cifar-10
Each element are of shape `3*32*32`:
`print(X.shape)` --> `(50000, 3, 32, 32)`

From the Caffe documentation:
"*The conventional blob dimensions for batches of image data are **number N x channel K x height H x width W**.*"

The labels are integer (not length 10 vectors with one-hot encoding). As an example, caffe does the encoding itself with the `SoftmaxWithLoss` layer when the previous layer is an `InnerProduct` with setting `num_output=10`.

### cifar-100
The data has the same shape than in the cifar-10 dataset. However, the dataset has two labels per image: while building caffe layers, the `label` data will not exist in the HDF5 format, it will be `label_coarse` and `label_fine`. Please refer to the original datasets' page for info about the number of classes in each coarse and fine labels. Also note that I did NOT tested the cifar-100 dataset, only the cifar-10.

## Link to original datasets
https://www.cs.toronto.edu/~kriz/cifar.html
