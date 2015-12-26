import copy
import os
from subprocess import call

import numpy as np
import sklearn
import sklearn.linear_model

import h5py

print("")

print("Downloading...")
if not os.path.exists("cifar-100-python.tar.gz"):
    call(
        "wget http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
        shell=True
    )
    print("Downloading done.\n")
else:
    print("Dataset already downloaded. Did not download twice.\n")


print("Extracting...")
cifar_python_directory = os.path.abspath("cifar-100-python")
if not os.path.exists(cifar_python_directory):
    call(
        "tar -zxvf cifar-100-python.tar.gz",
        shell=True
    )
    print("Extracting successfully done to {}.".format(cifar_python_directory))
else:
    print("Dataset already extracted. Did not extract twice.\n")


print("Converting...")
cifar_caffe_directory = os.path.abspath('cifar_100_caffe_hdf5/')
if not os.path.exists(cifar_caffe_directory):

    def unpickle(file):
        import cPickle
        fo = open(file, 'rb')
        dict = cPickle.load(fo)
        fo.close()
        return dict

    def shuffle_data(data, labels):
        data, _, labels, _ = sklearn.cross_validation.train_test_split(
            data, labels, test_size=0.0, random_state=42
        )
        return data, labels

    def load_data(train_file):
        data = []
        labels = []

        d = unpickle(
            os.path.join(cifar_python_directory, train_file)
        )

        data = d['data']
        coarse_labels = d['coarse_labels']
        fine_labels = d['fine_labels']
        length = len(d['fine_labels'])

        data, labels = shuffle_data(
            data,
            np.array(zip(coarse_labels, fine_labels))
        )
        coarse_labels, fine_labels = zip(*labels.tolist())

        return (
            data.reshape(length, 3, 32, 32),
            np.array(coarse_labels),
            np.array(fine_labels)
        )

    X, y_c, y_f = load_data("train")

    Xt, yt_c, yt_f = load_data("test")

    print("INFO: each dataset's element are of shape 3*32*32:")
    print('"print(X.shape)" --> "{}"\n'.format(X.shape))
    print("From the Caffe documentation: ")
    print("The conventional blob dimensions for batches of image data "
          "are number N x channel K x height H x width W.\n")

    print("Data is fully loaded, now truly converting.")

    os.makedirs(cifar_caffe_directory)
    train_filename = os.path.join(cifar_caffe_directory, 'train.h5')
    test_filename = os.path.join(cifar_caffe_directory, 'test.h5')

    comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
    # Train
    with h5py.File(train_filename, 'w') as f:
        f.create_dataset('data', data=X, **comp_kwargs)
        f.create_dataset('label_coarse', data=y_c.astype(np.int_), **comp_kwargs)
        f.create_dataset('label_fine', data=y_f.astype(np.int_), **comp_kwargs)
    with open(os.path.join(cifar_caffe_directory, 'train.txt'), 'w') as f:
        f.write(train_filename + '\n')
    # Test
    with h5py.File(test_filename, 'w') as f:
        f.create_dataset('data', data=Xt, **comp_kwargs)
        f.create_dataset('label_coarse', data=yt_c.astype(np.int_), **comp_kwargs)
        f.create_dataset('label_fine', data=yt_f.astype(np.int_), **comp_kwargs)
    with open(os.path.join(cifar_caffe_directory, 'test.txt'), 'w') as f:
        f.write(test_filename + '\n')

    print('Conversion successfully done to "{}".\n'.format(cifar_caffe_directory))
else:
    print("Conversion was already done. Did not convert twice.\n")
