#!/usr/bin/python3

import sys

import caffe
import numpy as np


if len(sys.argv) < 4:
    print("Usage: " + sys.argv[0] + " <input_deploy_file> <input_model_file> <output_model_file>")
    sys.exit()
else:
    prototxt = sys.argv[1]
    orig = sys.argv[2]  # original model (alex.caffemodel)
    comp = sys.argv[3]  # compressed model (comp.caffemodel)

net = caffe.Net(prototxt, orig, caffe.TEST)
layers = [l for l in net.params.keys() if 'fc' in l]


# Extract boundary value at ratio x while sorting data
def read_boundary_value_with_ratio(data, ratio):
    print("pruning off "+str(ratio*100)+" % of this layer\n")
    arr = data
    arr = list(arr.reshape(arr.size))
    arr.sort(key=abs)
    thresh = abs(arr[int(len(arr)*ratio)-1])
    return thresh


# Input: n-d dense array, Output: pruned array with threshold
def prune_dense(weight_arr, thresh=0.005):
    """Apply weight pruning with threshold """
    under_threshold = abs(weight_arr) < thresh
    weight_arr[under_threshold] = 0
    return weight_arr


# How many percentages you want to apply pruning
ratio = {"fc6": 0.91, "fc7": 0.91, "fc8": 0.75}


for idx, layer in enumerate(ratio):
    print("layer name: ", layer)

    temp2 = net.params[layer][0].data
    temp = np.zeros(net.params[layer][0].shape, np.float32)
    np.copyto(temp, temp2)

    nnz_before = np.sum(temp != 0)

    boundary = read_boundary_value_with_ratio(temp, ratio[layer])
    temp = prune_dense(temp, thresh=boundary)

    # re-write the compressed data on each network
    np.copyto(net.params[layer][0].data, temp)

    print("# of non-zero (before): ", nnz_before)
    print("# of non-zero (after): ", np.sum(temp != 0))
    print("")

net.save(comp)
print(" Compression is done! Output is dense model. ")
