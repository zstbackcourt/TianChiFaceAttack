#python3
# -*- coding: utf-8 -*-


import tensorflow as tf
import torch
import json
import os, sys

from dependencies.facenet.src import facenet
from dependencies.facenet.src.models import inception_resnet_v1 as tf_mdl
from dependencies.facenet.src.align import detect_face

from models.inception_resnet_v1 import InceptionResnetV1
from models.mtcnn import PNet, RNet, ONet


def import_tf_params(tf_mdl_dir, sess):
    """Import tensorflow model from save directory.

    Arguments:
        tf_mdl_dir {str} -- Location of protobuf, checkpoint, meta files.
        sess {tensorflow.Session} -- Tensorflow session object.

    Returns:
        (list, list, list) -- Tuple of lists containing the layer names,
            parameter arrays as numpy ndarrays, parameter shapes.
    """
    print('\nLoading tensorflow model\n')
    if callable(tf_mdl_dir):
        tf_mdl_dir(sess)
    else:
        facenet.load_model(tf_mdl_dir)

    print('\nGetting model weights\n')
    tf_layers = tf.trainable_variables()
    tf_params = sess.run(tf_layers)

    tf_shapes = [p.shape for p in tf_params]
    tf_layers = [l.name for l in tf_layers]

    if not callable(tf_mdl_dir):
        path = os.path.join(tf_mdl_dir, 'layer_description.json')
    else:
        path = 'data/layer_description.json'
    with open(path, 'w') as f:
        json.dump({l: s for l, s in zip(tf_layers, tf_shapes)}, f)

    return tf_layers, tf_params, tf_shapes


def get_layer_indices(layer_lookup, tf_layers):
    """Giving a lookup of model layer attribute names and tensorflow variable names,
    find matching parameters.

    Arguments:
        layer_lookup {dict} -- Dictionary mapping pytorch attribute names to (partial)
            tensorflow variable names. Expects dict of the form {'attr': ['tf_name', ...]}
            where the '...'s are ignored.
        tf_layers {list} -- List of tensorflow variable names.

    Returns:
        list -- The input dictionary with the list of matching inds appended to each item.
    """
    layer_inds = {}
    for name, value in layer_lookup.items():
        layer_inds[name] = value + [[i for i, n in enumerate(tf_layers) if value[0] in n]]
    return layer_inds


def load_tf_batchNorm(weights, layer):
    """Load tensorflow weights into nn.BatchNorm object.

    Arguments:
        weights {list} -- Tensorflow parameters.
        layer {torch.nn.Module} -- nn.BatchNorm.
    """
    layer.bias.data = torch.tensor(weights[0]).view(layer.bias.data.shape)
    layer.weight.data = torch.ones_like(layer.weight.data)
    layer.running_mean = torch.tensor(weights[1]).view(layer.running_mean.shape)
    layer.running_var = torch.tensor(weights[2]).view(layer.running_var.shape)


def load_tf_conv2d(weights, layer, transpose=False):
    """Load tensorflow weights into nn.Conv2d object.

    Arguments:
        weights {list} -- Tensorflow parameters.
        layer {torch.nn.Module} -- nn.Conv2d.
    """
    if isinstance(weights, list):
        if len(weights) == 2:
            layer.bias.data = (
                torch.tensor(weights[1])
                    .view(layer.bias.data.shape)
            )
        weights = weights[0]

    if transpose:
        dim_order = (3, 2, 1, 0)
    else:
        dim_order = (3, 2, 0, 1)

    layer.weight.data = (
        torch.tensor(weights)
            .permute(dim_order)
            .view(layer.weight.data.shape)
    )


def load_tf_conv2d_trans(weights, layer):
    return load_tf_conv2d(weights, layer, transpose=True)


def load_tf_basicConv2d(weights, layer):
    """Load tensorflow weights into grouped Conv2d+BatchNorm object.

    Arguments:
        weights {list} -- Tensorflow parameters.
        layer {torch.nn.Module} -- Object containing Conv2d+BatchNorm.
    """
    load_tf_conv2d(weights[0], layer.conv)
    load_tf_batchNorm(weights[1:], layer.bn)


def load_tf_linear(weights, layer):
    """Load tensorflow weights into nn.Linear object.

    Arguments:
        weights {list} -- Tensorflow parameters.
        layer {torch.nn.Module} -- nn.Linear.
    """
    if isinstance(weights, list):
        if len(weights) == 2:
            layer.bias.data = (
                torch.tensor(weights[1])
                    .view(layer.bias.data.shape)
            )
        weights = weights[0]
    layer.weight.data = (
        torch.tensor(weights)
            .transpose(-1, 0)
            .view(layer.weight.data.shape)
    )


# High-level parameter-loading functions:

def load_tf_block35(weights, layer):
    load_tf_basicConv2d(weights[:4], layer.branch0)
    load_tf_basicConv2d(weights[4:8], layer.branch1[0])
    load_tf_basicConv2d(weights[8:12], layer.branch1[1])
    load_tf_basicConv2d(weights[12:16], layer.branch2[0])
    load_tf_basicConv2d(weights[16:20], layer.branch2[1])
    load_tf_basicConv2d(weights[20:24], layer.branch2[2])
    load_tf_conv2d(weights[24:26], layer.conv2d)


def load_tf_block17_8(weights, layer):
    load_tf_basicConv2d(weights[:4], layer.branch0)
    load_tf_basicConv2d(weights[4:8], layer.branch1[0])
    load_tf_basicConv2d(weights[8:12], layer.branch1[1])
    load_tf_basicConv2d(weights[12:16], layer.branch1[2])
    load_tf_conv2d(weights[16:18], layer.conv2d)


def load_tf_mixed6a(weights, layer):
    if len(weights) != 16:
        raise ValueError(f'Number of weight arrays ({len(weights)}) not equal to 16')
    load_tf_basicConv2d(weights[:4], layer.branch0)
    load_tf_basicConv2d(weights[4:8], layer.branch1[0])
    load_tf_basicConv2d(weights[8:12], layer.branch1[1])
    load_tf_basicConv2d(weights[12:16], layer.branch1[2])


def load_tf_mixed7a(weights, layer):
    if len(weights) != 28:
        raise ValueError(f'Number of weight arrays ({len(weights)}) not equal to 28')
    load_tf_basicConv2d(weights[:4], layer.branch0[0])
    load_tf_basicConv2d(weights[4:8], layer.branch0[1])
    load_tf_basicConv2d(weights[8:12], layer.branch1[0])
    load_tf_basicConv2d(weights[12:16], layer.branch1[1])
    load_tf_basicConv2d(weights[16:20], layer.branch2[0])
    load_tf_basicConv2d(weights[20:24], layer.branch2[1])
    load_tf_basicConv2d(weights[24:28], layer.branch2[2])


def load_tf_repeats(weights, layer, rptlen, subfun):
    if len(weights) % rptlen != 0:
        raise ValueError(f'Number of weight arrays ({len(weights)}) not divisible by {rptlen}')
    weights_split = [weights[i:i + rptlen] for i in range(0, len(weights), rptlen)]
    for i, w in enumerate(weights_split):
        subfun(w, getattr(layer, str(i)))


def load_tf_repeat_1(weights, layer):
    load_tf_repeats(weights, layer, 26, load_tf_block35)


def load_tf_repeat_2(weights, layer):
    load_tf_repeats(weights, layer, 18, load_tf_block17_8)