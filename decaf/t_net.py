"""
load the net
"""
from decaf.scripts import imagenet
import numpy as np
from decaf.tests import unittest_imagenet_pipeline as tpp
from decaf.layers import core_layers
from collections import namedtuple
FLAGS = namedtuple("FLAGS", "net_file meta_file")
FLAGS.net_file = '../scripts/imagenet.decafnet.epoch90'
FLAGS.meta_file = '../scripts/imagenet.decafnet.meta'
net = imagenet.DecafNet(net_file=FLAGS.net_file, meta_file=FLAGS.meta_file)
