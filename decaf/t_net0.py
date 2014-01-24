"""
load the net and run unittest
"""
from decaf.tests import unittest_imagenet_pipeline as tpp
from decaf.layers import core_layers
from decaf import base
decaf_net = base.Net()
# add data layer
decaf_net.add_layers(tpp.imagenet_data(),
                     provides=['image', 'label'])
decaf_net.add_layers(tpp.imagenet_layers(),
                     needs='image',
                     provides='prediction')
loss_layer = core_layers.MultinomialLogisticLossLayer(
    name='loss')
decaf_net.add_layer(loss_layer,
                    needs=['prediction', 'label'])
decaf_net.finish()
loss = decaf_net.forward_backward()
print loss
