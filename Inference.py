#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        ### modify code the code from lesson 5 ex. 4 & 5 ###
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None
        self.infer_network = None
        
    #def load_model(self):
        # assign to variable
        #self.exec_network = plugin.load_network(network, device)
        ### TODO: Load the model ###
        ### TODO: Check for supported layers ###
          ### modify code the code from lesson 5 ex. 7 & 8 ###
    #def load_model(self, model, device="CPU", cpu_extension=None):
    def load_model(self, model, device="CPU", cpu_extension=None, num_requests=0):
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        #self.exec_network = infer_network.load_network(network, device)

        ### TODO: Add any necessary extensions ###
        
        self.infer_network = IECore()

        
        if cpu_extension and "CPU" in device:
            self.infer_network.add_extension(cpu_extension, device)
        ### TODO: Return the loaded inference plugin ###
        self.network = IENetwork(model=model_xml, weights=model_bin)
        self.exec_network = self.infer_network.load_network(self.network, device)
        ### Note: You may need to update the function parameters. ###
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

        return

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        ###return
        ### modify code the code from lesson 5 ex. 7 & 8 ###
        return self.network.inputs[self.input_blob].shape


    #def exec_net(self, image):
        ### TODO: Start an asynchronous request ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        #self.exec_network.start_async(request_id=0, 
            #inputs={self.input_blob: image})
        #return

    def exec_net(self):
    # now you can use it in another function
        self.exec_network.start_async()
        return
    
    def wait(self):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        ###return
        ### modify code the code from lesson 5 ex. 7 & 8 ###
        status = self.exec_network.requests[0].wait(-1)
        return status
    

    def get_output(self):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        ###return
         ### modify code the code from lesson 5 ex. 7 & 8 ###
        return self.exec_network.requests[0].outputs[self.output_blob]

