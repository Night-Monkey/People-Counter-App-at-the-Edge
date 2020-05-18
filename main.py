"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
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

import argparse
import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network
 ### modify code the code from lesson 5 ex. 13 & 14 ###

    
INPUT_STREAM = "Pedestrian_Detect_2_1_1.mp4"
CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3004
MQTT_KEEPALIVE_INTERVAL = 60

def build_argparser():
    """
   # Parse command line arguments.

   # :return: command line arguments
    """    
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type= str,help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device",type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    args = parser.parse_args() 
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    #client = None
    ### TODO: Connect to the MQTT server
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client

def draw_boxes(result,args,imgToDisplay,w,h):
    for box in result[0][0]:
        conf = box[2]

        print(box[1])
        if(conf >= args.prob_threshold and box[1]==1.):
            xmin = int(box[3] * w)  #box[3] - xmin
            ymin = int(box[4] * h) #box[4] - ymin
            xmax = int(box[5] * w)  #box[5] - xmax
            ymax = int(box[6] * h) #box[6] - ymax
            cv2.rectangle(imgToDisplay, (xmin, ymin), (xmax, ymax), (0, 0, 255),1)
    return imgToDisplay

def infer_on_stream(args,client):
    """
   # Initialize the inference network, stream video to network,
   # and output stats and video.

    #:param args: Command line arguments parsed by `build_argparser()`
    #:param client: MQTT client
    #:return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    total_count = 0
    last_count = 0 
    start_time = 0
    
    model = args.model
    DEVICE = args.device
    #CPU_EXTENSION = args.cpu_extension
    ### TODO: Load the model through `infer_network` ###
    ### modify code the code from lesson 5 ex. 7 & 8 ###
    #infer_network.load_model(args.model, args.device, args.cpu_extension)
    infer_network.load_model(model, DEVICE, CPU_EXTENSION)
    #infer_network.load_model(args.model, args.device, CPU_EXTENSION, num_requests=0)
    net_input_shape = infer_network.get_input_shape()
    ### TODO: Handle the input stream ###
    ### TODO: Loop  until stream is over ###
    #plugin = Network()
    #plugin.load_model(client, args.device, CPU_EXTENSION) 
    #net_input_shape = plugin.get_input_shape()

    ### TODO: Read from the video capture ###
    ### modify code the code from lesson 5 ex. 7 & 8 ###
    ### TODO: Handle the input stream ###
    
    '''
    single_image_mode = False
    if args.i == 'CAM':
        args.i = 0
    elif args.i.endswith('.jpg') or args.i.endswith('.png'):
        single_image_mode = True 
    '''       

    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)  
    
    width = int(cap.get(3))
    height = int(cap.get(4))
             
    #counter = 0
    #incident_flag = False
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)              
        ### TODO: Pre-process the image as needed ###th/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/e image as needed ###
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        ### TODO: Start asynchronous inference for specified request ###
        infer_network.async_inference(p_frame)
        inference_start = time.time()
        infer_network.exec_net(p_frame)                   
        ### TODO: Wait for the result ###
            
        ### TODO: Get the results of the inference request ###
        ### TODO: Extract any desired stats from the results ###
       #if infer_network.wait() == 0:
           #result = infer_network.extract_output()
        if infer_network.wait() == 0:
            determine_time = time.time() - inference_start
                             
        #if plugin.wait() == 0:
            #result = plugin.extract_output()
            result = infer_network.extract_output()
            ### TODO: Process the output
            #ncident_flag = assess_scene(result, counter, incident_flag)
            frame, current_count = rectangles(frame, result, width, height)
        #nference_message = "Inference: {:.3f}ms".format(determine_time * 1000)
        #v2.putText(frame, inference_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
        ### TODO: Calculate and send relevant information on ###
        ### current_count, total_count and duration to the MQTT server ###
        ### Topic "person": keys of "count" and "total" ###
        ### Topic "person/duration": key of "duration" ###
        # When new person enters the video
        if current_count > last_count:
            start_time = time.time()
            total_count = total_count + current_count - last_count
            client.publish("person", json.dumps({"total": total_count}))
                
        # Person duration in the video is calculated
        if current_count < last_count:
            duration = int(time.time() - start_time)
            
            # Publish messages to the MQTT server
        client.publish("person/duration",json.dumps({"duration": duration}))
        client.publish("person", json.dumps({"count": current_count}))
        last_count = current_count
        client.subscribe("person/duration")
        client.subscribe("person")
                
        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)  
        sys.stdout.flush()
        ### TODO: Write an output image if `single_image_mode` ###

        if key_pressed == 27:
            break
                        
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()
    
def main():
    """
    #Load the network and parse the output.

    #:return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args,client)

if __name__ == "__main__":
        main()