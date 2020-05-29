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
from random import randint


# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")

    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client

def draw_boxes(frame,output_result,prob_threshold, width, height):
    """
    Draw bounding boxes onto the frame.
    :param frame: frame from camera/video
    :param result: list contains the data comming from inference
    :return: person count and frame
    """
    counter=0
    # Initial coordinate, here (xmin, ymin) 
    initial_point = None

    # Ending coordinate, here (xmax, ymax) 
    ending_point = None

    # Bounding box color
    box_color = (255,0,0) # "BLUE": (255,0,0), "GREEN": (0,255,0), "RED": (0,0,255)

    # Box thickness
    box_thickness = 3

    for obj in output_result[0][0]: 
        if obj[2] > prob_threshold:
            xmin = int(obj[3] * width)
            ymin = int(obj[4] * height)
            xmax = int(obj[5] * width)
            ymax = int(obj[6] * height)
            initial_point = (xmin,ymin)
            ending_point = (xmax,ymax)
            # Using cv2.rectangle() method 
            # Draw a rectangle with Green line borders of thickness of 1 px
            frame = cv2.rectangle(frame, initial_point, ending_point, box_color, box_thickness)
            counter+=1
    return frame, counter


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class (Inference engine)
    infer_network = Network()

    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` into IE ###
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()

    single_image = False
    ### TODO: Handle the input_stream ###
    input_stream = args.input
    if input_stream == 'CAM':
        input_stream = 0
        
    elif input_stream.endswith('.jpg') or input_stream.endswith('.bmp'):
        single_image = True
    

    # Get and open ideo capture
    cap = cv2.VideoCapture(input_stream)
    cap.open(input_stream)
    
    #Grab the shape of the input
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Variables
    report = 0
    counter = 0
    counter_prev = 0
    duration_prev = 0
    counter_total = 0
    dur = 0
    request_id=0

    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the image as needed ###
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        ### TODO: Start asynchronous inference for specified request ###
        duration_report = None
        infer_start = time.time()
        infer_network.exec_net(p_frame)

        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
            det_time = time.time() - infer_start

            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output()
           
           #Update the frame to include detected bounding boxes
            frame_with_box, pointer = draw_boxes(frame, result, prob_threshold, width, height)

           #Display inference time
            inference_time_message = "Inference time: {:.3f}ms"\
            .format(det_time * 1000)
            cv2.putText(frame_with_box, inference_time_message, (15, 15),
                       cv2.FONT_HERSHEY_COMPLEX, 0.45, (200, 10, 10), 1)

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            if pointer != counter:
                counter_prev = counter
                counter = pointer
                if dur >= 3:
                    duration_prev = dur
                    dur = 0
                else:
                    dur = duration_prev + dur
                    duration_prev = 0  # unknown, not needed in this case
            else:
                dur += 1
                if dur >= 3:
                    report = counter
                    if dur == 3 and counter > counter_prev:
                        counter_total += counter - counter_prev
                    elif dur == 3 and counter < counter_prev:
                        duration_report = int((duration_prev / 10.0) * 1000)
            client.publish('person',
                           payload=json.dumps({
                               'count': report, 'total': counter_total}),
                           qos=0, retain=False)
            if duration_report is not None:
                client.publish('person/duration',
                                payload=json.dumps({'duration': duration_report}),
                                qos=0, retain=False)

            
            
            ### TODO: Send the frame to the FFMPEG server ###
            sys.stdout.buffer.write(frame_with_box)
            sys.stdout.flush()

            ### TODO: Write an output image if `single_image_mode` ###
            if single_image:
                cv2.imwrite('Output_Image.jpg', frame_with_box)
            

        # Break if escapturee key pressed
        if key_pressed == 27:
                    break

        
        
    # Release the capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()    
    client.disconnect()	


def main():
    """
    Load the network and parse the output.
    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
        main()
