# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you have a different formats/models that you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers
The process behind converting custom layers involves adding layers that are not part of any known layers. The model extraction extracts the informartion from an input. THe output is then handeled using model optimization based on a certain shape for the output. Finally, the xml and bin files are created as part of the IR output. which is needed by the inference Engine to run the model.
Some of the potential reasons for handling custom layers is to add extensions to both the Model Optimizer and the Inference Engine

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: SSD MobileNet V2 model
  - wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
  - tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
  - cd ssd_mobilenet_v2_coco_2018_03_29
  - python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
  - Python Main.py -m ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml| ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 1280x720 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

  
- Model 2: bvlc_alexnet
  - wget https://s3.amazonaws.com/download.onnx/models/opset_8/bvlc_alexnet.tar.gz
  - tar -xvf bvlc_alexnet.tar.gz
  - cd bvlc_alexnet
  - python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model model.onnx
  - python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m bvlc_alexnet/model.xml -l/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm


- Model 3: Person-detection-retail-0013 from Intel OpenVINO Model Zoo
  - https://docs.openvinotoolkit.org/latest/_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html
  - python3  /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name person-detection-retail-0013 -o /home/workspace/model/pre_trained/intel
  - python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m model/pre_trained/intel/person-detection-retail-0013/intel/FP16/person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
  
- Model 4:person-detection-retail-0002 from Intel OpenVINO Model Zoo
    - python3  /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name person-detection-retail-0002 -o /home/workspace/model/pre_trained/intel
    - python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m model/pre_trained/intel/intel/person-detection-retail-0002/FP16/person-detection-retail-0002.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

- Model 5:faster_rcnn_inception_v2

    - wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
    - tar -xvf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
    - cd faster_rcnn_inception_v2_coco_2018_01_28
    - python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json
    
- Model 6:ssd_mobilenet_v1_coco
    - wget
http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.g
    - tar -xvf ssd_mobilenet_v1_coco_2018_01_28.tar.gz
    - cd ssd_mobilenet_v1_coco_2018_01_28
    - python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_support.json
    
## Comparing Model Performance

Using each models in this comparison:
- faster_rcnn_inception_v2
- ssd_mobilenet_v2_coco
- ssd_mobilenet_v2_quantized_coco
- person-detection-retail-0002
- Person-detection-retail-0013

A simple comparison between the models are the following:

| Models           | Speed (ms) | Ex Time (s) | File Size (Mb) |
|------------------|------------|-------------|----------------|
| faster_rcnn      | 58         | 143         | 53             |
| ssd_mobilenet_v2 | 31         | 57          | 67             |
| ssd_mobilenet_v1 | 30         | 39          | 27             |
| retail-0002      | 217        | --          | 0.13           |
| retail-0013      | 45         | --          | 1.565          |

Create and explain my method behind comparing the performance of a model with and without the use of the OpenVINO™ Toolkit by its accuracy, size, speed, CPU overhead.
I make comparison by each models' average inference time. I will recommend ssd_mobilenet_v1 due to it time and speed. However, the only model that count the people in a frame will be Person-detection-retail-0013.

## Assess Model Use Cases

The use cases of a people counter app could be apply to make applications. You could use this model for cameras mounted at higher vantage points to count the people in a frame in the building. This could potentially help in limiting the number of people in a certain area, especially during the current convid situation. To track movement of people and footage activity in retail or warehouse space. Capture and record information on the number of people in stores or buildings.

## Assess Effects on End User Needs

Discuss lighting, model accuracy, and camera focal length/image size, and the effects these may have on an end user requirement.

- The lighting can influence the edge model on counting the people in frame. With different angles the model can be mistakenly ignore a person in the frame (or give it a lower confidence level) due to missing charectersitcs.
- The image size and camera focal length can infuence the overall processing. If the image size is too large, the edge compute might need more time to process the frame and thus exhaust mroe resources. A lower sized image can influence the accuracy of the output and thus jeopardize the project
- Model accuracy is proportional to compute power and memory consumption. The higher the model accuracy, the more resoureces it would potentially need. I believe this is more dependant on the use case. To be more accuracy, it will be using more computatal processing power. 
-  If end users would want to monitor wider area, then the high focal length camera is better option. The model can extract less information about object's in picture so it can lower the accuracy. If end users want to monitor very narrow place, then they can use low focal length camera.

## Output Results

After my investigation on those four models, I had conclude very good, suitable, and accurate model that was in Intermediate Representations provided by Intel® [person-detection-retail-0013]

Dowload the model 

```
python3  /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name person-detection-retail-0013 -o /home/workspace/model/pre_trained/intel
```

Running the app 

```
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m model/pre_trained/intel/person-detection-retail-0013/intel/FP16/person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```