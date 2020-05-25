# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are...

Each of these use cases would be useful because...

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

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
  
- Model 2: bvlc_alexnet
  - wget https://s3.amazonaws.com/download.onnx/models/opset_8/bvlc_alexnet.tar.gz
  - tar -xvf bvlc_alexnet.tar.gz
  - cd bvlc_alexnet
  - python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model model.onnx
  - python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m bvlc_alexnet/model.xml -l/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm


- Model 3: Person-detection-retail-0013 from Intel OpenVINO Model Zoo
  - https://docs.openvinotoolkit.org/latest/_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html
  - python3  /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name person-detection-retail-0013 -o /home/workspace/model/pre_trained/intel
  
  - 