#### Introduction
In the project, I want to give a tutorial for tensorflow object detection model fine-tuning, and also converting tensorflow saved_model to WebModel for tensorflow.js.


#### Tensorflow object dection model fine-tuning
Train the object detection model using wider face dataset
Usage:

    # Step 1:
        Download the WIDER face image dataset
    
    # Step 2:
        python process_img_data.py

    # Step 3:
        python create_wider_tf_record.py \
            --train_data_dir=./WIDER_train/images \
            --val_data_dir=./WIDER_val/images \
            --train_examples_path=./examples_files/train_examples.txt \
            --val_examples_path=./examples_files/val_examples.txt \
            --train_anno_path=./wider_face_split/wider_face_train_bbx_gt.txt \
            --val_anno_path=./wider_face_split/wider_face_val_bbx_gt.txt \
            --label_map_path=./wider_label_map.pbtxt \
            --output_dir=./tfrecord_data \
            --num_shards=10

    # Step 4:
        Download model.ckpt from the tensorflow model zoo

    # Step 5:
        # Under /research folder
        protoc object_detection/protos/*.proto --python_out=.
        export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

        python ./models/research/object_detection/train.py \
            --logtostderr \
            --pipeline_config_path=faster_rcnn_inception_v2_wider.config \
            --train_dir=model_output
        
        OR
        
        python ./models/research/object_detection/train.py \
            --logtostderr \
            --pipeline_config_path=ssd_mobilenet_v1_wider.config \
            --train_dir=model_output
            
    # Step 6: Generate inference model
        python ./models/research/object_detection/export_inference_graph.py \
            --input_type image_tensor \
            --pipeline_config_path faster_rcnn_inception_v2_wider.config \
            --trained_checkpoint_prefix model_output/model.ckpt-100 \
            --output_directory model/
            
        OR
        
        python ./models/research/object_detection/export_inference_graph.py \
            --input_type image_tensor \
            --pipeline_config_path ssd_mobilenet_v1_wider.config \
            --trained_checkpoint_prefix model_output/model.ckpt-100 \
            --output_directory model/
    

#### Convert to tensorflow saved_model to WebModel for tensorflow.js
[Reference](https://github.com/tensorflow/tfjs-converter)
Open Issue: [Tensorflow Object Detection API Model - Unsupported Ops in ssd_mobilenet_v2 model](https://github.com/tensorflow/tfjs/issues/188#issuecomment-403259774)
    
    
    # Step 1:
        tensorflowjs_converter --input_format=tf_saved_model \
            --output_node_names='MobilenetV1/Predictions/Reshape_1' \
            --saved_model_tags=serve /mobilenet/saved_model \
            /mobilenet/web_model
        
        OR
        
        tensorflowjs_converter --input_format=tf_frozen_model \
            --output_node_names='MobilenetV1/Predictions/Reshape_1' \
            --saved_model_tags=serve /model/frozen_inference_graph.pb \
            /model/web_model
