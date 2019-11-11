#!/usr/bin/env bash

gcloud ai-platform jobs submit training object_detection_r112_`date +%m_%d_%Y_%H_%M_%S` \
    --runtime-version 1.12 \
    --job-dir gs://ml-train-weapons/train \
    --packages models/research/dist/object_detection-0.1.tar.gz,models/research/dist/absl-py-0.8.1.tar.gz,models/research/slim/dist/slim-0.1.tar.gz,/tmp/pycocotools/pycocotools-2.0.tar.gz \
    --module-name object_detection.model_main \
    --region us-east1 \
    --config ./cmle/cloud.yml \
    -- \
    --model-dir gs://ml-train-weapons/train \
    --pipeline_config_path gs://ml-train-weapons/training/ssdlite_mobilenet_v2_coco.config