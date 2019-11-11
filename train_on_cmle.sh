#!/usr/bin/env bash

gcloud ai-platform jobs submit training object_detection_`date +%m_%d_%Y_%H_%M_%S` \
    --runtime-version 1.12 \
    --job-dir gs://ml-train-weapons/model \
    --packages models/research/dist/object_detection-0.1.tar.gz,models/research/slim/dist/slim-0.1.tar.gz,/tmp/pycocotools/pycocotools-2.0.tar.gz \
    --module-name object_detection.model_main \
    --region us-central1 \
    --config ./cmle/cloud.yml \
    -- \
    --model-dir gs://ml-train-weapons/model \
    --pipeline_config_path gs://ml-train-weapons/config/pipeline.config