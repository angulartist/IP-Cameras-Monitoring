#!/usr/bin/env bash

python pipeline.py --setup_file ./setup.py --machine_type=n1-standard-2 --max_num_workers=10 --disk_size_gb=30 --project alert-shape-256811 --temp_location gs://ml-video-cv/temp --staging_location gs://ml-video-cv/staging