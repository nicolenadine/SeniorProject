#!/bin/bash

# Create the output directory if it doesn't exist
mkdir -p sample_images/benign
mkdir -p sample_images/malware

# Run the single-process converter script
python convert_single.py \
  --model opcode_word2vec.model \
  --data-dir ./Data \
  --output-dir ./sample_images \
  --max-samples-per-family 971