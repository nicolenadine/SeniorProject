#!/bin/bash

# Create the output directory if it doesn't exist
mkdir -p ./generated_images/malware

# Run the Python script to process malware files with 750 samples per family
python malware_only_conversion.py \
  --data_dir ./Data \
  --w2v_model ./opcode_word2vec.model\
  --output_dir ./generated_images \
  --min_opcodes 1000 \
  --samples_per_family 750 \
  --batch_size 100 \
  --img_size 256

# Check if the script executed successfully
if [ $? -eq 0 ]; then
    echo "Malware image generation completed successfully"
else
    echo "Error: Malware image generation failed"
    exit 1
fi