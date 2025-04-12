#!/bin/bash

#!/bin/bash

# Create the output directory if it doesn't exist
mkdir -p sample_images/benign
mkdir -p sample_images/malware

# Run the converter script with the word2vec model and optimizations
python convert_opcodes_to_images.py \
  --model opcode_word2vec.model \
  --data-dir ./Data \
  --output-dir ./sample_images \
  --min-opcodes 1000 \
  --max-samples-per-family 971 \
  --workers 4 \
  --simplified

# If you encounter errors, try running with single process for debugging:
# python convert_opcodes_to_images.py \
#   --model opcode_word2vec.model \
#   --data-dir ./Data \
#   --output-dir ./sample_images \
#   --min-opcodes 1000 \
#   --max-samples-per-family 971 \
#   --workers 0 \
#   --simplified