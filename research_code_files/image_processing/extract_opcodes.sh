#!/bin/bash

# Check if correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_folder> <output_folder>"
    exit 1
fi

# Get input and output folder from command-line arguments
INPUT_FOLDER="$1"
OUTPUT_FOLDER="$2"

# Make sure the output folder exists
mkdir -p "$OUTPUT_FOLDER"

# Loop through all files in the input folder
for file in "$INPUT_FOLDER"/*; do
    if [ -f "$file" ]; then
        # Extract filename without path and extension
        filename=$(basename "$file" | cut -d. -f1)

        # Run objdump and clean output
        objdump -d --no-show-raw-insn "$file" | awk '/^ / {print $2}' | sed 's/[bwlq]$//' | grep -vE 'file|of|<.text>|<unknown>' | sed '/^$/d' > "$OUTPUT_FOLDER/${filename}_opcodes.txt"

        echo "Processed: $file -> $OUTPUT_FOLDER/${filename}_opcodes.txt"
    fi
done

