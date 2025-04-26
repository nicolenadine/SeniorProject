# Malware Classification Pipeline: Image Conversion

## Core Component

### 1. malware_image_generator.py
**Purpose**: Core class that handles converting opcode sequences into grayscale images using a Hilbert curve mapping algorithm.

**Inputs**:
- Word2Vec model file (trained on opcodes)
- Opcode sequences extracted from malware/benign files
- Output directory path
- Image size (default 256×256)

**Dependencies**:
- gensim (Word2Vec)
- PIL
- numpy
- matplotlib
- pandas
- sklearn
- umap
- seaborn

**Outputs**:
- Grayscale PNG images representing opcode sequences
- Visualization of sample images
- Hilbert curve visualization
- Embedding space visualization (UMAP)

## Conversion Scripts (Listed from Basic to Advanced)

### 2. convert_single.py
**Purpose**: Basic script for processing files one-by-one sequentially.

**Inputs**:
- Word2Vec model path
- Data directory containing benign and malware folders
- Output directory
- Optional parameters (image size, min opcodes, max samples per family)

**Dependencies**:
- malware_image_generator.py

**Outputs**:
- PNG images in benign/malware directories
- Sample visualizations

### 3. convert_batch.py
**Purpose**: Enhanced version of convert_single.py that processes files in batches and adds checkpointing.

**Inputs**: Same as convert_single.py, plus:
- Batch size parameter
- Checkpoint file path
- Resume flag

**Dependencies**:
- malware_image_generator.py

**Outputs**: Same as convert_single.py, with added checkpoint file for resuming

### 4. convert_fixed.py
**Purpose**: Improved script with better directory handling and more selective processing options.

**Inputs**: Same as convert_batch.py, plus:
- Options to process only benign or only malware files
- Samples per family parameter

**Dependencies**:
- malware_image_generator.py

**Outputs**: Same as convert_batch.py, with more organized file structure

### 5. full-generator.py
**Purpose**: Comprehensive script with advanced features like logging, controlled sampling, progress tracking, and checkpoint management.

**Inputs**:
- Word2Vec model path
- Data directory
- Output directory
- Batch size
- Samples per class
- Checkpoint directory
- Resume flag
- Skip visualizations flag

**Dependencies**:
- malware_image_generator.py
- umap

**Outputs**:
- PNG images in benign/malware directories
- Sample visualizations
- Sample registry (JSON)
- Problematic files list (JSON)
- Detailed log file

### 6. malware_only_conversion.py
**Purpose**: Specialized script for processing only malware samples with family-based organization.

**Inputs**:
- Data directory
- Word2Vec model path
- Output directory
- Batch size
- Minimum opcodes requirement
- Image size
- Samples per family
- Random seed

**Dependencies**:
- malware_image_generator.py
- PIL
- matplotlib

**Outputs**:
- PNG images organized by malware family
- Family sample visualization
- Embedding space visualization
- Hilbert curve visualization

## Utility Scripts

### 7. count.py
**Purpose**: Analyzes and visualizes the distribution of opcode file lengths.

**Inputs**:
- Directory paths for malware and benign files

**Dependencies**:
- numpy
- matplotlib

**Outputs**:
- Statistics on file lengths
- Box plot visualization of file length distributions

### 8. word2vecTraining.py
**Purpose**: Trains the Word2Vec model on opcode sequences from malware and benign samples.

**Inputs**:
- Base directory with malware and benign samples
- Vector size, window, min count parameters
- Samples counts for malware and benign

**Dependencies**:
- gensim
- numpy
- pandas
- matplotlib
- sklearn
- seaborn
- adjustText

**Outputs**:
- Trained Word2Vec model
- Opcode distribution visualization
- Embedding space visualization
- Analysis report (text file)
- Opcode frequencies (JSON)
- Complexity data (JSON)

  Shell Scripts
9. extract_opcodes.sh
**Purpose:** Bash script to extract opcodes from binary executable files using objdump.

**Inputs:**
- Input folder containing binary files
- Output folder for extracted opcode files

**Dependencies:**
- objdump utility
- Standard Unix tools (awk, sed, grep)

**Outputs:**
- Text files containing extracted opcodes for each binary file

## Workflow Summary

1. **Training Phase**:
   - Run word2vecTraining.py to create the opcode embedding model

2. **Generation Phase** (choose one):
   - For malware-only processing: run_fixed.sh 
   - For comprehensive processing: full-generator.py with appropriate parameters

3. **Analysis**: Use the generated images for downstream machine learning tasks
