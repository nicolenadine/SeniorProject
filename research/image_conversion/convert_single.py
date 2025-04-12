import os
import argparse
import random
import numpy as np
from tqdm import tqdm
from malware_image_generator import MalwareImageGenerator


def find_opcode_files(data_dir):
    """Get paths to all opcode files in the data directory with better path handling"""
    file_paths = []

    # Look for benign files in multiple possible locations
    benign_dirs = [
        os.path.join(data_dir, "Benign"),  # Capitalized
        os.path.join(data_dir, "benign"),  # Lowercase
        os.path.join(data_dir, "Benign", "extracted"),  # Original path with caps
        os.path.join(data_dir, "benign", "extracted")  # Original path
    ]

    # Look for malware files in multiple possible locations
    malware_dirs = [
        os.path.join(data_dir, "Malware"),  # Capitalized
        os.path.join(data_dir, "malware"),  # Lowercase
        os.path.join(data_dir, "v077_clean")  # Original path in code
    ]

    # Print the directories we're checking
    print("Looking for benign files in:", benign_dirs)
    print("Looking for malware files in:", malware_dirs)

    # Process benign files
    for benign_dir in benign_dirs:
        if os.path.exists(benign_dir):
            print(f"Found benign directory: {benign_dir}")
            for root, _, files in os.walk(benign_dir):
                for file in files:
                    file_paths.append(os.path.join(root, file))

    # Process malware files
    for malware_dir in malware_dirs:
        if os.path.exists(malware_dir):
            print(f"Found malware directory: {malware_dir}")
            for root, _, files in os.walk(malware_dir):
                for file in files:
                    file_paths.append(os.path.join(root, file))

    return file_paths


def main():
    parser = argparse.ArgumentParser(description='Convert opcode sequences to images')
    parser.add_argument('--model', type=str, default='opcode_word2vec.model',
                        help='Path to trained word2vec model')
    parser.add_argument('--data-dir', type=str, default='./Data',
                        help='Path to data directory containing benign and malware folders')
    parser.add_argument('--output-dir', type=str, default='./sample_images',
                        help='Directory to save generated images')
    parser.add_argument('--img-size', type=int, default=256,
                        help='Size of the generated images (width and height)')
    parser.add_argument('--min-opcodes', type=int, default=1000,
                        help='Minimum number of opcodes required to generate an image')
    parser.add_argument('--max-samples-per-family', type=int, default=0,
                        help='Maximum number of samples to process per malware family (0 for all)')

    args = parser.parse_args()

    # Create output directory and subdirectories
    os.makedirs(os.path.join(args.output_dir, "benign"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "malware"), exist_ok=True)

    # Initialize the generator
    print(f"Initializing generator with model: {args.model}")
    generator = MalwareImageGenerator(
        args.model,
        output_dir=args.output_dir,
        img_size=(args.img_size, args.img_size)
    )

    # Get all opcode files
    print("Finding opcode files...")
    all_files = find_opcode_files(args.data_dir)
    print(f"Found {len(all_files)} opcode files")

    # Split files by class (case-insensitive check for "benign")
    benign_files = [f for f in all_files if "benign" in f.lower()]
    malware_files = [f for f in all_files if "benign" not in f.lower()]

    print(f"Found {len(benign_files)} benign files and {len(malware_files)} malware files")

    # Group malware files by subfolder/family
    malware_by_family = {}
    for file_path in malware_files:
        # Extract directory path to use as family identifier
        family_dir = os.path.dirname(file_path)
        if family_dir not in malware_by_family:
            malware_by_family[family_dir] = []
        malware_by_family[family_dir].append(file_path)

    print(f"Found {len(malware_by_family)} malware families")
    for family, files in malware_by_family.items():
        print(f"  - {os.path.basename(family)}: {len(files)} files")

    # Limit samples per malware family if requested
    selected_malware_files = []
    if args.max_samples_per_family > 0:
        for family, files in malware_by_family.items():
            if len(files) > args.max_samples_per_family:
                family_samples = random.sample(files, args.max_samples_per_family)
            else:
                family_samples = files
            selected_malware_files.extend(family_samples)
            print(f"Family {os.path.basename(family)}: selected {len(family_samples)}/{len(files)} samples")
    else:
        selected_malware_files = malware_files

    print(f"Using all {len(benign_files)} benign samples and {len(selected_malware_files)} malware samples")

    # Combine files
    selected_files = benign_files + selected_malware_files

    # Calculate global opcode frequencies for better sampling
    print("Calculating global opcode frequencies...")
    global_frequencies = generator.calculate_global_frequencies(args.data_dir)

    # Process files sequentially
    processed = {"benign": 0, "malware": 0, "skipped": 0, "errors": 0}

    print("Starting image generation...")
    for file_path in tqdm(selected_files, desc="Converting"):
        try:
            # Determine if benign or malware
            if "benign" in file_path.lower():
                label = "benign"
            else:
                label = "malware"

            # Read opcodes
            with open(file_path, 'r', errors='ignore') as f:
                opcodes = f.read().strip().split()

            # Skip files with too few opcodes
            if len(opcodes) < args.min_opcodes:
                processed["skipped"] += 1
                continue

            # Generate image using the regular method
            img_data = generator.hilbert_mapping(opcodes, global_frequencies)

            # Save image
            generator._save_image(img_data, file_path, label)
            processed[label] += 1

        except Exception as e:
            import traceback
            print(f"Error processing {file_path}: {str(e)}")
            print(traceback.format_exc())
            processed["errors"] += 1

    # Generate visualization of sample images
    try:
        generator.visualize_sample_images(5)
    except Exception as e:
        print(f"Error generating sample visualizations: {e}")

    # Generate hilbert curve visualization
    try:
        generator.visualize_hilbert_curve()
    except Exception as e:
        print(f"Error generating Hilbert curve visualization: {e}")

    # Final report
    print("\nConversion complete!")
    print(f"Benign images generated: {processed['benign']}")
    print(f"Malware images generated: {processed['malware']}")
    print(f"Files skipped (too small): {processed['skipped']}")
    print(f"Errors encountered: {processed['errors']}")
    print(f"Images saved to: {args.output_dir}")


if __name__ == "__main__":
    main()