import os
import argparse
import random
import numpy as np
from tqdm import tqdm
import time
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


def process_batch(file_batch, generator, global_frequencies, min_opcodes):
    """Process a batch of files and return statistics"""
    batch_stats = {"benign": 0, "malware": 0, "skipped": 0, "errors": 0}

    for file_path in file_batch:
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
            if len(opcodes) < min_opcodes:
                batch_stats["skipped"] += 1
                continue

            # Generate image using the regular method
            img_data = generator.hilbert_mapping(opcodes, global_frequencies)

            # Save image
            generator._save_image(img_data, file_path, label)
            batch_stats[label] += 1

        except Exception as e:
            import traceback
            print(f"Error processing {file_path}: {str(e)}")
            print(traceback.format_exc())
            batch_stats["errors"] += 1

    return batch_stats


def main():
    parser = argparse.ArgumentParser(description='Convert opcode sequences to images in batches')
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
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Number of files to process in each batch')
    parser.add_argument('--checkpoint-file', type=str, default='conversion_progress.txt',
                        help='File to save progress for resuming')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint file')

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

    # Handle resuming from checkpoint
    processed_files = set()
    if args.resume and os.path.exists(args.checkpoint_file):
        print(f"Resuming from checkpoint file: {args.checkpoint_file}")
        with open(args.checkpoint_file, 'r') as f:
            processed_files = set(line.strip() for line in f)
        print(f"Found {len(processed_files)} processed files in checkpoint")

        # Filter out already processed files
        selected_files = [f for f in selected_files if f not in processed_files]
        print(f"Remaining files to process: {len(selected_files)}")

    # Calculate global opcode frequencies for better sampling
    print("Calculating global opcode frequencies...")
    global_frequencies = generator.calculate_global_frequencies(args.data_dir)

    # Process files in batches
    total_stats = {"benign": 0, "malware": 0, "skipped": 0, "errors": 0}
    checkpoint_file = open(args.checkpoint_file, 'a')

    print(f"Starting image generation in batches of {args.batch_size}...")
    start_time = time.time()

    # Split into batches
    num_batches = (len(selected_files) + args.batch_size - 1) // args.batch_size
    for batch_idx in range(num_batches):
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, len(selected_files))
        batch = selected_files[batch_start:batch_end]

        print(f"\nProcessing batch {batch_idx + 1}/{num_batches} ({len(batch)} files)")

        # Process this batch
        batch_stats = process_batch(batch, generator, global_frequencies, args.min_opcodes)

        # Update total stats
        for key in total_stats:
            total_stats[key] += batch_stats[key]

        # Save progress to checkpoint
        for file_path in batch:
            checkpoint_file.write(f"{file_path}\n")
        checkpoint_file.flush()

        # Report progress
        elapsed_time = time.time() - start_time
        processed_count = batch_idx * args.batch_size + len(batch)
        files_per_second = processed_count / elapsed_time if elapsed_time > 0 else 0

        print(
            f"Progress: {processed_count}/{len(selected_files)} files ({processed_count / len(selected_files) * 100:.1f}%)")
        print(
            f"Current stats: {total_stats['benign']} benign, {total_stats['malware']} malware, {total_stats['skipped']} skipped, {total_stats['errors']} errors")
        print(f"Processing speed: {files_per_second:.2f} files/second")

        remaining = len(selected_files) - processed_count
        if remaining > 0 and files_per_second > 0:
            remaining_time = remaining / files_per_second
            print(f"Estimated time remaining: {remaining_time / 60:.1f} minutes ({remaining_time / 3600:.1f} hours)")

    checkpoint_file.close()

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
    print(f"Benign images generated: {total_stats['benign']}")
    print(f"Malware images generated: {total_stats['malware']}")
    print(f"Files skipped (too small): {total_stats['skipped']}")
    print(f"Errors encountered: {total_stats['errors']}")
    print(f"Images saved to: {args.output_dir}")


if __name__ == "__main__":
    main()