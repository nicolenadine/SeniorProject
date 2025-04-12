import os
import argparse
import random
import numpy as np
from tqdm import tqdm
import time
from malware_image_generator import MalwareImageGenerator


def find_files_in_specific_paths(data_dir):
    """Get opcode files from the exact directory structure"""
    benign_files = []
    malware_files_by_family = {}

    # Find benign files - exact path only
    benign_dir = os.path.join(data_dir, "Benign")
    if os.path.exists(benign_dir):
        print(f"Found benign directory: {benign_dir}")

        # Get the subfolders in the Benign directory
        benign_subfolders = [os.path.join(benign_dir, d) for d in os.listdir(benign_dir)
                             if os.path.isdir(os.path.join(benign_dir, d))]

        print(f"Found {len(benign_subfolders)} benign subfolders: {[os.path.basename(d) for d in benign_subfolders]}")

        # Process each subfolder
        for subfolder in benign_subfolders:
            subfolder_files = []
            for root, _, files in os.walk(subfolder):
                for file in files:
                    subfolder_files.append(os.path.join(root, file))

            print(f"  - {os.path.basename(subfolder)}: {len(subfolder_files)} files")
            benign_files.extend(subfolder_files)
    else:
        print(f"WARNING: Benign directory not found at {benign_dir}")

    # Find malware files - exact path only
    malware_dir = os.path.join(data_dir, "Malware")
    if os.path.exists(malware_dir):
        print(f"Found malware directory: {malware_dir}")

        # Get the subfolders (malware families) in the Malware directory
        malware_subfolders = [os.path.join(malware_dir, d) for d in os.listdir(malware_dir)
                              if os.path.isdir(os.path.join(malware_dir, d))]

        print(f"Found {len(malware_subfolders)} malware families: {[os.path.basename(d) for d in malware_subfolders]}")

        # Process each malware family subfolder
        for subfolder in malware_subfolders:
            family_name = os.path.basename(subfolder)
            family_files = []

            for root, _, files in os.walk(subfolder):
                for file in files:
                    family_files.append(os.path.join(root, file))

            print(f"  - {family_name}: {len(family_files)} files")
            malware_files_by_family[family_name] = family_files
    else:
        print(f"WARNING: Malware directory not found at {malware_dir}")

    return benign_files, malware_files_by_family


def process_batch(file_batch, generator, global_frequencies, min_opcodes):
    """Process a batch of files and return statistics"""
    batch_stats = {"benign": 0, "malware": 0, "skipped": 0, "errors": 0}

    for file_path, file_type in file_batch:
        try:
            # Read opcodes
            with open(file_path, 'r', errors='ignore') as f:
                opcodes = f.read().strip().split()

            # Skip files with too few opcodes
            if len(opcodes) < min_opcodes:
                batch_stats["skipped"] += 1
                continue

            # Generate image
            img_data = generator.hilbert_mapping(opcodes, global_frequencies)

            # Save image
            generator._save_image(img_data, file_path, file_type)
            batch_stats[file_type] += 1

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
    parser.add_argument('--samples-per-family', type=int, default=500,
                        help='Number of samples to process per malware family')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Number of files to process in each batch')
    parser.add_argument('--checkpoint-file', type=str, default='conversion_progress.txt',
                        help='File to save progress for resuming')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint file')
    parser.add_argument('--benign-only', action='store_true',
                        help='Process only benign files')
    parser.add_argument('--malware-only', action='store_true',
                        help='Process only malware files')

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
    print("Finding opcode files in specified directories...")
    benign_files, malware_files_by_family = find_files_in_specific_paths(args.data_dir)

    print(
        f"Found {len(benign_files)} benign files and {sum(len(files) for files in malware_files_by_family.values())} total malware files")

    # Select files to process based on flags
    selected_files = []

    if not args.malware_only:
        # Process all benign files
        for file_path in benign_files:
            selected_files.append((file_path, "benign"))
        print(f"Selected all {len(benign_files)} benign files for processing")

    if not args.benign_only:
        # Process malware files, limited per family
        selected_malware_count = 0
        for family, files in malware_files_by_family.items():
            # Select samples from this family
            if len(files) > args.samples_per_family:
                family_samples = random.sample(files, args.samples_per_family)
            else:
                family_samples = files

            # Add to selected files
            for file_path in family_samples:
                selected_files.append((file_path, "malware"))

            selected_malware_count += len(family_samples)
            print(f"Family {family}: selected {len(family_samples)}/{len(files)} samples")

        print(f"Selected {selected_malware_count} malware files for processing")

    print(f"Total files selected for processing: {len(selected_files)}")

    # Handle resuming from checkpoint
    processed_files = set()
    if args.resume and os.path.exists(args.checkpoint_file):
        print(f"Resuming from checkpoint file: {args.checkpoint_file}")
        with open(args.checkpoint_file, 'r') as f:
            processed_files = set(line.strip() for line in f)
        print(f"Found {len(processed_files)} processed files in checkpoint")

        # Filter out already processed files
        selected_files = [(f, t) for f, t in selected_files if f not in processed_files]
        print(f"Remaining files to process: {len(selected_files)}")

    # Exit if no files to process
    if not selected_files:
        print("No files to process. Exiting.")
        return

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
        for file_path, _ in batch:
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