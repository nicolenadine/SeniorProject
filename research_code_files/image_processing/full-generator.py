import os
import argparse
import logging
from tqdm import tqdm
import time
import shutil
import json
import random

# Make sure UMAP is installed
try:
    import umap
except ImportError:
    print("UMAP is required. Please install it using: pip install umap-learn")
    exit(1)

from malware_image_generator import MalwareImageGenerator


def setup_logging(log_file="full_generation.log"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("FullGenerator")


def count_files(data_dir):
    """Count the total number of opcode files to process"""
    benign_count = 0
    malware_count = 0

    # Count benign files
    benign_dir = os.path.join(data_dir, "benign", "extracted")
    if os.path.exists(benign_dir):
        for root, _, files in os.walk(benign_dir):
            benign_count += len(files)

    # Count malware files
    malware_dir = os.path.join(data_dir, "v077_clean")
    if os.path.exists(malware_dir):
        for root, _, files in os.walk(malware_dir):
            malware_count += len(files)

    return benign_count, malware_count


def main():
    parser = argparse.ArgumentParser(description='Generate images from all malware and benign opcode sequences')

    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained word2vec model')

    parser.add_argument('--data-dir', type=str, default='./Data',
                        help='Path to data directory containing benign and malware folders')

    parser.add_argument('--output-dir', type=str, default='./malware_images',
                        help='Directory to save generated images')

    parser.add_argument('--img-size', type=int, default=256,
                        help='Size of the generated images (width and height)')

    parser.add_argument('--batch-size', type=int, default=1000,
                        help='Process files in batches to manage memory usage')

    parser.add_argument('--skip-visualizations', action='store_true',
                        help='Skip generating visualization plots')

    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Directory to save progress checkpoints')

    parser.add_argument('--resume', action='store_true',
                        help='Resume from last checkpoint if available')

    parser.add_argument('--samples-per-class', type=int, default=3000,
                        help='Number of samples to process per class (benign/malware)')

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging()

    logger.info("=== Starting Controlled Sample Image Generation ===")
    logger.info(f"Using word2vec model: {args.model}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Image size: {args.img_size}x{args.img_size}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Samples per class: {args.samples_per_class}")

    # Count total files
    benign_count, malware_count = count_files(args.data_dir)
    logger.info(f"Found {benign_count} benign files and {malware_count} malware files in dataset")

    # Create output and checkpoint directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # File to track selected samples
    sample_registry_file = os.path.join(args.checkpoint_dir, 'sample_registry.json')

    # Initialize the registry - tracks all files ever processed
    sample_registry = {"benign": [], "malware": []}

    # Initialize the generator
    generator = MalwareImageGenerator(
        args.model,
        output_dir=args.output_dir,
        img_size=(args.img_size, args.img_size)
    )

    # Get all opcode files
    all_files = generator._get_all_opcode_files(args.data_dir)

    # Separate benign and malware files
    benign_files = [f for f in all_files if "benign" in f]
    malware_files = [f for f in all_files if "v077_clean" in f]

    # Load existing registry if resuming
    if args.resume and os.path.exists(sample_registry_file):
        logger.info("Loading existing sample registry...")
        with open(sample_registry_file, 'r') as f:
            sample_registry = json.load(f)
        logger.info(
            f"Found {len(sample_registry['benign'])} benign and {len(sample_registry['malware'])} malware files in registry")

        # Remove already processed files from candidates
        benign_files = [f for f in benign_files if f not in sample_registry['benign']]
        malware_files = [f for f in malware_files if f not in sample_registry['malware']]

    # Calculate how many more samples we need
    benign_needed = max(0, args.samples_per_class - len(sample_registry['benign']))
    malware_needed = max(0, args.samples_per_class - len(sample_registry['malware']))

    logger.info(f"Need {benign_needed} more benign files and {malware_needed} more malware files to reach target")

    # Select required number of files
    selected_benign = random.sample(benign_files, min(benign_needed, len(benign_files)))
    selected_malware = random.sample(malware_files, min(malware_needed, len(malware_files)))

    # Combine selected samples for processing
    files_to_process = selected_benign + selected_malware

    if not files_to_process:
        logger.info("No new files to process. Target sample count already reached.")
        return

    logger.info(
        f"Selected {len(selected_benign)} new benign and {len(selected_malware)} new malware files for processing")

    # Calculate global opcode frequencies first (for sampling strategy)
    logger.info("Calculating global opcode frequencies...")
    global_frequencies = generator.calculate_global_frequencies(args.data_dir)

    # Process files in batches
    stats = {"benign": 0, "malware": 0, "errors": 0, "identical_embeddings": 0}

    # Track problematic files separately
    problematic_files = {"identical_embeddings": []}

    try:
        start_time = time.time()

        for i in range(0, len(files_to_process), args.batch_size):
            batch = files_to_process[i:i + args.batch_size]
            logger.info(
                f"Processing batch {i // args.batch_size + 1}/{(len(files_to_process) - 1) // args.batch_size + 1} ({len(batch)} files)")

            # Process batch
            for file_path in tqdm(batch, desc=f"Batch {i // args.batch_size + 1}"):
                try:
                    # Determine if benign or malware
                    if "benign" in file_path:
                        label = "benign"
                    else:
                        label = "malware"

                    # Read opcodes
                    with open(file_path, 'r', errors='ignore') as f:
                        opcodes = f.read().strip().split()

                    # Skip very small files
                    if len(opcodes) < 10:
                        logger.warning(f"Skipping {file_path} (too small: {len(opcodes)} opcodes)")
                        continue

                    # Generate grayscale image
                    try:
                        img_data = generator.hilbert_mapping(opcodes, global_frequencies)

                        # Check log for identical embeddings issue
                        if "identical - using random values" in generator.last_log_message:
                            problematic_files["identical_embeddings"].append(file_path)
                            stats["identical_embeddings"] += 1
                            logger.warning(f"File {file_path} has identical embeddings - using random pattern")
                    except Exception as e:
                        logger.error(f"Error in hilbert mapping for {file_path}: {str(e)}")
                        stats["errors"] += 1
                        continue

                    # Save image
                    output_path = generator._save_image(img_data, file_path, label)
                    stats[label] += 1

                    # Add to registry
                    sample_registry[label].append(file_path)

                    # Free up memory
                    del img_data, opcodes

                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    stats["errors"] += 1

            # Report progress after each batch
            elapsed_time = time.time() - start_time
            processed_count = stats["benign"] + stats["malware"]
            files_per_second = processed_count / elapsed_time if elapsed_time > 0 else 0

            logger.info(
                f"Progress: {processed_count}/{len(files_to_process)} files ({processed_count / len(files_to_process) * 100:.1f}%)")
            logger.info(
                f"Current stats: {stats['benign']} benign, {stats['malware']} malware, {stats['errors']} errors, {stats['identical_embeddings']} identical embeddings")
            logger.info(f"Processing speed: {files_per_second:.2f} files/second")

            remaining = len(files_to_process) - processed_count
            if remaining > 0 and files_per_second > 0:
                logger.info(f"Estimated time remaining: {remaining / files_per_second / 60:.1f} minutes")

            # Save the registry periodically
            with open(sample_registry_file, 'w') as f:
                json.dump(sample_registry, f, indent=2)

            # Save problematic files list
            problematic_files_path = os.path.join(args.checkpoint_dir, 'problematic_files.json')
            with open(problematic_files_path, 'w') as f:
                json.dump(problematic_files, f, indent=2)

        # Final stats
        logger.info(f"Image generation complete.")
        logger.info(
            f"Registry now contains {len(sample_registry['benign'])} benign and {len(sample_registry['malware'])} malware files")
        logger.info(f"Processed in this run: {stats['benign']} benign, {stats['malware']} malware")
        logger.info(
            f"Encountered {stats['errors']} errors and {stats['identical_embeddings']} files with identical embeddings")

    finally:
        # Save final registry
        with open(sample_registry_file, 'w') as f:
            json.dump(sample_registry, f, indent=2)

        # Save final problematic files list
        problematic_files_path = os.path.join(args.checkpoint_dir, 'problematic_files.json')
        with open(problematic_files_path, 'w') as f:
            json.dump(problematic_files, f, indent=2)

    # Create visualizations if not skipped
    if not args.skip_visualizations:
        logger.info("Generating visualizations...")
        generator.visualize_sample_images(10)  # Show more samples
        generator.visualize_embedding_space()
        generator.visualize_hilbert_curve()

    logger.info(f"All processing complete.")
    logger.info(f"Output saved to {args.output_dir}")
    logger.info(f"Sample registry saved to {sample_registry_file}")
    logger.info(f"Problematic files list saved to {problematic_files_path}")


if __name__ == "__main__":
    main()