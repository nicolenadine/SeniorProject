import os
import argparse
import multiprocessing
import random
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from malware_image_generator import MalwareImageGenerator


# Define process_file function outside of main to avoid pickling issues
def process_file(args):
    """Process a single file - needs to be outside main for multiprocessing"""
    file_path, generator, global_frequencies, min_opcodes = args

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
            return {"label": "skipped", "file": file_path}

        # Generate image
        img_data = generator.hilbert_mapping(opcodes, global_frequencies)

        # Save image
        output_path = generator._save_image(img_data, file_path, label)
        return {"label": label, "file": file_path, "output": output_path}

    except Exception as e:
        return {"label": "error", "file": file_path, "error": str(e)}


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
    parser.add_argument('--workers', type=int, default=multiprocessing.cpu_count() - 1,
                        help='Number of worker processes to use')
    parser.add_argument('--simplified', action='store_true',
                        help='Use simplified conversion with reduced calculations for speed')
    parser.add_argument('--cache-embeddings', action='store_true',
                        help='Cache embeddings in memory to avoid repeated lookups')

    args = parser.parse_args()

    # Create output directory and subdirectories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "benign"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "malware"), exist_ok=True)

    # Initialize the generator
    generator = MalwareImageGenerator(
        args.model,
        output_dir=args.output_dir,
        img_size=(args.img_size, args.img_size)
    )

    # Calculate global opcode frequencies for better sampling
    global_frequencies = generator.calculate_global_frequencies(args.data_dir)

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

    # Apply optimizations to the generator if needed
    if args.simplified:
        print("Using simplified conversion mode for better performance")

        # Monkey patch the generator with a simplified version of hilbert_mapping
        def simplified_hilbert_mapping(self, opcode_sequence, global_frequencies=None):
            """Simplified version of hilbert_mapping for better performance"""
            # Calculate the order based on image size
            order = int(np.log2(self.img_size[0]))
            n = 2 ** order
            total_pixels = n * n

            # Simple sampling strategy
            if len(opcode_sequence) > total_pixels:
                # Uniform sampling - much faster
                indices = np.linspace(0, len(opcode_sequence) - 1, total_pixels, dtype=int)
                sampled_opcodes = [opcode_sequence[i] for i in indices]
            else:
                # Use all opcodes and repeat if necessary
                repeats = int(np.ceil(total_pixels / len(opcode_sequence)))
                sampled_opcodes = (opcode_sequence * repeats)[:total_pixels]

            # Get embeddings
            embeddings = np.array([self.get_opcode_embedding(op) for op in sampled_opcodes])

            # Quick processing - use just the first dimension
            values = embeddings[:, 0]

            # Normalize to 0-1
            min_val = values.min()
            max_val = values.max()
            if max_val > min_val:
                values = (values - min_val) / (max_val - min_val)
            else:
                values = np.linspace(0, 1, len(values))

            # Create image data
            img_data = np.zeros(self.img_size)

            # Map using Hilbert curve
            for i in range(min(total_pixels, len(values))):
                x, y = self._d2xy(n, i)
                if x < self.img_size[0] and y < self.img_size[1]:
                    img_data[y, x] = values[i]

            return img_data

        # Replace the original method with our simplified version
        generator.hilbert_mapping = simplified_hilbert_mapping.__get__(generator, MalwareImageGenerator)

    # Cache embeddings if requested
    if args.cache_embeddings:
        print("Caching embeddings for faster lookups")
        # Create a cache for opcode embeddings
        embedding_cache = {}
        original_get_opcode_embedding = generator.get_opcode_embedding

        def cached_get_opcode_embedding(self, opcode):
            if opcode in embedding_cache:
                return embedding_cache[opcode]
            embedding = original_get_opcode_embedding(opcode)
            embedding_cache[opcode] = embedding
            return embedding

        generator.get_opcode_embedding = cached_get_opcode_embedding.__get__(generator, MalwareImageGenerator)

    # Combine files
    selected_files = benign_files + selected_malware_files

    # Process files in parallel
    processed = {"benign": 0, "malware": 0, "skipped": 0, "errors": 0}

    print(f"Starting image generation with {args.workers} workers...")

    # Prepare arguments for each file
    process_args = [(file_path, generator, global_frequencies, args.min_opcodes)
                    for file_path in selected_files]

    # Use single process mode for debugging if needed
    if args.workers <= 0:
        print("Running in single process mode for debugging")
        results = []
        for arg in tqdm(process_args, desc="Converting"):
            results.append(process_file(arg))

        for result in results:
            if result["label"] == "benign" or result["label"] == "malware":
                processed[result["label"]] += 1
            elif result["label"] == "skipped":
                processed["skipped"] += 1
            else:
                processed["errors"] += 1
                print(f"Error processing {result['file']}: {result.get('error', 'Unknown error')}")
    else:
        # Use multiprocessing
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_file, arg) for arg in process_args]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Converting"):
                try:
                    result = future.result()
                    if result["label"] == "benign" or result["label"] == "malware":
                        processed[result["label"]] += 1
                    elif result["label"] == "skipped":
                        processed["skipped"] += 1
                    else:
                        processed["errors"] += 1
                        print(f"Error processing {result['file']}: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    print(f"Error with future: {e}")
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
    # Required for multiprocessing on macOS
    multiprocessing.set_start_method('spawn', force=True)
    main()