import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from gensim.models import Word2Vec
import multiprocessing
from tqdm import tqdm
import seaborn as sns
from sklearn.manifold import TSNE
import re
import logging
import time
import json
import pickle
from adjustText import adjust_text

# Set up logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class OpcodeWord2VecTrainer:
    def __init__(self, base_dir, vector_size=100, window=5, min_count=5, workers=None, sg=1):
        """
        Initialize the Word2Vec trainer for opcode sequences.

        Args:
            base_dir (str): Base directory containing malware and benign samples
            vector_size (int): Dimensionality of the word vectors
            window (int): Maximum distance between current and predicted word
            min_count (int): Minimum count of words to consider
            workers (int, optional): Number of worker threads
            sg (int): Training algorithm: 1 for skip-gram; 0 for CBOW
        """
        self.base_dir = base_dir
        self.benign_dir = os.path.join(base_dir, 'Benign')
        self.malware_dir = os.path.join(base_dir, 'v077_clean')

        # Word2Vec parameters
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers if workers else multiprocessing.cpu_count() - 1
        self.sg = sg

        # Initialize storage for collected data
        self.file_lengths = defaultdict(dict)
        self.opcode_frequencies = defaultdict(Counter)
        self.unique_opcodes = set()
        self.model = None
        self.total_opcodes = {'malware': 0, 'benign': 0}

        # Initialize storage for sequence samples
        self.sequences = {
            'malware': [],
            'benign': []
        }

        # Mapping of filepath to label
        self.file_labels = {}

    def collect_file_paths(self):
        """
        Collect all file paths from the directory structure.

        Returns:
            dict: Dictionary with 'malware' and 'benign' keys, each containing a list of file paths
        """
        file_paths = {
            'malware': [],
            'benign': []
        }

        # Collect benign files
        for root, _, files in os.walk(self.benign_dir):
            for file in files:
                if file.endswith('.txt') or not file.endswith(('.py', '.md', '.json')):  # Adjust as needed
                    path = os.path.join(root, file)
                    file_paths['benign'].append(path)
                    self.file_labels[path] = 'benign'

        # Collect malware files
        for root, _, files in os.walk(self.malware_dir):
            for file in files:
                if file.endswith('.txt') or not file.endswith(('.py', '.md', '.json')):  # Adjust as needed
                    path = os.path.join(root, file)
                    file_paths['malware'].append(path)
                    self.file_labels[path] = 'malware'

        print(f"Found {len(file_paths['benign'])} benign files and {len(file_paths['malware'])} malware files")
        return file_paths

    def get_file_lengths(self, file_paths):
        """
        Get the length (number of lines) of each file.

        Args:
            file_paths (dict): Dictionary with 'malware' and 'benign' keys, each containing a list of file paths

        Returns:
            dict: Dictionary with file paths as keys and their lengths as values
        """
        for category in ['benign', 'malware']:
            for path in tqdm(file_paths[category], desc=f"Getting {category} file lengths"):
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        count = sum(1 for _ in f)
                        self.file_lengths[category][path] = count
                except Exception as e:
                    print(f"Error reading {path}: {e}")

        return self.file_lengths

    def select_stratified_samples(self, file_lengths, n_samples=1000, n_strata=4):
        """
        Select files with stratification based on length.

        Args:
            file_lengths (dict): Dictionary mapping filenames to their lengths
            n_samples (int): Number of files to select
            n_strata (int): Number of length-based strata

        Returns:
            list: List of selected filenames
        """
        # Get all lengths
        lengths = list(file_lengths.values())

        if not lengths:
            return []

        # Calculate percentiles for stratification
        percentiles = np.percentile(lengths, np.linspace(0, 100, n_strata + 1))

        # Group files by strata
        strata_files = defaultdict(list)
        for filename, length in file_lengths.items():
            for i in range(n_strata):
                if (i == n_strata - 1 and length >= percentiles[i]) or \
                        (percentiles[i] <= length < percentiles[i + 1]):
                    strata_files[i].append(filename)
                    break

        # Select samples from each stratum
        selected_files = []
        samples_per_stratum = n_samples // n_strata

        for stratum, files in strata_files.items():
            if len(files) <= samples_per_stratum:
                selected_files.extend(files)
            else:
                selected_files.extend(random.sample(files, samples_per_stratum))

        return selected_files

    def load_opcode_sequences(self, selected_files):
        sequences = []

        for path in tqdm(selected_files, desc="Loading opcode sequences"):
            try:
                category = self.file_labels[path]
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    sequence = []
                    for line in f:
                        line = line.strip()
                        if line:
                            opcode = re.split(r'[\s,;]', line)[0].strip()  # Extract first token
                            if opcode:
                                sequence.append(opcode)
                                self.unique_opcodes.add(opcode)
                                self.opcode_frequencies[category][opcode] += 1
                                self.total_opcodes[category] += 1

                    if sequence:
                        sequences.append(sequence)
                        self.sequences[category].append(sequence)
            except Exception as e:
                print(f" Error processing {path}: {e}")

        print(f"Loaded {len(sequences)} sequences.")
        print(f"Unique opcodes found: {len(self.unique_opcodes)}")  # Debug print

        return sequences

    def train_word2vec_model(self, sequences, epochs=5):
        """
        Train a Word2Vec model on opcode sequences.

        Args:
            sequences (list): List of opcode sequences
            epochs (int): Number of training epochs

        Returns:
            gensim.models.Word2Vec: Trained Word2Vec model
        """
        print(f"Training Word2Vec model with {len(sequences)} sequences...")
        start_time = time.time()

        model = Word2Vec(
            sentences=sequences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            #workers=self.workers,
            workers=min(4, multiprocessing.cpu_count()),
            sg=self.sg,  # Skip-gram model
            epochs= epochs
        )

        training_time = time.time() - start_time
        print(f"Model training completed in {training_time:.2f} seconds")
        print(f"Vocabulary size: {len(model.wv.key_to_index)}")

        self.model = model
        print(f"First 10 opcodes in vocabulary: {list(self.model.wv.index_to_key[:10])}")

        return model

    def save_model(self, output_path):
        """Save the trained model to disk."""
        if self.model:
            self.model.save(output_path)
            print(f"Model saved to {output_path}")
        else:
            print("No model to save. Train a model first.")

    def analyze_opcode_distributions(self):
        """
        Analyze and visualize opcode distributions.

        Returns:
            dict: Dictionary with analysis results
        """
        results = {}

        # 1. Calculate opcode distribution statistics
        for category in ['malware', 'benign']:
            if not self.opcode_frequencies[category]:
                continue

            # Get the most common opcodes
            most_common = self.opcode_frequencies[category].most_common(20)

            # Calculate percentage of total for each category
            total = sum(self.opcode_frequencies[category].values())
            percentages = [(op, count / total * 100) for op, count in most_common]

            results[f'{category}_top_opcodes'] = most_common
            results[f'{category}_percentages'] = percentages

        # 2. Visualize the distribution
        plt.figure(figsize=(15, 10))

        # Plot top opcodes for malware
        plt.subplot(2, 1, 1)
        if 'malware_percentages' in results:
            opcodes, percentages = zip(*results['malware_percentages'])
            plt.bar(opcodes, percentages, color='red', alpha=0.7)
            plt.title('Top 20 Opcodes in Malware Samples')
            plt.ylabel('Percentage of Total Opcodes')
            plt.xticks(rotation=45, ha='right')

        # Plot top opcodes for benign
        plt.subplot(2, 1, 2)
        if 'benign_percentages' in results:
            opcodes, percentages = zip(*results['benign_percentages'])
            plt.bar(opcodes, percentages, color='blue', alpha=0.7)
            plt.title('Top 20 Opcodes in Benign Samples')
            plt.ylabel('Percentage of Total Opcodes')
            plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig('opcode_distribution.png')
        plt.close()

        # 3. Calculate unique opcodes and overlap
        malware_opcodes = set(self.opcode_frequencies['malware'].keys())
        benign_opcodes = set(self.opcode_frequencies['benign'].keys())

        results['unique_malware_opcodes'] = len(malware_opcodes)
        results['unique_benign_opcodes'] = len(benign_opcodes)
        results['shared_opcodes'] = len(malware_opcodes.intersection(benign_opcodes))
        results['malware_only_opcodes'] = len(malware_opcodes - benign_opcodes)
        results['benign_only_opcodes'] = len(benign_opcodes - malware_opcodes)

        return results


    def visualize_embeddings(self, n_opcodes=100):
        """
        Visualize word embeddings using t-SNE.

        Args:
            n_opcodes (int): Number of most common opcodes to visualize

        Returns:
            None: Saves the visualization to a file
        """
        if self.model is None:
            print("No model available. Train a model first.")
            return

        # Get the most common opcodes
        all_frequencies = Counter()
        for category in ['malware', 'benign']:
            all_frequencies.update(self.opcode_frequencies[category])

        most_common = [op for op, _ in all_frequencies.most_common(n_opcodes)]

        # Get embeddings for these opcodes
        embeddings = []
        opcodes = []
        colors = []

        for opcode in most_common:
            if opcode in self.model.wv:
                opcodes.append(opcode)
                embeddings.append(self.model.wv[opcode])

                # Color based on which category uses this opcode more
                mal_freq = self.opcode_frequencies['malware'].get(opcode, 0) / max(1, self.total_opcodes['malware'])
                ben_freq = self.opcode_frequencies['benign'].get(opcode, 0) / max(1, self.total_opcodes['benign'])

                if mal_freq > ben_freq:
                    colors.append('red')  # More common in malware
                else:
                    colors.append('blue')  # More common in benign

        if not embeddings:
            print("No common opcodes found in the model.")
            return

        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
        reduced_embeddings = tsne.fit_transform(np.array(embeddings))

        # Plot
        plt.figure(figsize=(15, 12))  #Increased figure size for better label spacing
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=colors, alpha=0.7)

        # Store text objects for adjustment
        texts = []
        for i, opcode in enumerate(opcodes):
            x, y = reduced_embeddings[i]
            texts.append(plt.text(x, y, opcode, fontsize=8))  #Store text for adjustment

        # Automatically adjust labels to prevent overlap
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color='black', lw=0.5))

        # Legend
        plt.title('t-SNE Visualization of Opcode Embeddings')
        plt.legend(
            handles=[
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10,
                           label='More common in malware'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10,
                           label='More common in benign')
            ]
        )

        # Save and show
        plt.savefig('opcode_embeddings_tsne.png')
        plt.show()  #Display the plot interactively
        plt.close()

    def find_similar_opcodes(self, opcodes, top_n=10):
        """
        Find similar opcodes for a list of query opcodes.

        Args:
            opcodes (list): List of opcodes to query
            top_n (int): Number of similar opcodes to return

        Returns:
            dict: Dictionary mapping each query opcode to its similar opcodes
        """
        if self.model is None:
            print("No model available. Train a model first.")
            return {}

        results = {}

        for opcode in opcodes:
            if opcode in self.model.wv:
                similar = self.model.wv.most_similar(opcode, topn=top_n)
                results[opcode] = similar
            else:
                results[opcode] = f"Opcode '{opcode}' not in vocabulary"

        return results

    def analyze_sequence_complexity(self):
        """
        Analyze the complexity of opcode sequences.

        Returns:
            dict: Dictionary with complexity metrics
        """
        results = {}

        # Calculate metrics for each category
        for category in ['malware', 'benign']:
            if not self.sequences[category]:
                continue

            # Calculate sequence lengths
            lengths = [len(seq) for seq in self.sequences[category]]

            # Calculate unique opcodes per sequence
            unique_counts = [len(set(seq)) for seq in self.sequences[category]]

            # Calculate unique/total ratio (complexity)
            complexity = [len(set(seq)) / max(1, len(seq)) for seq in self.sequences[category]]

            results[f'{category}_length_mean'] = np.mean(lengths)
            results[f'{category}_length_median'] = np.median(lengths)
            results[f'{category}_length_std'] = np.std(lengths)
            results[f'{category}_unique_mean'] = np.mean(unique_counts)
            results[f'{category}_unique_median'] = np.median(unique_counts)
            results[f'{category}_complexity_mean'] = np.mean(complexity)
            results[f'{category}_complexity_median'] = np.median(complexity)

            # Visualize the distribution of complexity
            plt.figure(figsize=(10, 6))
            plt.hist(complexity, bins=30, alpha=0.7, color='red' if category == 'malware' else 'blue')
            plt.title(f'Opcode Sequence Complexity Distribution for {category.capitalize()}')
            plt.xlabel('Complexity (Unique/Total Ratio)')
            plt.ylabel('Number of Sequences')
            plt.savefig(f'{category}_complexity_distribution.png')
            plt.close()

        # Compare the distributions
        if self.sequences['malware'] and self.sequences['benign']:
            mal_complexity = [len(set(seq)) / max(1, len(seq)) for seq in self.sequences['malware']]
            ben_complexity = [len(set(seq)) / max(1, len(seq)) for seq in self.sequences['benign']]

            plt.figure(figsize=(12, 6))
            plt.hist(mal_complexity, bins=30, alpha=0.5, color='red', label='Malware')
            plt.hist(ben_complexity, bins=30, alpha=0.5, color='blue', label='Benign')
            plt.title('Comparison of Opcode Sequence Complexity')
            plt.xlabel('Complexity (Unique/Total Ratio)')
            plt.ylabel('Number of Sequences')
            plt.legend()
            plt.savefig('complexity_comparison.png')
            plt.close()

        return results

    def run_pipeline(self, malware_samples=1000, benign_samples=1000, output_model_path='opcode_word2vec.model'):
        """
        Run the complete pipeline from file collection to model training and analysis.

        Args:
            malware_samples (int): Number of malware samples to use
            benign_samples (int): Number of benign samples to use
            output_model_path (str): Path to save the trained model

        Returns:
            dict: Analysis results
        """
        # Collect file paths
        if os.path.exists('file_paths.pkl'):
            with open('file_paths.pkl', 'rb') as f:
                file_paths = pickle.load(f)
            print("Loaded file paths from cache.")
        else:
            file_paths = self.collect_file_paths()
            with open('file_paths.pkl', 'wb') as f:
                pickle.dump(file_paths, f)
            print("Saved file paths.")

        # Get file lengths
        if os.path.exists('file_lengths.pkl'):
            with open('file_lengths.pkl', 'rb') as f:
                file_lengths = pickle.load(f)
            print("Loaded file lengths from cache.")
        else:
            file_lengths = self.get_file_lengths(file_paths)
            with open('file_lengths.pkl', 'wb') as f:
                pickle.dump(file_lengths, f)
            print("Saved file lengths.")

        # Select balanced samples
        if os.path.exists('selected_files.pkl'):
            with open('selected_files.pkl', 'rb') as f:
                selected_files = pickle.load(f)
            print("Loaded selected files from cache.")

            # Reconstruct malware_files and benign_files correctly
            if os.path.exists('file_lengths.pkl'):
                with open('file_lengths.pkl', 'rb') as f:
                    file_lengths = pickle.load(f)
                print("Loaded file lengths from cache.")

                # Extract only the selected files that match the stratified sample distribution
                malware_files = [f for f in selected_files if f in file_lengths['malware']]
                benign_files = [f for f in selected_files if f in file_lengths['benign']]
            else:
                print("Warning: file_lengths.pkl not found. Using all selected files as a fallback.")
                malware_files = [f for f in selected_files if 'v077_clean' in f]
                benign_files = [f for f in selected_files if 'Benign' in f]

        else:
            # Correctly stratify samples when `selected_files.pkl` is missing
            malware_files = self.select_stratified_samples(file_lengths['malware'], n_samples=malware_samples)
            benign_files = self.select_stratified_samples(file_lengths['benign'], n_samples=benign_samples)
            selected_files = malware_files + benign_files
            with open('selected_files.pkl', 'wb') as f:
                pickle.dump(selected_files, f)
            print("Saved selected files.")

        print(f"Selected {len(selected_files)} files for training.")

        # Load opcode sequences
        if os.path.exists('sequences.pkl'):
            with open('sequences.pkl', 'rb') as f:
                sequences = pickle.load(f)

            print("Loaded opcode sequences from cache.")
            if len(sequences) == 0:
                print("Warning: sequences.pkl is empty. Reloading sequences...")
                sequences = self.load_opcode_sequences(selected_files)

        else:
            sequences = self.load_opcode_sequences(selected_files)
            with open('sequences.pkl', 'wb') as f:
                pickle.dump(sequences, f)
            print("Saved opcode sequences.")

        # Train Word2Vec model (only if it doesn't exist)

        print(f"Unique opcodes before training: {len(self.unique_opcodes)}")

        if os.path.exists(output_model_path):
            print(f"Found existing model at {output_model_path}. Loading instead of training...")
            self.model = Word2Vec.load(output_model_path)
        else:
            print("No existing model found. Training a new Word2Vec model...")
            self.model = self.train_word2vec_model(sequences)
            self.save_model(output_model_path)  # Save trained model
            print("Saved trained model.")

        # 7. Analyze and visualize
        print("Analyzing opcode distributions...")
        distribution_results = self.analyze_opcode_distributions()

        print("Analyzing sequence complexity...")
        complexity_results = self.analyze_sequence_complexity()

        print("Visualizing embeddings...")
        self.visualize_embeddings()

        # 8. Find similar opcodes for some common examples
        # First get a few common opcodes from each category
        common_malware = [op for op, _ in self.opcode_frequencies['malware'].most_common(5)]
        common_benign = [op for op, _ in self.opcode_frequencies['benign'].most_common(5)]

        similar_results = self.find_similar_opcodes(common_malware + common_benign)

        # 9. Combine all results
        # Ensure self.model is defined before using it
        if hasattr(self, 'model') and self.model is not None:
            model_info = {
                'vocabulary_size': len(self.model.wv.key_to_index),
                'vector_size': self.vector_size,
                'window': self.window,
                'min_count': self.min_count
            }
        else:
            print("Warning: Model is not loaded or trained yet.")
            model_info = {
                'vocabulary_size': 0,  # Default values to prevent errors
                'vector_size': self.vector_size,
                'window': self.window,
                'min_count': self.min_count
            }

        # Combine all results
        all_results = {
            'file_counts': {
                'malware_total': len(file_paths['malware']),
                'benign_total': len(file_paths['benign']),
                'malware_used': len(malware_files),
                'benign_used': len(benign_files)
            },
            'model_info': model_info,
            'distribution': distribution_results,
            'complexity': complexity_results,
            'similar_opcodes': similar_results
        }

        # 10 Save results to a text report
        with open('word2vec_analysis_report.txt', 'w') as f:
            f.write("Word2Vec Training and Analysis Report\n")
            f.write("=====================================\n\n")

            f.write("File Statistics\n")
            f.write("--------------\n")
            f.write(f"Total malware files: {all_results['file_counts']['malware_total']}\n")
            f.write(f"Total benign files: {all_results['file_counts']['benign_total']}\n")
            f.write(f"Malware files used for training: {all_results['file_counts']['malware_used']}\n")
            f.write(f"Benign files used for training: {all_results['file_counts']['benign_used']}\n\n")

            f.write("Model Information\n")
            f.write("-----------------\n")
            f.write(f"Vocabulary size: {all_results['model_info']['vocabulary_size']}\n")
            f.write(f"Vector size: {all_results['model_info']['vector_size']}\n")
            f.write(f"Context window: {all_results['model_info']['window']}\n")
            f.write(f"Minimum count: {all_results['model_info']['min_count']}\n\n")

            f.write("Opcode Distribution\n")
            f.write("------------------\n")
            f.write(f"Unique opcodes in malware: {distribution_results.get('unique_malware_opcodes', 'N/A')}\n")
            f.write(f"Unique opcodes in benign: {distribution_results.get('unique_benign_opcodes', 'N/A')}\n")
            f.write(f"Shared opcodes: {distribution_results.get('shared_opcodes', 'N/A')}\n")
            f.write(f"Opcodes only in malware: {distribution_results.get('malware_only_opcodes', 'N/A')}\n")
            f.write(f"Opcodes only in benign: {distribution_results.get('benign_only_opcodes', 'N/A')}\n\n")

            f.write("Sequence Complexity\n")
            f.write("------------------\n")

            # Ensure numeric values before formatting
            malware_length_mean = complexity_results.get('malware_length_mean', 'N/A')
            benign_length_mean = complexity_results.get('benign_length_mean', 'N/A')
            malware_complexity_mean = complexity_results.get('malware_complexity_mean', 'N/A')
            benign_complexity_mean = complexity_results.get('benign_complexity_mean', 'N/A')

            f.write(f"Malware mean length: {malware_length_mean:.2f}\n" if isinstance(malware_length_mean, (
            int, float)) else f"Malware mean length: {malware_length_mean}\n")
            f.write(f"Benign mean length: {benign_length_mean:.2f}\n" if isinstance(benign_length_mean, (
            int, float)) else f"Benign mean length: {benign_length_mean}\n")
            f.write(f"Malware mean complexity: {malware_complexity_mean:.2f}\n" if isinstance(malware_complexity_mean, (
            int, float)) else f"Malware mean complexity: {malware_complexity_mean}\n")
            f.write(f"Benign mean complexity: {benign_complexity_mean:.2f}\n" if isinstance(benign_complexity_mean, (
            int, float)) else f"Benign mean complexity: {benign_complexity_mean}\n")

            f.write("Similar Opcodes\n")
            f.write("--------------\n")
            for opcode, similar in similar_results.items():
                f.write(f"Similar to '{opcode}':\n")
                if isinstance(similar, list):
                    for sim_op, sim_score in similar:
                        f.write(f"  - {sim_op} (similarity: {sim_score:.4f})\n")
                else:
                    f.write(f"  {similar}\n")
                f.write("\n")

        # 10. save as JSON

        # Save opcode frequencies
        with open('opcode_frequencies.json', 'w') as f:
            json.dump({
                'malware': dict(self.opcode_frequencies['malware']),
                'benign': dict(self.opcode_frequencies['benign'])
            }, f)

        # Save complexity metrics
        complexity_data = {
            'malware': [len(set(seq)) / len(seq) for seq in self.sequences['malware']],
            'benign': [len(set(seq)) / len(seq) for seq in self.sequences['benign']]
        }
        with open('complexity_data.json', 'w') as f:
            json.dump(complexity_data, f)

        print("Analysis complete. Results saved to word2vec_analysis_report.txt")
        print("Visualizations saved as PNG files.")

        return all_results


# Example usage
if __name__ == "__main__":
    trainer = OpcodeWord2VecTrainer(
        base_dir="Data",  # Update this path to your actual base directory
        vector_size=100,
        window=5,
        min_count=1
    )

    results = trainer.run_pipeline(
        malware_samples=2500,  # Increased sample size
        benign_samples=2500,  # Using most of available benign samples
        output_model_path='opcode_word2vec.model'
    )
