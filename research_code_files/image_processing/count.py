import os
import numpy as np
import matplotlib.pyplot as plt


def count_lines_in_file(file_path):
    """Count the number of lines in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return sum(1 for _ in f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None  # Exclude problematic files


def get_file_lengths(directory):
    """Recursively collect line counts from all files in a directory."""
    lengths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            line_count = count_lines_in_file(file_path)
            if line_count is not None:
                lengths.append(line_count)
    return np.array(lengths)  # Convert to NumPy array for easier analysis


# Define paths
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "Data")
malware_dir = os.path.join(data_dir, "v077_clean")
benign_dir = os.path.join(data_dir, "Benign")

# Get line counts
malware_lengths = get_file_lengths(malware_dir)
benign_lengths = get_file_lengths(benign_dir)


# Compute statistics with quartile counts
def print_stats(name, data):
    print(f"\n{name} Statistics:")
    if len(data) == 0:
        print("No data available.")
        return

    # Compute quartiles
    q1 = np.percentile(data, 25)  # First quartile (25%)
    q2 = np.median(data)  # Median (50%)
    q3 = np.percentile(data, 75)  # Third quartile (75%)

    # Ensure mutually exclusive quartile counts
    q1_count = np.sum(data < q1)  # Less than Q1
    q2_count = np.sum((data >= q1) & (data < q2))  # Between Q1 and Q2
    q3_count = np.sum((data >= q2) & (data < q3))  # Between Q2 and Q3
    q4_count = np.sum(data >= q3)  # Greater than or equal to Q3

    print(f"  Min: {np.min(data)}")
    print(f"  Q1 (25th percentile): {q1:.1f}  | Count: {q1_count} files")
    print(f"  Median (Q2 - 50th percentile): {q2:.1f}  | Count: {q2_count} files")
    print(f"  Q3 (75th percentile): {q3:.1f}  | Count: {q3_count} files")
    print(f"  Max: {np.max(data)}  | Count: {q4_count} files")
    print(f"  Mean: {np.mean(data):.2f}")
    print(f"  Standard Deviation: {np.std(data):.2f}")


print_stats("Malware", malware_lengths)
print_stats("Benign", benign_lengths)


# Handle extreme outliers using percentiles (Optional)
def filter_outliers(data, lower=1, upper=99):
    """Filter extreme outliers using percentiles."""
    lower_bound = np.percentile(data, lower)
    upper_bound = np.percentile(data, upper)
    return data[(data >= lower_bound) & (data <= upper_bound)]


filtered_malware_lengths = filter_outliers(malware_lengths)
filtered_benign_lengths = filter_outliers(benign_lengths)

# Plot with logarithmic scale to better visualize range differences
plt.figure(figsize=(8, 6))
plt.boxplot([filtered_malware_lengths, filtered_benign_lengths], labels=["Malware", "Benign"])
plt.yscale("log")  # Log scale for better visualization
plt.ylabel("Number of Lines (Log Scale)")
plt.title("Distribution of Opcode File Lengths")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Show plot
plt.show()
