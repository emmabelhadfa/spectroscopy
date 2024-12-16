import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def read_dpt_file(file_path):
    data = np.loadtxt(file_path, skiprows=1, delimiter=',')
    wavelengths = data[:, 0]
    values = data[:, 1]
    values = np.nan_to_num(values, nan=0.0)
    return wavelengths, values

def spectral_angle(spectrum1, spectrum2):
    """
    Calculate the spectral angle between two spectra.
    Returns angle in degrees.
    """
    dot_product = np.dot(spectrum1, spectrum2)
    norm1 = np.linalg.norm(spectrum1)
    norm2 = np.linalg.norm(spectrum2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    angle = np.arccos(np.clip(dot_product / (norm1 * norm2), -1.0, 1.0))
    return np.degrees(angle)

# Directory containing the data files
data_dir = "/Users/emmabelhadfa/Documents/Oxford/spectrometer/orientation"
file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.dpt')]

# Create arrays to store all spectra and their labels
all_spectra = []
labels = []
wavelengths = None

# Read all spectra
for file_path in file_paths:
    label = os.path.basename(file_path).replace('.0.dpt', '')
    wl, values = read_dpt_file(file_path)
    
    if wavelengths is None:
        wavelengths = wl
    
    all_spectra.append(values)
    labels.append(label)

# Sort the labels and spectra numerically
# Convert labels to integers for sorting
label_numbers = [int(label) for label in labels]
sorted_indices = np.argsort(label_numbers)
labels = [labels[i] for i in sorted_indices]
all_spectra = [all_spectra[i] for i in sorted_indices]

# Convert to numpy array
spectra_array = np.array(all_spectra)

# Calculate SAM angles between all pairs
n_spectra = len(all_spectra)
sam_matrix = np.zeros((n_spectra, n_spectra))

for i in range(n_spectra):
    for j in range(n_spectra):
        sam_matrix[i, j] = spectral_angle(all_spectra[i], all_spectra[j])

# Create heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(sam_matrix, 
            xticklabels=labels, 
            yticklabels=labels, 
            annot=True, 
            fmt='.2f', 
            cmap='YlOrRd')
plt.title('Spectral Angle Mapper (SAM) Analysis\nAngles in Degrees - Orientation')
plt.xlabel('Spectrum')
plt.ylabel('Spectrum')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Adjust layout to prevent label cutoff
plt.tight_layout()
plt.show()

# Print summary statistics
print("\nSAM Analysis Summary:")
print(f"Maximum angle: {np.max(sam_matrix[~np.eye(n_spectra, dtype=bool)]):.2f}°")
print(f"Minimum angle (excluding self-comparison): {np.min(sam_matrix[~np.eye(n_spectra, dtype=bool)]):.2f}°")
print(f"Mean angle (excluding self-comparison): {np.mean(sam_matrix[~np.eye(n_spectra, dtype=bool)]):.2f}°")

# Find most different spectra
max_angle_idx = np.unravel_index(np.argmax(sam_matrix + np.eye(n_spectra) * -999), sam_matrix.shape)
print(f"\nMost different spectra:")
print(f"{labels[max_angle_idx[0]]} and {labels[max_angle_idx[1]]} with angle: {sam_matrix[max_angle_idx]:.2f}°")

# Calculate average angle for each spectrum
avg_angles = np.mean(sam_matrix, axis=1)
print("\nAverage angles for each spectrum:")
for label, avg_angle in zip(labels, avg_angles):
    print(f"{label}: {avg_angle:.2f}°")