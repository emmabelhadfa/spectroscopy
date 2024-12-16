import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def read_dpt_file(file_path):
    data = np.loadtxt(file_path, skiprows=1, delimiter=',')
    wavelengths = data[:, 0]
    values = data[:, 1]
    return wavelengths, values

# Directory containing the data files
data_dir = "/Users/emmabelhadfa/Documents/Oxford/spectrometer/surface"
file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.dpt')]

# Create arrays to store all spectra and their labels
all_spectra = []
labels = []
wavelengths = None

# Read all spectra
for file_path in file_paths:
    label = os.path.basename(file_path).replace('.*.dpt', '')
    wl, values = read_dpt_file(file_path)
    
    if wavelengths is None:
        wavelengths = wl
    
    all_spectra.append(values)
    labels.append(label)

# Convert to numpy array
X = np.array(all_spectra)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Create plots
plt.figure(figsize=(15, 10))

# Plot 1: Explained variance ratio
plt.subplot(2, 2, 1)
explained_variance = pca.explained_variance_ratio_ * 100
plt.bar(range(1, len(explained_variance) + 1), explained_variance)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance (%)')
plt.title('Explained Variance by Principal Components')

# Plot 2: Cumulative explained variance
plt.subplot(2, 2, 2)
cumulative_variance = np.cumsum(explained_variance)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance (%)')
plt.title('Cumulative Explained Variance')

# Plot 3: First two principal components
plt.subplot(2, 2, 3)
plt.scatter(X_pca[:, 0], X_pca[:, 1])
for i, label in enumerate(labels):
    plt.annotate(label, (X_pca[i, 0], X_pca[i, 1]))
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('First Two Principal Components')

# Plot 4: First three principal component loadings
plt.subplot(2, 2, 4)
for i in range(min(3, len(pca.components_))):
    plt.plot(wavelengths, pca.components_[i], label=f'PC{i+1}')
plt.xlabel('Wavenumber (cm⁻¹)')
plt.ylabel('Loading')
plt.title('Principal Component Loadings')
plt.legend()
plt.gca().invert_xaxis()

plt.tight_layout()
plt.show()

# Print the explained variance ratios
print("\nExplained variance ratios:")
for i, var in enumerate(explained_variance):
    print(f"PC{i+1}: {var:.2f}%")