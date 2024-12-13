import os
import numpy as np
import matplotlib.pyplot as plt

# Create a single figure before the loop
plt.figure(figsize=(10, 6))

def plot_data(file_path):
    wavelengths = []
    values = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) == 2:
                wavelength, value = map(float, parts)
                wavelengths.append(wavelength)
                values.append(value)

    # Plot on the existing figure
    site_name = os.path.basename(file_path).split('.')[0]
    plt.plot(wavelengths, values, label=site_name)

# Directory containing the data files
data_dir = "/Users/emmabelhadfa/Documents/Oxford/spectrometer/size"  # Replace with your folder path
file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.dpt')]

# Plot all files
for file_path in file_paths:
    plot_data(file_path)

# Add labels and show the plot after all data is plotted
plt.xlabel('Wavenumber (cm⁻¹)')
plt.ylabel('Emissivity')
plt.title('Emissivity Spectra of San Carlos Olivine - Grain Size Comparison')
plt.legend()
plt.grid(True)
plt.gca().invert_xaxis()
plt.show()
