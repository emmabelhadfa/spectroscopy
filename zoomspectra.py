import os
import numpy as np
import matplotlib.pyplot as plt

def read_dpt_file(file_path):
    data = np.loadtxt(file_path, skiprows=1, delimiter=',')
    wavelengths = data[:, 0]
    values = data[:, 1]
    return wavelengths, values

# Create figure with three subplots (original + two zoomed regions)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

# Directory containing the data files
data_dir = "/Users/emmabelhadfa/Documents/Oxford/spectrometer/orientation"
file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.dpt')]

# Plot all files in each subplot
for file_path in file_paths:
    label = os.path.basename(file_path).replace('.0.dpt', '')
    wavelengths, values = read_dpt_file(file_path)
    
    # Full range plot (150-2000)
    mask_full = (wavelengths >= 150) & (wavelengths <= 2000)
    ax1.plot(wavelengths[mask_full], values[mask_full], label=label)
    
    # First feature (800-1200)
    mask_feature1 = (wavelengths >= 800) & (wavelengths <= 1200)
    ax2.plot(wavelengths[mask_feature1], values[mask_feature1], label=label)
    
    # Second feature (200-800)
    mask_feature2 = (wavelengths >= 200) & (wavelengths <= 800)
    ax3.plot(wavelengths[mask_feature2], values[mask_feature2], label=label)

# Configure full range plot
ax1.set_xlabel('Wavenumber (cm⁻¹)')
ax1.set_ylabel('Reflectance')
ax1.set_title('Full Range Spectra (150-2000 cm⁻¹) - Orientation')
handles, labels = ax1.get_legend_handles_labels()
labels, handles = zip(*sorted(zip(labels, handles)))
ax1.legend(handles, labels)
ax1.grid(True)
ax1.invert_xaxis()

# Configure first feature plot
ax2.set_xlabel('Wavenumber (cm⁻¹)')
ax2.set_ylabel('Reflectance')
ax2.set_title('Zoomed Spectra (800-1200 cm⁻¹)')
ax2.grid(True)
ax2.invert_xaxis()

# Configure second feature plot
ax3.set_xlabel('Wavenumber (cm⁻¹)')
ax3.set_ylabel('Reflectance')
ax3.set_title('Zoomed Spectra (200-800 cm⁻¹)')
ax3.grid(True)
ax3.invert_xaxis()

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure
plt.savefig('/Users/emmabelhadfa/Documents/Oxford/spectrometer/orientation_zoomed.png', 
            dpi=300, 
            bbox_inches='tight')

# If you still want to display the plot as well, keep this line
plt.show()