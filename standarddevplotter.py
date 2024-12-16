import os
import numpy as np
import matplotlib.pyplot as plt

def read_dpt_file(file_path):
    data = np.loadtxt(file_path, skiprows=1, delimiter=',')
    wavelengths = data[:, 0]
    values = data[:, 1]
    return wavelengths, values

# Directory containing the data files
data_dir = "/Users/emmabelhadfa/Documents/Oxford/spectrometer/surface"
file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.dpt')]

# Create arrays to store all spectra
all_spectra = []
wavelengths = None

# Read all spectra
for file_path in file_paths:
    wl, values = read_dpt_file(file_path)
    
    if wavelengths is None:
        wavelengths = wl
    
    all_spectra.append(values)

# Convert to numpy array
spectra_array = np.array(all_spectra)

# Calculate mean spectrum and standard deviation
mean_spectrum = np.mean(spectra_array, axis=0)
std_spectrum = np.std(spectra_array, axis=0)

# Filter out data points where the wavenumber is less than or equal to 150 cm⁻¹
mask = wavelengths > 150

# Create figure with two subplotz
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Plot 1: Mean spectrum with standard deviation band
ax1.plot(wavelengths[mask], mean_spectrum[mask], 'b-', label='Mean Spectrum')
ax1.fill_between(wavelengths[mask], 
                 mean_spectrum[mask] - std_spectrum[mask], 
                 mean_spectrum[mask] + std_spectrum[mask], 
                 color='blue', alpha=0.2, 
                 label='±1 Standard Deviation')
ax1.set_xlabel('Wavenumber (cm⁻¹)')
ax1.set_ylabel('Reflectance')
ax1.set_title('Mean Spectrum with Standard Deviation - Surface Features')
ax1.legend()
ax1.grid(True)
ax1.invert_xaxis()

# Plot 2: Standard deviation vs wavenumber
ax2.plot(wavelengths[mask], std_spectrum[mask], 'r-')
ax2.set_xlabel('Wavenumber (cm⁻¹)')
ax2.set_ylabel('Standard Deviation')
ax2.set_title('Standard Deviation vs Wavenumber - Surface Features')
ax2.grid(True)
ax2.invert_xaxis()

# Adjust layout and display
plt.tight_layout()
plt.show()

# Optional: Print some statistics
print("\nSpectral Statistics (for wavenumbers > 150 cm⁻¹):")
print(f"Maximum standard deviation: {np.max(std_spectrum[mask]):.4f}")
print(f"Minimum standard deviation: {np.min(std_spectrum[mask]):.4f}")
print(f"Mean standard deviation: {np.mean(std_spectrum[mask]):.4f}")
print(f"Wavenumber at maximum standard deviation: {wavelengths[mask][np.argmax(std_spectrum[mask])]:.1f} cm⁻¹")