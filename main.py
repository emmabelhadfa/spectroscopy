import os
import numpy as np
from driftquantifier import analyze_emissivity_stability, plot_stability_summary
from emissivityplotter import plot_data
import matplotlib.pyplot as plt

def load_data_from_file(file_path):
    """Load data from a single .dpt file"""
    wavelengths = []
    values = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) == 2:
                wavelength, value = map(float, parts)
                wavelengths.append(wavelength)
                values.append(value)
    return np.array(wavelengths), np.array(values)

def main():
    # Define the directory containing your data files
    data_dir = "/Users/emmabelhadfa/Documents/Oxford/spectrometer/drift"
    
    # Get all .dpt files in the directory, but filter out incomplete paths
    file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                 if f.endswith('.dpt') and not f.endswith('_')]
    
    # Sort the file paths to ensure consistent ordering
    file_paths.sort()
    
    # Debug print
    print(f"Found {len(file_paths)} valid .dpt files:")
    for file in file_paths:
        print(f"- {file}")
    
    if len(file_paths) == 0:
        print("No valid .dpt files found in the directory!")
        return
        
    # Create figure for reflectance plots
    plt.figure(figsize=(12, 8))
    
    # Load and process all data
    all_data = []
    wavenumbers = None
    
    print(f"\nProcessing {len(file_paths)} files...")
    for i, file_path in enumerate(file_paths):
        try:
            # Load the data for stability analysis
            wn, values = load_data_from_file(file_path)
            if wavenumbers is None:
                wavenumbers = wn
            all_data.append(values)
            
            # Plot the data with a label
            plt.plot(wn, values, label=f'Scan {i+1}', alpha=0.5)
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    # Convert all data to numpy array
    reflectance_data = np.array(all_data)
    
    # Perform stability analysis
    results = analyze_emissivity_stability(reflectance_data, wavenumbers, data_dir=data_dir)
    
    # Print stability summary
    plot_stability_summary(results)

    
    # Add call to plot_data
    plot_data(wavenumbers, reflectance_data)


if __name__ == "__main__":
    main()