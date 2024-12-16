import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import os

def analyze_emissivity_stability(emissivity_data, wavelengths, data_dir=None, gold_standard=None):
    """
    Analyze stability of emissivity measurements across multiple scans
    
    Parameters:
    emissivity_data: np.array of shape (n_scans, n_wavelengths)
    wavelengths: np.array of wavelength values
    data_dir: str, directory where the data is stored (optional)
    gold_standard: np.array of reference emissivity values (optional)
    """
    # Create output directory if it doesn't exist
    output_dir = "stability_analysis_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    n_scans, n_wavelengths = emissivity_data.shape
    scan_numbers = np.arange(n_scans)
    
    # 1. Plot emissivity over time for selected wavelengths
    plt.figure(figsize=(12, 6))
    selected_indices = np.linspace(0, n_wavelengths-1, 5, dtype=int)
    for idx in selected_indices:
        plt.plot(scan_numbers, emissivity_data[:, idx], 
                label=f'Wavenumber {wavelengths[idx]:.2f}')
    plt.xlabel('Scan Number')
    plt.ylabel('Reflectance')
    plt.title('Reflectance Variation Over Time for Selected Wavenumbers')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'time_variation.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # 2. Drift Analysis
    mean_emissivity = np.mean(emissivity_data, axis=1)
    linear_fit = np.polyfit(scan_numbers, mean_emissivity, 1)
    poly_fit = np.polyfit(scan_numbers, mean_emissivity, 3)
    
    plt.figure(figsize=(12, 6))
    plt.plot(scan_numbers, mean_emissivity, 'o', label='Mean Reflectance')
    plt.plot(scan_numbers, np.polyval(linear_fit, scan_numbers), 
             'r-', label=f'Linear Drift (rate={linear_fit[0]:.2e}/scan)')
    plt.plot(scan_numbers, np.polyval(poly_fit, scan_numbers), 
             'g-', label='Polynomial Fit')
    plt.xlabel('Scan Number')
    plt.ylabel('Mean Reflectance')
    plt.title('Overall Reflectance Drift')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'drift_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # 3. Systematic Error Analysis
    mean_spectrum = np.mean(emissivity_data, axis=0)
    deviations = emissivity_data - mean_spectrum
    
    plt.figure(figsize=(12, 6))
    for i in range(len(deviations)):
        plt.plot(wavelengths, deviations[i], alpha=0.2, color='blue')
    plt.plot(wavelengths, np.zeros_like(wavelengths), 'r--', label='Mean', linewidth=2)
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Deviation from Mean')
    plt.title('Distribution of Scan Deviations from Mean')
    plt.grid(True)
    plt.legend()
    plt.gca().invert_xaxis()
    plt.savefig(os.path.join(output_dir, 'systematic_deviations.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # 4. Random Error Analysis
    std_dev = np.std(emissivity_data, axis=0)
    snr = mean_spectrum / std_dev

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    ax1.plot(wavelengths, std_dev)
    ax1.set_xlabel('Wavenumber (cm⁻¹)')
    ax1.set_ylabel('Standard Deviation')
    ax1.set_title('Random Error vs Wavenumber')
    ax1.grid(True)
    ax1.invert_xaxis()
    
    ax2.plot(wavelengths, snr)
    ax2.set_xlabel('Wavenumber (cm⁻¹)')
    ax2.set_ylabel('Signal-to-Noise Ratio')
    ax2.set_title('Signal-to-Noise Ratio vs Wavenumber')
    ax2.grid(True)
    ax2.invert_xaxis()
    
    plt.tight_layout()
    
    # Save in both locations
    plt.savefig(os.path.join(output_dir, 'random_error_and_snr.png'), dpi=300, bbox_inches='tight')
    if data_dir:
        plt.savefig(os.path.join(data_dir, 'random_error_and_snr.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # 5. Residual Analysis
    linear_trend = np.outer(np.polyval(linear_fit, scan_numbers), np.ones(n_wavelengths))
    residuals = emissivity_data - linear_trend
    
    plt.figure(figsize=(12, 6))
    plt.hist(residuals.flatten(), bins=50, density=True)
    plt.xlabel('Residual Value')
    plt.ylabel('Density')
    plt.title('Distribution of Residuals')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'residuals.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Return summary statistics
    results = {
        'drift_rate': linear_fit[0],
        'mean_systematic_error': np.mean(np.abs(deviations)),
        'mean_random_error': np.mean(std_dev),
        'mean_snr': np.mean(snr)
    }
    
    return results

def plot_stability_summary(results):
    """Print summary of stability analysis"""
    print("\nStability Analysis Summary:")
    print(f"Drift Rate: {results['drift_rate']:.2e} per scan")
    if results['mean_systematic_error'] is not None:
        print(f"Mean Systematic Error: {results['mean_systematic_error']:.2e}")
    print(f"Mean Random Error: {results['mean_random_error']:.2e}")
    print(f"Mean SNR: {results['mean_snr']:.2f}")
