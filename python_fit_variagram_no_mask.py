import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.optimize import curve_fit
from sklearn.preprocessing import PolynomialFeatures
from pyproj import Transformer
import LiCSBAS_io_lib as LiCS_lib
import os

def detrend(data):
    # Fit a polynomial of degree 1 (linear) and subtract to detrend
    poly = PolynomialFeatures(degree=1)
    X = np.arange(len(data)).reshape(-1, 1)
    poly_X = poly.fit_transform(X)
    coefs = np.linalg.lstsq(poly_X, data, rcond=None)[0]
    trend = np.dot(poly_X, coefs)
    detrended = data - trend
    return trend, detrended

def calculate_variogram(lons, lats, values):
    # Convert lat/lon to distances
    transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
    x, y = transformer.transform(lons, lats)
    
    # Compute distance matrix
    dist_matrix = distance_matrix(np.column_stack((x, y)), np.column_stack((x, y)))
    distances = dist_matrix[np.triu_indices(len(dist_matrix), k=1)]
    
    # Compute semi-variograms
    trend, detrended = detrend(values)
    semi_variances = []
    
    for i in range(len(detrended)):
        for j in range(i + 1, len(detrended)):
            semi_variances.append(0.5 * (detrended[i] - detrended[j]) ** 2)
    
    semi_variances = np.array(semi_variances)
    
    # Check for NaNs or Infs
    if np.any(np.isnan(distances)) or np.any(np.isinf(distances)):
        print("Distance matrix contains NaNs or Infs.")
        distances = np.nan_to_num(distances)  # Replace NaNs/Infs with zeros
    
    if np.any(np.isnan(semi_variances)) or np.any(np.isinf(semi_variances)):
        print("Semi-variances contain NaNs or Infs.")
        semi_variances = np.nan_to_num(semi_variances)  # Replace NaNs/Infs with zeros
    
    return distances, semi_variances, trend, detrended

def exponential_model(h, sill, nugget, range_):
    return nugget + (sill - nugget) * (1 - np.exp(-h / range_))

def estimate_initial_params(distances, semi_variograms):
    # Estimate Nugget
    nugget = np.min(semi_variograms)
    
    # Estimate Sill
    sill = np.max(semi_variograms)
    
    # Estimate Range
    sill_diff = 0.95 * (sill - nugget)
    indices = np.where(semi_variograms >= sill_diff)[0]
    
    if len(indices) == 0:
        # If no distances exceed sill_diff, choose a large distance as the range
        range_ = np.max(distances)
    else:
        range_ = np.max(distances[indices])
    
    return [sill, nugget, range_]

def fit_variogram_model(distances, semi_variograms):
    # Remove NaNs and Infs from both arrays
    valid_indices = np.isfinite(distances) & np.isfinite(semi_variograms)
    distances = distances[valid_indices]
    semi_variograms = semi_variograms[valid_indices]

    # Ensure arrays are the same length
    if len(distances) != len(semi_variograms):
        raise ValueError("Distances and semi-variograms must have the same length.")

    # Estimate initial parameters
    initial_params = estimate_initial_params(distances, semi_variograms)
    
    # Fit the variogram model
    try:
        params, _ = curve_fit(exponential_model, distances, semi_variograms, p0=initial_params)
    except Exception as e:
        print(f"Error fitting variogram model: {e}")
        params = [np.nan, np.nan, np.nan]  # Set to NaNs if fitting fails
    
    return params

def plot_results(lons, lats, original_data, trend, detrended_data, distances, semi_variograms, fitted_params,length,width):
    plt.figure(figsize=(18, 12))

    # Reshape the data to 2D using length and width
    
    # original_data_2d = original_data.reshape((length, width))
    # trend_2d = trend.reshape((length, width))
    # detrended_data_2d = detrended_data.reshape((length, width))

    # Plot original data
    plt.subplot(2, 2, 1)
    plt.scatter(lons,lats,c=original_data, cmap='viridis')
    plt.colorbar(label='Value')
    plt.title('Original Data')

    # Plot trend
    plt.subplot(2, 2, 2)
    plt.scatter(lons,lats,c=trend, cmap='viridis')
    plt.colorbar(label='Value')
    plt.title('Trend')

    # Plot detrended data
    plt.subplot(2, 2, 3)
    plt.scatter(lons,lats,c=detrended_data, cmap='viridis')
    plt.colorbar(label='Value')
    plt.title('Detrended Data')

    # Plot experimental and fitted variogram
    plt.subplot(2, 2, 4)
    plt.scatter(distances, semi_variograms, label='Experimental Variogram', color='blue', s=10)
    h = np.linspace(0, np.max(distances), 100)
    fitted_variogram = exponential_model(h, *fitted_params)
    plt.plot(h, fitted_variogram, label='Fitted Exponential', color='red')
    plt.xlabel('Distance')
    plt.ylabel('Semivariance')
    plt.title('Variogram Model Fit')
    plt.legend()

    plt.tight_layout()
    plt.show()

def process_subset(lons, lats, values, subset_size=10000):
    # Randomly sample a subset of the data
    indices = np.random.choice(len(lons), size=subset_size, replace=False)
    return lons[indices], lats[indices], values[indices]

def main(binary_file_path, lons, lats,length,width):
    # Read binary file (example assuming single channel of floats)
    data = np.fromfile(binary_file_path, dtype=np.float32)
    
    # Process a subset of the data
    subset_lons, subset_lats, subset_data = process_subset(lons, lats, data)
    
    # Compute variogram
    distances, semi_variograms, trend, detrended_data = calculate_variogram(subset_lons, subset_lats, subset_data)
    
    # Check lengths of distances and semi_variograms
    print(f"Length of distances: {len(distances)}")
    print(f"Length of semi_variograms: {len(semi_variograms)}")
    
    # Fit variogram model
    fitted_params = fit_variogram_model(distances, semi_variograms)
    
    # Extract parameters
    sill, nugget, range_ = fitted_params
    print(f"Sill: {sill:.2f}")
    print(f"Nugget: {nugget:.2f}")
    print(f"Range: {range_:.2f}")
    
    # Plot results
    plot_results(subset_lons, subset_lats, subset_data, trend, detrended_data, distances, semi_variograms, fitted_params,length,width)

if __name__ == "__main__":
    # Example usage:
    binary_file_path = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us6000iaqi_insar_processing/GEOC_077D_05685_121313_floatml_masked_GACOS_Corrected_clipped/20220712_20220829/20220712_20220829.unw'
    EQA_dem_par = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us6000iaqi_insar_processing/GEOC_077D_05685_121313_floatml_masked_GACOS_Corrected_clipped/EQA.dem_par'
    
    width = int(LiCS_lib.get_param_par(EQA_dem_par, 'width'))
    length = int(LiCS_lib.get_param_par(EQA_dem_par, 'nlines'))
    dlat = float(LiCS_lib.get_param_par(EQA_dem_par, 'post_lat'))
    dlon = float(LiCS_lib.get_param_par(EQA_dem_par, 'post_lon'))
    lat1 = float(LiCS_lib.get_param_par(EQA_dem_par, 'corner_lat'))
    lon1 = float(LiCS_lib.get_param_par(EQA_dem_par, 'corner_lon'))

    centerlat = lat1 + dlat * (length / 2)
    ra = float(LiCS_lib.get_param_par(EQA_dem_par, 'ellipsoid_ra'))
    recip_f = float(LiCS_lib.get_param_par(EQA_dem_par, 'ellipsoid_reciprocal_flattening'))
    rb = ra * (1 - 1 / recip_f)  # polar radius
    pixsp_a = 2 * np.pi * rb / 360 * abs(dlat)
    pixsp_r = 2 * np.pi * ra / 360 * dlon * np.cos(np.deg2rad(centerlat))
    
    Lat = np.arange(0, (length + 1) * pixsp_r, pixsp_r)
    Lon = np.arange(0, (width + 1) * pixsp_a, pixsp_a)
    
    Lat = Lat[:length]
    Lon = Lon[:width]
    lats_orig = np.array(Lat, dtype=float)
    lons_orig = np.array(Lon, dtype=float)

    lons, lats = np.meshgrid(lons_orig, lats_orig)
   
    main(binary_file_path, lons.flatten(), lats.flatten(),length,width)