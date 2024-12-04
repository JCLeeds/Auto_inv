import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.spatial import distance
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
from scipy.optimize import curve_fit
from scipy.spatial import distance
import llh2local as llh2local

def load_cpt_cmap(cpt_file, n_colors):
    # Attempt to load a CPT file or use a default colormap if the file is not available.
    try:
        with open(cpt_file, 'r') as f:
            # Parse the CPT file to extract the colormap (this is a simplified example)
            colors = []
            for line in f:
                if not line.startswith('#') and len(line.strip()) > 0:
                    parts = line.split()
                    if len(parts) >= 4:
                        r, g, b = map(float, parts[:3])
                        colors.append((r/255, g/255, b/255))
            cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=n_colors)
    except FileNotFoundError:
        print(f"CPT file not found: {cpt_file}. Using default seismic colormap.")
        cmap = plt.get_cmap('seismic', n_colors)
    return cmap


# def llh2local(sll, ref_point):
#     R = 6371e3  # Earth radius in meters
#     lat1 = np.deg2rad(ref_point[1])
#     lat2 = np.deg2rad(sll[:, 1])
#     delta_lat = np.deg2rad(sll[:, 1] - ref_point[1])
#     delta_lon = np.deg2rad(sll[:, 0] - ref_point[0])

#     a = np.sin(delta_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
#     c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
#     d = R * c

#     return np.column_stack((d * np.cos(delta_lon), d * np.sin(delta_lat)))

def variogram(x, y, bins=30, subsample=3000):
    # Subsample the data if necessary
    if len(x) > subsample:
        idx = np.random.choice(len(x), subsample, replace=False)
        x, y = x[idx], y[idx]

    # Compute the pairwise distances between points in x
    dists = distance.pdist(x, metric='euclidean')
    
    # Compute the differences in values for y
    vals = distance.pdist(y[:, np.newaxis], metric='euclidean')
    
    # Define the bin edges and centers
    bin_edges = np.linspace(0, dists.max(), bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    variogram_vals = np.zeros_like(bin_centers)

    # Calculate the average semivariance for each bin
    for i in range(len(bin_centers)):
        idx = (dists >= bin_edges[i]) & (dists < bin_edges[i + 1])
        if np.any(idx):
            variogram_vals[i] = np.mean(vals[idx])

    return bin_centers, variogram_vals

def exponential_model(h, range_, sill, nugget):
    """Exponential variogram model"""
    return nugget + sill * (1.0 - np.exp(-h / range_))


def advanced_detrend(xy, los):
    """Remove a 2nd order polynomial trend from the data"""
    A = np.vstack([xy[:, 0]**2, xy[:, 1]**2, xy[:, 0]*xy[:, 1], xy[:, 0], xy[:, 1], np.ones(xy.shape[0])]).T
    coeff = np.linalg.lstsq(A, los, rcond=None)[0]
    trend = A.dot(coeff)
    detrended = los - trend
    return detrended, trend

def onselect(vertices):
    global polygon_path
    polygon_path = Path(vertices)

def select_circle():
    plt.title('Click to select the center and radius of the circle')
    points = plt.ginput(2)
    center = points[0]
    radius = np.sqrt((points[1][0] - center[0]) ** 2 + (points[1][1] - center[1]) ** 2)
    return center, radius



def plot_local_coordinates(xy, values, wavelength, title):
    """Plot data in the local coordinate system"""
    plt.figure()
    plt.scatter(xy[:, 0], xy[:, 1], c=np.mod(values, wavelength / 2), cmap='seismic', s=1)
    plt.colorbar(label='Displacement (m)')
    plt.title(title)
    plt.xlabel('Local X (m)')
    plt.ylabel('Local Y (m)')
    plt.axis('equal')
    plt.show()


def fit_variogram(input_file, wavelength, auto=None, dtype=np.float32):
    print('Ingesting data to estimate (semi-)variogram ...')
    insar_data = sio.loadmat(input_file)
    
    cmap_seismo = load_cpt_cmap('GMT_seis.cpt', 100)
    
    ref_point = [np.min(insar_data['Lon']), np.min(insar_data['Lat'])]

    converted_phase = (insar_data['Phase'] / (4 * np.pi)) * wavelength
    los = -converted_phase.astype(np.float32)
    
    sampling = 1
    if 400000 < len(los) < 1000000:
        sampling = 2
    elif len(los) > 1000000:
        sampling = 5


    plt.figure()
    scatter_plot = plt.scatter(insar_data['Lon'][::sampling], insar_data['Lat'][::sampling], 
                               c=np.mod(los[::sampling], wavelength / 2), cmap=cmap_seismo)
    plt.colorbar(label='Displacement (m)')
    plt.title('Wrapped Interferogram')
    plt.xlabel('Longitude (degrees)')
    plt.ylabel('Latitude (degrees)')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    if auto is None:
        choice = input('Would you like to select a circular area or mask out a region? (Circle/Mask): ')
        if choice.lower() == 'circle':
            plt.figure()
            plt.scatter(insar_data['Lon'][::sampling], insar_data['Lat'][::sampling], 
                        c=np.mod(los[::sampling], wavelength / 2), cmap=cmap_seismo)
            plt.title('Click to select the center and radius of the circle')
            plt.xlabel('Longitude (degrees)')
            plt.ylabel('Latitude (degrees)')
            plt.gca().set_aspect('equal', adjustable='box')
            center, radius = select_circle()
            dists = np.sqrt((insar_data['Lon'] - center[0])**2 + (insar_data['Lat'] - center[1])**2)
            ix_subset = np.where(dists >= radius)
        else:
            fig, ax = plt.subplots()
            scatter_plot = ax.scatter(insar_data['Lon'][::sampling], insar_data['Lat'][::sampling], 
                                      c=np.mod(los[::sampling], wavelength / 2), cmap=cmap_seismo)
            plt.title('Draw a polygon to mask out a region')
            plt.xlabel('Longitude (degrees)')
            plt.ylabel('Latitude (degrees)')
            plt.gca().set_aspect('equal', adjustable='box')

            global polygon_path
            polygon_selector = PolygonSelector(ax, onselect)
            plt.show()

            in_polygon = polygon_path.contains_points(np.column_stack((insar_data['Lon'], insar_data['Lat'])))
            ix_subset = np.where(in_polygon == 1)
    else:
        max_lon = auto[0] + 0.4
        min_lon = auto[0] - 0.4
        max_lat = auto[1] + 0.4
        min_lat = auto[1] - 0.4
        
        ix_subset = np.where((insar_data['Lat'] < min_lat) | (insar_data['Lat'] > max_lat) |
                             (insar_data['Lon'] < min_lon) | (insar_data['Lon'] > max_lon))

    subset = los[ix_subset]
    llon = insar_data['Lon'][ix_subset]
    llat = insar_data['Lat'][ix_subset]

    plt.figure(figsize=(12, 10))
    plt.subplot(2, 3, 1)
    plt.scatter(llon, llat, c=np.mod(subset, wavelength / 2), cmap=cmap_seismo)
    plt.colorbar(label='Displacement (m)')
    plt.title('Selected region, NON-DETRENDED')
    plt.xlabel('Longitude (degrees)')
    plt.ylabel('Latitude (degrees)')
    plt.gca().set_aspect('equal', adjustable='box')



    sll_col = np.column_stack((llon, llat))
    sll = np.array([llon,llat])
    print(np.shape(sll))
    print(np.shape(sll_col))
    xy = llh2local.llh2local(sll, np.array(ref_point)) 
    print(np.shape(xy))
    xy = xy.T
  


    # A = np.column_stack((xy, np.ones(len(xy))))
    # coeff, _, _, _ = np.linalg.lstsq(A, subset, rcond=None)
    # deramped = subset - np.dot(A, coeff)

    deramped, trend = advanced_detrend(xy,subset)

    plt.subplot(2, 3, 2)
    plt.scatter(llon, llat, c=np.mod(trend, wavelength / 2), cmap=cmap_seismo)
    plt.colorbar(label='Displacement (m)')
    plt.title('Selected region, ESTIMATED TREND')
    plt.xlabel('Longitude (degrees)')
    plt.ylabel('Latitude (degrees)')
    plt.gca().set_aspect('equal', adjustable='box')

    plt.subplot(2, 3, 3)
    plt.scatter(llon, llat, c=np.mod(deramped, wavelength / 2), cmap=cmap_seismo)
    plt.colorbar(label='Displacement (m)')
    plt.title('Selected region, DETRENDED')
    plt.xlabel('Longitude (degrees)')
    plt.ylabel('Latitude (degrees)')
    plt.gca().set_aspect('equal', adjustable='box')

    h_vals, variog_vals = variogram(xy, subset, bins=30)
    plt.subplot(2, 3, 4)
    plt.plot(h_vals, variog_vals, 'o')
    plt.title('Semi-variogram, NON-DETRENDED')

    h_vals_dtrnd, variog_vals_dtrnd = variogram(xy, deramped, bins=100)

    sill_guess = np.var(deramped)
    range_guess = h_vals_dtrnd[np.argmax(variog_vals_dtrnd)]  # Approximate range where the variogram reaches its maximum
    nugget_guess = np.min(variog_vals_dtrnd)

    popt, _ = curve_fit(exponential_model,h_vals_dtrnd, variog_vals_dtrnd, p0=[range_guess, sill_guess, nugget_guess])

    plt.subplot(2, 3, 5)
    plt.plot(h_vals_dtrnd, variog_vals_dtrnd, 'o', label='Data')
    plt.plot(h_vals_dtrnd, exponential_model(h_vals_dtrnd, *popt), 'r-', label='Fit')
    plt.title('Semi-variogram and fit, DETRENDED')
    plt.legend()

    plt.subplot(2, 3, 6)
    plt.axis('off')
    plt.text(0.1, 1.0, 'Fitted exponential semi-variogram parameters:', fontsize=14)
    plt.text(0.1, 0.8, f'Sill:  {popt[1]:.4f}', fontsize=14)
    plt.text(0.1, 0.6, f'Range:  {popt[0]:.4f}', fontsize=14)
    plt.text(0.1, 0.4, f'Nugget:  {popt[2]:.4f}', fontsize=14)

    plot_local_coordinates(xy, subset, wavelength, 'Data in Local Coordinate System')
    plt.show()

    print(f'Sill:  {popt[1]:.4f}')
    print(f'Range:  {popt[0]:.4f}')
    print(f'Nugget:  {popt[2]:.4f}')

    return popt[0], popt[1], popt[2]
if __name__ == '__main__':
    a, c, n = fit_variogram('/uolstore/Research/a/a285/homes/ee18jwc/code/synthetic_data/NP_ambiguity_tests/NP_ambiguity_test_thrust_str0_dip45_rake-180_NP1_A.mat', 0.056)