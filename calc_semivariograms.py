#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import LiCSBAS_io_lib as LiCS_lib
import os 
import LiCSBAS_tools_lib as LiCS_tools
import multiprocessing as multi
from lmfit import Model
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap as LSC
from matplotlib import pyplot as plt
from cmcrameri import cm
import shutil



def calculate_semivarigrams(geoc_ml_path):
        
    try:
        n_para = len(os.sched_getaffinity(0))
    except:
        n_para = multi.cpu_count()
    

    global ifgdates2, outdir, pixsp_a, pixsp_r, width,length, output_dict
    output_dict = {}
    

    q = multi.get_context('fork')

    ifgdates = LiCS_tools.get_ifgdates(geoc_ml_path)
    n_ifg = len(ifgdates)



    EQA_dem_par = os.path.join(geoc_ml_path,"EQA.dem_par")
    width = int(LiCS_lib.get_param_par(EQA_dem_par, 'width'))
    length = int(LiCS_lib.get_param_par(EQA_dem_par, 'nlines'))
    dlat = float(LiCS_lib.get_param_par(EQA_dem_par, 'post_lat')) #negative
    dlon = float(LiCS_lib.get_param_par(EQA_dem_par, 'post_lon')) #positive
    lat1 = float(LiCS_lib.get_param_par(EQA_dem_par, 'corner_lat'))
    lon1 = float(LiCS_lib.get_param_par(EQA_dem_par, 'corner_lon'))
    

    print('\nIn geographical coordinates', flush=True)

    centerlat = lat1+dlat*(length/2)
    ra = float(LiCS_lib.get_param_par(EQA_dem_par, 'ellipsoid_ra'))
    recip_f = float(LiCS_lib.get_param_par(EQA_dem_par, 'ellipsoid_reciprocal_flattening'))
    rb = ra*(1-1/recip_f) ## polar radius
    pixsp_a = 2*np.pi*rb/360*abs(dlat)
    pixsp_r = 2*np.pi*ra/360*dlon*np.cos(np.deg2rad(centerlat))



    outdir = geoc_ml_path
    slc_mli_par_path = os.path.join(geoc_ml_path,"slc.mli.par")

    if os.path.exists(os.path.join(outdir, 'semivariograms')):
        shutil.rmtree(os.path.join(outdir, 'semivariograms'))

    # if not os.path.exists(os.path.join(outdir, 'semivariograms')):
    os.mkdir(os.path.join(outdir, 'semivariograms'))


    ifgdates2 = ifgdates
    n_ifg2 = len(ifgdates2)
    dict_total = []
    if n_ifg-n_ifg2 > 0:
        print("  {0:3}/{1:3} masked unw and cc already exist. Skip".format(n_ifg-n_ifg2, n_ifg), flush=True)

    if n_ifg2 > 0:
        ### Mask with parallel processing
        if n_para > n_ifg2:
            n_para = n_ifg2
         
        # print('  {} parallel processing...'.format(n_para), flush=True)
        # p = q.Pool(n_para)
        # dates_and_noise_dict = p.map(calc_semi_para, range(n_ifg2))
        # p.close()
        dates_and_noise_dict = {}
        for ii in range(n_ifg2):
            dict_total.append(calc_semi_para(ii))
        for ii in range(len(dict_total)):
            dates_and_noise_dict = dates_and_noise_dict | dict_total[ii]
        # for ii in range(len(output_dict)):
        #     dates_and_noise_dict = dates_and_noise_dict | output_dict[ii]

    return dates_and_noise_dict


# def calculate_semivariogram(Lat, Lon, ifgm):
#         #untested function
#         """
#         Calculate the semivariogram of a grid of Lat Lons with a value ifgm at each point
#         @param Lat: 1D array of latitudes
#         @param Lon: 1D array of longitudes
#         @param ifgm: 2D array of values at each (Lat, Lon) point
#         @return: semivariogram model
#         """
#         XX, YY = np.meshgrid(Lon, Lat)
#         XX = XX.flatten()
#         YY = YY.flatten()
#         ifgm = ifgm.flatten()

#         # Drop all nan data
#         xdist = XX[~np.isnan(ifgm)]
#         ydist = YY[~np.isnan(ifgm)]
#         ifgm = ifgm[~np.isnan(ifgm)]

#         # Calculate distances between random points
#         n_pix = int(1e6)
#         pix_1 = np.random.choice(np.arange(ifgm.shape[0]), n_pix)
#         pix_2 = np.random.choice(np.arange(ifgm.shape[0]), n_pix)
#         dists = np.sqrt(((xdist[pix_1] - xdist[pix_2]) ** 2) + ((ydist[pix_1] - ydist[pix_2]) ** 2))
#         vals = abs((ifgm[pix_1] - ifgm[pix_2])) ** 2

#         medians, binedges = stats.binned_statistic(dists, vals, 'median', bins=500)[:-1]
#         stds = stats.binned_statistic(dists, vals, 'std', bins=500)[0]
#         bincenters = (binedges[0:-1] + binedges[1:]) / 2

#         mod = Model(spherical)
#         mod.set_param_hint('p', value=np.percentile(medians, 75))
#         mod.set_param_hint('n', value=0)
#         mod.set_param_hint('r', value=8000)
#         result = mod.fit(medians, d=bincenters)

#         try:
#             sill = result.best_values['p']
#             model_semi = (result.best_values['n'] + sill * ((3 * bincenters) / (2 * result.best_values['r']) - 0.5 * ((bincenters ** 3) / (result.best_values['r'] ** 3))))
#             model_semi[np.where(bincenters > result.best_values['r'])[0]] = result.best_values['n'] + sill
#         except:
#             sill = 100
#             model_semi = np.zeros(bincenters.shape) * np.nan

#         return bincenters, medians, model_semi


def calc_semi_para(ifgix):
    ifgd = ifgdates2[ifgix]
    unw_path = os.path.join(outdir, ifgd, f"{ifgd}.unw")
    ifgm = LiCS_lib.read_img(unw_path, length, width)
    ifgm = -ifgm * 0.0555 / (4 * np.pi)  # Convert to meters deformation

    Lat = np.arange(0, length * pixsp_r, pixsp_r)
    Lon = np.arange(0, width * pixsp_a, pixsp_a)

    XX, YY = np.meshgrid(Lon, Lat)
    XX = XX[:length,:width]
    YY = YY[:length,:width]

    print(np.shape(XX), np.shape(YY), np.shape(ifgm))  
    mask_sig = LiCS_lib.read_img(os.path.join(outdir, "signal_mask"), length, width)
    ifgm[mask_sig == 0] = np.nan

    mask = LiCS_lib.read_img(os.path.join(outdir, "mask"), length, width)
    ifgm[mask == 0] = np.nan

    Afit, _ = LiCS_tools.fit2d(ifgm, w=None, deg="1")
    ifgm = (ifgm - Afit)

    XX, YY = XX.flatten(), YY.flatten()
    ifgm = ifgm.flatten()
    ifgm_not_masked = ifgm
    valid_mask = ~np.isnan(ifgm)
    print(np.shape(XX), np.shape(YY), np.shape(ifgm), np.shape(valid_mask))
    xdist, ydist = XX[valid_mask], YY[valid_mask]
    ifgm = ifgm[valid_mask]

    maximum_dist = np.sqrt((xdist.ptp() ** 2) + (ydist.ptp() ** 2))

    n_pix = int(1e6)
    pix_1, pix_2 = np.array([]), np.array([])

    its = 0
    while pix_1.size < n_pix and its < 5:
        its += 1
        pix_1 = np.concatenate([pix_1, np.random.choice(ifgm.size, n_pix * 2)])
        pix_2 = np.concatenate([pix_2, np.random.choice(ifgm.size, n_pix * 2)])

        unique_pix = np.unique(np.vstack([pix_1, pix_2]).T, axis=0)
        pix_1, pix_2 = unique_pix[:, 0].astype(int), unique_pix[:, 1].astype(int)

        dists = np.sqrt((xdist[pix_1] - xdist[pix_2]) ** 2 + (ydist[pix_1] - ydist[pix_2]) ** 2)
        valid = dists <= (maximum_dist * 0.5)
        pix_1, pix_2 = pix_1[valid], pix_2[valid]

    if n_pix > pix_1.size:
        n_pix = pix_1.size

    pix_1, pix_2 = pix_1[:n_pix].astype(int), pix_2[:n_pix].astype(int)
    dists = np.sqrt((xdist[pix_1] - xdist[pix_2]) ** 2 + (ydist[pix_1] - ydist[pix_2]) ** 2)
    vals = (ifgm[pix_1] - ifgm[pix_2]) ** 2

    medians, binedges, _ = stats.binned_statistic(dists, vals, 'median', bins=50)
    stds, _, _ = stats.binned_statistic(dists, vals, 'std', bins=50)
    bincenters = (binedges[:-1] + binedges[1:]) / 2

    mod = Model(spherical)
    mod.set_param_hint('p', value=np.percentile(medians, 75))
    mod.set_param_hint('n', value=0)
    mod.set_param_hint('r', value=8000)
    result = mod.fit(medians, d=bincenters)

    try:
        sill = result.best_values['p']
        model_semi = (result.best_values['n'] + sill * ((3 * bincenters) / (2 * result.best_values['r']) - 0.5 * (bincenters ** 3) / (result.best_values['r'] ** 3)))
        model_semi[bincenters > result.best_values['r']] = result.best_values['n'] + sill
    except:
        sill = 100
        model_semi = np.full(bincenters.shape, np.nan)

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    axs[0, 0].imshow(ifgm_not_masked.reshape(length, width))
    axs[0, 0].set_title(f'Original {ifgd}')
    axs[0, 1].imshow(ifgm_not_masked.reshape(length, width))
    axs[0, 1].set_title(f'NaN {ifgd}')
    scatter = axs[1, 0].scatter(bincenters, medians, c=stds)
    axs[1, 0].plot(bincenters, model_semi)
    axs[1, 0].set_title(f'Partial Sill: {sill:.6f}, Nugget: {result.best_values["n"]:.6f}, Range: {result.best_values["r"] / 1000:.6f} km')
    fig.colorbar(scatter, ax=axs[1, 0])

    # Remove the unused axis
    fig.delaxes(axs[1, 1])

    plt.savefig(os.path.join(outdir, 'semivariograms', f'semivariogram_{ifgd}.png'))
    plt.close()

    output_dict[ifgd] = [sill, result.best_values['n'], result.best_values['r']]
    return output_dict



def spherical(d, p, n, r):

    """
    Compute spherical variogram model
    @param d: 1D distance array
    @param p: partial sill
    @param n: nugget
    @param r: range
    @return: spherical variogram model
    """
    if r>d.max():
        r=d.max()-1
    return np.where(d > r, p + n, p * (3/2 * d/r - 1/2 * d**3 / r**3) + n)

def exponential(d,p,n,r):
    """
    Compute exponential variogram model
    @param d: 1D distance array
    @param p: partial sill
    @param n: nugget
    @param r: range
    @return: exponential variogram model
    """
    if r>d.max():
        r=d.max()-1
    return p*(1-np.exp((-d)/r)) + n



if __name__ == "__main__":
    dates_and_noise_dict = calculate_semivarigrams("/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us70007v9g_insar_processing/GEOC_072A_05090_131313_floatml_masked_GACOS_Corrected_clipped_signal_masked_QAed")
    print(dates_and_noise_dict)