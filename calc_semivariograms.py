#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import LiCSBAS_io_lib as LiCS_lib
import os 
import LiCSBAS_tools_lib as LiCS_tools
import multiprocessing as multi
from lmfit.model import *
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap as LSC
from matplotlib import pyplot as plt




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


    ifgdates2 = ifgdates
    n_ifg2 = len(ifgdates2)

    if n_ifg-n_ifg2 > 0:
        print("  {0:3}/{1:3} masked unw and cc already exist. Skip".format(n_ifg-n_ifg2, n_ifg), flush=True)

    if n_ifg2 > 0:
        ### Mask with parallel processing
        if n_para > n_ifg2:
            n_para = n_ifg2
         
            print('  {} parallel processing...'.format(n_para), flush=True)
            p = q.Pool(n_para)
            output_dict = p.map(calc_semi_para, range(n_ifg2))
            p.close()
            dates_and_noise_dict = {}
            for ii in range(len(output_dict)):
                dates_and_noise_dict = dates_and_noise_dict | output_dict[ii]

    return dates_and_noise_dict


    


def calc_semi_para(ifgix):
        ifgd = ifgdates2[ifgix]
        unw_path = os.path.join(os.path.join(outdir,ifgd),ifgd+".unw")   
        ifgm = LiCS_lib.read_img(unw_path,length,width)
      
        ifgm = -ifgm/4/np.pi*0.0555 # Added by JC to convert to meters deformation. ----> rad to m posative is -LOS

        Lat = np.arange(0, (length + 1) * pixsp_r, pixsp_r)
        Lon = np.arange(0, (width + 1) * pixsp_a, pixsp_a)
        Lat = Lat[:length]
        Lon = Lon[:width]


        XX, YY = np.meshgrid(Lon, Lat)
        XX = XX.flatten()
        YY = YY.flatten()
      
        mask_sig = LiCS_lib.read_img(os.path.join(outdir,"signal_mask"),length,width)
        masked_pixels = np.where(mask_sig==0)
        ifgm[masked_pixels] = np.nan
        ifgm_orig = ifgm.copy()
      

      
        mask = LiCS_lib.read_img(os.path.join(outdir,"mask"),length,width)
        masked_pixels = np.where(mask==0)
        ifgm[masked_pixels] = np.nan
        ifgm_orig = ifgm.copy()
        

        # ifgm[abs(ifgm) > (semi_mask_thresh)] = np.nan
        ifgm_nan = ifgm.copy()
        ifgm_deramp = ifgm.copy()
        ifgm = ifgm.flatten()
        # Drop all nan data
        xdist = XX[~np.isnan(ifgm)]
        ydist = YY[~np.isnan(ifgm)]

        maximum_dist = np.sqrt(((np.max(xdist) - np.min(xdist)) ** 2) + ((np.max(ydist) - np.min(ydist) ** 2)))
        ifgm = ifgm[~np.isnan(ifgm)]


        # ifgm = scipy.signal.detrend(ifgm)
        # Detrend code written by J Condon  
        ifgm_deramp, ydist, xdist = LiCS_tools.invert_plane(ifgm,ydist,xdist)
        ifgm = ifgm_deramp
    

        # calc from lmfit
        # mod = Model(spherical)
        medians = np.array([])
        bincenters = np.array([])
        stds = np.array([])

        # Find random pairings of pixels to check
        # Number of random checks
        n_pix = int(1e6)

        pix_1 = np.array([])
        pix_2 = np.array([])

        # Going to look at n_pix pairs. Only iterate 5 times. Life is short
        its = 0
         # Default Value
        while pix_1.shape[0] < n_pix and its < 5:
            its += 1
            # Create n_pix random selection of data points (Random selection with replacement)
            # Work out too many in case we need to remove duplicates
            pix_1 = np.concatenate([pix_1, np.random.choice(np.arange(ifgm.shape[0]), n_pix * 2)])
            pix_2 = np.concatenate([pix_2, np.random.choice(np.arange(ifgm.shape[0]), n_pix * 2)])

            # Find where the same pixel is selected twice
            duplicate = np.where(pix_1 == pix_2)[0]
            pix_1 = np.delete(pix_1, duplicate)
            pix_2 = np.delete(pix_2, duplicate)

            # Drop duplicate pairings
            unique_pix = np.unique(np.vstack([pix_1, pix_2]).T, axis=0)
            pix_1 = unique_pix[:, 0].astype('int')
            pix_2 = unique_pix[:, 1].astype('int')

            # Remove pixels with a seperation of more than 225 km 
            dists = np.sqrt(((xdist[pix_1] - xdist[pix_2]) ** 2) + ((ydist[pix_1] - ydist[pix_2]) ** 2))
            # # Max Lag solution to end member issue from J. McGrath
            # pix_1 = np.delete(pix_1, np.where(dists > (max_lag * 1000))[0])
            # pix_2 = np.delete(pix_2, np.where(dists > (max_lag * 1000))[0])

            # Max Dist solution to end member issue J. Condon 
            pix_1 = np.delete(pix_1, np.where(dists > (maximum_dist*0.85))[0])
            pix_2 = np.delete(pix_2, np.where(dists > (maximum_dist*0.85))[0])

        # In case of early ending
        if n_pix > len(pix_1):
            n_pix = len(pix_1)

        # Trim to n_pix, and create integer array
        pix_1 = pix_1[:n_pix].astype('int')
        pix_2 = pix_2[:n_pix].astype('int')

        # Calculate distances between random points
        dists = np.sqrt(((xdist[pix_1] - xdist[pix_2]) ** 2) + ((ydist[pix_1] - ydist[pix_2]) ** 2))
        # Calculate squared difference between random points
        vals = abs((ifgm[pix_1] - ifgm[pix_2])) ** 2

        medians, binedges = stats.binned_statistic(dists, vals, 'median', bins=1000)[:-1]
        stds = stats.binned_statistic(dists, vals, 'std', bins=1000)[0]
        bincenters = (binedges[0:-1] + binedges[1:]) / 2
        mod = Model(spherical)

        # try:
        mod.set_param_hint('p', value=np.percentile(medians, 75))  # guess maximum variance
        mod.set_param_hint('n', value=0)  # guess 0
        mod.set_param_hint('r', value=8000)  # guess 100 km
        sigma = stds + np.power(bincenters / max(bincenters), 2)
        sigma = stds * (1 + (max(bincenters) / bincenters))
        result = mod.fit(medians, d=bincenters, weights=sigma)
    

        # except:
        #     # Try smaller ranges
        #     n_bins = len(bincenters)
        #     try:
        #         bincenters = bincenters[:int(n_bins * 3 / 4)]
        #         stds = stds[:int(n_bins * 3 / 4)]
        #         medians = medians[:int(n_bins * 3 / 4)]
        #         sigma = stds + np.power(bincenters / max(bincenters), 3)
        #         sigma = stds * (1 + (max(bincenters) / bincenters))
        #         result = mod.fit(medians, d=bincenters, weights=sigma)
        #     except:
        #         try:
        #             bincenters = bincenters[:int(n_bins / 2)]
        #             stds = stds[:int(n_bins / 2)]
        #             medians = medians[:int(n_bins / 2)]
        #             sigma = stds + np.power(bincenters / max(bincenters), 3)
        #             sigma = stds * (1 + (max(bincenters) / bincenters))
        #             result = mod.fit(medians, d=bincenters, weights=sigma)
        #         except:
        #             sill = 100
        #             print('Ifgm  Failed to solve - setting sill to {}'.format(sill))

        try:
            # Print Sill (ie variance)
            sill = result.best_values['p']
            model_semi = (result.best_values['n'] + sill * ((3 * bincenters)/ (2 * result.best_values['r']) - 0.5*((bincenters**3) / (result.best_values['r']**3))))
            model_semi[np.where(bincenters > result.best_values['r'])[0]] = result.best_values['n'] + sill
        except:
            sill = 100
            model_semi = np.zeros(bincenters.shape) * np.nan

        if not os.path.exists(os.path.join(outdir, 'semivariograms')):
            os.mkdir(os.path.join(outdir, 'semivariograms'))

        fig=plt.figure(figsize=(12,12))
        ax=fig.add_subplot(2,2,1)
        im = ax.imshow(ifgm_orig)
        plt.title('Original {}'.format(ifgd))
        fig.colorbar(im, ax=ax)
        ax=fig.add_subplot(2,2,2)
        im = ax.imshow(ifgm_nan)
        plt.title('NaN {}'.format(ifgd))
        fig.colorbar(im, ax=ax)
        ax=fig.add_subplot(2,2,3)
        im = ax.scatter(xdist,ydist,c=ifgm_deramp) # remeber this might be breaking my code
        plt.title('NaN + Deramp {}'.format(ifgd))
        fig.colorbar(im, ax=ax)
        ax=fig.add_subplot(2,2,4)
        im = ax.scatter(bincenters, medians, c=sigma, label=ifgd)
        ax.plot(bincenters, model_semi, label='{} model'.format(ifgd))
        fig.colorbar(im, ax=ax)
        try:
            plt.title('Partial Sill: {:.6f}, Nugget: {:.6f}, Range: {:.6f} km'.format(sill, result.best_values['n'],result.best_values['r']/1000))
        except:
            plt.title('Semivariogram Failed')
        if sill == sill:
            plt.savefig(os.path.join(outdir, 'semivariograms', 'semivarigram{}X.png'.format(ifgd)))
        else:
            plt.savefig(os.path.join(outdir, 'semivariograms', 'semivarigram{}.png'.format(ifgd)))
        plt.close()
        output_dict[ifgd] = [sill ,result.best_values['n'], result.best_values['r']]
        print(output_dict)
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