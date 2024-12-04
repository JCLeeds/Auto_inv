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
from cmcrameri import cm
import shutil
import numpy as np
import matplotlib.pyplot as plt
# plt.style.use('ggplot')
import skgstat as skg
from skgstat.binning import even_width_lags, uniform_count_lags
from scipy.spatial.distance import pdist


def calculate_semivarigrams_skgstat(geoc_ml_path):

    # try:
    #     n_para = len(os.sched_getaffinity(0))
    # except:
    #     n_para = multi.cpu_count()
    

    global ifgdates2, outdir, pixsp_a, pixsp_r, width,length, output_dict
    output_dict = {}
    

    # q = multi.get_context('fork')

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

    if n_ifg-n_ifg2 > 0:
        print("  {0:3}/{1:3} masked unw and cc already exist. Skip".format(n_ifg-n_ifg2, n_ifg), flush=True)

    # for ii in range(n_ifg2):
    calc_semi_para(1)
    # if n_ifg2 > 0:
    #     ### Mask with parallel processing
    #     if n_para > n_ifg2:
    #         n_para = n_ifg2
         
    #     print('  {} parallel processing...'.format(n_para), flush=True)
    #     p = q.Pool(n_para)
    #     output_dict = p.map(calc_semi_para, range(n_ifg2))
    #     p.close()
    #     dates_and_noise_dict = {}
    #     for ii in range(len(output_dict)):
    #         dates_and_noise_dict = dates_and_noise_dict | output_dict[ii]

    return

def calc_semi_para(ifgix):
    ifgd = ifgdates2[ifgix]
    unw_path = os.path.join(os.path.join(outdir,ifgd),ifgd+".unw")   
    ifgm = LiCS_lib.read_img(unw_path,length,width)
    
    ifgm = -ifgm*0.0555/(4*np.pi)   # Added by JC to convert to meters deformation. ----> rad to m posative is -LOS

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
    ifgm = np.array(ifgm.flatten())
    xdist = XX[~np.isnan(ifgm)]
    ydist = YY[~np.isnan(ifgm)]
    ifgm = ifgm[~np.isnan(ifgm)]

    # length_new = ifgm.shape[0]
    # width_new = ifgm.shape[1]
    # Lat = np.arange(0, (length_new + 1) * pixsp_r, pixsp_r)
    # Lon = np.arange(0, (width_new + 1) * pixsp_a, pixsp_a)
    # Lat = Lat[:length_new]
    # Lon = Lon[:width_new]



    # indexs_to_remove_for_decimation = np.random.randint(low=0,high=len(ifgm),size=int(len(ifgm)*0.001)) # change to smaller value decimation to harsh this needs to be sped up 
    # # ifgm_orig = ifgm.copy()
    # ifgm = ifgm[~indexs_to_remove_for_decimation]
    # xdist = xdist[~indexs_to_remove_for_decimation]
    # ydist = ydist[~indexs_to_remove_for_decimation]   
    # coords = np.array([xdist,ydist]).T
    # print(coords.shape)
    # print(ifgm.shape)
    # ifgm_orig = ifgm.copy()

  
    # Lon = np.array(Lon)
    # Lat = np.array(Lat).reshape((-1, 1))
    # coords = Lon + Lat.repeat(len(Lon), 1)
    # coords = np.array(coords)
    # print(Lon.shape)
    # print(Lat.shape)
    # print(coords.shape)
    # print(ifgm.shape)
    # V = skg.Variogram(coords, ifgm)    
    # V.plot()
    
    # np.random.seed(42)
    # xx, yy = np.mgrid[0:0.5 * np.pi:500j, 0:0.8 * np.pi:500j]
    # _field = np.sin(xx)**2 + np.cos(yy)**2 + 10
    # z = _field + np.random.normal(0, 0.15, (500,  500))
    # coords = np.random.randint(0, 500, (300, 2))

    # values = np.fromiter((z[c[0], c[1]] for c in coords), dtype=float)
    # distances = pdist(coords)
    # V = skg.Variogram(coords, ifgm,n_lags=25,bin_func='even')
    coordinates = np.vstack((xdist, ydist)).T
    V = skg.Variogram(coordinates, ifgm, model='exponential', normalize=False)
    nugget = V.parameters[0]  # Nugget
    sill = V.parameters[0] + V.parameters[1]  # Sill (nugget + partial sill)
    range_val = V.parameters[2]  # Range

    # Output the fitted parameters
    print(f"Nugget: {nugget}")
    print(f"Sill: {sill}")
    print(f"Range: {range_val}")

    # Step 4: (Optional) Visualize the experimental and fitted variogram
    V.plot(show=False)
    plt.title('Variogram and Fitted Model')
    plt.xlabel('Distance')
    plt.ylabel('Semi-variance')
    plt.show()
    # Vg.plot()
    # plt.show()
    # print(values.shape)
    # print(coords.shape)
   


if __name__ == '__main__':
    calculate_semivarigrams_skgstat('/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/working_reported_recent/us70008cld_insar_processing/GEOC_012A_06041_131313_floatml_masked_GACOS_Corrected_clipped_signal_masked_QAed')
