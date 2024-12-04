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
       
    return dates_and_noise_dict


    


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
        ifgm_orig = ifgm.copy()
      

      
        mask = LiCS_lib.read_img(os.path.join(outdir,"mask"),length,width)
        masked_pixels = np.where(mask==0)
        ifgm[masked_pixels] = np.nan
        ifgm_orig = ifgm.copy()
        

   
        return output_dict


