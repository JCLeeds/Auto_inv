import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# from scipy import signal
import time
from skimage.measure import block_reduce
import os
import LiCSBAS_io_lib as LiCS_lib
import LiCSBAS_tools_lib as LiCS_tools



def resample(data, rate):
    return block_reduce(data, block_size=(rate, rate), func=np.nanmean,cval=np.nan)
def calc_rates(data,ds_total_points):
    rate = int(np.sqrt(len(data.flatten())/(ds_total_points)))
    print(rate)
    return  rate
def resample_all(data,X, Y, ds_total_points):
    sample_rate = calc_rates(data,ds_total_points)
    # downsample your X and Y meshgrids
    X_inside_downsampled = resample(X, sample_rate)
    Y_inside_downsampled = resample(Y, sample_rate)
    data = resampled(data,sample_rate)
    return data, X, Y 
def slipburi_uniform(data, E, N, U, X, Y, n_points,filename):
    sample_rate = calc_rates(data,n_points)
    data = resample(data, sample_rate).flatten()
    E = resample(E,sample_rate).flatten()
    N = resample(N,sample_rate).flatten()
    U = resample(U,sample_rate).flatten()
    X = resample(X,sample_rate).flatten()
    Y = resample(Y, sample_rate).flatten()
    indx = np.isnan(data)
    data = data[~indx]
    E = E[~indx]    
    N = N[~indx]
    U = U[~indx]
    X = X[~indx]
    Y = Y[~indx]

    print('len of data  ' + str(len(data)))
    f = open(filename, "w")
    for ii in range(len(data)):
        string = (str(X[ii]) + ' ' 
                + str(Y[ii]) + ' ' 
                + str(data[ii]) + ' ' 
                + str(E[ii]) + ' ' 
                + str(N[ii]) + ' ' 
                + str(U[ii]) + ' ' + '\n')     
        f.write(string)
    f.close()
    return 


if __name__ == '__main__':
    #asc
    # EQA_dem_par = os.path.join('/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/slip_buri_files/GEOC_013A_05597_131313_floatml_masked_clipped',"EQA.dem_par")
    # E_geo = os.path.join('/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/slip_buri_files/GEOC_013A_05597_131313_floatml_masked_clipped',"E.geo")
    # N_geo = os.path.join('/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/slip_buri_files/GEOC_013A_05597_131313_floatml_masked_clipped',"N.geo")
    # U_geo = os.path.join('/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/slip_buri_files/GEOC_013A_05597_131313_floatml_masked_clipped',"U.geo")
    # # #dsc
    EQA_dem_par = os.path.join('/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/slip_buri_files/GEOC_020D_05533_131313_floatml_masked_clipped',"EQA.dem_par")
    E_geo = os.path.join('/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/slip_buri_files/GEOC_020D_05533_131313_floatml_masked_clipped',"E.geo")
    N_geo = os.path.join('/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/slip_buri_files/GEOC_020D_05533_131313_floatml_masked_clipped',"N.geo")
    U_geo = os.path.join('/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/slip_buri_files/GEOC_020D_05533_131313_floatml_masked_clipped',"U.geo")


    width = int(LiCS_lib.get_param_par(EQA_dem_par, 'width'))
    length = int(LiCS_lib.get_param_par(EQA_dem_par, 'nlines'))
    dlat = float(LiCS_lib.get_param_par(EQA_dem_par, 'post_lat')) #negative
    dlon = float(LiCS_lib.get_param_par(EQA_dem_par, 'post_lon')) #positive
    lat1 = float(LiCS_lib.get_param_par(EQA_dem_par, 'corner_lat'))
    lon1 = float(LiCS_lib.get_param_par(EQA_dem_par, 'corner_lon'))
    lat2 = lat1+dlat*(length-1) # south # Remove
    lon2 = lon1+dlon*(width-1) # east # Remove
    #lon, lat = np.arange(lon1, lon2+postlon, postlon), np.arange(lat1, lat2+postlat, postlat)
    
    
    # centerlat = lat1+dlat*(length/2)
    # ra = float(LiCS_lib.get_param_par(EQA_dem_par, 'ellipsoid_ra'))
    # recip_f = float(LiCS_lib.get_param_par(EQA_dem_par, 'ellipsoid_reciprocal_flattening'))
    # rb = ra*(1-1/recip_f) ## polar radius
    # pixsp_a = 2*np.pi*rb/360*abs(dlat)
    # pixsp_r = 2*np.pi*ra/360*dlon*np.cos(np.deg2rad(centerlat))

    E = LiCS_lib.read_img(E_geo, length, width)
    N = LiCS_lib.read_img(N_geo, length, width)
    U = LiCS_lib.read_img(U_geo, length, width)

    # Lat = np.arange(0, (length + 1) * pixsp_r, pixsp_r)
    # Lon = np.arange(0, (width + 1) * pixsp_a, pixsp_a) 
    # Lat = Lat[:length]
    # Lon = Lon[:width]

    #lon, lat = np.arange(lon1, lon2+postlon, postlon), np.arange(lat1, lat2+postlat, postlat)
    lon, lat = np.linspace(lon1, lon2, width), np.linspace(lat1, lat2, length)

    # Lon, Lat = np.linspace(lon1, lon2, width), np.linspace(lat1, lat2, length) # Remove
    total_points = length*width
    print('TOTAL Points === ' + str(total_points))
  
    # generate coordinate mesh
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # asc
    # unw_files = ['/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/slip_buri_files/GEOC_013A_05597_131313_floatml_masked_clipped/20230901_20231019/20230901_20231019.unw',
    #             '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/slip_buri_files/GEOC_013A_05597_131313_floatml_masked_clipped/20230913_20231019/20230913_20231019.unw',
    #             '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/slip_buri_files/GEOC_013A_05597_131313_floatml_masked_clipped/20230913_20231031/20230913_20231031.unw',
    #             '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/slip_buri_files/GEOC_013A_05597_131313_floatml_masked_clipped/20230925_20231019/20230925_20231019.unw',
    #             '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/slip_buri_files/GEOC_013A_05597_131313_floatml_masked_clipped/20230925_20231031/20230925_20231031.unw']
    # dsc
    unw_files = ['/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/slip_buri_files/GEOC_020D_05533_131313_floatml_masked_clipped/20230902_20231020/20230902_20231020.unw',
                 '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/slip_buri_files/GEOC_020D_05533_131313_floatml_masked_clipped/20230902_20231101/20230902_20231101.unw',
                 '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/slip_buri_files/GEOC_020D_05533_131313_floatml_masked_clipped/20230926_20231020/20230926_20231020.unw',
                 '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/slip_buri_files/GEOC_020D_05533_131313_floatml_masked_clipped/20230926_20231101/20230926_20231101.unw']


    for ii in range(len(unw_files)):
        unw = LiCS_lib.read_img(unw_files[ii], length, width)
        unw[unw==0] = np.nan
        slipburi_uniform(unw,E,N,U,lon_grid,lat_grid,2000,('ds_'+ str(unw_files[ii].split('/')[-1])))


