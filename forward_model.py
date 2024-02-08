#!/usr/bin/env python
import coseis_lib as lib
import numpy as np
import matplotlib.pyplot as plt
import LiCSBAS_io_lib as LiCS_lib
import os 
import LiCSBAS_tools_lib as LiCS_tools
# this is additional scientific colour maps, see "https://www.fabiocrameri.ch/colourmaps/"
from cmcrameri import cm
import multiprocessing as multi



def forward_model(input_dir,strike_F,dip_F,rake_F,slip_F,depth_F,width_F,length_F,location):
    try:
        n_para = len(os.sched_getaffinity(0))
    except:
        n_para = multi.cpu_count()
    

    global ifgdates2, length_ifgm, width_ifgm, geoc_ml_path, pixsp_a, pixsp_r, xcent, ycent,strike,dip,rake,depth,width,length,slip
    geoc_ml_path = input_dir
    strike = strike_F
    dip = dip_F 
    rake = rake_F 
    slip = slip_F 
    depth = depth_F 
    width = width_F 
    length = length_F
    

    q = multi.get_context('fork')

    ifgdates = LiCS_tools.get_ifgdates(geoc_ml_path)
    n_ifg = len(ifgdates)



    # x, y, unw, diff = lib.load_ifgs(path_to_i) # pull for dims of data for model
    EQA_dem_par = os.path.join(geoc_ml_path,"EQA.dem_par")
    width_ifgm = int(LiCS_lib.get_param_par(EQA_dem_par, 'width'))
    length_ifgm = int(LiCS_lib.get_param_par(EQA_dem_par, 'nlines'))
    dlat = float(LiCS_lib.get_param_par(EQA_dem_par, 'post_lat')) #negative
    dlon = float(LiCS_lib.get_param_par(EQA_dem_par, 'post_lon')) #positive
    lat1 = float(LiCS_lib.get_param_par(EQA_dem_par, 'corner_lat'))
    lon1 = float(LiCS_lib.get_param_par(EQA_dem_par, 'corner_lon'))
    lat2 = lat1+dlat*(length_ifgm-1) # south # Remove
    lon2 = lon1+dlon*(width_ifgm-1) # east # Remove
    lon, lat = np.arange(lon1, lon2+dlon, dlon), np.arange(lat1, lat2+dlat, dlat)
    centerlat = lat1+dlat*(length_ifgm/2)
    ra = float(LiCS_lib.get_param_par(EQA_dem_par, 'ellipsoid_ra'))
    recip_f = float(LiCS_lib.get_param_par(EQA_dem_par, 'ellipsoid_reciprocal_flattening'))
    rb = ra*(1-1/recip_f) ## polar radius
    pixsp_a = 2*np.pi*rb/360*abs(dlat)
    pixsp_r = 2*np.pi*ra/360*dlon*np.cos(np.deg2rad(centerlat))


    x_usgs,y_usgs = LiCS_tools.bl2xy(location[1],location[0],width_ifgm,length_ifgm,lat1,dlat,lon1,dlon)
    xcent = x_usgs * pixsp_a
    ycent = y_usgs * pixsp_r
    m_cent = [xcent,ycent]
    print(xcent)
    print(ycent)
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
        p.map(forward_modelling_para, range(n_ifg2))
        p.close()




def forward_modelling_para(ifgix):
    ifgd = ifgdates2[ifgix]
    Inc_file = os.path.join(geoc_ml_path,'Theta.geo') 
    Head_file = os.path.join(geoc_ml_path,'Phi.geo')
    Inc = np.fromfile(Inc_file, dtype='float32').reshape((length_ifgm, width_ifgm))
    Head = np.fromfile(Head_file, dtype='float32').reshape((length_ifgm, width_ifgm))
    date_dir = geoc_ml_path + "/" + ifgd
    file_name = ifgd + '.unw'
    unw_file = os.path.join(date_dir,file_name) 
    unw = np.fromfile(unw_file, dtype='float32').reshape((length_ifgm, width_ifgm))       

    y = np.arange(0, (length_ifgm + 1) * pixsp_r, pixsp_r)
    x = np.arange(0, (width_ifgm + 1) * pixsp_a, pixsp_a) 
    y = y[:length_ifgm]
    x = x[:width_ifgm]
    xx, yy = np.meshgrid(x, y)
    # convert to pair of coordinates for our dislocation model
    xx_vec = np.reshape(xx, -1)
    yy_vec = np.reshape(yy, -1)
    # Define the fault model
    #####################################
    # xcen = xcent          # vertical surface projection of fault centroid in x (m)
    # ycen = ycent        # vertical surface projection of fault centroid in y (m)
    # strike = strike       # strike in degrees (0-360)
    # dip =    dip        # dip in degrees (0-90)
    # rake =   rake       # rake in degrees (-180 - 180)
    # slip =     slip       # magnitude of slip vector in metres
    centroid_depth = depth       # depth (measured vertically) to fault centroid in metres
    # width = width    # width of fault measured along-dip in metres
    # length =  length        # fault length in metres
    #####################################
    model = [xcent, ycent, strike, dip, rake, slip, length, centroid_depth, width]
    # Calcualte displacements
    disp = lib.disloc3d3(xx_vec, yy_vec, xoff=xcent, yoff=ycent, depth=centroid_depth,
                    length=length, width=width, slip=slip, opening=0, 
                    strike=strike, dip=dip, rake=rake, nu=0.25)
    #####################################
    incidence_angle = np.nanmean(Inc)
    azimuth_angle = np.nanmean(Head)
    #####################################

    # Convert to unit vector components
    e2los = np.cos(np.deg2rad(azimuth_angle)) * np.sin(np.deg2rad(incidence_angle))
    n2los = -np.sin(np.deg2rad(azimuth_angle)) * np.sin(np.deg2rad(incidence_angle))
    u2los = -np.cos(np.deg2rad(incidence_angle))
    fig = lib.plot_data_model(x, y, disp, model, unw, e2los, n2los, u2los, show_grid=True)
    plt.savefig(os.path.join(os.path.join(geoc_ml_path),ifgd+"forward_model_comp.png"))
    # plt.show()
    return 


if __name__ == "__main__":
    strike = 312
    dip = 74
    rake = -168
    slip = 10
    depth = 21500
    width = 10000
    length = 10000
    location = [38.420,44.910]
    dates = '20230108_20230213'
    forward_model("/Users/jcondon/phd/code/auto_inv/us6000jk0t_insar_processing/GEOC_072A_05090_131313_floatml_masked_GACOS_Corrected_clipped",strike,dip,rake,slip,depth,length,width,location,dates)