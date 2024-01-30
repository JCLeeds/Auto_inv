#!/usr/bin/env python
import coseis_lib as lib
import numpy as np
import matplotlib.pyplot as plt
import LiCSBAS_io_lib as LiCS_lib
import os 
import LiCSBAS_tools_lib as LiCS_tools
# this is additional scientific colour maps, see "https://www.fabiocrameri.ch/colourmaps/"
from cmcrameri import cm



def forward_model(geoc_ml_path,strike,dip,rake,slip,depth,width,length,location,dates):
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

    y = np.arange(0, (length_ifgm + 1) * pixsp_r, pixsp_r)
    x = np.arange(0, (width_ifgm + 1) * pixsp_a, pixsp_a) 
    y = y[:length_ifgm]
    x = x[:width_ifgm]
    x_usgs,y_usgs = LiCS_tools.bl2xy(location[1],location[0],width_ifgm,length_ifgm,lat1,dlat,lon1,dlon)
    xcent = x_usgs * pixsp_a
    ycent = y_usgs * pixsp_r
    m_cent = [xcent,ycent]
    print(xcent)
    print(ycent)
    Inc_file = os.path.join(geoc_ml_path,'Theta.geo') 
    Head_file = os.path.join(geoc_ml_path,'Phi.geo')
    Inc = np.fromfile(Inc_file, dtype='float32').reshape((length_ifgm, width_ifgm))
    Head = np.fromfile(Head_file, dtype='float32').reshape((length_ifgm, width_ifgm))
    date_dir = geoc_ml_path + "/" + dates
    file_name = dates + '.unw'
    unw_file = os.path.join(date_dir,file_name) 
    unw = np.fromfile(unw_file, dtype='float32').reshape((length_ifgm, width_ifgm))       
    xx, yy = np.meshgrid(x, y)
    # convert to pair of coordinates for our dislocation model
    xx_vec = np.reshape(xx, -1)
    yy_vec = np.reshape(yy, -1)
    # Define the fault model
    #####################################
    xcen = xcent          # vertical surface projection of fault centroid in x (m)
    ycen = ycent        # vertical surface projection of fault centroid in y (m)
    strike = strike       # strike in degrees (0-360)
    dip =    dip        # dip in degrees (0-90)
    rake =   rake       # rake in degrees (-180 - 180)
    slip =     slip       # magnitude of slip vector in metres
    centroid_depth = depth       # depth (measured vertically) to fault centroid in metres
    width = width    # width of fault measured along-dip in metres
    length =  length        # fault length in metres
    #####################################
    model = [xcen, ycen, strike, dip, rake, slip, length, centroid_depth, width]
    # Calcualte displacements
    disp = lib.disloc3d3(xx_vec, yy_vec, xoff=xcen, yoff=ycen, depth=centroid_depth,
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
    plt.show()
    return fig


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