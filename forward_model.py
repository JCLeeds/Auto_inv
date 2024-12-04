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
import llh2local as llh
import local2llh as l2llh
import pygmt
from importlib import reload



def forward_model(input_dir,strike_F,dip_F,rake_F,slip_F,depth_F,width_F,length_F,location,NP):
    # try:
    #     n_para = len(os.sched_getaffinity(0))
    # except:
    #     n_para = multi.cpu_count()
    

    global ifgdates2, length_ifgm, width_ifgm, geoc_ml_path, pixsp_a, pixsp_r, xcent, ycent,strike,dip,rake,depth,width,length,slip,EQA_dem_par, locations,nodal_plane
    geoc_ml_path = input_dir
    strike = strike_F
    # strike = strike - 180 
    # if strike < 0: 
    #     strike = strike + 360
    locations = location
    dip = dip_F 
    rake = rake_F 
    slip = slip_F 
    depth = depth_F 
    width = width_F 
    length = length_F
    nodal_plane = NP
    

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
        # if n_para > n_ifg2:
        #     n_para = n_ifg2

        for ii in range(n_ifg2):
            forward_modelling_para_gmt(ii)
        # print('  {} parallel processing...'.format(n_para), flush=True)
        # p = q.Pool(n_para)
        # p.map(forward_modelling_para_gmt, range(n_ifg2))
        # p.close()


def forward_modelling_para_gmt(ifgix):
    # import pygmt
    # reload(pygmt)
    ifgd = ifgdates2[ifgix]
    Inc_file = os.path.join(geoc_ml_path,'theta.geo') 
    Head_file = os.path.join(geoc_ml_path,'phi.geo')
    Inc = np.fromfile(Inc_file, dtype='float32').reshape((length_ifgm, width_ifgm))
    Head = np.fromfile(Head_file, dtype='float32').reshape((length_ifgm, width_ifgm))
    date_dir = geoc_ml_path + "/" + ifgd
    file_name = ifgd + '.unw'
    unw_file = os.path.join(date_dir,file_name) 
    unw = np.fromfile(unw_file, dtype='float32').reshape((length_ifgm, width_ifgm))
    unw = -unw*(0.0555/(4*np.pi))
    # width_ifgm = int(LiCS_lib.get_param_par(EQA_dem_par, 'width'))
    # length_ifgm = int(LiCS_lib.get_param_par(EQA_dem_par, 'nlines'))
    dlat = float(LiCS_lib.get_param_par(EQA_dem_par, 'post_lat')) #negative
    dlon = float(LiCS_lib.get_param_par(EQA_dem_par, 'post_lon')) #positive
    lat1 = float(LiCS_lib.get_param_par(EQA_dem_par, 'corner_lat'))
    lon1 = float(LiCS_lib.get_param_par(EQA_dem_par, 'corner_lon'))

    lat2 = lat1+dlat*(length_ifgm-1) # south # Remove
    lon2 = lon1+dlon*(width_ifgm-1) # east # Remove
    lons, lats = np.arange(lon1, lon2+dlon, dlon), np.arange(lat1, lat2+dlat, dlat)
    # centerlat = lat1+dlat*(length_ifgm/2)
    # ra = float(LiCS_lib.get_param_par(EQA_dem_par, 'ellipsoid_ra'))
    # recip_f = float(LiCS_lib.get_param_par(EQA_dem_par, 'ellipsoid_reciprocal_flattening'))
    # rb = ra*(1-1/recip_f) ## polar radius
    # pixsp_a = 2*np.pi*rb/360*abs(dlat)
    # pixsp_r = 2*np.pi*ra/360*dlon*np.cos(np.deg2rad(centerlat))
    lats_orig = np.array(lats,dtype=float)[:length_ifgm]
    lons_orig = np.array(lons,dtype=float)[:width_ifgm]

    lons, lats = np.meshgrid(lons_orig,lats_orig)
    lons = lons.flatten() 
    lats = lats.flatten()
    ll = [lons.flatten(),lats.flatten()]
    ll = np.array(ll,dtype=float)
    xy = llh.llh2local(ll,np.array([locations[1],locations[0]],dtype=float))
    xx_vec = xy[0,:].reshape((length_ifgm, width_ifgm))
    yy_vec = xy[1,:].reshape((length_ifgm, width_ifgm))
    xx_original = xx_vec[0,:]
    yy_original = yy_vec[:,0]
    xx_original = xx_original[:width_ifgm]
    yy_original = yy_original[:length_ifgm]
    xx_vec = xy[0,:].flatten() 
    yy_vec = xy[1,:].flatten()
    unw = unw.flatten()
    if len(xx_vec) > 100000:  
        indexs_to_remove_for_decimation = np.random.randint(low=0,high=len(xx_vec),size=int(len(xx_vec)*0.5))
        xx_vec = xx_vec[indexs_to_remove_for_decimation]
        yy_vec = yy_vec[indexs_to_remove_for_decimation]
        unw_ds = unw[indexs_to_remove_for_decimation]
        lons_ds = lons[indexs_to_remove_for_decimation]
        lats_ds = lats[indexs_to_remove_for_decimation]
    
        

    centroid_depth = depth
    # centroid_depth=5000
    print(depth)
    model = [0, 0, strike, dip, rake, slip, length, centroid_depth, width]
    # disp = lib.disloc3d3(xx_vec, yy_vec, xoff=xcent, yoff=ycent, depth=centroid_depth,
    #                 length=length, width=width, slip=slip, opening=0, 
    #                 strike=strike, dip=dip, rake=rake, nu=0.25)
    
    disp_usgs =lib.disloc3d3(xx_vec,yy_vec,xoff=0,yoff=0,depth=model[7],
                            length=model[6],width=model[8],slip=model[5],
                            opening=0,strike=model[2],dip=model[3],rake=model[4],nu=0.25)
    end1x, end2x, end1y, end2y, c1x, c2x, c3x, c4x, c1y, c2y, c3y, c4y = lib.fault_for_plotting(model)
    x_usgs = np.array([c1x,c2x,c3x,c4x])/ 1000 #local2llh needs results in km 
    y_usgs = np.array([c1y,c2y,c3y,c4y])/ 1000 #local2llh needs results in km 
    x_usgs_end = np.array([end1x,end2x]) / 1000 
    y_usgs_end = np.array([end1y,end2y]) / 1000

    local_usgs_source = [x_usgs,y_usgs]  
    local_end_usgs_source = [x_usgs_end,y_usgs_end]
    print(np.shape(local_usgs_source))
    print("LOCATIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~ = " + str(locations))
    llh_usgs = l2llh.local2llh(local_usgs_source,[locations[1],locations[0]])
    fault_end = l2llh.local2llh(local_end_usgs_source,[locations[1],locations[0]])

    incidence_angle = np.nanmean(Inc)
    azimuth_angle = np.nanmean(Head)

   
    e2los = np.cos(np.deg2rad(azimuth_angle)) * np.sin(np.deg2rad(incidence_angle))
    n2los = -np.sin(np.deg2rad(azimuth_angle)) * np.sin(np.deg2rad(incidence_angle))
    u2los = -np.cos(np.deg2rad(incidence_angle))
    los_grid_usgs = (disp_usgs[0,:] * e2los) + (disp_usgs[1,:] * n2los) + (disp_usgs[2,:] * u2los) 
    los_grid_usgs = -los_grid_usgs
    if len(xx_vec) > 100000:  
        resid_usgs = unw_ds.flatten() - los_grid_usgs
    else:
        resid_usgs = unw.flatten() - los_grid_usgs

    rms = np.round(np.sqrt(np.mean(np.square(resid_usgs[~np.isnan(resid_usgs)]))),decimals=4)
    rms_unw = np.round(np.sqrt(np.mean(np.square(unw[~np.isnan(unw)]))),decimals=4)
    region = [np.min(lons),np.max(lons),np.min(lats),np.max(lats)] # GMT region  [xmin,xmax,ymin,ymax].
    
    file_path_data = os.path.join(date_dir,'data_meters.grd')
    file_path_res = os.path.join(date_dir,'residual_usgs_model_meters.grd')
    file_path_model = os.path.join(date_dir,'model_usgs_meters.grd')

    if len(xx_vec) > 100000:  
        pygmt.xyz2grd(x=lons_ds.flatten(),y=lats_ds.flatten(),z=resid_usgs,outgrid=file_path_res,region=region,spacing=(0.005,0.005))
        pygmt.xyz2grd(x=lons_ds.flatten(),y=lats_ds.flatten(),z=los_grid_usgs,outgrid=file_path_model,region=region,spacing=(0.01,0.01))
        pygmt.xyz2grd(x=lons.flatten(),y=lats.flatten(),z=unw,outgrid=file_path_data,region=region,spacing=(0.001,0.001))
    else:
        pygmt.xyz2grd(x=lons.flatten(),y=lats.flatten(),z=unw,outgrid=file_path_data,region=region,spacing=(0.001,0.001))
        pygmt.xyz2grd(x=lons.flatten(),y=lats.flatten(),z=resid_usgs,outgrid=file_path_res,region=region,spacing=(0.001,0.001))
        pygmt.xyz2grd(x=lons.flatten(),y=lats.flatten(),z=los_grid_usgs,outgrid=file_path_model,region=region,spacing=(0.001,0.001))

    max_data = np.nanmax(unw) 
    print(max_data)
    min_data = np.nanmin(unw) 

    if max_data < min_data:
        max_data, min_data = min_data, max_data
    data_series = str(min_data) + '/' + str(max_data) +'/' + str((max_data - min_data)/100)
    print('LOOK HERE DATA SERIES!!!')
    print(data_series)
    max_model = np.max(los_grid_usgs) 
    min_model = np.min(los_grid_usgs) 
    model_series = str(min_model) + '/' + str(max_model) +'/' + str((max_model - min_model)/100)
    print("model cpt")
    print(model_series)

    fig = pygmt.Figure()
    pygmt.config(MAP_FRAME_TYPE="plain")
    pygmt.config(FORMAT_GEO_MAP="ddd.xx")
    pygmt.config(FORMAT_FLOAT_OUT='%.12lg')
    vik = '/uolstore/Research/a/a285/homes/ee18jwc/code/colormaps/vik/vik.cpt'
    cmap_output_data = os.path.join(date_dir,'data_meters.cpt')
    pygmt.makecpt(cmap=vik,series=data_series, continuous=True,output=cmap_output_data,background=True)
    cmap_output_model = os.path.join(date_dir,'model_meters.cpt')
    pygmt.makecpt(cmap=vik,series=model_series, continuous=True,output=cmap_output_model,background=True)

    if np.abs(min_data) > max_data:
        range_limit = np.abs(min_data)
    else:
        range_limit = max_data    

    pygmt.makecpt(series=[-range_limit, range_limit], cmap=vik,output=cmap_output_data,background=True)


    if np.abs(min_model) > max_model:
        range_limit_model = np.abs(min_model)
    else:
        range_limit_model = max_model    

    pygmt.makecpt(series=[-range_limit_model, range_limit_model], cmap=vik,output=cmap_output_model,background=True)

    with fig.subplot(
        nrows=1,
        ncols=3,
        figsize=("45c", "45c","45c"),
        autolabel=False,
        frame=["f","WSne","+tData"],
        margins=["0.05c", "0.05c"],
        # title="Geodetic Moderling Sequence one 2023/09/25 - 2023/10/08",
    ):
        
        fig.grdimage(grid=file_path_data,cmap=cmap_output_data,region=region,projection='M?c',panel=[0,0])
        fig.basemap(frame=['a','+tData (rms: '+str(rms_unw) + ')'],panel=[0,0],region=region,projection='M?c')
        fig.grdimage(grid=file_path_model,cmap=cmap_output_model,region=region,projection='M?c',panel=[0,1])
        fig.basemap(frame=['xa','+tModel USGS'],panel=[0,1],region=region,projection='M?c')
        fig.grdimage(grid=file_path_res,cmap=cmap_output_data,region=region,projection='M?c',panel=[0,2])
        fig.basemap(frame=['xa','+tResidual (rms: '+str(rms) + ')'],panel=[0,2],region=region,projection='M?c')
        
        print(llh_usgs[0,:])
        print(llh_usgs[1,:])
        for ii in range(0,3):
            fig.plot(x=llh_usgs[0,:],
                        y=llh_usgs[1,:],
                        pen='2p,black',
                        fill='gray',
                        transparency=80,
                        region=region,
                        projection='M?c',
                        panel=[0,ii]
            )

            fig.plot(x=fault_end[0,:],
                        y=fault_end[1,:],
                        pen='2p,black,-',
                        fill='gray',
                        transparency=80,
                        region=region,
                        projection='M?c',
                        panel=[0,ii]
            )
         
        for ii in range(0,3):
                fig.colorbar(frame=["x+lLOS displacment(m)", "y+lm"], position="JMB",projection='M?c',panel=[0,ii]) # ,panel=[1,0]
        # fig.show(method='external')
        fig.savefig(os.path.join(os.path.join(geoc_ml_path),ifgd+"forward_model_comp_" + str(nodal_plane) + ".png"))



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
    test = -unw*(0.0555/(4*np.pi))
    e2los = np.cos(np.deg2rad(azimuth_angle)) * np.sin(np.deg2rad(incidence_angle))
    n2los = -np.sin(np.deg2rad(azimuth_angle)) * np.sin(np.deg2rad(incidence_angle))
    u2los = -np.cos(np.deg2rad(incidence_angle))




    fig = lib.plot_data_model(x, y, disp, model, test, e2los, n2los, u2los, show_grid=True)
    plt.savefig(os.path.join(os.path.join(geoc_ml_path),ifgd+"forward_model_comp" + str(nodal_plane) +".png"))
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
    dates = '20200110_20200614'
    forward_model("/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/test/GEOC_043A_05008_161514_floatml_masked_GACOS_Corrected_clipped_signal_masked",strike,dip,rake,slip,depth,length,width,location,dates)