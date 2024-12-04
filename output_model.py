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

import logging
import scipy
from scipy import stats, interpolate, io
import shutil
from scipy.io import loadmat

try:
    import h5py
except ImportError as e:
    raise e('Please install h5py library')
log = logging.getLogger('mat2npz')



def read_mat(filename):
    try:
        mat = h5py.File(filename, 'r')
    except OSError:
        log.debug('using old scipy import for %s', filename)
        mat = io.loadmat(filename)
    return mat




def output_model(geoc_ml_path,output_geoc,model,location,vertex_path,const_offs, date_order):
    # import pygmt
    # reload(pygmt)

   
    try:
        n_para = len(os.sched_getaffinity(0))
    except:
        n_para = multi.cpu_count()
    

    # global ifgdates2,in_dir,out_dir, locations ,opt_model, vertex
    in_dir = geoc_ml_path

    opt_model = model
    in_dir = geoc_ml_path
    out_dir = output_geoc
    locations = location
    
    if not os.path.exists(out_dir):
            os.mkdir(out_dir)
       
  

    q = multi.get_context('fork')

    ifgdates = LiCS_tools.get_ifgdates(in_dir)
    n_ifg = len(ifgdates)


    # x, y, unw, diff = lib.load_ifgs(path_to_i) # pull for dims of data for model
    # EQA_dem_par = os.path.join(in_dir,"EQA.dem_par")
  
    # dlat = float(LiCS_lib.get_param_par(EQA_dem_par, 'post_lat')) #negative
    # dlon = float(LiCS_lib.get_param_par(EQA_dem_par, 'post_lon')) #positive
    # lat1 = float(LiCS_lib.get_param_par(EQA_dem_par, 'corner_lat'))
    # lon1 = float(LiCS_lib.get_param_par(EQA_dem_par, 'corner_lon'))
    # lat2 = lat1+dlat*(length_ifgm-1) # south # Remove
    # lon2 = lon1+dlon*(width_ifgm-1) # east # Remove
    # lon, lat = np.arange(lon1, lon2+dlon, dlon), np.arange(lat1, lat2+dlat, dlat)
    # centerlat = lat1+dlat*(length_ifgm/2)
    # ra = float(LiCS_lib.get_param_par(EQA_dem_par, 'ellipsoid_ra'))
    # recip_f = float(LiCS_lib.get_param_par(EQA_dem_par, 'ellipsoid_reciprocal_flattening'))
    # rb = ra*(1-1/recip_f) ## polar radius
    # pixsp_a = 2*np.pi*rb/360*abs(dlat)
    # pixsp_r = 2*np.pi*ra/360*dlon*np.cos(np.deg2rad(centerlat))

   
    vertex_data = read_mat(vertex_path)
    vertex = np.asarray(vertex_data['vertex_total_list'])
   
  
    ifgdates2 = ifgdates
    n_ifg2 = len(ifgdates2)
    print(n_ifg2)

    # if n_ifg-n_ifg2 > 0:
    #     print("  {0:3}/{1:3} masked unw and cc already exist. Skip".format(n_ifg-n_ifg2, n_ifg), flush=True)

    # if n_ifg2 > 0:
    #     ### Mask with parallel processing
    #     if n_para > n_ifg2:
    #         n_para = n_ifg2

            
    #     print('  {} parallel processing...'.format(n_para), flush=True)
    #     p = q.Pool(n_para)
    #     p.map(forward_modelling_para_gmt, range(n_ifg2))
    #     p.close()
    print('here')
    for ii in range(n_ifg2):
        print(ii)
        forward_modelling_para_gmt(in_dir,out_dir,locations ,opt_model, vertex,ii,ifgdates2,const_offs,date_order)


    # del ifgdates2, in_dir,out_dir,locations ,opt_model, vertex

 

def forward_modelling_para_gmt(in_dir,out_dir,locations ,opt_model, vertex,ifgix,ifgdates2,const_offs,date_order):
    print('here')
    # import pygmt
    # reload(pygmt)
    ifgd = ifgdates2[ifgix]
    print(ifgd)
    print(date_order)
  

    if ifgd in date_order:
        idx = date_order.index(ifgd)
        offset = const_offs[idx]
        print('Constant offset used for ifgm  ' + ifgd )
    else:
        offset = 0    
    EQA_dem_par = os.path.join(in_dir,"EQA.dem_par")
    width_ifgm = int(LiCS_lib.get_param_par(EQA_dem_par, 'width'))
    length_ifgm = int(LiCS_lib.get_param_par(EQA_dem_par, 'nlines'))

    Inc_file = os.path.join(in_dir,'theta.geo') 
    Head_file = os.path.join(in_dir,'phi.geo')
    Inc = np.fromfile(Inc_file, dtype='float32').reshape((length_ifgm, width_ifgm))
    Head = np.fromfile(Head_file, dtype='float32').reshape((length_ifgm, width_ifgm))
    date_dir = in_dir + "/" + ifgd
    file_name = ifgd + '.unw'
    unw_file = os.path.join(date_dir,file_name) 
    unw = np.fromfile(unw_file, dtype='float32').reshape((length_ifgm, width_ifgm)) 



    out_dir1 = os.path.join(out_dir, ifgd)
    if not os.path.exists(out_dir1): 
        os.mkdir(out_dir1)
       

    # if not os.path.exists(os.path.join(out_dir1, ifgd+'.unw')):
    #     unwfile = os.path.join(in_dir, ifgd, ifgd+'.unw')
    #     os.symlink(os.path.relpath(unwfile, out_dir1), os.path.join(out_dir1, ifgd+'.unw')) 

    if not os.path.exists(os.path.join(out_dir1, ifgd+'.unw.png')):
        unwfilepng = os.path.join(in_dir, ifgd, ifgd+'.unw.png')
        shutil.copy(unwfilepng, os.path.join(out_dir1, ifgd+'.unw.png'))  

    # if not os.path.exists(os.path.join(out_dir1, ifgd+'.cc')):
    #     ccfile = os.path.join(in_dir, ifgd, ifgd+'.cc')
    #     os.symlink(os.path.relpath(ccfile, out_dir1), os.path.join(out_dir1, ifgd+'.cc'))

  

    dlat = float(LiCS_lib.get_param_par(EQA_dem_par, 'post_lat')) #negative
    dlon = float(LiCS_lib.get_param_par(EQA_dem_par, 'post_lon')) #positive
    lat1 = float(LiCS_lib.get_param_par(EQA_dem_par, 'corner_lat'))
    lon1 = float(LiCS_lib.get_param_par(EQA_dem_par, 'corner_lon'))

    lat2 = lat1+dlat*(length_ifgm-1) # south # Remove
    lon2 = lon1+dlon*(width_ifgm-1) # east # Remove
    lons, lats = np.arange(lon1, lon2+dlon, dlon), np.arange(lat1, lat2+dlat, dlat)
 
    lats_orig = np.array(lats,dtype=float)[:length_ifgm]
    lons_orig = np.array(lons,dtype=float)[:width_ifgm]

    lons, lats = np.meshgrid(lons_orig,lats_orig)
    ll = [lons.flatten(),lats.flatten()]
    ll = np.array(ll,dtype=float)
    xy = llh.llh2local(ll,np.array([locations[0],locations[1]],dtype=float))
    xx_vec = xy[0,:].reshape((length_ifgm, width_ifgm))
    yy_vec = xy[1,:].reshape((length_ifgm, width_ifgm))
    xx_original = xx_vec[0,:]
    yy_original = yy_vec[:,0]
    xx_original = xx_original[:width_ifgm]
    yy_original = yy_original[:length_ifgm]
    xx_vec = xy[0,:].flatten() 
    yy_vec = xy[1,:].flatten()
  
    length = opt_model[0]
    width = opt_model[1]
    depth = opt_model[2]
    dip = -opt_model[3]
    depth =  depth + ((width/2)*np.sin(np.abs(dip)*(np.pi/180)))
    strike = (opt_model[4] + 180) % 360
    # strike = opt_model[4]
    X = opt_model[5]
    Y = opt_model[6]
    SS = opt_model[7]
    DS = opt_model[8]
    # if const_offs:
    #     const_offset = opt_model[9]
    # else:
    #     const_offs = 0

    rake = np.degrees(np.arctan2(DS,SS))
    print('Rake for forward model')
    print(rake)
    total_slip = np.sqrt(SS**2 + DS**2)

    # model = [X, Y, strike, dip, rake, total_slip, length, depth, width]
    # print(model)
    # print(depth)
    # print(length)
    # print(width)
    # print(total_slip)
    # print(strike)
    # print(dip)
    # print(rake)
    lons_vertex = vertex[:][0]
    lats_vertex = vertex[:][1]
    print('this one')
    print(np.array([lons_vertex,lats_vertex]))
    vertex_meters = llh.llh2local(np.array([lons_vertex,lats_vertex]), np.array([locations[0],locations[1]])) 
    # print(vertex[0:2,:])
    # print(vertex_meters)
    X_vertex = vertex_meters[0][:]
    Y_vertex = vertex_meters[1][:]
   
    # depth_vertex = vertex_meters[2][:]

    
    X_cent, Y_cent = np.mean(X_vertex[0:len(X_vertex)-2]),np.mean(Y_vertex[0:len(Y_vertex)-2])
    print(X)
    print(X_cent)
  

    disp =lib.disloc3d3(xx_vec,yy_vec,xoff=X_cent,yoff=Y_cent,depth=depth,
                            length=length,width=width,slip=total_slip,
                            opening=0,strike=strike,dip=dip,rake=rake,nu=0.25)


    incidence_angle = np.nanmean(Inc)
    azimuth_angle = np.nanmean(Head)

    unw = -unw*(0.0555/(4*np.pi)) - offset
    # unw = unw[~np.isnan(unw)]
    # lats_flat = lats.flatten()
    # lats_flat = lats_flat[~np.isnan(unw)]
    # lons_flat = lons.flatten()
    # lons_flat = lons_flat[~np.isnan(unw)]
    
    # unw,lats,lons =  LiCS_tools.invert_plane(unw.flatten(), lats.flatten(), lons.flatten()) # invert for plane and remove constant offset and ramp from data, this is done for the modelling at step downsampling
    e2los = np.cos(np.deg2rad(azimuth_angle)) * np.sin(np.deg2rad(incidence_angle))
    n2los = -np.sin(np.deg2rad(azimuth_angle)) * np.sin(np.deg2rad(incidence_angle))
    u2los = -np.cos(np.deg2rad(incidence_angle))
    los_grid = (disp[0,:] * e2los) + (disp[1,:] * n2los) + (disp[2,:] * u2los) 
    los_grid = los_grid 
    resid = unw.flatten() - los_grid
    rms = np.round(np.sqrt(np.mean(np.square(resid[~np.isnan(resid)]))),decimals=4)
    rms_unw = np.round(np.sqrt(np.mean(np.square(unw[~np.isnan(unw)]))),decimals=4)

    wrapped_los = np.mod(unw,0.056/2)
    wrapped_res = np.mod(resid, 0.056/2)
    wrapped_model = np.mod(los_grid,0.056/2)

    region = [np.min(lons),np.max(lons),np.min(lats),np.max(lats)] # GMT region  [xmin,xmax,ymin,ymax].
    
    file_path_data = os.path.join(out_dir1,'data_meters.grd')
    file_path_res = os.path.join(out_dir1,'residual_model_meters.grd')
    file_path_model = os.path.join(out_dir1,'model_meters.grd')

    pygmt.xyz2grd(x=lons.flatten(),y=lats.flatten(),z=unw.flatten(),outgrid=file_path_data,region=region,spacing=(0.001,0.001))
    pygmt.xyz2grd(x=lons.flatten(),y=lats.flatten(),z=resid,outgrid=file_path_res,region=region,spacing=(0.001,0.001))
    pygmt.xyz2grd(x=lons.flatten(),y=lats.flatten(),z=los_grid,outgrid=file_path_model,region=region,spacing=(0.001,0.001))

    pygmt.xyz2grd(x=lons.flatten(),y=lats.flatten(),z=wrapped_los.flatten(),outgrid='wrapped_los.unw.grd',region=region,spacing=(0.001,0.001))
    pygmt.xyz2grd(x=lons.flatten(),y=lats.flatten(),z=wrapped_res,outgrid='wrapped_residual.unw.grd',region=region,spacing=(0.001,0.001))
    pygmt.xyz2grd(x=lons.flatten(),y=lats.flatten(),z=wrapped_model,outgrid='wrapped_model.unw.grd',region=region,spacing=(0.001,0.001))

    max_data = np.max(unw[~np.isnan(unw)].flatten()) 
    # print(max_data)
    min_data = np.min(unw[~np.isnan(unw)].flatten()) 
    data_series = str(min_data) + '/' + str(max_data*1.5) +'/' + str((max_data - min_data)/100)
    # print("data cpt")
    # print(data_series)

    max_model = np.max(los_grid) 
    min_model = np.min(los_grid) 
    model_series = str(min_model) + '/' + str(max_model*1.5) +'/' + str((max_model - min_model)/100)
    # print("model cpt")
    # print(model_series)

    max_resid = np.max(resid[~np.isnan(resid)]) 
    min_resid = np.min(resid[~np.isnan(resid)]) 
    resid_series = str(min_resid) + '/' + str(max_resid*1.5) +'/' + str((max_resid - min_resid)/100)
    # print("resid cpt")
    # print(resid_series)


    fig = pygmt.Figure()
    pygmt.config(MAP_FRAME_TYPE="plain")
    pygmt.config(FORMAT_GEO_MAP="ddd.xx")
    pygmt.config(FORMAT_FLOAT_OUT='%.12lg') 
    pygmt.config(GMT_VERBOSE='e')
    
    
    cmap_output_data = os.path.join(out_dir1,'data_meters.cpt')
    cmap_output_wrapped = os.path.join(out_dir1,'wrapped_data.cpt')
    # pygmt.makecpt(cmap='polar',series=data_series, continuous=True,output=cmap_output_data,background=True)
    # cmap_output_model = os.path.join(out_dir1,'model_meters.cpt')
    # pygmt.makecpt(cmap='polar',series=model_series, continuous=True,output=cmap_output_model,background=True)
    # cmap_output_resid = os.path.join(out_dir1,'resid_meters.cpt')
    # pygmt.makecpt(cmap='polar',series=resid_series, continuous=True,output=cmap_output_resid,background=True)

    if np.abs(min_data) > max_data:
        range_limit = np.abs(min_data)
    else:
        range_limit = max_data    

    if range_limit > 2:
        range_limit = 1

  
    vik = '/uolstore/Research/a/a285/homes/ee18jwc/code/colormaps/vik/vik.cpt'
    roma = '/uolstore/Research/a/a285/homes/ee18jwc/code/colormaps/roma/roma.cpt'
    pygmt.makecpt(series=[-range_limit, range_limit], cmap=vik,output=cmap_output_data,background=True)
    pygmt.makecpt(cmap=roma,series=[0,0.056/2], continuous=True,output=cmap_output_wrapped,background=True) 

    lons_vertex = vertex[0][:]
    lats_vertex = vertex[1][:]
    depth_vertex = vertex[2][:]
   
    # print(lons_vertex)
    # print(lats_vertex)
    # print(depth_vertex)

    with fig.subplot(
        nrows=2,
        ncols=3,
        figsize=("45c", "45c","45c"),
        autolabel=False,
        frame=["f","WSne","+tData"],
        margins=["0.05c", "0.05c"],
        # title="Geodetic Moderling Sequence one 2023/09/25 - 2023/10/08",
    ):
        
        fig.grdimage(grid=file_path_data,cmap=cmap_output_data,region=region,projection='M?c',panel=[0,0])
        fig.basemap(frame=['a','+tData (rms: '+str(rms_unw) + ')'],panel=[0,0],region=region,projection='M?c')
        fig.grdimage(grid=file_path_model,cmap=cmap_output_data,region=region,projection='M?c',panel=[0,1])
        fig.basemap(frame=['xa','+tModel Auto_Inv'],panel=[0,1],region=region,projection='M?c')
        fig.grdimage(grid=file_path_res,cmap=cmap_output_data,region=region,projection='M?c',panel=[0,2])
        fig.basemap(frame=['xa','+tResidual (rms: '+str(rms) + ')'],panel=[0,2],region=region,projection='M?c')
        
        
        for ii in range(0,3):
            fig.plot(x=lons_vertex[:-2],
                        y=lats_vertex[:-2],
                        pen='1p,black',
                        fill='gray',
                        transparency=80,
                        region=region,
                        projection='M?c',
                        panel=[0,ii]
            )

            fig.plot(x=lons_vertex[len(lons_vertex)-2:len(lons_vertex)],
                        y=lats_vertex[len(lats_vertex)-2:len(lats_vertex)],
                        pen='4p,black,-',
                        fill='gray',
                        # transparency=,
                        region=region,
                        projection='M?c',
                        panel=[0,ii]
            )
         
        for ii in range(0,3):
                fig.colorbar(frame=["x+lLOS Displacement(m)", "y+lm"], position="JMB",projection='M?c',panel=[0,ii]) # ,panel=[1,0]
        # fig.show(method='external')
       



        fig.grdimage(grid='wrapped_los.unw.grd',cmap='Wrapped_CPT.cpt',region=region,projection='M?c',panel=[1,0])
        fig.basemap(frame=['a'],panel=[1,0],region=region,projection='M?c')
        fig.grdimage(grid='wrapped_model.unw.grd',cmap='Wrapped_CPT.cpt',region=region,projection='M?c',panel=[1,1])
        fig.basemap(frame=['xa'],panel=[1,1],region=region,projection='M?c')
        fig.grdimage(grid='wrapped_residual.unw.grd',cmap='Wrapped_CPT.cpt',region=region,projection='M?c',panel=[1,2])
        fig.basemap(frame=['xa'],panel=[1,2],region=region,projection='M?c')


                
        
        for ii in range(0,3):
            fig.plot(x=lons_vertex[:-2],
                        y=lats_vertex[:-2],
                        pen='1p,black',
                        fill='gray',
                        transparency=80,
                        region=region,
                        projection='M?c',
                        panel=[1,ii]
            )

            fig.plot(x=lons_vertex[len(lons_vertex)-2:len(lons_vertex)],
                        y=lats_vertex[len(lats_vertex)-2:len(lats_vertex)],
                        pen='4p,black,-',
                        fill='gray',
                        # transparency=,
                        region=region,
                        projection='M?c',
                        panel=[1,ii]
            )
         
        for ii in range(0,3):
                fig.colorbar(frame=["x+lLOS Displacement(m)", "y+lm"], position="JMB",projection='M?c',panel=[1,ii]) # ,panel=[1,0]

    fig.savefig(os.path.join(os.path.join(out_dir1),"output_model_comp.png"))

# def forward_modelling_2D_plot(resampled_terrain,gradiant_terrain,ifgix):
#     import pygmt
#     reload(pygmt)

#     ifgd = ifgdates2[ifgix]

#     if np.mod(ifgix,100) == 0:
#         print("  {0:3}/{1:3}th unw...".format(ifgix, len(ifgdates2)), flush=True)

    
#     out_dir1 = os.path.join(out_dir, ifgd)
#     if not os.path.exists(out_dir1): 
#         os.mkdir(out_dir1)
       

#     dlat = float(LiCS_lib.get_param_par(EQA_dem_par, 'post_lat')) #negative
#     dlon = float(LiCS_lib.get_param_par(EQA_dem_par, 'post_lon')) #positive
#     lat1 = float(LiCS_lib.get_param_par(EQA_dem_par, 'corner_lat'))
#     lon1 = float(LiCS_lib.get_param_par(EQA_dem_par, 'corner_lon'))

#     lat2 = lat1+dlat*(length_ifgm-1) # south # Remove
#     lon2 = lon1+dlon*(width_ifgm-1) # east # Remove
#     lons, lats = np.arange(lon1, lon2+dlon, dlon), np.arange(lat1, lat2+dlat, dlat)
 
#     lats_orig = np.array(lats,dtype=float)[:length_ifgm]
#     lons_orig = np.array(lons,dtype=float)[:width_ifgm]

#     lons, lats = np.meshgrid(lons_orig,lats_orig)


#     out_dir1 = os.path.join(out_dir, ifgd)
#     print('######################################## 2D locations plot ##############################################################')
#     x1 = round((np.min(lons) - (np.min(lons) % 0.03) - 2*0.03),2)
#     x2 = round((np.max(lons) - (np.max(lons) % 0.03) + 2*0.03),2)
#     y1 = round((np.min(lats) - (np.min(lats) % 0.03) - 2*0.03),2)
#     y2 = round((np.max(lats) - (np.max(lats) % 0.03) + 2*0.03),2)
#     print("y1")
#     print(y2%0.03)

#     region = [np.min(lons),np.max(lons),np.min(lats),np.max(lats)] 
#     dem_region = [x1,x2,y1,y2]
#     dem_region = [np.min(lons)-0.1,np.max(lons)+0.1,np.min(lats)-0.1,np.max(lats)+0.1] 

#     date_dir = in_dir + "/" + ifgd
#     file_name = ifgd + '.unw'
#     unw_file = os.path.join(date_dir,file_name) 
#     unw = np.fromfile(unw_file, dtype='float32').reshape((length_ifgm, width_ifgm)) 
#     if opt_model[9]:
#         const_offset = opt_model[9]
#     else:
#         const_offset = 0
#     file_path_data = os.path.join(out_dir1,'data_meters.grd')

#     unw = -unw*(0.0555/(4*np.pi)) - const_offset
#     pygmt.xyz2grd(x=lons.flatten(),y=lats.flatten(),z=unw.flatten(),outgrid=file_path_data,region=region,spacing='03s')

   
#     print('surface project')
#     # pygmt.xyz2grd(x=lons.flatten(),y=lats.flatten(),z=unw.flatten(),outgrid=file_path_data,region=region,spacing=(0.001,0.001))
#     lons_vertex = vertex[0][:]
#     lats_vertex = vertex[1][:]
#     depth_vertex = vertex[2][:]
#     surface_projection = pygmt.project(center=[lons_vertex[4],lats_vertex[4]],endpoint=[lons_vertex[5],lats_vertex[5]], generate=0.01)
 
#     max_data = np.nanmax(unw[~np.isnan(unw)].flatten()) 
#     print(max_data)
#     min_data = np.nanmin(unw[~np.isnan(unw)].flatten()) 
#     data_series = str(min_data) + '/' + str(max_data*1.5) +'/' + str((max_data - min_data)/100)
#     print("data cpt")
#     print(data_series)
    
#     # print("data cpt")
#     # print(data_series)
#     print('make color pallette')
#     fig = pygmt.Figure()
#     pygmt.config(MAP_FRAME_TYPE="plain")
#     pygmt.config(FORMAT_GEO_MAP="ddd.xx")
#     pygmt.config(FORMAT_FLOAT_OUT='%.12lg') 
#     # cmap_output_data =  out_dir1 + '/' +'InSAR_CPT_data.cpt'
#     topo_output_cpt = out_dir1 + '/topo.cpt'
#     topo_cpt_series = '0/5000/100' 
#     cmap_output_data = os.path.join(out_dir1,'InSAR_CPT_data.cpt')
#     file_path_data = os.path.join(out_dir1,'data_meters.grd')


   

#     # resampled_terrain = out_dir+'/'+'terrain.grd'
#     # resamp = pygmt.grdsample(terrain,region=region,outgrid=resampled_terrain,spacing='10s')
 
    
#     pygmt.makecpt(cmap='oleron',series=topo_cpt_series, continuous=True,output=topo_output_cpt,background=True) 
#     pygmt.makecpt(cmap='polar',series=data_series, continuous=True,output=cmap_output_data,background=True) 

    
#     print('add basemap, gridimage and coast to 2D figure')
  
#     fig.grdimage(resampled_terrain,cmap=topo_output_cpt,region=region,projection='M8c',shading=gradiant_terrain)
  
   
#     print('Down sample InSAR grd')
#     unw_grd_ds = pygmt.grdsample(grid=file_path_data,spacing='03s',registration='gridline',region=region)
#     print('grdimage InSAR grid')
#     fig.grdimage(grid=unw_grd_ds,cmap=cmap_output_data,region=region,projection='M8c',shading=gradiant_terrain)
#     fig.basemap(frame=['a','+tUSGS Location Comparison'],region=region,projection='M8c',map_scale="jBL+w10k+o0.5c/0.5c+f+lkm")
#     fig.coast(region=region, projection = 'M8c', water='lightblue')
#     print('plotting fault plane')
#     fig.plot(x=lons_vertex[:-2],
#                         y=lats_vertex[:-2],
#                         pen='1p,black',
#                         fill='gray',
#                         transparency=20,
#                         region=region,
#                         projection='M8c',
#                         )
#     print('plotting surface projection')
#     fig.plot(x=lons_vertex[len(lons_vertex)-2:len(lons_vertex)],
#                 y=lats_vertex[len(lats_vertex)-2:len(lats_vertex)],
#                 pen='1p,black,.',
#                 no_clip=False,
#                 #    sizes=0.1 * (2**np.array(eqMagAll)),
#                 #    style="uc",
#                 fill="gray",
#                 projection="M8c",
#                 transparency=20,
#                 # frame=["xa", "yaf", "za", "wSnEZ"],
#                 region=region)
#     print('plotting USGS location')
#     fig.plot(x=locations[0],
#              y=locations[1],
#              pen='1p,red',
#              style='a0.4c',
#              fill='darkorange')
#     font = "2p,Helvetica-Bold"
#     fig.text(x=locations[0], y=locations[1] + np.abs(0.001*locations[1]), text="USGS", font=font, region=region,projection='M8c',fill="white")
#     fig.colorbar(frame=["x+lLOS displacment(m)", "y+lm"], position="JMB",projection='M8c') # ,panel=[1,0]
#     # fig_2D.savefig('2D_test.png')
#     print('saving figure')
#     fig.savefig(os.path.join(out_dir1,'2D_locations_plot.png'))
#     print('figure saved')
#     # fig_2D.close()
#     # fig.show()

#     # return 
  

if __name__ == '__main__':
    model = [
        7221.23,
       11694.6,
          14930.3,
       -53.9059,
          16.0605,
        2457.95,
        1977.82,
     0.661505	,
      -1.64494	,
            0]


    
    # model = [ 5676.77,
    #           6019.58,
    #           3605.16,
    #           -30.6425,
    #           187.587,
    #           -2.67297,
    #           211.366,
    #           -0.47416,
    #         -0.28685,
    #         0
    # ]
    geoc_ml_path = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us7000lt29_insar_processing/GEOC_034D_04913_131313_floatml_masked_GACOS_Corrected_clipped'
    output_geoc = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/test'
    # location = [44.9097,38.4199]
    # location = [28.5896,87.3081]
    # location = [2.6675,-59.5879]
    date_order = []
    outputfile = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us7000lt29_NP2/invert_2_3_5_6_F/invert_2_3_5_6_F.mat'
    output_matrix = loadmat(outputfile,squeeze_me=True)
    opt_model = np.array(output_matrix['invResults'][()].item()[-1])
    const_offs =  np.array(output_matrix['invResults'][()].item()[2])[()].item()[10]
    print(const_offs)
    const_offs = const_offs[9:len(const_offs)]
    InSAR_codes = []
    with open('/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us7000lt29_GBIS_area' + '/ifgms_used_in_inversion.txt','r') as file:
            lines = file.readlines()
    for line in lines:
            InSAR_codes.append(int(line.split('file')[0])+1)
            date_order.append(line.split('_QAed_')[-1].split('.ds')[0])
    location = [7.6222,124.9103]
    location = [124.9103,7.6222]
    location = [19.5256,41.5138]

    vertex_path = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us7000lt29_NP2/invert_2_3_5_6_F/optmodel_vertex.mat'
    # try:
    output_model(geoc_ml_path,output_geoc,model,location,vertex_path,const_offs, date_order)
    # except:
    #     pass