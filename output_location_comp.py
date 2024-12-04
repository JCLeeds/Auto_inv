#!/usr/bin/env python
import coseis_lib as lib
import numpy as np
import matplotlib.pyplot as plt
import LiCSBAS_io_lib as LiCS_lib
import os 
import LiCSBAS_tools_lib as LiCS_tools
# this is additional scientific colour maps, see "https://www.fabiocrameri.ch/colourmaps/"
from cmcrameri import cm
# import multiprocessing as multi
import llh2local as llh
import local2llh as l2llh
import pygmt
# from importlib import reload
import logging
import scipy
from scipy import stats, interpolate, io

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




def output_location_comp(geoc_ml_path,output_geoc,model,location,vertex_path):
    # import pygmt
    # reload(pygmt)

   
    # try:
    #     n_para = len(os.sched_getaffinity(0))
    # except:
    #     n_para = multi.cpu_count()
    

    global ifgdates2, length_ifgm, width_ifgm,in_dir,out_dir, pixsp_a, pixsp_r,EQA_dem_par, locations,opt_model, vertex
    in_dir = geoc_ml_path

    opt_model = model
    in_dir = geoc_ml_path
    out_dir = output_geoc
    locations = location
    
    if not os.path.exists(out_dir):
            os.mkdir(out_dir)
       
  

    # q = multi.get_context('fork')

    ifgdates = LiCS_tools.get_ifgdates(in_dir)
    n_ifg = len(ifgdates)


    # x, y, unw, diff = lib.load_ifgs(path_to_i) # pull for dims of data for model
    EQA_dem_par = os.path.join(in_dir,"EQA.dem_par")
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

   
    vertex_data = read_mat(vertex_path)
    vertex = np.asarray(vertex_data['vertex_total_list'])
   
  
    ifgdates2 = ifgdates
    n_ifg2 = len(ifgdates2)


    

    # global resampled_terrain, gradiant_terrain
    dem_region = [np.min(lon)-0.4,np.max(lon)+0.4,np.min(lat)-0.4,np.max(lat)+0.4] 
    region = [np.min(lon),np.max(lon),np.min(lat),np.max(lat)] 
    terrain = pygmt.datasets.load_earth_relief(
    resolution="03s",
    region=dem_region,
    registration="gridline",
    ) #### This needs removing from paralellel wasted 
    global resampled_terrain, gradiant_terrain
    resampled_terrain = out_dir+'/'+'terrain.grd'
    gradiant_terrain = out_dir+'/'+'gradiant.grd'
    resamp = pygmt.grdsample(terrain,region=region,outgrid=resampled_terrain,spacing='03s',registration='gridline')
    pygmt.grdgradient(grid=resampled_terrain,azimuth=-35,outgrid=gradiant_terrain)


    for ii in range(n_ifg2):
        # try:
            forward_modelling_2D_plot(resampled_terrain,gradiant_terrain,ii)
            # except:
            #     print(ii)
    # except:
    #     pass 
    



def forward_modelling_2D_plot(resampled_terrain,gradiant_terrain,ifgix):

    ifgd = ifgdates2[ifgix]

    if np.mod(ifgix,100) == 0:
        print("  {0:3}/{1:3}th unw...".format(ifgix, len(ifgdates2)), flush=True)

    
    out_dir1 = os.path.join(out_dir, ifgd)
    if not os.path.exists(out_dir1): 
        os.mkdir(out_dir1)
       

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


    out_dir1 = os.path.join(out_dir, ifgd)
    print('######################################## 2D locations plot ##############################################################')
    x1 = round((np.min(lons) - (np.min(lons) % 0.03) - 2*0.03),2)
    x2 = round((np.max(lons) - (np.max(lons) % 0.03) + 2*0.03),2)
    y1 = round((np.min(lats) - (np.min(lats) % 0.03) - 2*0.03),2)
    y2 = round((np.max(lats) - (np.max(lats) % 0.03) + 2*0.03),2)
    print("y1")
    print(y2%0.03)

    region = [np.min(lons),np.max(lons),np.min(lats),np.max(lats)] 
    dem_region = [x1,x2,y1,y2]
    dem_region = [np.min(lons)-0.1,np.max(lons)+0.1,np.min(lats)-0.1,np.max(lats)+0.1] 

    date_dir = in_dir + "/" + ifgd
    file_name = ifgd + '.unw'
    unw_file = os.path.join(date_dir,file_name) 
    unw = np.fromfile(unw_file, dtype='float32').reshape((length_ifgm, width_ifgm)) 
    if opt_model[9]:
        const_offset = opt_model[9]
    else:
        const_offset = 0
    file_path_data = os.path.join(out_dir1,'data_meters.grd')

    unw = -unw*(0.0555/(4*np.pi)) - const_offset
    pygmt.xyz2grd(x=lons.flatten(),y=lats.flatten(),z=unw.flatten(),outgrid=file_path_data,region=region,spacing=(0.001,0.001))

   
    print('surface project')
    # pygmt.xyz2grd(x=lons.flatten(),y=lats.flatten(),z=unw.flatten(),outgrid=file_path_data,region=region,spacing=(0.001,0.001))
    lons_vertex = vertex[0][:]
    lats_vertex = vertex[1][:]
    depth_vertex = vertex[2][:]
    surface_projection = pygmt.project(center=[lons_vertex[4],lats_vertex[4]],endpoint=[lons_vertex[5],lats_vertex[5]], generate=0.01)
 
    max_data = np.nanmax(unw[~np.isnan(unw)].flatten()) 
    print(max_data)
    min_data = np.nanmin(unw[~np.isnan(unw)].flatten()) 
    data_series = str(min_data) + '/' + str(max_data*1.5) +'/' + str((max_data - min_data)/100)
    print("data cpt")
    print(data_series)
    
    # print("data cpt")
    # print(data_series)
    print('make color pallette')
 
    # cmap_output_data =  out_dir1 + '/' +'InSAR_CPT_data.cpt'
    topo_output_cpt = out_dir1 + '/topo.cpt'
    topo_cpt_series = '0/5000/100' 
    cmap_output_data = os.path.join(out_dir1,'InSAR_CPT_data.cpt')
    file_path_data = os.path.join(out_dir1,'data_meters.grd')


   

    # resampled_terrain = out_dir+'/'+'terrain.grd'
    # resamp = pygmt.grdsample(terrain,region=region,outgrid=resampled_terrain,spacing='10s')
 
    
    pygmt.makecpt(cmap='oleron',series=topo_cpt_series, continuous=True,output=topo_output_cpt,background=True) 
    
    # pygmt.makecpt(cmap='polar',series=data_series, continuous=True,output=cmap_output_data,background=True) 

    if np.abs(min_data) > max_data:
        range_limit = np.abs(min_data)
    else:
        range_limit = max_data    

    if range_limit > 2:
        range_limit = 1

    pygmt.makecpt(series=[-range_limit, range_limit], cmap="polar",output=cmap_output_data)

    fig = pygmt.Figure()
    pygmt.config(MAP_FRAME_TYPE="plain")
    pygmt.config(FORMAT_GEO_MAP="ddd.xx")
    pygmt.config(FORMAT_FLOAT_OUT='%.12lg') 
    
    print('add basemap, gridimage and coast to 2D figure')
  
    # fig.grdimage(resampled_terrain,cmap=topo_output_cpt,region=region,projection='M8c',shading=gradiant_terrain)
  
    
    print('Down sample InSAR grd')
    unw_grd_ds = pygmt.grdsample(grid=file_path_data,spacing='03s',registration='gridline',region=region)
    print('grdimage InSAR grid')
    fig.grdimage(grid=unw_grd_ds,cmap=cmap_output_data,region=region,projection='M8c',shading=gradiant_terrain)
    fig.basemap(frame=['a','+tUSGS Location Comparison'],region=region,projection='M8c',map_scale="jBL+w10k+o0.5c/0.5c+f+lkm")
    fig.coast(region=region, projection = 'M8c', water='lightblue')
    print('plotting fault plane')
    fig.plot(x=lons_vertex[:-2],
                        y=lats_vertex[:-2],
                        pen='1p,black',
                        fill='gray',
                        transparency=60,
                        region=region,
                        projection='M8c',
                        )
    print('plotting surface projection')
    fig.plot(x=lons_vertex[len(lons_vertex)-2:len(lons_vertex)],
                y=lats_vertex[len(lats_vertex)-2:len(lats_vertex)],
                pen='1p,black,.',
                no_clip=False,
                #    sizes=0.1 * (2**np.array(eqMagAll)),
                #    style="uc",
                fill="gray",
                projection="M8c",
                transparency=20,
                # frame=["xa", "yaf", "za", "wSnEZ"],
                region=region)
    print('plotting USGS location')
    fig.plot(x=locations[0],
             y=locations[1],
             pen='1p,black',
             style='a0.4c',
             fill='white')
    font = "2p,Helvetica-Bold"
    fig.text(x=locations[0], y=locations[1] + np.abs(0.001*locations[1]), text="USGS", font=font, region=region,projection='M8c',fill="white")
    fig.colorbar(frame=["x+lLOS Displacement(m)", "y+lm"], position="JMB",projection='M8c') # ,panel=[1,0]
    # fig_2D.savefig('2D_test.png')
    print('saving figure')
    fig.savefig(os.path.join(out_dir1,'2D_locations_plot.png'))
    print('figure saved')
    # fig_2D.close()
    # fig.show()


if __name__ == '__main__':
    # try:
        # import pygmt
        # reload(pygmt)
        model = [
        6578.2,
        6498.4,
        6177.6,
        -58.315,
        411.55,
        -5267.1,
            10216,
        -0.47416,
        -0.28685,
                0]
        geoc_ml_path = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us6000abnv_insar_processing/GEOC_043A_05008_161514_floatml_masked_GACOS_Corrected_clipped'
        output_geoc = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/test'
        location = [40.7073,39.4229]
        vertex_path = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us6000abnv_NP1/invert_16_21_26_32_37_46_F/optmodel_vertex.mat'
        output_location_comp(geoc_ml_path,output_geoc,model,location,vertex_path)
        print('MADE IT PAST O')
    # except:
    #     print('What the hell')