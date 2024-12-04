#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import LiCSBAS_io_lib as LiCS_lib
import os 
import LiCSBAS_tools_lib as LiCS_tools
import multiprocessing as multi
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap as LSC
from matplotlib import pyplot as plt
from cmcrameri import cm
import sys
import local2llh as l2llh
from scipy import stats, interpolate, io
import logging
import coseis_lib as cl
import numpy as np
import pygmt as pygmt
import LiCSBAS_tools_lib as LiCS_tools
import LiCSBAS_io_lib as LiCS_lib
import pylab as plt
import llh2local as llh 
from cmcrameri import cm
import shutil 
import glob
import pygmt
from importlib import reload
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


def produce_final_GBISoutput(geoc_ml_path,output_geoc_ml_path, model, vertex_path, loc ,seis_model, USGS_loc ,argv=None):
    import pygmt
    reload(pygmt)
    
        
    try:
        n_para = len(os.sched_getaffinity(0))
    except:
        n_para = multi.cpu_count()

    if argv == None:
            argv = sys.argv
    

    global ifgdates2,in_dir, out_dir, pixsp_a, pixsp_r, opt_model, vertex, xy, lats, lons, locations, usgs_model, resampled_terrain, gradiant, terrain , region, usgs_loc
    opt_model = model
    usgs_model = seis_model
    usgs_loc = USGS_loc

    q = multi.get_context('fork')

    ifgdates = LiCS_tools.get_ifgdates(geoc_ml_path)
    n_ifg = len(ifgdates)

    EQA_dem_par = os.path.join(geoc_ml_path,"EQA.dem_par")
    width_ifgm = int(LiCS_lib.get_param_par(EQA_dem_par, 'width'))
    length_ifgm = int(LiCS_lib.get_param_par(EQA_dem_par, 'nlines'))
    dlat = float(LiCS_lib.get_param_par(EQA_dem_par, 'post_lat')) #negative
    dlon = float(LiCS_lib.get_param_par(EQA_dem_par, 'post_lon')) #positive
    lat1 = float(LiCS_lib.get_param_par(EQA_dem_par, 'corner_lat'))
    lon1 = float(LiCS_lib.get_param_par(EQA_dem_par, 'corner_lon'))

    vertex_data = read_mat(vertex_path)
    vertex = np.asarray(vertex_data['vertex_total_list'])
    locations = loc
    in_dir = geoc_ml_path
    out_dir = output_geoc_ml_path
    

    if not os.path.exists(out_dir):
            os.mkdir(out_dir)
       

    print('\nIn geographical coordinates', flush=True)

    centerlat = lat1+dlat*(length_ifgm/2)
    ra = float(LiCS_lib.get_param_par(EQA_dem_par, 'ellipsoid_ra'))
    recip_f = float(LiCS_lib.get_param_par(EQA_dem_par, 'ellipsoid_reciprocal_flattening'))
    rb = ra*(1-1/recip_f) ## polar radius
    pixsp_a = 2*np.pi*rb/360*abs(dlat)
    pixsp_r = 2*np.pi*ra/360*dlon*np.cos(np.deg2rad(centerlat))

    lat2 = lat1+dlat*(length_ifgm-1) # south # Remove
    lon2 = lon1+dlon*(width_ifgm-1) # east # Remove
    lons, lats = np.arange(lon1, lon2+dlon, dlon), np.arange(lat1, lat2+dlat, dlat)
    centerlat = lat1+dlat*(length_ifgm/2)
    ra = float(LiCS_lib.get_param_par(EQA_dem_par, 'ellipsoid_ra'))
    recip_f = float(LiCS_lib.get_param_par(EQA_dem_par, 'ellipsoid_reciprocal_flattening'))
    rb = ra*(1-1/recip_f) ## polar radius
    pixsp_a = 2*np.pi*rb/360*abs(dlat)
    pixsp_r = 2*np.pi*ra/360*dlon*np.cos(np.deg2rad(centerlat))
    lats_orig = np.array(lats,dtype=float)[:length_ifgm]
    lons_orig = np.array(lons,dtype=float)[:width_ifgm]

    lons, lats = np.meshgrid(lons_orig,lats_orig)
    ll = [lons.flatten(),lats.flatten()]
    ll = np.array(ll,dtype=float)
    xy = llh.llh2local(ll,np.array([location[0],location[1]],dtype=float))
    # xy_GBIS_reloc = llh.llh2local(ll,np.array([reloc[0],reloc[1]],dtype=float))

    #### pygmt ##### 
    x1 = round((np.min(lons) - (np.min(lons) % 0.03) - 2*0.03),2)
    x2 = round((np.max(lons) - (np.max(lons) % 0.03) + 2*0.03),2)
    y1 = round((np.min(lats) - (np.min(lats) % 0.03) - 2*0.03),2)
    y2 = round((np.max(lats) - (np.max(lats) % 0.03) + 2*0.03),2)
    print("y1")
    print(y2%0.03)

    region = [np.min(lons),np.max(lons),np.min(lats),np.max(lats)] 
    dem_region = [x1,x2,y1,y2]
    depth_vertex = vertex[2][:]
    dem_region = [np.min(lons)-0.1,np.max(lons)+0.1,np.min(lats)-0.1,np.max(lats)+0.1,(np.min(depth_vertex)-2000),3000] 
    
    # region.append(np.min(depth_vertex)-2000)
    # region.append(3000)
   

    terrain = pygmt.datasets.load_earth_relief(
    resolution="03s",
    region=dem_region,
    registration="gridline",
    ) #### This needs removing from paralellel wasted 
 

    resampled_terrain = out_dir+'/'+'terrain.grd'
    pygmt.grdsample(terrain,region=region,outgrid=resampled_terrain,spacing='10s')
    gradiant = pygmt.grdgradient(grid=resampled_terrain,azimuth=-35)

 
    ifgdates2 = ifgdates
    n_ifg2 = len(ifgdates2)


    ifgdates = LiCS_tools.get_ifgdates(in_dir)
    n_ifg = len(ifgdates)
    
    files = glob.glob(os.path.join(in_dir, '*'))
    for file in files:
        if not os.path.isdir(file): #not copy directory, only file
            print('Copy {}'.format(os.path.basename(file)), flush=True)
            shutil.copy(file, out_dir)

        #%% Mask unw
    print('\nMask unw and link cc', flush=True)
    ### First, check if already exist
    ifgdates2 = []
    for ifgix, ifgd in enumerate(ifgdates): 
        out_dir1 = os.path.join(out_dir, ifgd)
        unwfile_m = os.path.join(out_dir1, ifgd+'.unw')
        ccfile_m = os.path.join(out_dir1, ifgd+'.cc')
        if not (os.path.exists(unwfile_m) and os.path.exists(ccfile_m)):
            ifgdates2.append(ifgd)

        
    n_ifg2 = len(ifgdates2)


    if n_ifg-n_ifg2 > 0:
        print("  {0:3}/{1:3} masked unw and cc already exist. Skip".format(n_ifg-n_ifg2, n_ifg), flush=True)

    if n_ifg2 > 0:
        ### Mask with parallel processing
        if n_para > n_ifg2:
            n_para = n_ifg2
         
        print('  {} parallel processing...'.format(n_para), flush=True)
        p = q.Pool(n_para)
        p.map(GBIS_GMT_OUTPUT_FORWARD_MODEL, range(n_ifg2))
        p.close()

        print("", flush=True)
        for ii in range(n_ifg2):
            twoD_locations_plot(ii)
            six_panel_plot(ii)
      
    import pygmt
    reload(pygmt)
 
       
#%% Copy other files
    files = glob.glob(os.path.join(in_dir, '*'))
    for file in files:
        if not os.path.isdir(file): #not copy directory, only file
            print('Copy {}'.format(os.path.basename(file)), flush=True)
            shutil.copy(file, output_geoc_ml_path)

    print('\n{} Successfully finished!!\n'.format(os.path.basename(argv[0])))
    print('Output directory: {}\n'.format(os.path.relpath(out_dir)))



    print("3D Plots ")
    lons_vertex = vertex[0][:]
    lats_vertex = vertex[1][:]
    depth_vertex = vertex[2][:]
    surface_projection = pygmt.project(center=[lons_vertex[4],lats_vertex[4]],endpoint=[lons_vertex[5],lats_vertex[5]], generate=0.01)
    region.append(np.min(depth_vertex)-2000)
    region.append(3000)

    new_fig = pygmt.Figure()
    print(region)
    print('finding surface projection of tracks')
    down_terrain = out_dir+'/'+'down_terrain.grd'
    pygmt.grdsample(terrain,region=region,outgrid=down_terrain,spacing='50s')
    pygmt.grdtrack(
            grid = down_terrain, 
            points = surface_projection,
            outfile=os.path.join(out_dir,"surface_projection_of_fault.txt"),
            skiprows=False,
            newcolname='surface_proj')
    x_project = []
    y_project = []
    z_project = [] 

    with open(os.path.join(out_dir,"surface_projection_of_fault.txt"), 'r') as f:
        lines = f.readlines()
    f.close()
    for line in lines:
        # print(line.split(' '))
        x_project.append(float(line.split()[0]))
        y_project.append(float(line.split()[1]))
        z_project.append(float(line.split()[3]))
    opt_model[4] = opt_model[4] + 180 
    if opt_model[4] > 360: 
        opt_model[4] = opt_model[4] - 360
    total_depth = np.abs(region[-2]-region[len(region)-1])
    print(total_depth)
    z_scaling = str(round(6/total_depth,8)) +'c'
    # z_scaling = str(round(6/np.abs(region[-2] - region[len(region)-1]),5)) +'c'
    print('Calculating z_scaling')
    print(z_scaling)
    perspective=[opt_model[4]+20,20]
    print('perspective = ' + str(perspective))
    print('plotting basemap')
    new_fig.basemap(region=region, projection="M8c",map_scale="jBL+w10k+o0.5c/0.5c+f+lkm")
    print('plotting 3D topo')
    new_fig.grdview(
            grid=down_terrain,
            perspective=perspective,
            projection="M8c",
            zscale=z_scaling,
            # Set the surftype to "surface"
            surftype="s",
            shading="+a45",
            # Set the CPT to "geo"
            cmap="geo",
            region=region,
            frame=["xa", "yaf", "za+lDepth(m)", "wSnEZ"],
            transparency=20
            )
    print('plotting fault plane on 2D projection on bottom')
    new_fig.plot(x=lons_vertex[:-2],
                        y=lats_vertex[:-2],
                        pen='1p,red,-',
                        fill='white',
                        transparency=20,
                        region=region[:-2],
                        projection='M8c',
                        perspective=perspective)
    print('plotting fault plane at depth')
    new_fig.plot3d(x=lons_vertex[:-2],
                    y=lats_vertex[:-2],
                    z=depth_vertex[:-2],
                    pen='2p,red',
                    no_clip=False,
                    #    sizes=0.1 * (2**np.array(eqMagAll)),
                    #    style="uc",
                    fill="gray",
                    zscale=z_scaling,
                    perspective=perspective,
                    projection="M8c",
                    transparency=20,
                    #    frame=["xa", "yaf", "za", "wSnEZ"],
                    region=region)
    print('plotting surface projection of Fault')
    new_fig.plot3d(x=x_project,
                    y=y_project,
                    z=z_project,
                    pen='2p,red',
                    no_clip=False,
                    style="c",
                    size=(np.zeros(np.shape(x_project))+0.075),
                    #    sizes=0.1 * (2**np.array(eqMagAll)),
                    #    style="uc",
                    fill="gray",
                    zscale=z_scaling,
                    perspective=perspective,    
                    projection="M8c",
                    transparency=20,
                    region=region)

    # new_fig.show()
    # new_fig.savefig('3D_test.png')
    print('saving figure')
    # try:
    new_fig.savefig(os.path.join(out_dir,'3D_fault_plane.png'))
    import pygmt
    reload(pygmt)
    # except:
    #     print('need this because throws error after working for some reason')
    # new_fig.close()
    


def GBIS_GMT_OUTPUT_FORWARD_MODEL(ifgix):
    import pygmt
    reload(pygmt)
    ifgd = ifgdates2[ifgix]

    if np.mod(ifgix,100) == 0:
        print("  {0:3}/{1:3}th unw...".format(ifgix, len(ifgdates2)), flush=True)

   
    EQA_dem_par = os.path.join(in_dir,"EQA.dem_par")
    width_ifgm = int(LiCS_lib.get_param_par(EQA_dem_par, 'width'))
    length_ifgm = int(LiCS_lib.get_param_par(EQA_dem_par, 'nlines'))
    dlat = float(LiCS_lib.get_param_par(EQA_dem_par, 'post_lat')) #negative
    dlon = float(LiCS_lib.get_param_par(EQA_dem_par, 'post_lon')) #positive
    lat1 = float(LiCS_lib.get_param_par(EQA_dem_par, 'corner_lat'))
    lon1 = float(LiCS_lib.get_param_par(EQA_dem_par, 'corner_lon'))


    out_dir1 = os.path.join(out_dir, ifgd)
    if not os.path.exists(out_dir1): os.mkdir(out_dir1)
    unwfile = os.path.join(in_dir, ifgd, ifgd+'.unw')
    dtype = np.float32
    unw = LiCS_lib.read_img(unwfile, length_ifgm, width_ifgm, dtype=dtype)
    pngfile_unw = os.path.join(out_dir1, ifgd+'.unw.png')
    unw.tofile(os.path.join(out_dir1, ifgd+'.unw'))

    if not os.path.exists(os.path.join(out_dir1, ifgd+'.cc')):
        ccfile = os.path.join(in_dir, ifgd, ifgd+'.cc')
        os.symlink(os.path.relpath(ccfile, out_dir1), os.path.join(out_dir1, ifgd+'.cc'))

    out_dir1 = os.path.join(out_dir, ifgd)
    unwfile = os.path.join(in_dir, ifgd, ifgd+'.unw')
    dtype = np.float32
    unw = LiCS_lib.read_img(unwfile, length_ifgm, width_ifgm, dtype=dtype)

    thetafile = os.path.join(in_dir,"theta.geo")
    phifile = os.path.join(in_dir,"phi.geo")

    Inc = LiCS_lib.read_img(phifile, length_ifgm, width_ifgm)
    Head = LiCS_lib.read_img(thetafile, length_ifgm, width_ifgm)
    unw[unw==0] = np.nan
    

    length = opt_model[0]
    width = opt_model[1]
    depth = opt_model[2]
    dip = -opt_model[3]
    depth =  depth + ((width/2)*np.sin(np.abs(dip)*(np.pi/180)))
    strike = opt_model[4] - 180 
    # strike = opt_model[4]
    X = opt_model[5]
    Y = opt_model[6]
    SS = opt_model[7]
    DS = opt_model[8]


    print(opt_model)
    rake = -np.degrees(np.arctan2(DS/SS))
    total_slip = np.sqrt(SS**2 + DS**2)
    print(total_slip)
    if strike < 0: 
        strike = strike + 360 

    xx_vec = xy[0,:].reshape((length_ifgm, width_ifgm))
    yy_vec = xy[1,:].reshape((length_ifgm, width_ifgm))
   

    xx_original = xx_vec[0,:]
    yy_original = yy_vec[:,0]
    xx_original = xx_original[:width_ifgm]
    yy_original = yy_original[:length_ifgm]

    xx_vec = xy[0,:].flatten() 
    yy_vec = xy[1,:].flatten()

    # x_usgs,y_usgs = LiCS_tools.bl2xy(locations[1],locations[0],width_ifgm,length_ifgm,lat1,dlat,lon1,dlon)
    # xcent = x_usgs * pixsp_a
    # ycent = y_usgs * pixsp_r
    # end1x, end2x, end1y, end2y, c1x, c2x, c3x, c4x, c1y, c2y, c3y, c4y = cl.fault_for_plotting(usgs_model)
    # x_usgs = np.array([c1x,c2x,c3x,c4x])/ 1000 #local2llh needs results in km 
    # y_usgs = np.array([c1y,c2y,c3y,c4y])/ 1000 #local2llh needs results in km 
    # x_usgs_end = np.array([end1x,end2x]) / 1000 
    # y_usgs_end = np.array([end1y,end2y]) / 1000

    # print(x_usgs)
    # print(y_usgs)
    # local_usgs_source = [x_usgs,y_usgs]  
    # local_end_usgs_source = [x_usgs_end,y_usgs_end]
    # print(np.shape(local_usgs_source))
    # print("LOCATIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~ = " + str(locations))
    # llh_usgs = l2llh.local2llh(local_usgs_source,[locations[0],locations[1]])
    # fault_end = l2llh.local2llh(local_end_usgs_source,[locations[0],locations[1]])


    model = [X, Y, strike, dip, rake, total_slip, length, depth, width]
    # Calcualte displacements
    disp = cl.disloc3d3(xx_vec, yy_vec, xoff=X, yoff=Y, depth=depth,
                    length=length, width=width, slip=total_slip, opening=0, 
                    strike=strike, dip=dip, rake=rake, nu=0.25)
    print(depth)
    
    # disp_usgs =cl.disloc3d3(xx_vec,yy_vec,xoff=0,yoff=0,depth=usgs_model[7],
    #                         length=usgs_model[6],width=usgs_model[8],slip=usgs_model[5],
    #                         opening=0,strike=usgs_model[2],dip=usgs_model[3],rake=usgs_model[4],nu=0.25)
    np.shape(disp)
    disp = -disp
    disp_usgs = -disp_usgs
    incidence_angle = np.nanmean(Inc)
    azimuth_angle = np.nanmean(Head)
    print('AZIMUTH AND INCIDENCE AVERAGE')
    print(azimuth_angle)
    print(incidence_angle)
    #####################################

    # Convert to unit vector components
    unw = -unw*(0.0555/(4*np.pi))
    e2los = np.cos(np.deg2rad(azimuth_angle)) * np.sin(np.deg2rad(incidence_angle))
    n2los = -np.sin(np.deg2rad(azimuth_angle)) * np.sin(np.deg2rad(incidence_angle))
    u2los = -np.cos(np.deg2rad(incidence_angle))
    los_grid = (disp[0,:] * e2los) + (disp[1,:] * n2los) + (disp[2,:] * u2los) 
    resid = unw.flatten() - los_grid
    region = [np.min(lons),np.max(lons),np.min(lats),np.max(lats)] # GMT region  [xmin,xmax,ymin,ymax].

    e2los = np.cos(np.deg2rad(azimuth_angle)) * np.sin(np.deg2rad(incidence_angle))
    n2los = -np.sin(np.deg2rad(azimuth_angle)) * np.sin(np.deg2rad(incidence_angle))
    u2los = -np.cos(np.deg2rad(incidence_angle))
    los_grid_usgs = (disp_usgs[0,:] * e2los) + (disp_usgs[1,:] * n2los) + (disp_usgs[2,:] * u2los) 
    resid_usgs = unw.flatten() - los_grid_usgs
    region = [np.min(lons),np.max(lons),np.min(lats),np.max(lats)] # GMT region  [xmin,xmax,ymin,ymax].


    # file_path_data = 'A_data.grd'
    # file_path_res = 'A_res.grd'
    # file_path_model = 'A_mod.grd'

    file_path_data = os.path.join(out_dir1,'data_meters.grd')
    file_path_res = os.path.join(out_dir1,'residual_model_meters.grd')
    file_path_model = os.path.join(out_dir1,'model_model_meters.grd')

    # file_path_res_usgs = os.path.join(out_dir1,'residual_usgs_model_meters.grd')
    # file_path_model_usgs = os.path.join(out_dir1,'model_usgs_meters.grd')


    pygmt.xyz2grd(x=lons.flatten(),y=lats.flatten(),z=unw.flatten(),outgrid=file_path_data,region=region,spacing=(0.001,0.001))
    pygmt.xyz2grd(x=lons.flatten(),y=lats.flatten(),z=resid,outgrid=file_path_res,region=region,spacing=(0.001,0.001))
    pygmt.xyz2grd(x=lons.flatten(),y=lats.flatten(),z=los_grid,outgrid=file_path_model,region=region,spacing=(0.001,0.001))

    # pygmt.xyz2grd(x=lons.flatten(),y=lats.flatten(),z=resid_usgs,outgrid=file_path_res_usgs,region=region,spacing=(0.001,0.001))
    # pygmt.xyz2grd(x=lons.flatten(),y=lats.flatten(),z=los_grid_usgs,outgrid=file_path_model_usgs,region=region,spacing=(0.001,0.001))



    lons_vertex = vertex[0][:]
    lats_vertex = vertex[1][:]
    depth_vertex = vertex[2][:]
    print(lons_vertex)
    max_los = np.max(los_grid) 
    min_los = np.min(los_grid) 
    model_series = str(min_los) + '/' + str(max_los*1.25) +'/' + str((max_los - min_los)/50)

    max_res = np.max(resid[~np.isnan(resid)]) 
    min_res = np.min(resid[~np.isnan(resid)]) 
    res_series = str(min_res) + '/' + str(max_res*1.25) +'/' + str((max_res - min_res)/50)

    max_data = np.max(unw[~np.isnan(unw)].flatten()) 
    print(max_data)
    min_data = np.min(unw[~np.isnan(unw)].flatten()) 
    data_series = str(min_data) + '/' + str(max_data*1.25) +'/' + str((max_data - min_data)/50)
    print(data_series)

    # max_los_usgs = np.max(los_grid_usgs) 
    # min_los_usgs = np.min(los_grid_usgs) 
    # model_series_usgs = str(min_los_usgs) + '/' + str(max_los_usgs*1.25) +'/' + str((max_los_usgs - min_los_usgs)/50)

    # max_res_usgs = np.max(resid_usgs[~np.isnan(resid_usgs)]) 
    # min_res_usgs = np.min(resid_usgs[~np.isnan(resid_usgs)]) 
    # res_series_usgs = str(min_res_usgs) + '/' + str(max_res_usgs*1.25) +'/' + str((max_res_usgs - min_res_usgs)/50)

    # print(file_path_data.split('/')[-1])   
    cmap_output_los = out_dir1 +'/' + 'InSAR_CPT_los.cpt'
    pygmt.makecpt(cmap='polar',series=model_series, continuous=True,output=cmap_output_los) 
    cmap_output_data = out_dir1 + '/' + 'InSAR_CPT_data.cpt'
    pygmt.makecpt(cmap='polar',series=data_series, continuous=True,output=cmap_output_data) 
    cmap_output_res = out_dir1 + '/' + 'InSAR_CPT_res.cpt'
    pygmt.makecpt(cmap='polar',series=res_series, continuous=True,output=cmap_output_res) 

    # cmap_output_los_usgs = out_dir1 + '/' + 'InSAR_CPT_los_usgs.cpt'
    # pygmt.makecpt(cmap='polar',series=model_series_usgs, continuous=True,output=cmap_output_los_usgs) 
    # cmap_output_res_usgs = out_dir1 + '/' + 'InSAR_CPT_res_usgs.cpt'
    # pygmt.makecpt(cmap='polar',series=res_series_usgs, continuous=True,output=cmap_output_res_usgs) 



def six_panel_plot(ifgix):
    import pygmt
    reload(pygmt)
    ifgd = ifgdates2[ifgix]
    out_dir1 = os.path.join(out_dir, ifgd)
    in_dir1 = os.path.join(in_dir, ifgd)

    EQA_dem_par = os.path.join(in_dir,"EQA.dem_par")
    width_ifgm = int(LiCS_lib.get_param_par(EQA_dem_par, 'width'))
    length_ifgm = int(LiCS_lib.get_param_par(EQA_dem_par, 'nlines'))
    dlat = float(LiCS_lib.get_param_par(EQA_dem_par, 'post_lat')) #negative
    dlon = float(LiCS_lib.get_param_par(EQA_dem_par, 'post_lon')) #positive
    lat1 = float(LiCS_lib.get_param_par(EQA_dem_par, 'corner_lat'))
    lon1 = float(LiCS_lib.get_param_par(EQA_dem_par, 'corner_lon'))

    file_path_data = os.path.join(out_dir1,'data_meters.grd')
    cmap_output_data = os.path.join(out_dir1,'InSAR_CPT_data.cpt')
    
    file_path_model = os.path.join(out_dir1,'model_model_meters.grd')
    file_path_res = os.path.join(out_dir1,'residual_model_meters.grd')
    cmap_output_los = os.path.join(out_dir1,'InSAR_CPT_los.cpt')
    cmap_output_res = os.path.join(out_dir1,'InSAR_CPT_res.cpt')

    # file_path_model_usgs = os.path.join(out_dir1,'model_usgs_meters.grd')
    # file_path_res_usgs = os.path.join(out_dir1,'residual_usgs_model_meters.grd')
    # cmap_output_los_usgs = os.path.join(out_dir1,'InSAR_CPT_los_usgs.cpt')
    # cmap_output_res_usgs = os.path.join(out_dir1,'InSAR_CPT_res_usgs.cpt')

    lons_vertex = vertex[0][:]
    lats_vertex = vertex[1][:]
    depth_vertex = vertex[2][:]


    # x_usgs,y_usgs = LiCS_tools.bl2xy(locations[1],locations[0],width_ifgm,length_ifgm,lat1,dlat,lon1,dlon)
    # xcent = x_usgs * pixsp_a
    # ycent = y_usgs * pixsp_r
    # end1x, end2x, end1y, end2y, c1x, c2x, c3x, c4x, c1y, c2y, c3y, c4y = cl.fault_for_plotting(usgs_model)
    # x_usgs = np.array([c1x,c2x,c3x,c4x])/ 1000 #local2llh needs results in km 
    # y_usgs = np.array([c1y,c2y,c3y,c4y])/ 1000 #local2llh needs results in km 
    # x_usgs_end = np.array([end1x,end2x]) / 1000 
    # y_usgs_end = np.array([end1y,end2y]) / 1000

    # print(x_usgs)
    # print(y_usgs)
    # local_usgs_source = [x_usgs,y_usgs]  
    # local_end_usgs_source = [x_usgs_end,y_usgs_end]
    # print(np.shape(local_usgs_source))
    # print("LOCATIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~ = " + str(locations))
    # llh_usgs = l2llh.local2llh(local_usgs_source,[reloc[0],reloc[1]])
    # fault_end = l2llh.local2llh(local_end_usgs_source,[reloc[0],reloc[1]])


    fig = pygmt.Figure()
    pygmt.config(MAP_FRAME_TYPE="plain")
    pygmt.config(FORMAT_GEO_MAP="ddd.xx")
    pygmt.config(FORMAT_FLOAT_OUT='%.12lg') 

    with fig.subplot(
        nrows=1,
        ncols=3,
        figsize=("45c", "45c","45c"),
        autolabel=False,
        frame=["f","WSne","+tData"],
        margins=["0.05c", "0.05c"],
        # title="Geodetic Moderling Sequence one 2023/09/25 - 2023/10/08",
    ):

        # Configuration for the 'current figure'.
    
        # ,rose=["JBL+w5c+f2+l"]
        print('first grdimages')
        fig.grdimage(grid=file_path_data,cmap=cmap_output_data,region=region,projection='M?c',panel=[0,0])
        print('first data')
        fig.basemap(frame=['a','+tData'],panel=[0,0],region=region,projection='M?c',)
        fig.grdimage(grid=file_path_model,cmap=cmap_output_los,region=region,projection='M?c',panel=[0,1])
        print('first model')
        fig.basemap(frame=['xa','+tModel'],panel=[0,1],region=region,projection='M?c')
        fig.grdimage(grid=file_path_res,cmap=cmap_output_res,region=region,projection='M?c',panel=[0,2])
        print('first res')
        fig.basemap(frame=['xa','+tResidual'],panel=[0,2],region=region,projection='M?c')
        print('completed first grdimages')
        for ii in range(0,3):
            fig.plot(x=lons_vertex[:-2],
                        y=lats_vertex[:-2],
                        pen='2p,black',
                        fill='gray',
                        transparency=80,
                        region=region,
                        projection='M10c',
                        panel=[0,ii]
            )

            fig.plot(x=lons_vertex[len(lons_vertex)-2:len(lons_vertex)],
                        y=lats_vertex[len(lats_vertex)-2:len(lats_vertex)],
                        pen='2p,black,-',
                        fill='gray',
                        transparency=80,
                        region=region,
                        projection='M10c',
                        panel=[0,ii]
            )
        print('first grdimages after loop')
        for ii in range(0,3):
                fig.colorbar(frame=["x+lLOS Displacement(m)", "y+lm"], position="JMB",projection='M10c',panel=[0,ii]) 
                
        # print('second grdimages')  
        # fig.grdimage(grid=file_path_data,cmap=cmap_output_data,region=region,projection='M?c',panel=[1,0])
        # # # fig.basemap(frame=['a','+tData'],panel=[1,0],region=region,projection='M10c')
        # fig.grdimage(grid=file_path_model_usgs,cmap=cmap_output_los_usgs,region=region,projection='M?c',panel=[1,1])
        # # # fig.basemap(frame=['xa','+tModel'],panel=[1,1],region=region,projection='M10c')
        # fig.grdimage(grid=file_path_res_usgs,cmap=cmap_output_res_usgs,region=region,projection='M?c',panel=[1,2])
        # # # fig.basemap(frame=['xa','+tResidual'],panel=[1,2],region=region,projection='M10c')
        # print('second grdimages after grd image')  
        # for ii in range(0,3):
        #     fig.plot(x=llh_usgs[0,:],
        #                 y=llh_usgs[1,:],
        #                 pen='2p,black',
        #                 fill='gray',
        #                 transparency=80,
        #                 region=region,
        #                 projection='M10c',
        #                 panel=[1,ii]
        #     )

        #     fig.plot(x=fault_end[0,:],
        #                 y=fault_end[1,:],
        #                 pen='2p,black,-',
        #                 fill='gray',
        #                 transparency=80,
        #                 region=region,
        #                 projection='M10c',
        #                 panel=[1,ii]
        #     )
        # for ii in range(0,3):
        #         fig.colorbar(frame=["x+lLOS displacment(m)", "y+lm"], position="JMB",projection='M10c',panel=[1,ii])# ,panel=[1,0]
        # # fig.show(method='external')
        fig.savefig(os.path.join(out_dir1,'output.png'))
    return 

def twoD_locations_plot(ifgix):
    import pygmt
    reload(pygmt)

    ifgd = ifgdates2[ifgix]

    if np.mod(ifgix,100) == 0:
        print("  {0:3}/{1:3}th unw...".format(ifgix, len(ifgdates2)), flush=True)


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
   
    print('surface project')
    # pygmt.xyz2grd(x=lons.flatten(),y=lats.flatten(),z=unw.flatten(),outgrid=file_path_data,region=region,spacing=(0.001,0.001))
    lons_vertex = vertex[0][:]
    lats_vertex = vertex[1][:]
    depth_vertex = vertex[2][:]
    surface_projection = pygmt.project(center=[lons_vertex[4],lats_vertex[4]],endpoint=[lons_vertex[5],lats_vertex[5]], generate=0.01)
 
    
    # max_data = np.max(unw[~np.isnan(unw)].flatten()) 
    # # print(max_data)
    # min_data = np.min(unw[~np.isnan(unw)].flatten()) 
    # data_series = str(min_data) + '/' + str(max_data*1.5) +'/' + str((max_data - min_data)/100)
    # # print("data cpt")
    # # print(data_series)
    print('make color pallette')
    fig_2D = pygmt.Figure()
    pygmt.config(MAP_FRAME_TYPE="plain")
    pygmt.config(FORMAT_GEO_MAP="ddd.xx")
    pygmt.config(FORMAT_FLOAT_OUT='%.12lg') 
    # cmap_output_data =  out_dir1 + '/' +'InSAR_CPT_data.cpt'
    topo_output_cpt = out_dir1 + '/topo.cpt'
    topo_cpt_series = '0/5000/100' 
    cmap_output_data = os.path.join(out_dir1,'InSAR_CPT_data.cpt')
    file_path_data = os.path.join(out_dir1,'data_meters.grd')
    
    pygmt.makecpt(cmap='oleron',series=topo_cpt_series, continuous=True,output=topo_output_cpt) 
    # pygmt.makecpt(cmap='polar',series=data_series, continuous=True,output=cmap_output_data) 

    
    print('add basemap, gridimage and coast to 2D figure')
  
    fig_2D.grdimage(terrain,cmap=topo_output_cpt,region=region,projection='M8c',shading='+a-35')
  
   
    print('Down sample InSAR grd')
    unw_grd_ds = pygmt.grdsample(grid=file_path_data,spacing='10s',registration='gridline',region=region)
    print('grdimage InSAR grid')
    fig_2D.grdimage(grid=unw_grd_ds,cmap=cmap_output_data,region=region,projection='M8c',shading=gradiant,transparency=80,nan_transparent=True)
    fig_2D.basemap(frame=['a','+tUSGS Location Comparison'],region=region,projection='M8c',map_scale="jBL+w10k+o0.5c/0.5c+f+lkm")
    fig_2D.coast(region=region, projection = 'M8c', water='lightblue')
    print('plotting fault plane')
    fig_2D.plot(x=lons_vertex[:-2],
                        y=lats_vertex[:-2],
                        pen='1p,black',
                        fill='gray',
                        transparency=20,
                        region=region,
                        projection='M8c',
                        )
    print('plotting surface projection')
    fig_2D.plot(x=lons_vertex[len(lons_vertex)-2:len(lons_vertex)],
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
    fig_2D.plot(x=locations[0],
             y=locations[1],
             pen='1p,red',
             style='a0.4c',
             fill='darkorange')
    font = "2p,Helvetica-Bold"
    fig_2D.text(x=locations[0], y=locations[1] + np.abs(0.001*locations[1]), text="USGS", font=font, region=region,projection='M8c',fill="white")
    # fig_2D.savefig('2D_test.png')
    print('saving figure')
    fig_2D.savefig(os.path.join(out_dir1,'2D_locations_plot.png'))
    # fig.show()

  
    return 
  




if __name__ == '__main__':
    import pygmt
    reload(pygmt)
    geoc_ml_path = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us6000jk0t_insar_processing/GEOC_072A_05090_131313_floatml_masked_GACOS_Corrected_clipped'
    output_geoc_ml_path = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us6000jk0t_GBIS_area/GEOC_072A_05090_131313__INVERSION_Results_NP1'
    event_file_path = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us6000jk0t_insar_processing/us6000jk0t.txt'
    with open(event_file_path,'r') as file:
        params = file.readlines()
    file.close()
    name = params[0].split('=')[-1]
    time = params[1].split('=')[-1]
    latitude = float(params[2].split('=')[-1])
    longitude = float(params[3].split('=')[-1])
    magnitude = float(params[4].split('=')[-1])
    magnitude_type = params[5].split('=')[-1]
    moment = float(params[6].split('=')[-1])
    depth = float(params[7].split('=')[-1])
    catalog = params[8].split('=')[-1]
    strike1 = float(params[9].split('=')[-1])
    dip1 = float(params[10].split('=')[-1])
    rake1 = float(params[11].split('=')[-1])
    strike2 = float(params[12].split('=')[-1])
    dip2 = float(params[13].split('=')[-1])
    rake2 = float(params[14].split('=')[-1])
    slip_rate=5.5e-5
    slip = 8000 * slip_rate
    opt_model = [  8.21570490e+03,   8.96776468e+03,   2.58874315e+04,  -8.80402084e+01,
    1.67827131e+02,   1.97834972e+03,   1.77592369e+04,  -1.69630229e+00,
    1.13374081e+00,   0.00000000e+00]
    vertex_path = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us6000jk0t_NP1/invert_1_2_3_4_5_F/optmodel_vertex.mat'
    usgs_loc = [latitude,longitude]
    grids = ['/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us6000jk0t_GBIS_area/GEOC_072A_05090_131313__INVERSION_Results_NP1/20221203_20230201/',
             '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us6000jk0t_GBIS_area/GEOC_072A_05090_131313__INVERSION_Results_NP1/20221215_20230201/',
             '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us6000jk0t_GBIS_area/GEOC_072A_05090_131313__INVERSION_Results_NP1/20221227_20230201/',
             '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us6000jk0t_GBIS_area/GEOC_072A_05090_131313__INVERSION_Results_NP1/20230108_20230201/',
             '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us6000jk0t_GBIS_area/GEOC_072A_05090_131313__INVERSION_Results_NP1/20230120_20230201/'
    ]

    # for ii in range(len(grids)):
    #     fig = pygmt.Figure()
    #     # fig.grdimage(grid=grids[ii]+'data_meters.grd',cmap=grids[ii]+'InSAR_CPT_data.cpt')
    #     # fig.show()
    #     # fig.grdimage(grid=grids[ii]+'model_model_meters.grd',cmap=grids[ii]+'InSAR_CPT_los.cpt')
    #     # fig.show()
    #     # print(pygmt.grdinfo(grid=grids[ii]+'model_model_meters.grd'))
    #     print(pygmt.grdinfo(grid=grids[ii]+'data_meters.grd'))
    #     # fig.grdimage(grid=grids[ii]+'residual_model_meters.grd',cmap=grids[ii]+'InSAR_CPT_res.cpt')
    #     # fig.show()

    # if NP ==1:
    usgs_model = [ 0,0,strike1,dip1,rake1,slip,8000,depth,8000]
    
    print(depth)    # elif NP == 2:
    #     usgs_model = [ 0,0,strike2,dip2,rake2,slip,self.estimate_length,depth,self.estimate_length]
  
    produce_final_GBISoutput( geoc_ml_path,output_geoc_ml_path, opt_model, vertex_path,usgs_loc, usgs_model)