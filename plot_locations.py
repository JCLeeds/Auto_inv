import pygmt 
import numpy as np 
from scipy import stats, interpolate, io
import logging
import coseis_lib as cl
import numpy as np
import pygmt as pygmt
import os 
import LiCSBAS_tools_lib as LiCS_tools
import LiCSBAS_io_lib as LiCS_lib
import pylab as plt
import llh2local as llh 
from cmcrameri import cm
import local2llh as l2llh
from pygmt.datasets import load_earth_relief
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

def plot_location_comparisons(vertex_path,
                                  opt_model,
                                  EQA_dem_par,
                                  unw_file,
                                  Inc_file,
                                  Head_file,
                                  locations,
                                  usgs_model):
    width_ifgm = int(LiCS_lib.get_param_par(EQA_dem_par, 'width'))
    length_ifgm = int(LiCS_lib.get_param_par(EQA_dem_par, 'nlines'))
    dlat = float(LiCS_lib.get_param_par(EQA_dem_par, 'post_lat')) #negative
    dlon = float(LiCS_lib.get_param_par(EQA_dem_par, 'post_lon')) #positive
    lat1 = float(LiCS_lib.get_param_par(EQA_dem_par, 'corner_lat'))
    lon1 = float(LiCS_lib.get_param_par(EQA_dem_par, 'corner_lon'))
    lat2 = lat1+dlat*(length_ifgm-1) # south # Remove
    lon2 = lon1+dlon*(width_ifgm-1) # east # Remove
    lons, lats = np.arange(lon1, lon2+dlon, dlon), np.arange(lat1, lat2+dlat, dlat)
    font = "4p,Helvetica-Bold"


    
    centerlat = lat1+dlat*(length_ifgm/2)
    ra = float(LiCS_lib.get_param_par(EQA_dem_par, 'ellipsoid_ra'))
    recip_f = float(LiCS_lib.get_param_par(EQA_dem_par, 'ellipsoid_reciprocal_flattening'))
    rb = ra*(1-1/recip_f) ## polar radius
    pixsp_a = 2*np.pi*rb/360*abs(dlat)
    pixsp_r = 2*np.pi*ra/360*dlon*np.cos(np.deg2rad(centerlat))
    lats_orig = np.array(lats,dtype=float)[:length_ifgm]
    lons_orig = np.array(lons,dtype=float)[:width_ifgm]
    unw_file = os.path.join(unw_file) 
    unw = np.fromfile(unw_file, dtype='float32').reshape((length_ifgm, width_ifgm)) 


    lats_orig = np.array(lats,dtype=float)[:length_ifgm]
    lons_orig = np.array(lons,dtype=float)[:width_ifgm]

    lons, lats = np.meshgrid(lons_orig,lats_orig)

 


    unw = -unw*(0.0555/(4*np.pi))
   

    x1 = round((np.min(lons) - (np.min(lons) % 0.03) - 2*0.03),2)
    x2 = round((np.max(lons) - (np.max(lons) % 0.03) + 2*0.03),2)
    y1 = round((np.min(lats) - (np.min(lats) % 0.03) - 2*0.03),2)
    y2 = round((np.max(lats) - (np.max(lats) % 0.03) + 2*0.03),2)
    print("y1")
    print(y2%0.03)

    region = [np.min(lons),np.max(lons),np.min(lats),np.max(lats)] 
    dem_region = [x1,x2,y1,y2]
    dem_region = [np.min(lons)-0.1,np.max(lons)+0.1,np.min(lats)-0.1,np.max(lats)+0.1] 
    file_path_data = 'A_los.grd'
  
    pygmt.xyz2grd(x=lons.flatten(),y=lats.flatten(),z=unw.flatten(),outgrid=file_path_data,region=region,spacing=(0.001,0.001))
    vertex_data = read_mat(vertex_path)
    vertex = np.asarray(vertex_data['vertex_total_list'])
    lons_vertex = vertex[0][:]
    lats_vertex = vertex[1][:]
    depth_vertex = vertex[2][:]
    surface_projection = pygmt.project(center=[lons_vertex[4],lats_vertex[4]],endpoint=[lons_vertex[5],lats_vertex[5]], generate=0.01)
 
    
    max_data = np.max(unw[~np.isnan(unw)].flatten()) 
    # print(max_data)
    min_data = np.min(unw[~np.isnan(unw)].flatten()) 
    data_series = str(min_data) + '/' + str(max_data*1.5) +'/' + str((max_data - min_data)/100)
    # print("data cpt")
    # print(data_series)

    fig = pygmt.Figure()
    pygmt.config(MAP_FRAME_TYPE="plain")
    pygmt.config(FORMAT_GEO_MAP="ddd.xx")
    pygmt.config(FORMAT_FLOAT_OUT='%.12lg') 
    cmap_output_data =  './InSAR_CPT_data.cpt'
    topo_output_cpt = './topo.cpt'
    topo_cpt_series = '0/5000/100' 
    pygmt.makecpt(cmap='oleron',series=topo_cpt_series, continuous=True,output=topo_output_cpt) 
    pygmt.makecpt(cmap='polar',series=data_series, continuous=True,output=cmap_output_data) 
    terrain = load_earth_relief(
    resolution="03s",
    region=dem_region,
    registration="gridline",
    )
 

    resampled_terrain = './terrain.grd'
    pygmt.grdsample(terrain,region=region,outgrid=resampled_terrain,spacing='10s')
  
    fig.basemap(frame=['a','+tUSGS Location Comparison'],region=region,projection='M8c',map_scale="jBL+w10k+o0.5c/0.5c+f+lkm")
    fig.grdimage(grid=resampled_terrain,cmap=topo_output_cpt,region=region,projection='M8c',shading='+a-35')
    fig.coast(region=region, projection = 'M8c', water='lightblue')
   
    gradiant = pygmt.grdgradient(grid=resampled_terrain,azimuth=-35)
    unw_grd_ds = pygmt.grdsample(grid=file_path_data,spacing='10s',registration='gridline',region=region)
 
    fig.grdimage(grid=unw_grd_ds,cmap=cmap_output_data,region=region,projection='M8c',shading=gradiant,transparency=80,nan_transparent=True)

    fig.plot(x=lons_vertex[:-2],
                        y=lats_vertex[:-2],
                        pen='1p,black',
                        fill='gray',
                        transparency=20,
                        region=region,
                        projection='M8c',
                        )
    
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
    
    fig.plot(x=locations[0],
             y=locations[1],
             pen='1p,red',
             style='a0.4c',
             fill='darkorange')
    fig.text(x=locations[0], y=locations[1] + np.abs(0.001*locations[1]), text="USGS", font=font, region=region,projection='M8c',fill="white")
    fig.savefig('2D_test.png')
    # fig.show()

############################################# 3D section ##############################################
    
    region.append(np.min(depth_vertex)-2000)
    region.append(3000)
    new_fig = pygmt.Figure()
    print(region)
    pygmt.grdtrack(
            grid = terrain, 
            points = surface_projection,
            outfile="tmp_NP1.txt",
            skiprows=False,
            newcolname='surface_proj')
    x_project = []
    y_project = []
    z_project = [] 

    with open("tmp_NP1.txt", 'r') as f:
        lines = f.readlines()
    for line in lines:
        # print(line.split(' '))
        x_project.append(float(line.split()[0]))
        y_project.append(float(line.split()[1]))
        z_project.append(float(line.split()[3]))
    opt_model[4] = opt_model[4] + 180 
    if opt_model[4] > 360: 
        opt_model[4] = opt_model[4] - 360

    z_scaling = str(round(6/np.abs(region[-2] - region[len(region)-1]),5)) +'c'
    print('z_scaling')
    print(z_scaling)
    perspective=[opt_model[4]+20,15]
    new_fig.basemap(region=region, projection="M8i",map_scale="jBL+w10k+o0.5c/0.5c+f+lkm")
    new_fig.grdview(
            grid=terrain,
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
    new_fig.plot(x=lons_vertex[:-2],
                        y=lats_vertex[:-2],
                        pen='1p,red,-',
                        fill='white',
                        transparency=20,
                        region=region[:-2],
                        projection='M8i',
                        perspective=perspective)

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
    new_fig.savefig('3D_test.png')
    return 


if __name__ == "__main__":

    vertex_path = '/Users/jcondon/phd/code/auto_inv/us6000jk0t/invert_1_2_3_4_F/optmodel_vertex.mat'
    # GBIS_GMT_OUTPUT('/Users/jcondon/phd/code/auto_inv/us6000jk0t/invert_1_F/Figures/res_los_modlos_lonlat072A_20230108_20230213.ds_unw_Lon_Lat_Inc_Heading.GBIS.mat',vertex_path)
    output_matrix = io.loadmat('/Users/jcondon/phd/code/auto_inv/us6000jk0t/invert_1_2_3_4_F/invert_1_2_3_4_F.mat',squeeze_me=True)
    opt_model = output_matrix['invResults'][()].item()[-1]
    unw_file = '/Users/jcondon/phd/code/auto_inv/us6000jk0t_insar_processing/GEOC_072A_05090_131313_floatml_masked_GACOS_Corrected_clipped/20230120_20230201/20230120_20230201.unw'
    Inc_file = '/Users/jcondon/phd/code/auto_inv/us6000jk0t_insar_processing/GEOC_072A_05090_131313_floatml_masked_GACOS_Corrected_clipped/theta.geo'
    Head_file = '/Users/jcondon/phd/code/auto_inv/us6000jk0t_insar_processing/GEOC_072A_05090_131313_floatml_masked_GACOS_Corrected_clipped/phi.geo'
    EQA_dem_par = '/Users/jcondon/phd/code/auto_inv/us6000jk0t_insar_processing/GEOC_072A_05090_131313_floatml_masked_GACOS_Corrected_clipped/EQA.dem_par'
    locations = [44.9097,38.4199]
    print(opt_model)
    depth = 13000.0 
    strike1 = 13.0
    dip1 = 75.41
    rake1 = 177.86
    strike2 = 103.54
    dip2 = 87.93
    rake2 = 14.6
    mu = 3.2e10
    slip_rate=5.5e-5
    moment = 1.05e+19
    width = np.cbrt(float(moment)/(slip_rate*mu))
    length = width 
    slip = length * slip_rate
    usgs_model = [ 0,0,strike2,dip2,rake2,slip,length,depth,width]
    # locations = [37.8283,101.29]
    plot_location_comparisons(vertex_path,
                                  opt_model,
                                  EQA_dem_par,
                                  unw_file,
                                  Inc_file,
                                  Head_file,
                                  locations,
                                  usgs_model)