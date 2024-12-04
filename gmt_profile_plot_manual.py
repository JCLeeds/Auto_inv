from scipy import stats, interpolate, io
import logging
try:
    import h5py
except ImportError as e:
    raise e('Please install h5py library')
from osgeo import gdal, osr
import os 
import numpy as np
import pygmt 
log = logging.getLogger('mat2npz')
import random 
import pylab as plt
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


def read_mat(filename):
    try:
        mat = h5py.File(filename, 'r')
    except OSError:
        log.debug('using old scipy import for %s', filename)
        mat = io.loadmat(filename)
    return mat

def open_print_vertex(vertex):
     data = read_mat(vertex)
     print(data['vertex_total_list'])




def plot_location_and_profile(unw_file,Inc_file,Head_file,vertexs,EQA_dem_par,opt_model,locations,ifgix):
  
    # data = read_mat(datapath)
    # los = np.asarray(data['los']).flatten()
    # modLos_save = np.asarray(data['modLos_save']).flatten()
    # residual = np.asarray(data['residual']).flatten()
    # lonlat = np.asarray(data['ll'])
    # lons = lonlat[:, 0]
    # lats = lonlat[:, 1]
    # print(lons.shape)
    # print(lats.shape)
    # print(los.shape)
    # los = los[~np.isnan(los)]
    # lons = lons[~np.isnan(los)]
    # lats = lats[~np.isnan(los)]
    # modLos_save = modLos_save[~np.isnan(los)]
    # residual = residual[~np.isnan(los)]


    vertexs = read_mat(vertex_data)
    vertexs = vertexs['vertex_total_list']
    print(vertexs)
    x_lon_NP1 = vertexs[0,:]
    y_lat_NP1 = vertexs[1,:]
    z_depth_NP1 = vertexs[2,:]
    print(x_lon_NP1)


    
    width_ifgm = int(LiCS_lib.get_param_par(EQA_dem_par, 'width'))
    length_ifgm = int(LiCS_lib.get_param_par(EQA_dem_par, 'nlines'))

  
    Inc = np.fromfile(Inc_file, dtype='float32').reshape((length_ifgm, width_ifgm))
    Head = np.fromfile(Head_file, dtype='float32').reshape((length_ifgm, width_ifgm))
    unw = np.fromfile(unw_file, dtype='float32').reshape((length_ifgm, width_ifgm)) 



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
    # if opt_model[9]:
    #     const_offset = opt_model[9]
    # else:
    #     const_offset = 0

    rake = np.degrees(np.arctan2(DS,SS))
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
    disp =lib.disloc3d3(xx_vec,yy_vec,xoff=X,yoff=Y,depth=depth,
                            length=length,width=width,slip=total_slip,
                            opening=0,strike=strike,dip=dip,rake=rake,nu=0.25)


    incidence_angle = np.nanmean(Inc)
    azimuth_angle = np.nanmean(Head)

    # unw = -unw*(0.0555/(4*np.pi)) - const_offset
    unw = -unw*(0.0555/(4*np.pi)) 
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
    modLos_save = los_grid 
    los = unw
    residual = resid 




    USGS_mecas_seq_1 = dict(
        strike =[253,297,265],
        dip=[26,25,21],
        rake=[75,111,93],
        magnitude=[6.3,6.3,5.9],
    )
    mid_top_x = (x_lon_NP1[0] + x_lon_NP1[1])/2
    mid_top_y = (y_lat_NP1[0] + y_lat_NP1[1])/2

    mid_bottom_x = (x_lon_NP1[2] + x_lon_NP1[3])/2
    mid_bottom_y = (y_lat_NP1[2] + y_lat_NP1[3])/2

    m = (mid_top_y-mid_bottom_y)/(mid_top_x-mid_bottom_x)
    c = mid_top_y - m*mid_top_x
    if mid_top_x > mid_bottom_x:
        low_bound = mid_bottom_x-0.05
        high_bound = mid_top_x+0.05
    else:
        low_bound = mid_top_x-0.05
        high_bound = mid_bottom_x+0.05

    x = np.random.uniform(low=low_bound, high=high_bound, size=(100,))
    y = m*x + c
    y_line_end_one = m*low_bound + c
    y_line_end_two = m*high_bound + c
    start = [ high_bound,y_line_end_one]
    end = [low_bound,y_line_end_two]
    # plt.scatter(x,y)
    # plt.scatter(low_bound,y_line_end_two,c='red')
    # plt.scatter(high_bound,y_line_end_one,c='green')
    # plt.show()
    

    strike =  98.2176	
    print('######################################## Profile plot ##############################################################')
    x1 = round((np.min(lons) - (np.min(lons) % 0.03) - 1*0.03),2)
    x2 = round((np.max(lons) - (np.max(lons) % 0.03) + 1*0.03),2)
    y1 = round((np.min(lats) - (np.min(lats) % 0.03) - 1*0.03),2)
    y2 = round((np.max(lats) - (np.max(lats) % 0.03) + 1*0.03),2)
    print("y1")
    print(y2%0.03)

    region = [np.min(lons),np.max(lons),np.min(lats),np.max(lats)] 
    dem_region = [x1,x2,y1,y2]
    dem_region = [np.min(lons)-0.4,np.max(lons)+0.4,np.min(lats)-0.4,np.max(lats)+0.4] 


 
    file_path_data = 'profile_data_meters.grd'
    file_path_model = 'profile_model_meters.grd'
    file_path_res = 'profile_res_meters.grd'
   
    pygmt.xyz2grd(x=lons.flatten(),y=lats.flatten(),z=los.flatten(),outgrid=file_path_data,region=region,spacing=(0.001,0.001))
    print('surface project')
    pygmt.xyz2grd(x=lons.flatten(),y=lats.flatten(),z=modLos_save.flatten(),outgrid=file_path_model,region=region,spacing=(0.001,0.001))
    pygmt.xyz2grd(x=lons.flatten(),y=lats.flatten(),z=residual.flatten(),outgrid=file_path_res,region=region,spacing=(0.001,0.001))
   
    #   dem_region = [np.min(lon)-0.4,np.max(lon)+0.4,np.min(lat)-0.4,np.max(lat)+0.4] 
    # region = [np.min(lon),np.max(lon),np.min(lat),np.max(lat)] 
    terrain = pygmt.datasets.load_earth_relief(
    resolution="03s",
    region=dem_region,
    registration="gridline",
    ) #### This needs removing from paralellel wasted 

    resampled_terrain ='terrain.grd'
    gradiant_terrain = 'gradiant.grd'
    resamp = pygmt.grdsample(terrain,region=region,outgrid=resampled_terrain,spacing='03s',registration='gridline')
    pygmt.grdgradient(grid=resampled_terrain,azimuth=-35,outgrid=gradiant_terrain)

    max_data = np.nanmax(los) 
    print(max_data)
    min_data = np.nanmin(los) 
    data_series = str(min_data) + '/' + str(max_data*1.5) +'/' + str((max_data - min_data)/100)
    print("data cpt")
    print(data_series)
    
    # print("data cpt")
    # print(data_series)
    print('make color pallette')
 
    # cmap_output_data =  out_dir1 + '/' +'InSAR_CPT_data.cpt'
    topo_output_cpt =  'topo.cpt'
    topo_cpt_series = '0/5000/100' 
    cmap_output_data ='InSAR_CPT_data.cpt'
 

    # resampled_terrain = out_dir+'/'+'terrain.grd'
    # resamp = pygmt.grdsample(terrain,region=region,outgrid=resampled_terrain,spacing='10s')
 
    
    pygmt.makecpt(cmap='oleron',series=topo_cpt_series, continuous=True,output=topo_output_cpt,background=True) 
    pygmt.makecpt(cmap='polar',series=data_series, continuous=True,output=cmap_output_data,background=True) 

    fig = pygmt.Figure()
    pygmt.config(MAP_FRAME_TYPE="plain")
    pygmt.config(FORMAT_GEO_MAP="ddd.xx")
    pygmt.config(FORMAT_FLOAT_OUT='%.12lg') 
    
    print('add basemap, gridimage and coast to 2D figure')
  
    # fig.grdimage(resampled_terrain,cmap=topo_output_cpt,region=region,projection='M8c',shading=gradiant_terrain)
    



    #### LARGE LOCATION PLOT 
    title_dates = '013A 20230925_20231007'
    print('Down sample InSAR grd')
    unw_grd_ds = pygmt.grdsample(grid=file_path_data,spacing='03s',registration='gridline',region=region)
    print('grdimage InSAR grid')
    fig.grdimage(grid=unw_grd_ds,cmap=cmap_output_data,region=region,projection='M10c',shading=gradiant_terrain)

  

    fig.basemap(frame=['a'],region=region,projection='M10c',map_scale="jBL+w10k+o0.5c/0.5c+f+lkm")
    fig.coast(region=region, projection = 'M10c', water='lightblue')
    print('plotting fault plane')
    fig.plot(x=x_lon_NP1[:-2],
                        y=y_lat_NP1[:-2],
                        pen='1p,black',
                        fill='gray',
                        transparency=75,
                        region=region,
                        projection='M10c',
                        )
    print('plotting surface projection')
    fig.plot(x=x_lon_NP1[len(x_lon_NP1)-2:len(x_lon_NP1)],
                y=y_lat_NP1[len(y_lat_NP1)-2:len(y_lat_NP1)],
                pen='1p,black,.',
                no_clip=False,
                #    sizes=0.1 * (2**np.array(eqMagAll)),
                #    style="uc",
                fill="gray",
                projection="M10c",
                transparency=20,
                # frame=["xa", "yaf", "za", "wSnEZ"],
                region=region)
    
    fig.colorbar(frame=["x", "y+lm"],projection='M10c')

    profile_line = pygmt.project(
    center=start,
    endpoint=end,
    generate=(0.005)
    )

    profile_line = pygmt.project(
    center=start,
    endpoint=end,
    generate=(0.005)
    )

    print(np.min(x), np.max(y))
    print(np.max(x), np.min(y))
    pygmt.grdtrack(
            grid = file_path_data, 
            points = profile_line,
            outfile="tmp_data_profile.txt",
            skiprows=False,
            newcolname='profile',
    )

    pygmt.grdtrack(
            grid = file_path_model, 
            points = profile_line,
            outfile="tmp_model_profile.txt",
            skiprows=False,
            newcolname='profile',
    )

    x_NP1_project = []
    y_NP1_project = []
    z_NP1_project = [] 
    distance_NP1_project = [] 

    with open("tmp_data_profile.txt", 'r') as f:
        lines = f.readlines()
    for line in lines:
        # print(line.split(' '))
        x_NP1_project.append(float(line.split()[0]))
        y_NP1_project.append(float(line.split()[1]))
        z_NP1_project.append(float(line.split()[3]))
        distance_NP1_project.append(float(line.split()[2]))

    x_NP1_project = np.array(x_NP1_project)
    y_NP1_project = np.array(y_NP1_project)
    z_NP1_project = np.array(z_NP1_project)
    distance_NP1_project=np.array(distance_NP1_project)
    print(len(x_NP1_project))
    print(len(y_NP1_project))
    print(len(z_NP1_project))
    x_NP1_project = x_NP1_project[~np.isnan(z_NP1_project)]
    y_NP1_project = y_NP1_project[~np.isnan(z_NP1_project)]
    distance_NP1_project = distance_NP1_project[~np.isnan(z_NP1_project)]
    z_NP1_project = z_NP1_project[~np.isnan(z_NP1_project)]


    x_NP1_project_model = []
    y_NP1_project_model = []
    z_NP1_project_model = [] 
    distance_NP1_project_model = [] 

    with open("tmp_model_profile.txt", 'r') as f:
        lines = f.readlines()
    for line in lines:
        # print(line.split(' '))
        x_NP1_project_model.append(float(line.split()[0]))
        y_NP1_project_model.append(float(line.split()[1]))
        z_NP1_project_model.append(float(line.split()[3]))
        distance_NP1_project_model.append(float(line.split()[2]))

    x_NP1_project_model = np.array(x_NP1_project_model)
    y_NP1_project_model = np.array(y_NP1_project_model)
    z_NP1_project_model = np.array(z_NP1_project_model)
    distance_NP1_project_model= np.array(distance_NP1_project_model)
    # print(len(x_NP1_project))
    # print(len(y_NP1_project))
    # print(len(z_NP1_project))
    x_NP1_project_model = x_NP1_project_model[~np.isnan(z_NP1_project_model)]
    y_NP1_project_model = y_NP1_project_model[~np.isnan(z_NP1_project_model)]
    distance_NP1_project_model = distance_NP1_project_model[~np.isnan(z_NP1_project_model)]
    z_NP1_project_model = z_NP1_project_model[~np.isnan(z_NP1_project_model)]
    
    fig.plot(x=[low_bound,high_bound], y=[y_line_end_one, y_line_end_two], projection="M10", pen=2)
    if y_line_end_one > y_line_end_two:
        text_upper = y_line_end_one + 0.05
        text_lower = y_line_end_two - 0.05
    else:
        text_lower = y_line_end_one - 0.05 
        text_upper = y_line_end_two + 0.05 
    fig.text(x=low_bound, y=text_upper, text="B", font="8,Helvetica")
    fig.text(x=high_bound, y=text_lower, text="A", font="8,Helvetica")
  
    #### Model small panel 
      
    fig.shift_origin(xshift=11.5)
    fig.shift_origin(yshift=5.75)
    mod_grd_ds = pygmt.grdsample(grid=file_path_model,spacing='03s',registration='gridline',region=region)
    fig.grdimage(grid=mod_grd_ds,cmap=cmap_output_data,region=region,projection='M5c',shading=gradiant_terrain)
    fig.basemap(
      
       frame=["WStr","ya0.5f1"],
       region=region,
       map_scale="jBL+w10k+o0.5c/0.5c+f+lkm",
       projection='M5c',  
    )
    map_scale="jBL+w10k+o0.5c/0.5c+f+lkm"
   
    fig.shift_origin(yshift=-5.75)
    res_grd_ds = pygmt.grdsample(grid=file_path_res,spacing='03s',registration='gridline',region=region)
    fig.grdimage(grid=res_grd_ds,cmap=cmap_output_data,region=region,projection='M5c',shading=gradiant_terrain)
    fig.basemap(
       frame=['a'],
       region=region,
       map_scale="jBL+w10k+o0.5c/0.5c+f+lkm",
       projection='M5c',
    )
  
   
    fig.shift_origin(yshift=-8.25)
    fig.shift_origin(xshift=-11.5)
    fig.basemap(
        projection="X16.5/-6",
        region=[np.min(distance_NP1_project), np.max(distance_NP1_project), np.nanmin(los), np.nanmax(los)],
        frame=['xafg100+l"Distance (Degrees)"', 'yafg50+l"Displacement in Line-of-Sight (m)"', "WSen"],
        # yshift=-7,
        # xshift=-11.5
        
    )
    fig.plot(x=distance_NP1_project,y=z_NP1_project, projection="X", style="c0.02", pen='thick,-', fill="black")
    fig.plot(x=distance_NP1_project, y=z_NP1_project, pen="0.5p,black",projection='X',label='Data in Line-of-Sight (m)')
    # fig.plot(x=distance_NP1_project_model,y=z_NP1_project_model, projection="X", style="c0.02", pen='thick,-', fill="red")
    fig.plot(x=distance_NP1_project_model, y=z_NP1_project_model, pen="1p,red",projection='X',label='Model in Line-of-Sight (m)')
    fig.text(x=0.01, y=-0.05, text="B", font="10,Helvetica")
    fig.text(x=np.max(distance_NP1_project)-0.01, y=-0.05, text="A", font="10,Helvetica")
    fig.legend(position='JBL+jBL+o0.2c')


    fig.savefig(os.path.join(out_dir1,'profile_plot.png'))
    # fig.show()


if __name__ == '__main__':
  
    vertex_data = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us60007anp_NP1/invert_1_2_3_4_5_6_F_location_run/optmodel_vertex.mat'
    unw_file = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us60007anp_insar_processing/GEOC_034D_04913_131313_floatml_masked_clipped/20191229_20200122/20191229_20200122.unw'
    Head_file = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us60007anp_insar_processing/GEOC_034D_04913_131313_floatml_masked_clipped/phi.geo'
    Inc_file = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us60007anp_insar_processing/GEOC_034D_04913_131313_floatml_masked_clipped/theta.geo'
    # plot_location_and_profile(data_locations_seq2[1],2)
    EQA_dem_par = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us60007anp_insar_processing/GEOC_034D_04913_131313_floatml_masked_clipped/EQA.dem_par'
    model = [   	  
      	  4622.97,	 
      	  25001.2,	  
       	  6146.63,	  
  	      -24.1593,	 
     	  -14.9842,	
           21672,	  
          486.089,	  
     	0.264796,	
     	 -0.300814]	 


    model = [
    	  4432.82,	 
     	  6647.78,	  
   	  7605.09,	  
     	 -11.6904,	
     	  24.6423,	  
       	  18405.6,	  
          	  6021.84,	  
   	 -0.900769,	
   	 -0.531673
     ]	 
    location = [77.1084,39.8353]
    # open_print_vertex('/uolstore/Research/a/a285/homes/ee18jwc/projects/afghanistan/Final_products_checked/Afphgan_seq2_NP2_newdata/optmodel_vertex.mat')
    # plot_location_and_profile(unw_file,Inc_file,Head_file,vertex_data,EQA_dem_par,model,location,'profile_test.png')
    # plot_location_and_profile(data_locations_seq1[1],vertex_data_seq2,1,'profile_seq1_Dsc.png')
    # plot_location_and_profile(data_locations_seq2[0],vertex_data_seq2,0,'profile_seq2_Asc.png')
    # plot_location_and_profile(data_locations_seq2[1],vertex_data_seq2,0,'profile_seq2_Dsc.png')
    geoc_ml_path = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us60007anp_insar_processing/GEOC_034D_04913_131313_floatml_masked_clipped'
    output_geoc = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/test_profile'
    plot_location_and_profile(geoc_ml_path,output_geoc,model,location,vertex_data)