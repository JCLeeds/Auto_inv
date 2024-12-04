import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box
from shapely.affinity import rotate, scale
from scipy.io import loadmat
import matlab.engine 
import llh2local
import local2llh
import pygmt 
import LiCSBAS_io_lib as LiCS_lib
import pygmt
from importlib import reload
import geopandas as gpd
from matplotlib.path import Path


def start_matlab_set_path(path_to_GBIS):
        eng = matlab.engine.start_matlab()
        s = eng.genpath(path_to_GBIS)
        eng.addpath(s, nargout=0)
        return eng 

def read_in_invert(filepath,path_to_GBIS,burn_in):
    inv_results = loadmat(filepath,squeeze_me=True)
    models_total = inv_results['invResults'][()].item()[0].T
    print(models_total)
    geo = np.array(inv_results['geo'][()].item()[0])
    print(np.shape(models_total))
    print(np.shape(models_total)[1] - burn_in)
    print(np.shape(models_total))
    model_indices = np.array(np.arange(burn_in, 1e6),dtype=int)

    model_random_indices = np.random.choice(model_indices, size=int(1e4), replace=False)
    print(model_random_indices)
    vertex_total_list = model_to_verticies(models_total[model_random_indices,:],geo)
    print(len(vertex_total_list))

    # vertex_total_list = model_to_verticies(models_total[int(0):int(burn_in),:],geo)
   
    return vertex_total_list


def plot_vertices(vertex_total_list):
    print(len(vertex_total_list))
    for vertex in vertex_total_list:
        # print(vertex)
      
        plt.plot(vertex[0],vertex[1])
    # plt.show()



def heatmap_new(vertices,grid_resolution,lon_grid,lat_grid):
    heatmap = np.zeros_like(lat_grid, dtype=int)

    # for vertex in vertices:
    #     min_x, min_y = np.min(vertex[0]), np.min(vertex[1])
    #     max_x, max_y = np.max(vertex[0]), np.max(vertex[1])
    #     fault_planes.append((min_y,max_y,min_x,max_x))

    fault_planes = [(np.min(v[1]), np.max(v[1]), np.min(v[0]), np.max(v[0])) for v in vertices]
    for  min_lat, max_lat, min_lon, max_lon in fault_planes:
        # min_lat, max_lat, min_lon, max_lon = rect
        
        lat_in_rect = (lat_grid >= min_lat) & (lat_grid <= max_lat)
        lon_in_rect = (lon_grid >= min_lon) & (lon_grid <= max_lon)
    
        # Update the heatmap where the rectangle overlaps the grid
        heatmap += (lat_in_rect & lon_in_rect).astype(int)
        # Plotting the heatmap
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(heatmap,cmap='plasma')
    for vertex in vertices:
        # print(vertex)
        plt.plot(vertex[0],vertex[1])
  
    plt.title('Heatmap of Rectangle Overlaps')
 
    # plt.pcolormesh(lon_grid, lat_grid, heatmap, shading='auto', cmap='hot')  # Use shading='auto' for better visuals
    plt.colorbar(label='Number of overlaps')  # Add a colorbar to indicate heatmap values
    # plt.xlabel('Longitude')
    # plt.ylabel('Latitude')
    # plt.title('Heatmap of Rectangle Overlaps')
    plt.show()
    return heatmap
    

def heatmap_new_new(vertices, grid_resolution, lon_grid, lat_grid):
    heatmap = np.zeros_like(lat_grid, dtype=int)
    
    # Generate a grid of points (lon_grid, lat_grid)
    points = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))
    
    for vertex in vertices:
        # Create a Path (polygon) from the vertices
        polygon_path = Path(np.column_stack((vertex[0], vertex[1])))
        
        # Get the bounding box of the polygon (min/max lat/lon)
        min_lon, max_lon = np.min(vertex[0]), np.max(vertex[0])
        min_lat, max_lat = np.min(vertex[1]), np.max(vertex[1])
        
        # Filter points within the bounding box for efficiency
        in_bbox = (lon_grid >= min_lon) & (lon_grid <= max_lon) & (lat_grid >= min_lat) & (lat_grid <= max_lat)
        
        # Get the flattened indices of points inside the bounding box
        bbox_indices = np.where(in_bbox.ravel())[0]
        filtered_points = points[bbox_indices]
        
        # Check if points inside the bounding box are within the polygon
        inside_polygon = polygon_path.contains_points(filtered_points)
        
        # Update heatmap by reshaping the results and adding to the heatmap
        heatmap.ravel()[bbox_indices[inside_polygon]] += 1
    heatmap = heatmap / len(vertices)
    # Plot the heatmap
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(heatmap, cmap='plasma', origin='lower', extent=[lon_grid.min(), lon_grid.max(), lat_grid.min(), lat_grid.max()])
    
    # Plot the polygons on top of the heatmap
    for vertex in vertices:
        plt.plot(vertex[0], vertex[1], 'r-')

    plt.title('Heatmap of Polygon Overlaps')
    plt.colorbar(label='Number of overlaps')  # Add a colorbar to indicate heatmap values
    plt.show()

    return heatmap


def model_to_verticies(m,origin):
    origin = np.array(origin)  # Convert origin list to numpy array
    print('length m ')
    print(np.shape(m))
    # m is assumed to be a 2D numpy array
    vertex_total_list = []
    slipvectors = []
    centerpoint = []
    # print(m)
    for model in m:
        if np.all(model == 0):
            continue 
        else:
            vertices = np.array([[0, 0, -model[1], -model[1]],
                                [model[0] / 2, -model[0] / 2, -model[0] / 2, model[0] / 2],
                                [0, 0, 0, 0]])  
                            

            slipvec = np.array([0, 0, 0])
            sp = np.sin(np.deg2rad(model[4]))
            cp = np.cos(np.deg2rad(model[4]))
            cp = np.where(np.abs(cp) < 1e-12, 0, cp)
            sp = np.where(np.abs(sp) < 1e-12, 0, sp)

            cd = np.cos(np.deg2rad(model[3]))
            sd = np.sin(np.deg2rad(model[3]))
            cd = np.where(np.abs(cd) < 1e-12, 0, cd)
            sd = np.where(np.abs(sd) < 1e-12, 0, sd)

            R2 = np.array([[cd, 0, sd],
                        [0, 1, 0],
                        [-sd, 0, cd]])

            R1 = np.array([[cp, sp, 0],
                        [-sp, cp, 0],
                        [0, 0, 1]])

            vertices = R1 @ R2 @ vertices + np.tile(np.array([model[5], model[6], -model[2]]).reshape(3, 1), (1, 4))
            # print(vertices)
            # slipvec = R1 @ R2 @ slipvec
            # slipvectors.append(slipvec)

            centerpoint.append(np.mean(vertices, axis=1))  # Store center point of patch
            # print(np.shape(vertices))
            # Convert to llh if origin is supplied
            vertices[0:2,:] = local2llh.local2llh(vertices * 0.001, origin)[0:2,:]
            # print(vertices)
            # print(vertices)

            distances_one_bottom = np.sqrt((vertices[0, 2] - vertices[0, 0])**2 + (vertices[1, 2] - vertices[1, 0])**2)
            distance_x = vertices[0, 1] - vertices[0, 2]
            distance_y = vertices[1, 1] - vertices[1, 2]
            distance_z = vertices[2, 1] - vertices[2, 2]

            l = distance_x + 0.00001
            ym = distance_y + 0.00001
            n = distance_z + 0.00001
        

            x_surf = -(vertices[2, 2] * l / n) + vertices[0, 2]
            y_surf = -(vertices[2, 2] * ym / n) + vertices[1, 2]

            # vertices = np.hstack((vertices, np.zeros((3, 2))))  # Add two more columns to vertices

            # vertices[0, 4] = x_surf
            # vertices[1, 4] = y_surf
            # vertices[2, 4] = 0

            # distance_x = vertices[0, 0] - vertices[0, 3]
            # distance_y = vertices[1, 0] - vertices[1, 3]
            # distance_z = vertices[2, 0] - vertices[2, 3]

            # l = distance_x
            # ym = distance_y/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us6000abnv_insar_processing/GEOC_043A_05008_161514_floatml_masked_GACOS_Corrected_clipped
            # y_surf_two = -(vertices[2, 3] * ym / n) + vertices[1, 3]

            # vertices[0, 5] = x_surf_two
            # vertices[1, 5] = y_surf_two
            # vertices[2, 5] = 0

            vertex_total_list.append(vertices)


    return vertex_total_list

def pygmt_forward_model(unw_file,geoc,EQA_dem_par):
    dlat = float(LiCS_lib.get_param_par(EQA_dem_par, 'post_lat')) #negative
    dlon = float(LiCS_lib.get_param_par(EQA_dem_par, 'post_lon')) #positive
    lat1 = float(LiCS_lib.get_param_par(EQA_dem_par, 'corner_lat'))
    lon1 = float(LiCS_lib.get_param_par(EQA_dem_par, 'corner_lon'))
    width_ifgm = int(LiCS_lib.get_param_par(EQA_dem_par, 'width'))
    length_ifgm = int(LiCS_lib.get_param_par(EQA_dem_par, 'nlines'))

    lat2 = lat1+dlat*(length_ifgm-1) # south # Remove
    lon2 = lon1+dlon*(width_ifgm-1) # east # Remove
    lons, lats = np.arange(lon1, lon2+dlon, dlon), np.arange(lat1, lat2+dlat, dlat)
    lats_orig = np.array(lats,dtype=float)[:length_ifgm]
    lons_orig = np.array(lons,dtype=float)[:width_ifgm]

    # # Decimation step 
    # lons_orig = lons_orig[::10]
    # lats_orig = lats_orig[::10]

    lons, lats = np.meshgrid(lons_orig,lats_orig)
    unw = np.fromfile(unw_file, dtype='float32').reshape((length_ifgm, width_ifgm))
    unw = -unw*(0.0555/(4*np.pi)) 
    region = [np.min(lons),np.max(lons),np.min(lats),np.max(lats)] 
    file_path_data = 'unw.grd'
    # pygmt.xyz2grd(x=lons.flatten(),y=lats.flatten(),z=unw.flatten(),outgrid=file_path_data,region=region,spacing=(0.001,0.001))


    return lons,lats

def pygmt_figure(unw_file,geoc,EQA_dem_par,heatmap,vertex_path,vertex_total_list,usgs_location,USGS_mecas,event_name):
    dlat = float(LiCS_lib.get_param_par(EQA_dem_par, 'post_lat')) #negative
    dlon = float(LiCS_lib.get_param_par(EQA_dem_par, 'post_lon')) #positive
    lat1 = float(LiCS_lib.get_param_par(EQA_dem_par, 'corner_lat'))
    lon1 = float(LiCS_lib.get_param_par(EQA_dem_par, 'corner_lon'))
    width_ifgm = int(LiCS_lib.get_param_par(EQA_dem_par, 'width'))
    length_ifgm = int(LiCS_lib.get_param_par(EQA_dem_par, 'nlines'))

    lat2 = lat1+dlat*(length_ifgm-1) # south # Remove
    lon2 = lon1+dlon*(width_ifgm-1) # east # Remove
    lons, lats = np.arange(lon1, lon2+dlon, dlon), np.arange(lat1, lat2+dlat, dlat)
    lats_orig = np.array(lats,dtype=float)[:length_ifgm]
    lons_orig = np.array(lons,dtype=float)[:width_ifgm]

    # # Decimation step 
    # lons_orig = lons_orig[::10]
    # lats_orig = lats_orig[::10]

    lons, lats = np.meshgrid(lons_orig,lats_orig)
    unw = np.fromfile(unw_file, dtype='float32').reshape((length_ifgm, width_ifgm))
    unw = -unw*(0.0555/(4*np.pi)) 
    region = [np.min(lons)+0.1,np.max(lons)-0.1,np.min(lats)+0.1,np.max(lats)-0.1] 
    file_path_data = 'unw.grd'
    file_path_heatmap = 'heat.grd'
    pygmt.xyz2grd(x=lons.flatten(),y=lats.flatten(),z=unw.flatten(),outgrid=file_path_data,region=region,spacing=(0.001,0.001))
    pygmt.xyz2grd(x=lons.flatten(),y=lats.flatten(),z=heatmap.flatten(),outgrid=file_path_heatmap,region=region,spacing=(0.001,0.001))
    dem_region = [np.min(lons)-0.4,np.max(lons)+0.4,np.min(lats)-0.4,np.max(lats)+0.4] 
    terrain = pygmt.datasets.load_earth_relief(
    resolution="03s",
    region=dem_region,
    registration="gridline",
    ) #### This needs removing from paralellel wasted 
    resampled_terrain ='terrain.grd'
    gradiant_terrain = 'gradiant.grd'
    resamp = pygmt.grdsample(terrain,region=region,outgrid=resampled_terrain,spacing='03s',registration='gridline')
    pygmt.grdgradient(grid=resampled_terrain,azimuth=-35,outgrid=gradiant_terrain)

    max_data = np.nanmax(unw[~np.isnan(unw)].flatten()) 
    print(max_data)
    min_data = np.nanmin(unw[~np.isnan(unw)].flatten()) 
    data_series = str(min_data) + '/' + str(max_data*1.5) +'/' + str((max_data - min_data)/100)
    print("data cpt")
    print(data_series)
    if np.abs(min_data) > max_data:
        range_limit = np.abs(min_data)
    else:
        range_limit = max_data    

    if range_limit > 2:
        range_limit = 1
    cmap_output_data ='InSAR_CPT_data.cpt'
    pygmt.makecpt(series=[-range_limit, range_limit], cmap="polar",output=cmap_output_data)



    # if range_limit > 2:
    #     range_limit = 1

    
    vertex_data = loadmat(vertex_path)
    vertex = np.asarray(vertex_data['vertex_total_list'])
    lons_vertex = vertex[0][:]
    lats_vertex = vertex[1][:]
    depth_vertex = vertex[2][:]

    cmap_output_heat ='HEAT_CPT_data.cpt'
    pygmt.makecpt(series=[0, 1], cmap="plasma",output=cmap_output_heat)
    fig = pygmt.Figure()
    pygmt.config(MAP_FRAME_TYPE="plain")
    pygmt.config(FORMAT_GEO_MAP="ddd.xx")
    pygmt.config(FORMAT_FLOAT_OUT='%.12lg') 
    
    print('add basemap, gridimage and coast to 2D figure')
  
    # fig.grdimage(resampled_terrain,cmap=topo_output_cpt,region=region,projection='M8c',shading=gradiant_terrain)
  
    
    print('Down sample InSAR grd')
    unw_grd_ds = pygmt.grdsample(grid=file_path_data,spacing='03s',registration='gridline',region=region)
    # heatmap_grd_ds = pygmt.grdsample(grid=file_path_heatmap,spacing='03s',registration='gridline',region=region)
    print('grdimage InSAR grid')
    fig.grdimage(grid=unw_grd_ds,cmap=cmap_output_data,region=region,projection='M8c',shading=gradiant_terrain)
    fig.colorbar(frame=["x+lLOS Displacement(m)", "y+lm"], position="JMR",projection='M8c')
    fig.grdimage(grid=file_path_heatmap,cmap=cmap_output_heat,region=region,projection='M8c',nan_transparent='+z0')
    fig.basemap(frame=['a','+tUSGS Location Comparison'],region=region,projection='M8c',map_scale="jBL+w10k+o0.5c/0.5c+f+lkm")
    fig.coast(region=region, projection = 'M8c', water='lightblue')
    fig.plot(x=lons_vertex[len(lons_vertex)-2:len(lons_vertex)],
                y=lats_vertex[len(lats_vertex)-2:len(lats_vertex)],
                pen='1p,black,.',
                no_clip=False,
                #    sizes=0.1 * (2**np.array(eqMagAll)),
                #    style="uc",
                fill="gray",
                projection="M8c",
                transparency=60,
                # frame=["xa", "yaf", "za", "wSnEZ"],
                region=region)
    fig.plot(x=lons_vertex[:-2],
                        y=lats_vertex[:-2],
                        pen='1p,black',
                        fill='gray',
                        transparency=60,
                        region=region,
                        projection='M8c',
                        )
    

    fig.meca(
            spec=USGS_mecas, # <<< use dictionary
            scale="0.1c+f4p,Helvetica-Bold", 
            longitude=usgs_location[0], # event longitude
            latitude=usgs_location[1], # event latitude
            depth=[0],
            event_name=[event_name],
            compressionfill ='red',
            labelbox=True,
            # perspective=perspective,
            region=region,
        
            projection='M8c',
        
            
    )
    fig.colorbar(frame=["x+lModel Agreement", "y"], position="JMB",projection='M8c')
    # for vertex in vertex_total_list:
    #     lons_vertex = vertex[0][:]
    #     lats_vertex = vertex[1][:]
    #     depth_vertex = vertex[2][:]

    #     fig.plot(x=lons_vertex,
    #             y=lats_vertex,
    #             pen='1p,black,.',
    #             no_clip=False,
    #             #    sizes=0.1 * (2**np.array(eqMagAll)),
    #             #    style="uc",
    #             fill="gray",
    #             projection="M8c",
    #             transparency=20,
    #             # frame=["xa", "yaf", "za", "wSnEZ"],
    #             region=region)
    fig.show()


def run_heatmap(burn_in,GBIS_path,invert_path,opt_vertex_path,EQA_dem_par,unw_file,geoc_path,usgs_location,USGS_mecas,event_name):
    vertex_total_list = read_in_invert(invert_path,GBIS_path,burn_in)
    lons,lats = pygmt_forward_model(unw_file,geoc_ml_path,EQA_dem_par)
    heatmap = heatmap_new_new(vertex_total_list,100,lons,lats)
    pygmt_figure(unw_file,geoc_ml_path,EQA_dem_par,heatmap,opt_vertex_path,vertex_total_list,usgs_location,USGS_mecas,event_name)


if __name__ == '__main__':
    burn_in = 1e5

    GBIS_path = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/GBIS'
    # invert_path = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us6000ddge_NP1/invert_1_2_F/invert_1_2_F.mat'
    # opt_vertex_path = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us6000ddge_NP1/invert_1_2_F/optmodel_vertex.mat'
    # # vertex_total_list = read_in_invert(invert_path,GBIS_path,burn_in)
    # # print(len(vertex_total_list))
    # # plot_vertices(vertex_total_list)
    # print('finshed vertex list')
    # EQA_dem_par = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us6000ddge_insar_processing/GEOC_083D_08666_131313_floatml_masked_GACOS_Corrected_clipped/EQA.dem_par'
    # unw_file = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us6000ddge_insar_processing/GEOC_083D_08666_131313_floatml_masked_GACOS_Corrected_clipped/20210131_20210212/20210131_20210212.unw'
    # geoc_ml_path = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us6000ddge_insar_processing/GEOC_083D_08666_131313_floatml_masked_GACOS_Corrected_clipped'
    # # lons,lats = pygmt_forward_model(unw_file,geoc_ml_path,EQA_dem_par)
    # # heatmap = heatmap_new_new(vertex_total_list,100,lons,lats)
    # # pygmt_figure(unw_file,geoc_ml_path,EQA_dem_par,heatmap,opt_vertex_path,vertex_total_list)

    # invert_path = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us6000e2k3_NP1/invert_1_5_7_8_9_10_13_14_F/invert_1_5_7_8_9_10_13_14_F.mat'
    # opt_vertex_path = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us6000e2k3_NP1/invert_1_5_7_8_9_10_13_14_F/optmodel_vertex.mat'
    # EQA_dem_par = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us6000e2k3_insar_processing/GEOC_035D_05978_131209_floatml_masked_GACOS_Corrected_clipped/EQA.dem_par'
    # unw_file = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us6000e2k3_insar_processing/GEOC_035D_05978_131209_floatml_masked_GACOS_Corrected_clipped/20210329_20210422/20210329_20210422.unw'
    # geoc_ml_path = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us6000e2k3_insar_processing/GEOC_035D_05978_131209_floatml_masked_GACOS_Corrected_clipped'
    # usgs_location = [50.6784,29.7531]
    # USGS_mecas = dict(
    #     strike =[295.38],
    #     dip=[16.08],
    #     rake=[70.01],
    #     magnitude=[6.0],
    # )
    # event_name = 'us6000e2k3'

    # # run_heatmap(burn_in,GBIS_path,invert_path,opt_vertex_path,EQA_dem_par,unw_file,geoc_ml_path,usgs_location,USGS_mecas,event_name)
    

    # usgs_location = [18.1803,43.0742]
    # USGS_mecas = dict(
    #     strike =[278.66],
    #     dip=[21.83],
    #     rake=[75.08],
    #     magnitude=[5.6],
    # )
    # event_name = 'us6000hfqj'
    # invert_path = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us6000hfqj_NP2/invert_1_3_5_6_F/invert_1_3_5_6_F.mat'
    # opt_vertex_path = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us6000hfqj_NP2/invert_1_3_5_6_F/optmodel_vertex.mat'
    # EQA_dem_par = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us6000hfqj_insar_processing/GEOC_051D_04568_131313_floatml_masked_GACOS_Corrected_clipped/EQA.dem_par'
    # unw_file = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us6000hfqj_insar_processing/GEOC_051D_04568_131313_floatml_masked_GACOS_Corrected_clipped/20220325_20220430/20220325_20220430.unw'
    # geoc_ml_path = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us6000hfqj_insar_processing/GEOC_051D_04568_131313_floatml_masked_GACOS_Corrected_clipped'
 
    # run_heatmap(burn_in,GBIS_path,invert_path,opt_vertex_path,EQA_dem_par,unw_file,geoc_ml_path,usgs_location,USGS_mecas,event_name)

    # abnv test
    usgs_location = [39.4229,40.7073]
    USGS_mecas = dict(
        strike =[263.56],
        dip=[85.49],
        rake=[169.52],
        magnitude=[5.7],
    )
    event_name = 'us6000abnv'
    invert_path = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/Paper_run/us6000abnv_NP1/invert_21_27_33_40_43_46_57_F/invert_21_27_33_40_43_46_57_F.mat'
    opt_vertex_path = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/Paper_run/us6000abnv_NP1/invert_21_27_33_40_43_46_57_F/optmodel_vertex.mat'
    EQA_dem_par = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/Paper_run/us6000abnv_insar_processing/GEOC_050D_05046_141313_floatml_masked_GACOS_Corrected_clipped/EQA.dem_par'
    unw_file = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/Paper_run/us6000abnv_insar_processing/GEOC_050D_05046_141313_floatml_masked_GACOS_Corrected_clipped/20200609_20200709/20200609_20200709.unw'
    geoc_ml_path = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/Paper_run/us6000abnv_insar_processing/GEOC_050D_05046_141313_floatml_masked_GACOS_Corrected_clipped'
 
    run_heatmap(burn_in,GBIS_path,invert_path,opt_vertex_path,EQA_dem_par,unw_file,geoc_ml_path,usgs_location,USGS_mecas,event_name)


    # ####### afghan events ####### 
    # # event         lat     lon 
    # # us6000ldpm 	34.55 	61.88 
    # # us6000ldpg 	34.6 	61.93
    # # us6000ldpv 	34.63 	62 
    # # us6000lfn5 	34.62 	62.05 
    # # us6000len8 	34.56 	62.04 

    # # ASC Seq 1
    # event_name = ['us6000ldpm ','us6000ldpg','us6000ldpv','us6000lfn5','us6000len8']
    # lats = [34.55,34.6,34.63,34.62,34.56]
    # lons = [61.88,61.93,62,62.05,62.04]
   
    # invert_path = '/uolstore/Research/a/a285/homes/ee18jwc/projects/afghanistan/Final_products_checked/Afphgan_seq1_NP1_width_unlimited_THIS_ONE_USED/invert_1_2_F/invert_1_2_F.mat'
    # opt_vertex_path = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us6000hfqj_NP2/invert_1_3_5_6_F/optmodel_vertex.mat'
    # EQA_dem_par = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us6000hfqj_insar_processing/GEOC_051D_04568_131313_floatml_masked_GACOS_Corrected_clipped/EQA.dem_par'
    # unw_file = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us6000hfqj_insar_processing/GEOC_051D_04568_131313_floatml_masked_GACOS_Corrected_clipped/20220325_20220430/20220325_20220430.unw'
    # geoc_ml_path = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us6000hfqj_insar_processing/GEOC_051D_04568_131313_floatml_masked_GACOS_Corrected_clipped'
 
    # run_heatmap(burn_in,GBIS_path,invert_path,opt_vertex_path,EQA_dem_par,unw_file,geoc_ml_path,usgs_location,USGS_mecas,event_name)





     