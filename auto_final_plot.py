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



# def load_from_mat(filename=None, data={}, loaded=None):
#     if filename:
#         vrs = io.whosmat(filename)
#         name = vrs[0][0]
#         loaded = io.loadmat(filename,struct_as_record=True)
#         loaded = loaded[name]
#     whats_inside = loaded.dtype.fields
#     fields = list(whats_inside.keys())
#     for field in fields:
#         if len(loaded[0,0][field].dtype) > 0: # it's a struct
#             data[field] = {}
#             data[field] = load_from_mat(data=data[field], loaded=loaded[0,0][field])
#         else: # it's a variable
#             data[field] = loaded[0,0][field]
#     return data


def GBIS_GMT_OUTPUT(file_path_data,vertex_path):
    data = read_mat(file_path_data)
    vertex_data = read_mat(vertex_path)
    vertex = np.asarray(vertex_data['vertex_total_list'])

    los = np.asarray(data['los']).flatten()
    modLos_save = np.asarray(data['modLos_save']).flatten()
    residual = np.asarray(data['residual']).flatten()
    lonlat = np.asarray(data['ll'])
    lons = lonlat[:, 0]
    lats = lonlat[:, 1]


    # USGS_mecas_seq_1 = dict(
    #     strike =[253,297,265],
    #     dip=[26,25,21],
    #     rake=[75,111,93],
    #     magnitude=[6.3,6.3,5.9],
    # )
    region = [np.min(lons),np.max(lons),np.min(lats),np.max(lats)] # GMT region  [xmin,xmax,ymin,ymax].

    if "A_" in file_path_data:
        file_path_los = file_path_data + 'A_los.grd'
        file_path_res = file_path_data + 'A_res.grd'
        file_path_model = file_path_data + 'A_mod.grd'
    elif "D_" in file_path_data:
        file_path_los = file_path_data + 'D_los.grd'
        file_path_res = file_path_data + 'D_res.grd'
        file_path_model = file_path_data+ 'D_mod.grd'
    
    
    pygmt.xyz2grd(x=lons,y=lats,z=los,outgrid=file_path_los,region=region,spacing=(0.03,0.03))
    pygmt.xyz2grd(x=lons,y=lats,z=residual,outgrid=file_path_res,region=region,spacing=(0.03,0.03))
    pygmt.xyz2grd(x=lons,y=lats,z=modLos_save,outgrid=file_path_model,region=region,spacing=(0.03,0.03))
    print(pygmt.grdinfo(file_path_los))
    print(vertex)
    lons_vertex = vertex[0][:]
    lats_vertex = vertex[1][:]
    depth_vertex = vertex[2][:]
    print(lons_vertex)
    max_los = np.max(los) 
    min_los = np.min(los) 
    print(max(lats))
    print(max(lons))
    series = str(min_los) + '/' + str(max_los*1.25) +'/' +'0.01'

    fig = pygmt.Figure()
    pygmt.config(MAP_FRAME_TYPE="plain")
    pygmt.config(FORMAT_GEO_MAP="ddd.xx")
    pygmt.config(FORMAT_FLOAT_OUT='%.12lg') 
    # print(file_path_data.split('/')[-1])   
    cmap_output = file_path_data + 'InSAR_CPT.cpt'
    print(cmap_output)
    pygmt.makecpt(cmap='polar',series=series, continuous=True,output=cmap_output) 
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
        print(max(los))
        fig.grdimage(grid=file_path_los,cmap=cmap_output,region=region,projection='M?c',panel=[0,0])
        fig.basemap(frame=['a','+tData'],panel=[0,0],region=region,projection='M?c')
        fig.grdimage(grid=file_path_model,cmap=cmap_output,region=region,projection='M?c',panel=[0,1])
        fig.basemap(frame=['xa','+tModel'],panel=[0,1],region=region,projection='M?c')
        fig.grdimage(grid=file_path_res,cmap=cmap_output,region=region,projection='M?c',panel=[0,2])
        fig.basemap(frame=['xa','+tResidual'],panel=[0,2],region=region,projection='M?c')

        for ii in range(0,3):
            fig.plot(x=lons_vertex[:-2],
                        y=lats_vertex[:-2],
                        pen='2p,black',
                        fill='gray',
                        transparency=80,
                        region=region,
                        projection='M?c',
                        panel=[0,ii]
            )

            fig.plot(x=lons_vertex[len(lons_vertex)-2:len(lons_vertex)],
                        y=lats_vertex[len(lats_vertex)-2:len(lats_vertex)],
                        pen='2p,black,-',
                        fill='gray',
                        transparency=80,
                        region=region,
                        projection='M?c',
                        panel=[0,ii]
            )
        for ii in range(0,3):
                fig.colorbar(frame=["x+lLOS displacment(m)", "y+lm"], position="JMB",projection='M?c',panel=[0,ii]) # ,panel=[1,0]
        fig.show(method='external')
        fig.savefig(file_path_data+'gmt_output.png')

        
def GBIS_GMT_OUTPUT_FORWARD_MODEL(vertex_path,opt_model,EQA_dem_par,unw_file,Inc_file,Head_file,locations,usgs_model):
    # EQA_dem_par = '/Users/jcondon/phd/code/auto_inv/us6000jk0t_insar_processing/GEOC_072A_05090_131313_floatml_masked_GACOS_Corrected_clipped_signal_masked_down_sampled/EQA.dem_par'
    # x_usgs,y_usgs = LiCS_tools.bl2xy(location[1],location[0],width_ifgm,length_ifgm,lat1,dlat,lon1,dlon)
    vertex_data = read_mat(vertex_path)
    vertex = np.asarray(vertex_data['vertex_total_list'])
    # xcent = x_usgs * pixsp_a
    # ycent = y_usgs * pixsp_r
    # xcent
    # data = read_mat(file_path_data)
    # los = np.asarray(data['los']).flatten()
    # modLos_save = np.asarray(data['modLos_save']).flatten()
    # residual = np.asarray(data['residual']).flatten()
    # lonlat = np.asarray(data['ll'])
    # lons = lonlat[:, 0]
    # lats = lonlat[:, 1]

    width_ifgm = int(LiCS_lib.get_param_par(EQA_dem_par, 'width'))
    length_ifgm = int(LiCS_lib.get_param_par(EQA_dem_par, 'nlines'))
    dlat = float(LiCS_lib.get_param_par(EQA_dem_par, 'post_lat')) #negative
    dlon = float(LiCS_lib.get_param_par(EQA_dem_par, 'post_lon')) #positive
    lat1 = float(LiCS_lib.get_param_par(EQA_dem_par, 'corner_lat'))
    lon1 = float(LiCS_lib.get_param_par(EQA_dem_par, 'corner_lon'))
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
    xy = llh.llh2local(ll,np.array([locations[0],locations[1]],dtype=float))
    

    length = opt_model[0]
    width = opt_model[1]
    depth = opt_model[2] # this gives top depth middle depth is given as w/2cos(dip) + depth
    print(depth)
    dip = -opt_model[3]
    print(dip)
    print(np.sin(dip*(np.pi/180)))
    depth =  depth + ((width/2)*np.sin(dip*(np.pi/180))) # this may need editing for sin or cos depending on dip check 
    print(depth)
    print(width)
    strike = opt_model[4] - 180 
    # strike = opt_model[4]
    X = opt_model[5]
    Y = opt_model[6]
    SS = opt_model[7]
    DS = opt_model[8]


    print(opt_model)
    rake = np.degrees(np.arctan(DS/SS))
    total_slip = np.sqrt(SS**2 + DS**2)
    print("total slip")
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


    model = [X, Y, strike, dip, rake, total_slip, length, depth, width]
    # Calcualte displacements
    disp = cl.disloc3d3(xx_vec, yy_vec, xoff=X, yoff=Y, depth=depth,
                    length=length, width=width, slip=total_slip, opening=0, 
                    strike=strike, dip=dip, rake=rake, nu=0.25)

    disp_usgs =cl.disloc3d3(xx_vec,yy_vec,xoff=0,yoff=0,depth=usgs_model[7],
                            length=usgs_model[6],width=usgs_model[8],slip=usgs_model[5],
                            opening=0,strike=usgs_model[2],dip=usgs_model[3],rake=usgs_model[4],nu=0.25)
    end1x, end2x, end1y, end2y, c1x, c2x, c3x, c4x, c1y, c2y, c3y, c4y = cl.fault_for_plotting(usgs_model)
    x_usgs = np.array([c1x,c2x,c3x,c4x])/ 1000 #local2llh needs results in km 
    y_usgs = np.array([c1y,c2y,c3y,c4y])/ 1000 #local2llh needs results in km 
    x_usgs_end = np.array([end1x,end2x]) / 1000 
    y_usgs_end = np.array([end1y,end2y]) / 1000

    print(x_usgs)
    print(y_usgs)
    local_usgs_source = [x_usgs,y_usgs]  
    local_end_usgs_source = [x_usgs_end,y_usgs_end]
    print(np.shape(local_usgs_source))
    print("LOCATIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~ = " + str(locations))
    llh_usgs = l2llh.local2llh(local_usgs_source,[locations[0],locations[1]])
    fault_end = l2llh.local2llh(local_end_usgs_source,[locations[0],locations[1]])
    print(llh_usgs)
    print(np.shape(llh_usgs))

    np.shape(disp)
    disp = -disp
    unw_file = os.path.join(unw_file) 
    unw = np.fromfile(unw_file, dtype='float32').reshape((length_ifgm, width_ifgm)) 
    Inc = np.fromfile(Inc_file, dtype='float32').reshape((length_ifgm, width_ifgm))
    Head = np.fromfile(Head_file, dtype='float32').reshape((length_ifgm, width_ifgm))  

   
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

    los_grid_usgs = (disp_usgs[0,:] * e2los) + (disp_usgs[1,:] * n2los) + (disp_usgs[2,:] * u2los) 
    resid_usgs = unw.flatten() - los_grid_usgs

    file_path_los = 'A_los.grd'
    file_path_res = 'A_res.grd'
    file_path_model = 'A_mod.grd'

    file_path_res_usgs = 'A_res_usgs.grd'
    file_path_model_usgs = 'A_mod_usgs.grd'


    pygmt.xyz2grd(x=lons.flatten(),y=lats.flatten(),z=unw.flatten(),outgrid=file_path_los,region=region,spacing=(0.001,0.001))
    pygmt.xyz2grd(x=lons.flatten(),y=lats.flatten(),z=resid,outgrid=file_path_res,region=region,spacing=(0.001,0.001))
    pygmt.xyz2grd(x=lons.flatten(),y=lats.flatten(),z=los_grid,outgrid=file_path_model,region=region,spacing=(0.001,0.001))


      
    pygmt.xyz2grd(x=lons.flatten(),y=lats.flatten(),z=resid_usgs,outgrid=file_path_res_usgs,region=region,spacing=(0.001,0.001))
    pygmt.xyz2grd(x=lons.flatten(),y=lats.flatten(),z=los_grid_usgs,outgrid=file_path_model_usgs,region=region,spacing=(0.001,0.001))


    lons_vertex = vertex[0][:]
    lats_vertex = vertex[1][:]
    depth_vertex = vertex[2][:]
    print(lons_vertex)
    max_los = np.max(los_grid) 
    min_los = np.min(los_grid) 
    model_series = str(min_los) + '/' + str(max_los*1.5) +'/' +'0.01'
    print("model cpt")
    print(model_series)

    max_res = np.max(resid[~np.isnan(resid)]) 
    min_res = np.min(resid[~np.isnan(resid)]) 
    res_series = str(min_res) + '/' + str(max_res*1.5) +'/' + str((max_res - min_res)/100)
    print("res cpt")
    print(res_series)

    max_data = np.max(unw[~np.isnan(unw)].flatten()) 
    print(max_data)
    min_data = np.min(unw[~np.isnan(unw)].flatten()) 
    data_series = str(min_data) + '/' + str(max_data*1.5) +'/' + str((max_data - min_data)/100)
    print("data cpt")
    print(data_series)



    fig = pygmt.Figure()
    pygmt.config(MAP_FRAME_TYPE="plain")
    pygmt.config(FORMAT_GEO_MAP="ddd.xx")
    pygmt.config(FORMAT_FLOAT_OUT='%.12lg') 
    # print(file_path_data.split('/')[-1])   
    cmap_output_los = './InSAR_CPT_los.cpt'
    pygmt.makecpt(cmap='polar',series=model_series, continuous=True,output=cmap_output_los) 
    cmap_output_data =  './InSAR_CPT_data.cpt'
    pygmt.makecpt(cmap='polar',series=data_series, continuous=True,output=cmap_output_data) 
    cmap_output_res = './InSAR_CPT_res.cpt'
    pygmt.makecpt(cmap='polar',series=res_series, continuous=True,output=cmap_output_res) 
    with fig.subplot(
        nrows=2,
        ncols=3,
        figsize=("45c", "45c","45c"),
        autolabel=False,
        frame=["f","WSne","+tData"],
        margins=["0.05c", "0.05c"],
        # title="Geodetic Moderling Sequence one 2023/09/25 - 2023/10/08",
    ):

        # Configuration for the 'current figure'.
    
        # ,rose=["JBL+w5c+f2+l"]
        # print(max(los))
        fig.grdimage(grid=file_path_los,cmap=cmap_output_data,region=region,projection='M?c',panel=[0,0])
        fig.basemap(frame=['a','+tData'],panel=[0,0],region=region,projection='M?c')
        fig.grdimage(grid=file_path_model,cmap=cmap_output_data,region=region,projection='M?c',panel=[0,1])
        fig.basemap(frame=['xa','+tModel'],panel=[0,1],region=region,projection='M?c')
        fig.grdimage(grid=file_path_res,cmap=cmap_output_data,region=region,projection='M?c',panel=[0,2])
        fig.basemap(frame=['xa','+tResidual'],panel=[0,2],region=region,projection='M?c')

        for ii in range(0,3):
            fig.plot(x=lons_vertex[:-2],
                        y=lats_vertex[:-2],
                        pen='2p,black',
                        fill='gray',
                        transparency=80,
                        region=region,
                        projection='M?c',
                        panel=[0,ii]
            )

            fig.plot(x=lons_vertex[len(lons_vertex)-2:len(lons_vertex)],
                        y=lats_vertex[len(lats_vertex)-2:len(lats_vertex)],
                        pen='2p,black,-',
                        fill='gray',
                        transparency=80,
                        region=region,
                        projection='M?c',
                        panel=[0,ii]
            )

          
        for ii in range(0,3):
                fig.colorbar(frame=["x+lLOS displacment(m)", "y+lm"], position="JMB",projection='M?c',panel=[0,ii]) # ,panel=[1,0]
     



        fig.grdimage(grid=file_path_los,cmap=cmap_output_data,region=region,projection='M?c',panel=[1,0])
        # fig.basemap(frame=['a','+tData'],panel=[0,0],region=region,projection='M?c')
        fig.grdimage(grid=file_path_model_usgs,cmap=cmap_output_data,region=region,projection='M?c',panel=[1,1])
        # fig.basemap(frame=['xa','+tModel USGS'],panel=[0,1],region=region,projection='M?c')
        fig.grdimage(grid=file_path_res_usgs,cmap=cmap_output_data,region=region,projection='M?c',panel=[1,2])
        # fig.basemap(frame=['xa','+tResidual'],panel=[0,2],region=region,projection='M?c')
        
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
                        panel=[1,ii]
            )

            fig.plot(x=fault_end[0,:],
                        y=fault_end[1,:],
                        pen='2p,black,-',
                        fill='gray',
                        transparency=80,
                        region=region,
                        projection='M?c',
                        panel=[1,ii]
            )
         
        for ii in range(0,3):
                fig.colorbar(frame=["x+lLOS displacment(m)", "y+lm"], position="JMB",projection='M?c',panel=[1,ii]) # ,panel=[1,0]
        fig.show(method='external')
        fig.savefig('./gmt_output.png')


    return 


if __name__ == "__main__":
    # vertex_path = '/Users/jcondon/phd/code/auto_inv/us7000g9zq/vertex.mat'

    # # GBIS_GMT_OUTPUT('/Users/jcondon/phd/code/auto_inv/us6000jk0t/invert_1_F/Figures/res_los_modlos_lonlat072A_20230108_20230213.ds_unw_Lon_Lat_Inc_Heading.GBIS.mat',vertex_path)
    # output_matrix = io.loadmat('/Users/jcondon/phd/code/auto_inv/us7000g9zq/invert_1_F/invert_1_F.mat',squeeze_me=True)
    # opt_model = output_matrix['invResults'][()].item()[-1]
    # unw_file = '/Users/jcondon/phd/code/auto_inv/us7000g9zq_insar_processing/GEOC_128A_05172_131313_floatml_masked_GACOS_Corrected_clipped/20211224_20220117/20211224_20220117.unw'
    # Inc_file = '/Users/jcondon/phd/code/auto_inv/us7000g9zq_insar_processing/GEOC_128A_05172_131313_floatml_masked_GACOS_Corrected_clipped/theta.geo'
    # Head_file = '/Users/jcondon/phd/code/auto_inv/us7000g9zq_insar_processing/GEOC_128A_05172_131313_floatml_masked_GACOS_Corrected_clipped/phi.geo'
    # EQA_dem_par = '//Users/jcondon/phd/code/auto_inv/us7000g9zq_insar_processing/GEOC_128A_05172_131313_floatml_masked_GACOS_Corrected_clipped/EQA.dem_par'
    # locations = [101.29,37.8283]

    # depth = 13000.0 
    # strike1 = 13.0
    # dip1 = 75.41
    # rake1 = 177.86
    # strike2 = 103.54
    # dip2 = 87.93
    # rake2 = 14.6
    # mu = 3.2e10
    # slip_rate=5.5e-5
    # moment = 1.05e+19
    # width = np.cbrt(float(moment)/(slip_rate*mu))
    # length = width 
    # slip = length * slip_rate
    # usgs_model = [ 0,0,strike2,dip2,rake2,slip,length,depth,width]
    # # locations = [37.8283,101.29]
    # GBIS_GMT_OUTPUT_FORWARD_MODEL(vertex_path,
    #                               opt_model,
    #                               EQA_dem_par,
    #                               unw_file,
    #                               Inc_file,
    #                               Head_file,
    #                               locations,
    #                               usgs_model)
    




    vertex_path = '/Users/jcondon/phd/code/auto_inv/us6000jk0t/vertex_list.mat'

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
    GBIS_GMT_OUTPUT_FORWARD_MODEL(vertex_path,
                                  opt_model,
                                  EQA_dem_par,
                                  unw_file,
                                  Inc_file,
                                  Head_file,
                                  locations,
                                  usgs_model)