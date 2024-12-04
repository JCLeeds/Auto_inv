import pygmt
import numpy as np

def location_plot(input_txt_file,poly_list,save_location):
    with open(input_txt_file,'r') as file:
        params = file.readlines()

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

    # list_poly_files = [poly_file]
    frame_coords = []
    total_max_lat = [] 
    total_min_lat = [] 
    total_max_lon = [] 
    total_min_lon = []

    for poly_file in poly_list:
        print(poly_file)
        with open(poly_file[0],'r') as file:
            latlons = file.readlines()
        x = [] 
        y = [] 
        for latlon in latlons:
            y.append(float(latlon.split(' ')[1]))
            x.append(float(latlon.split(' ')[0]))
        y.append(float(latlons[0].split(' ')[1]))    
        x.append(float(latlons[0].split(' ')[0]))
        total_min_lon.append(np.min(x))
        total_max_lon.append(np.max(x))
        total_min_lat.append(np.min(y))
        total_max_lat.append(np.max(y))


        frame_coords.append([x,y,poly_file[0].split('/')[-1].split('-')[0]])

    min_lon = np.min(total_min_lon)
    max_lon = np.max(total_max_lon)
    min_lat = np.min(total_min_lat)
    max_lat = np.max(total_max_lat)

            
    fig = pygmt.Figure() 
    region = [min_lon-0.5, max_lon+0.5, min_lat-0.5, max_lat+0.5]

    fig.basemap(
        region=region,
        projection="M12c",  # Mercator projection with a width of 12 centimeters
        frame="a",
        map_scale="jBL+w10k+o0.5c/0.5c+f+lkm")

    grid_map = pygmt.datasets.load_earth_relief(
    resolution="15s",
    region=region,
    )
    fig.grdimage(grid=grid_map, cmap="dem3")

    fig.plot(
        data="gem_active_faults_harmonized.gmt",
        style="f1c/0.1c+l+t",
        fill="black",
        pen="0.5p,black",
        projection="M12c",
        region=region,
        )
    
    color_array = ['red','blue','green','yellow','brown','lightblue','gray','lightgrey','black']
    ii = 0
    edge_frames = []
    for frame in frame_coords:
        fig.plot(x=frame[0],
                y=frame[1],
                label=frame[2],
            # color="black",
            pen="1p," + color_array[ii],
            projection="M12c",
            region=region,
            )
        ii += 1
        if np.max(frame[1]) - 0.2 >= latitude and np.min(frame[1]) + 0.2 <= latitude:
            edge_frames.append(frame[2])

    fig.colorbar(
    # Place the colorbar inside the plot (lower-case "j") with justification
    # Bottom Right and an offset ("+o") of 0.7 centimeters and
    # 0.3 centimeters in x or y directions, respectively
    # Move the x label above the horizontal colorbar ("+ml")
    position="jBR+o0.7c/0.8c+h+w5c/0.3c+ml",
    # Add a box around the colobar with a fill ("+g") in "white" color and
    # a transparency ("@") of 30 % and with a 0.8-points thick black
    # outline ("+p")
    box="+gwhite@30+p0.8p,black",
    # Add x and y labels ("+l")
    frame=["x+lElevation", "y+lm"],
    )
    fig.legend(position='JTR+jTR+o0.2c', box='+gwhite+p1p',region=region, projection='M12c')




    USGS_mecas_seq_1 = dict(
        strike =[strike1],
        dip=[dip1],
        rake=[rake1],
        magnitude=[magnitude],
    )
    fig.meca(
            spec=USGS_mecas_seq_1, # <<< use dictionary
            scale="0.5c", 
            longitude=[longitude], # event longitude
            latitude=[latitude], # event latitude
            depth=[depth],
            event_name=['USGS'],
            compressionfill ='blue',
            labelbox=True,
            # perspective=perspective,
            region=region,
            projection='M12c',    
    )


    projection='G' + str(longitude)  + '/' + str(latitude) +'/' + '3c'
    with fig.inset(
    position="jTL+o0.1c",
    box="+gwhite+p1p",
    region='g',
    projection=projection,
    # 
    ):  
       
        fig.coast(
        projection=projection, region="g", frame="g", land="gray",borders="1/0.4p,black"
        )
        rectangle = [[region[0], region[2], region[1], region[3]]]
        fig.plot(data=rectangle, style="r+s", pen="0.75p,red")


    # fig.show(method='external')
    fig.savefig(save_location)
    return edge_frames

if __name__ == '__main__':
    location_plot('/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/working_6/us7000gebb_insar_processing/us7000gebb.txt',[['/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/test_poly.txt'],['/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/test_frame.txt']],'test.png')
