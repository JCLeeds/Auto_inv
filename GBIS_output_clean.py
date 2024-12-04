import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde
import seaborn as sns
from scipy.stats import binned_statistic_2d
import llh2local 
import local2llh 
import pygmt






def calculate_bins(data):
        """Calculates the number of bins based on the size of the data."""
        n = len(data)
        bins = max(1, int(n ** 0.5))
        return bins



# Custom pairplot function with hexbin
def hexbin_pairgrid(x, y, **kwargs):
    plt.hexbin(x, y, gridsize=50, cmap='Blues')

def generate_GBIS_plots(filepath,burn_in,no_iterations,num_faults,out_dir):

    inv_results = loadmat(filepath, squeeze_me=True)
    models_total = inv_results['invResults'][()].item()[0].T
    model_indices = np.arange(burn_in, no_iterations, dtype=int)
    opt_models = inv_results['invResults'][()].item()[3].T
    opt_models = np.array(opt_models).flatten().T
    geo = inv_results['geo'][()].item()[0]
    print(np.shape(opt_models))

    # model_random_indices = np.random.choice(model_indices, size=int(no_models), replace=False)
    # models = models_total[model_indices, :]
    models= models_total

   
    params_each = 9 # subtract ramp and constant params 
    



    for ii in range(num_faults):
        if num_faults > 1:
            opt_model = opt_models[ii]
        else:
            opt_model = opt_models
        # print(np.shape(opt_model))
        print(opt_model)
        length = models[:,0+params_each*ii]
        width = models[:,1+params_each*ii]
        top_depth = models[:,2+params_each*ii] # this gives top depth, middle depth is given as w/2cos(dip) + depth
        dip = -models[:,3+params_each*ii]
        middle_depth =  top_depth + ((width/2)*np.sin(dip*(np.pi/180))) # this may need editing for sin or cos depending on dip check 
        bottom_depth =  top_depth + ((width)*np.sin(dip*(np.pi/180))) 
        strike = (models[:,4+params_each*ii] +180) % 360 
        X = models[:,5+params_each*ii]
        Y = models[:,6+params_each*ii]
        SS = models[:,7+params_each*ii]
        DS = models[:,8+params_each*ii]
        rake = np.degrees(np.arctan2(-DS,-SS))
        total_slip = np.sqrt(SS**2 + DS**2)
        mu = 3.2e10
        M0_assuming_mu = mu*length*width*total_slip
        Mw = (2/3)*np.log10(M0_assuming_mu*10**7) - 10.7 
        
        XY = np.array([X,Y,top_depth]) * 0.001
        # XY = np.array([[np.max(X)],[np.max(Y)]]) * 0.001
        

        print(np.shape(XY))
        print(np.shape([[1,2,3],[1,2,3]]))
        print(XY)
        Lat_lon_top_cent = local2llh.local2llh(XY, geo)[0:2,:]
        Lat_top_cent = Lat_lon_top_cent[1,:]
        Lon_top_cent = Lat_lon_top_cent[0,:]
        x_centroid = X  - (width/2) * np.cos(dip*(np.pi/180)) * np.cos(strike*(np.pi/180))  # Horizontal shift along x
        y_centroid = Y - (width/2) * np.cos(dip*(np.pi/180)) * np.sin(strike*(np.pi/180))  # Horizontal shift along y
        XY_cent = np.array([x_centroid,y_centroid,middle_depth]) * 0.001
        cent_lat_lon = local2llh.local2llh(XY_cent, geo)[0:2,:]
        cent_lat = cent_lat_lon[1,:]
        cent_long = cent_lat_lon[0,:]


        length_opt = opt_model[0]
        width_opt = opt_model[1]
        top_depth_opt = opt_model[2] # this gives top depth, middle depth is given as w/2cos(dip) + depth
        dip_opt = -opt_model[3]
        middle_depth_opt =  top_depth_opt + ((width_opt/2)*np.sin(dip_opt*(np.pi/180))) # this may need editing for sin or cos depending on dip check 
        bottom_depth_opt =  top_depth_opt + ((width_opt)*np.sin(dip_opt*(np.pi/180))) 
        strike_opt = (opt_model[4] +180) % 360 
        X_opt = opt_model[5]
        Y_opt = opt_model[6]
        SS_opt = opt_model[7]
        DS_opt = opt_model[8]
        rake_opt = np.degrees(np.arctan2(-DS_opt,-SS_opt))
        total_slip_opt = np.sqrt(SS_opt**2 + DS_opt**2)
        mu = 3.2e10
        M0_assuming_mu_opt = mu*length_opt*width_opt*total_slip_opt
        Mw_opt = (2/3)*np.log10(M0_assuming_mu_opt*10**7) - 10.7 
        XY_opt = np.array([[X_opt],[Y_opt]])*0.001
        Lat_lon_top_cent_opt = local2llh.local2llh(XY_opt, geo)[0:2,:]
        Lat_top_cent_opt = Lat_lon_top_cent_opt[1,:]
        Lon_top_cent_opt = Lat_lon_top_cent_opt[0,:]
        x_centroid_opt = X_opt  - (width_opt/2) * np.cos(dip_opt*(np.pi/180)) * np.cos(strike_opt*(np.pi/180))  # Horizontal shift along x
        y_centroid_opt = Y_opt - (width_opt/2) * np.cos(dip_opt*(np.pi/180)) * np.sin(strike_opt*(np.pi/180))  # Horizontal shift along y
        XY_cent_opt = np.array([[x_centroid_opt],[y_centroid_opt]]) * 0.001
        cent_lat_lon_opt = local2llh.local2llh(XY_cent_opt, geo)[0:2,:]
        cent_lat_opt = cent_lat_lon_opt[1,:]
        cent_long_opt = cent_lat_lon_opt[0,:]
    
        
        optimal_model = [length_opt,width_opt,top_depth_opt,middle_depth_opt,bottom_depth_opt,dip_opt,strike_opt,cent_long_opt,cent_lat_opt,total_slip_opt,Mw_opt,rake_opt]
        print(optimal_model)

        df_data = np.array([length,width,top_depth,middle_depth,bottom_depth,dip,strike,cent_long,cent_lat,total_slip,Mw,rake]).T
        df_data = df_data[0:no_iterations-10,:]
        
        # Create DataFrame
        columns = ['Length', 'Width', 'T Depth', 'M Depth', 'B Depth', 'Dip', 'Strike', 'cent_lon', 'cent_lat', 'Slip', 'Mw', 'Rake']
        units = ['m','m','m','m','m','degrees','degrees','degrees','degrees','m','Mw','degrees','m']
        df = pd.DataFrame(df_data, columns=columns)
        df_clipped = df[burn_in:len(df_data)]
        # df = df[0:no_iterations-10]



        # Create a single figure with subplots for histograms
        n_cols = 3  # Number of columns for the subplots
        n_rows = (len(df.columns) + n_cols - 1) // n_cols  # Calculate number of rows needed
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))  # Create subplots

        # Flatten the axs array for easy indexing
        axs = axs.flatten()

        # Plotting histograms for all columns in the DataFrame
        # print(len(df_clipped.columns))
        # print(len(opt_model))
        for i, column in enumerate(df_clipped.columns):
            bins = calculate_bins(df[column])  # Get dynamic bins based on the data size
            sns.histplot(df_clipped[column], bins=50, ax=axs[i], color='blue', alpha=0.5, kde=True)
            print(optimal_model[i])
            axs[i].axvline(optimal_model[i], color='red', linestyle='--', label=f'optimal {column}')
            axs[i].set_title(f'{column}')
            axs[i].set_xlabel(units[i])
            axs[i].set_ylabel('Frequency')
            # axs[i].grid(axis='y', alpha=0.75)
            # Set x-ticks to show only the minimum and maximum values
            # axs[i].set_xlim(np.nanmin(df_data[:, i]), np.nanmax(df_data[:, i]))


        # Adjust layout
        plt.tight_layout()
        # plt.show()
        save_location = out_dir + '/pdf_fault' +str(ii)+ '.png'
        plt.savefig(save_location)


        n_variables = df_data.shape[1]
        # Create a figure and subplots
        # fig, axes = plt.subplots(n_variables, 1, figsize=(10, 20))
        # fig.tight_layout(pad=4.0)

        n_cols = 3  # Number of columns for the subplots
        n_rows = (len(df.columns) + n_cols - 1) // n_cols  # Calculate number of rows needed
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))  # Create subplots

        # Flatten the axs array for easy indexing
        axes = axes.flatten()

        # Loop through each variable and plot
        for i in range(n_variables):
            x = np.arange(0,len(df_data[:, i]),1)
            axes[i].scatter(x,df_data[:, i], label=columns[i],s=0.001,color='red')
            axes[i].axvline(burn_in, color='black', linestyle='--', label=f'Index {burn_in}')
            axes[i].set_title(columns[i])
            axes[i].set_ylabel(columns[i])
            axes[i].set_xlabel('Iteration')
            axes[i].set_ylim(np.nanmin(df_data[:, i])-0.1*np.nanmax(df_data[:, i]), np.nanmax(df_data[:, i])+0.1*np.nanmax(df_data[:, i]))
            axes[i].legend()

        # Adjust layout
        plt.tight_layout()
        save_location = out_dir + '/convergence_fault'+str(ii)+ '.png'
        plt.savefig(save_location)
        # plt.show()

                # Define the number of bins you want (e.g., 10)
        df_clipped = df_clipped.replace([np.inf, -np.inf], np.nan).dropna()
        # df_clipped = df_clipped.sample(int(len(df_clipped)*0.1))
    


        

        print(len(df_clipped))
        # Create a PairGrid for KDE in the lower triangle and histograms on the diagonal
        # g = sns.PairGrid(df_clipped,height=20,aspect=2,despine=True,layout_pad=2)
        g = sns.PairGrid(df_clipped,height=1,aspect=1, despine=True, layout_pad=1)
        print('here 1')
        # Plot KDEs on the lower triangle
        g.map_lower(plot_2d_histogram)
        print('here 2')
        # Plot histograms on the diagonal
        g.map_diag(sns.histplot, bins=50, kde=True)
        print('here 3')
        # Turn off the upper triangle using a mask
        g.map_upper(lambda *args, **kwargs: plt.gca().set_visible(False))
        print('here 4')
        # Adjust layout for better visualization
        plt.tight_layout()
        save_location = out_dir + '/probability-density-distributions_fault'+str(ii)+ '.png'
        plt.savefig(save_location)
        # plt.show()
        



#         # Dictionary to hold pre-computed KDE values
#         kde_values = {}
#         print('About to compute KDEs')
# # Compute KDE for each pair of columns
#         for i in range(df_clipped.shape[1]):
#             for j in range(i + 1, df_clipped.shape[1]):
#                 x = df_clipped.iloc[:, i]
#                 y = df_clipped.iloc[:, j]
#                 kde_values[(i, j)] = compute_kde(x, y)
#         print('KDEs Done')
#         plot_pairplot_with_kde(kde_values, df_clipped)
#         plt.tight_layout()
#         plt.show()
#         # binned_pairplot(df_clipped, bins=bins)

# Function to compute KDE for a pair of variables

def generate_summary(filepath,burn_in,no_iterations,num_faults,output_dir):
    inv_results = loadmat(filepath, squeeze_me=True)
    models_total = inv_results['invResults'][()].item()[0].T
    model_indices = np.arange(burn_in, no_iterations, dtype=int)
    opt_models = inv_results['invResults'][()].item()[3].T
    opt_models = np.array(opt_models).flatten().T
    geo = inv_results['geo'][()].item()[0]
    print(np.shape(opt_models))

    # model_random_indices = np.random.choice(model_indices, size=int(no_models), replace=False)
    # models = models_total[model_indices, :]
    models= models_total

   
    params_each = 9 # subtract ramp and constant params 
    params_each = 9 # subtract ramp and constant params 

    for ii in range(num_faults):
        if num_faults > 1:
            opt_model = opt_models[ii]
        else:
            opt_model = opt_models
        # print(np.shape(opt_model))
        print(opt_model)
        length = models[:,0+params_each*ii]
        width = models[:,1+params_each*ii]
        top_depth = models[:,2+params_each*ii] # this gives top depth, middle depth is given as w/2cos(dip) + depth
        dip = -models[:,3+params_each*ii]
        middle_depth =  top_depth + ((width/2)*np.sin(dip*(np.pi/180))) # this may need editing for sin or cos depending on dip check 
        bottom_depth =  top_depth + ((width)*np.sin(dip*(np.pi/180))) 
        strike = (models[:,4+params_each*ii] +180) % 360 
        X = models[:,5+params_each*ii]
        Y = models[:,6+params_each*ii]
        SS = models[:,7+params_each*ii]
        DS = models[:,8+params_each*ii]
        rake = np.degrees(np.arctan2(-DS,-SS))
        total_slip = np.sqrt(SS**2 + DS**2)
        mu = 3.2e10
        M0_assuming_mu = mu*length*width*total_slip
        Mw = (2/3)*np.log10(M0_assuming_mu*10**7) - 10.7 
        
        XY = np.array([X,Y,top_depth]) * 0.001
        # XY = np.array([[np.max(X)],[np.max(Y)]]) * 0.001
        

        print(np.shape(XY))
        print(np.shape([[1,2,3],[1,2,3]]))
        print(XY)
        Lat_lon_top_cent = local2llh.local2llh(XY, geo)[0:2,:]
        Lat_top_cent = Lat_lon_top_cent[1,:]
        Lon_top_cent = Lat_lon_top_cent[0,:]
        x_centroid = X  - ((width/2) * np.cos(dip*(np.pi/180)) * np.cos(strike*(np.pi/180)))  # Horizontal shift along x
        y_centroid = Y - ((width/2) * np.cos(dip*(np.pi/180)) * np.sin(strike*(np.pi/180))) # Horizontal shift along y
        XY_cent = np.array([x_centroid,y_centroid,middle_depth]) * 0.001
        cent_lat_lon = local2llh.local2llh(XY_cent, geo)[0:2,:]
        cent_lat = cent_lat_lon[1,:]
        cent_long = cent_lat_lon[0,:]


        length_opt = opt_model[0]
        width_opt = opt_model[1]
        top_depth_opt = opt_model[2] # this gives top depth, middle depth is given as w/2cos(dip) + depth
        dip_opt = -opt_model[3]
        middle_depth_opt =  top_depth_opt + ((width_opt/2)*np.sin(dip_opt*(np.pi/180))) # this may need editing for sin or cos depending on dip check 
        bottom_depth_opt =  top_depth_opt + ((width_opt)*np.sin(dip_opt*(np.pi/180))) 
        strike_opt = (opt_model[4] +180) % 360 
        X_opt = opt_model[5]
        Y_opt = opt_model[6]
        SS_opt = opt_model[7]
        DS_opt = opt_model[8]
        rake_opt = np.degrees(np.arctan2(-DS_opt,-SS_opt))
        total_slip_opt = np.sqrt(SS_opt**2 + DS_opt**2)
        mu = 3.2e10
        M0_assuming_mu_opt = mu*length_opt*width_opt*total_slip_opt
        Mw_opt = (2/3)*np.log10(M0_assuming_mu_opt*10**7) - 10.7 
        XY_opt = np.array([[X_opt],[Y_opt]])*0.001
        Lat_lon_top_cent_opt = local2llh.local2llh(XY_opt, geo)[0:2,:]
        Lat_top_cent_opt = Lat_lon_top_cent_opt[1,:]
        Lon_top_cent_opt = Lat_lon_top_cent_opt[0,:]
        x_centroid_opt = X_opt  - ((width_opt/2) * np.cos(dip_opt*(np.pi/180)) * np.cos(strike_opt*(np.pi/180)))  # Horizontal shift along x
        y_centroid_opt = Y_opt - ((width_opt/2) * np.cos(dip_opt*(np.pi/180)) * np.sin(strike_opt*(np.pi/180)))  # Horizontal shift along y
        XY_cent_opt = np.array([[x_centroid_opt],[y_centroid_opt]]) * 0.001
        cent_lat_lon_opt = local2llh.local2llh(XY_cent_opt, geo)[0:2,:]
        cent_lat_opt = cent_lat_lon_opt[1,:][0]
        cent_long_opt = cent_lat_lon_opt[0,:][0]
    
        
        optimal_model = [length_opt,width_opt,top_depth_opt,middle_depth_opt,bottom_depth_opt,dip_opt,strike_opt,cent_long_opt,cent_lat_opt,total_slip_opt,Mw_opt,rake_opt]
        # print(optimal_model)

        df_data = np.array([length,width,top_depth,middle_depth,bottom_depth,dip,strike,cent_long,cent_lat,total_slip,Mw,rake]).T
        df_data = df_data[0:no_iterations-10,:]
        
        # Create DataFrame
        columns = ['Length', 'Width', 'T Depth', 'M Depth', 'B Depth', 'Dip', 'Strike', 'cent_lon', 'cent_lat', 'Slip', 'Mw', 'Rake']
        units = ['m','m','m','m','m','degrees','degrees','degrees','degrees','m','Mw','degrees','m']
        df = pd.DataFrame(df_data, columns=columns)
        df_clipped = df[burn_in:len(df_data)]
   
        save_location = output_dir + '/summary_' + str(burn_in) + 'Fault'+str(ii)+ '.txt'
        fileID = open(save_location, "w") 
        # plt.savefig(save_location)
        fileID.write("Par Name\t Optimal Value \t Mean_value \t Median_value \t 2.5% \t 97.5%  \n")

        for jj in range(len(columns)):
            par_name = columns[jj]
            optimal_value = optimal_model[jj]
            if isinstance(optimal_value,list):
                optimal_value = optimal_value[0]
    
            # Extract the relevant slice for statistical calculations
            # data_slice = invResults["mKeep"][i, burning: -blankCells]
            data_slice = df_clipped[par_name].values
            
            # Calculate statistics
            mean_value = np.mean(data_slice)
            median_value = np.median(data_slice)
            percentile_2_5 = np.percentile(data_slice, 2.5)
            percentile_97_5 = np.percentile(data_slice, 97.5)
            print(mean_value)
            print(median_value)
            print(percentile_2_5)
            print(percentile_97_5)
            print(par_name)
            print(optimal_value)

            # model_m_value = model["m"][jj]

            # Write to file with formatting
            fileID.write(f"{par_name:12}\t {float(optimal_value):8.3f}\t {float(mean_value):8.3f}\t "
                        f"{float(median_value):8.3f}\t {float(percentile_2_5):8.3f}\t {float(percentile_97_5):8.3f}\t\r\n")

        fileID.close()  # Close the file after writing



def plot_2d_histogram(x, y, **kwargs):
    # Calculate the counts and bins for histogram
    bins = calculate_bins(x) 
    counts, x_edges, y_edges = np.histogram2d(x, y, bins=100)
    max_count = np.max(counts)  # Find the maximum count
    
    # Automatically set pmax to 90% of the max count
    pmax_value = np.percentile(counts[counts > 0], 95) if np.any(counts > 0) else 0
    print(pmax_value)

    # Use histplot with calculated pmax
    sns.histplot(x=x, y=y, bins=100, pmax=pmax_value/100, cmap='viridis', color='blue')

def compute_kde(x, y, num_points=100):
    kde = gaussian_kde(np.vstack([x, y]))
    x_grid = np.linspace(np.min(x), np.max(x), num_points)
    y_grid = np.linspace(np.min(y), np.max(y), num_points)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = kde(np.vstack([X.ravel(), Y.ravel()]))
    return X, Y, Z.reshape(X.shape)

def plot_mod_los_res(filepath,filepath_results,output_dir,num_faults):
    inv_results = loadmat(filepath, squeeze_me=True)
    opt_models = inv_results['invResults'][()].item()[3].T
    opt_models = np.array(opt_models).flatten()
    geo = inv_results['geo'][()].item()[0]
    all_fault_planes = []
    if num_faults > 1:
        for model in opt_models:
            model = [model]
            print(model)
            print(geo)
            vertex_total_list = model_to_verticies(model,geo)
            all_fault_planes.append(vertex_total_list)
    else:
        model = [opt_models]
        vertex_total_list = model_to_verticies(model,geo)
        all_fault_planes.append(vertex_total_list)
    
    data = loadmat(filepath_results)
    los = np.asarray(data['los']).flatten()
    modLos_save = np.asarray(data['modLos_save']).flatten()
    residual = np.asarray(data['residual']).flatten()
    lonlat = np.asarray(data['ll'])
    lons = lonlat[:, 0]
    lats = lonlat[:, 1]
    region = [np.nanmin(lons),np.nanmax(lons),np.nanmin(lats),np.nanmax(lats)]
    #wrapped
    wrapped_los = np.mod(los,0.056/2)
    wrapped_res = np.mod(residual, 0.056/2)
    wrapped_model = np.mod(modLos_save,0.056/2)
    rms = np.round(np.sqrt(np.mean(np.square(residual[~np.isnan(residual)]))),decimals=4)
    rms_unw = np.round(np.sqrt(np.mean(np.square(los[~np.isnan(los)]))),decimals=4)

    # region = [61.2,62.7,33.8,35.3]
    #unwrapped 

    intervals = np.diff(lons)
    sample_lons = np.min(np.abs(intervals[intervals>0.001]))+0.005
    # Sort the intervals to help identify clusters
    sample_lats = sample_lons

    # intervals = np.diff(lats)
    # # Sort the intervals to help identify clusters
    # sample_lats = np.min(np.abs(intervals[intervals>0.001])) +0.005
    print('SAMPLE RATES')
    print(sample_lats,sample_lons)
    pygmt.xyz2grd(x=lons,y=lats,z=los,outgrid='los.unw.grd',region=region,spacing=(sample_lons ,sample_lats))
    pygmt.xyz2grd(x=lons,y=lats,z=residual,outgrid='residual.unw.grd',region=region,spacing=(sample_lons ,sample_lats))
    pygmt.xyz2grd(x=lons,y=lats,z=modLos_save,outgrid='model.unw.grd',region=region,spacing=(sample_lons ,sample_lats))

    pygmt.xyz2grd(x=lons,y=lats,z=wrapped_los,outgrid='wrapped_los.unw.grd',region=region,spacing=(sample_lons ,sample_lats))
    pygmt.xyz2grd(x=lons,y=lats,z=wrapped_res,outgrid='wrapped_residual.unw.grd',region=region,spacing=(sample_lons ,sample_lats))
    pygmt.xyz2grd(x=lons,y=lats,z=wrapped_model,outgrid='wrapped_model.unw.grd',region=region,spacing=(sample_lons ,sample_lats))

    fig = pygmt.Figure()
    pygmt.config(MAP_FRAME_TYPE="plain")
    pygmt.config(FORMAT_GEO_MAP="ddd.xx")
    pygmt.config(FORMAT_FLOAT_OUT='%.12lg')  
    max_los = np.max(los) 
    min_los = np.min(los) 
    if abs(min_los) > abs(max_los):
        range_value = np.abs(min_los)
    else:
        range_value = max_los

    # series = str(min_los) + '/' + str(max_los) +'/' +'0.001'
    roma = '/uolstore/Research/a/a285/homes/ee18jwc/code/colormaps/roma/roma.cpt'
    vik = '/uolstore/Research/a/a285/homes/ee18jwc/code/colormaps/vik/vik.cpt'
    pygmt.makecpt(cmap=roma,series=[0,0.056/2], continuous=True,output='Wrapped_CPT.cpt',background=True) 
    print([-range_value,range_value])
    pygmt.makecpt(cmap=vik,series=[-range_value,range_value], continuous=True,output='InSAR_CPT.cpt',background=True) 
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
        fig.grdimage(grid='los.unw.grd',cmap='InSAR_CPT.cpt',region=region,projection='M?c',panel=[0,0])
        fig.basemap(frame=['a','+tData (rms: '+str(rms_unw) + ')'],panel=[0,0],region=region,projection='M?c')
        fig.grdimage(grid='model.unw.grd',cmap='InSAR_CPT.cpt',region=region,projection='M?c',panel=[0,1])
        fig.basemap(frame=['xa','+tModel'],panel=[0,1],region=region,projection='M?c')
        fig.grdimage(grid='residual.unw.grd',cmap='InSAR_CPT.cpt',region=region,projection='M?c',panel=[0,2])
        fig.basemap(frame=['xa','+tResidual (rms: ' + str(rms)+ ')'],panel=[0,2],region=region,projection='M?c')


        for ii in range(0,3):
            for jj, vertex in enumerate(all_fault_planes):
                vertex = vertex[0]
                y_lat = np.array(vertex)[1,:]
                x_lon = np.array(vertex)[0,:]
                if len(all_fault_planes) > 1: 
                    fig.text(x=np.mean(x_lon[:-2]),
                            y=np.mean(y_lat[:-2]),
                            pen='0.2p,black',
                            # fill='gray',
                            text=str(jj),
                            # transparency=75,
                            region=region,
                            projection='M?c',
                            panel=[0,ii]
                )


                fig.plot(x=x_lon[:-2],
                            y=y_lat[:-2],
                            close=True,
                            pen='1p,black',
                            # fill='gray',
                            # transparency=75,
                            region=region,
                            projection='M?c',
                            panel=[0,ii]
                )

                fig.plot(x=x_lon[len(x_lon)-2:len(x_lon)],
                            y=y_lat[len(x_lon)-2:len(x_lon)],
                            pen='1p,black',
                            # fill='gray',
                            # transparency=75,
                            region=region,
                            projection='M?c',
                            panel=[0,ii]
                )

        for ii in range(0,3):
                fig.colorbar(frame=["x+lLOS displacment(m)", "y+lm"], position="JMB",projection='M?c',panel=[0,ii]) # ,panel=[1,0]


        fig.grdimage(grid='wrapped_los.unw.grd',cmap='Wrapped_CPT.cpt',region=region,projection='M?c',panel=[1,0])
        fig.basemap(frame=['a','+tData'],panel=[1,0],region=region,projection='M?c')
        fig.grdimage(grid='wrapped_model.unw.grd',cmap='Wrapped_CPT.cpt',region=region,projection='M?c',panel=[1,1])
        fig.basemap(frame=['xa','+tModel'],panel=[1,1],region=region,projection='M?c')
        fig.grdimage(grid='wrapped_residual.unw.grd',cmap='Wrapped_CPT.cpt',region=region,projection='M?c',panel=[1,2])
        fig.basemap(frame=['xa','+tResidual'],panel=[1,2],region=region,projection='M?c')

        for ii in range(0,3):
            for jj, vertex in enumerate(all_fault_planes):
                vertex = vertex[0]
                y_lat = np.array(vertex)[1,:]
                x_lon = np.array(vertex)[0,:]
                if len(all_fault_planes) > 1: 
                    fig.text(x=np.mean(x_lon[:-2]),
                            y=np.mean(y_lat[:-2]),
                            pen='0.2p,black',
                            # fill='gray',
                            text=str(jj),
                            # transparency=75,
                            region=region,
                            projection='M?c',
                            panel=[1,ii]
                )


                fig.plot(x=x_lon[:-2],
                            y=y_lat[:-2],
                            close=True,
                            pen='1p,black',
                            # fill='gray',
                            # transparency=75,
                            region=region,
                            projection='M?c',
                            panel=[1,ii]
                )

                fig.plot(x=x_lon[len(x_lon)-2:len(x_lon)],
                            y=y_lat[len(x_lon)-2:len(x_lon)],
                            pen='1p,black',
                            # fill='gray',
                            # transparency=75,
                            region=region,
                            projection='M?c',
                            panel=[1,ii]
                )
        for ii in range(0,3):
                fig.colorbar(frame=["x+lLOS displacment(m)", "y+lm"], position="JMB",projection='M?c',panel=[1,ii]) # ,panel=[1,0]




    # fig.show()
    filepath_out = filepath_results.split('/')[-1][:-4] 
    save_location = output_dir + '/' + filepath_out + '_data_mod_residual.png'
    fig.savefig(save_location)
def model_to_verticies(m,origin):
    origin = np.array(origin)  # Convert origin list to numpy array
    # print('length m ')
    # print(np.shape(m))
    # m is assumed to be a 2D numpy array
    vertex_total_list = []
    slipvectors = []
    centerpoint = []
    # print()
    # print(m)
    for model in m:
        # print(model)
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

            vertices = np.hstack((vertices, np.zeros((3, 2))))  # Add two more columns to vertices

            vertices[0, 4] = x_surf
            vertices[1, 4] = y_surf
            vertices[2, 4] = 0

            distance_x = vertices[0, 0] - vertices[0, 3]
            distance_y = vertices[1, 0] - vertices[1, 3]
            distance_z = vertices[2, 0] - vertices[2, 3]

            l = distance_x
            ym = distance_y
            y_surf_two = -(vertices[2, 3] * ym / n) + vertices[1, 3]
            x_surf_two = -(vertices[2, 3] * l / n) + vertices[0, 3]

            vertices[0, 5] = x_surf_two
            vertices[1, 5] = y_surf_two
            vertices[2, 5] = 0

            vertex_total_list.append(vertices)


    return vertex_total_list


 

def plot_location_and_profile(filepath,filepath_results,plot_title,num_faults,select_fault=1):

    data = loadmat(filepath_results)
    los = np.asarray(data['los']).flatten()
    modLos_save = np.asarray(data['modLos_save']).flatten()
    residual = np.asarray(data['residual']).flatten()
    lonlat = np.asarray(data['ll'])
    lons = lonlat[:, 0]
    lats = lonlat[:, 1]
    # print(lons.shape)
    # print(lats.shape)
    # print(los.shape)
    los = los[~np.isnan(los)]
    lons = lons[~np.isnan(los)]
    lats = lats[~np.isnan(los)]
    modLos_save = modLos_save[~np.isnan(los)]
    residual = residual[~np.isnan(los)]
    inv_results = loadmat(filepath, squeeze_me=True)
    opt_models = inv_results['invResults'][()].item()[3].T
    opt_models = np.array(opt_models).flatten()
    geo = inv_results['geo'][()].item()[0]
    
    all_fault_planes = []
    if num_faults > 1:
        for model in opt_models:
            model = [model]
            # print(model)
            # print(geo)
            vertex_total_list = model_to_verticies(model,geo)
            all_fault_planes.append(vertex_total_list)
    else:
        model = [opt_models]
        vertex_total_list = model_to_verticies(model,geo)
        all_fault_planes.append(vertex_total_list)
    
    if num_faults > 1:
        select_fault = 1 
   
    
    vertexs = np.array(all_fault_planes[int(select_fault-1)])
    vertexs = vertexs[0]
    # print(vertexs)
    x_lon_NP1 = vertexs[0,:]
    y_lat_NP1 = vertexs[1,:]
    z_depth_NP1 = vertexs[2,:]
    # print(x_lon_NP1)



    mid_top_x = (x_lon_NP1[0] + x_lon_NP1[2])/2
    mid_top_y = (y_lat_NP1[0] + y_lat_NP1[2])/2

    mid_bottom_x = (x_lon_NP1[1] + x_lon_NP1[3])/2
    mid_bottom_y = (y_lat_NP1[1] + y_lat_NP1[3])/2

    m = (mid_top_y-mid_bottom_y)/(mid_top_x-mid_bottom_x)
    c = mid_top_y - m*mid_top_x
    if mid_top_x > mid_bottom_x:
        low_bound = mid_bottom_x-0.05
        high_bound = mid_top_x+0.05
    else:
        low_bound = mid_top_x-0.04
        high_bound = mid_bottom_x+0.04

    low_bound = low_bound+0.05
    high_bound = high_bound-0.05

    x = np.random.uniform(low=low_bound, high=high_bound, size=(100,))
    y = m*x + c
    y_line_end_one = m*low_bound + c
    y_line_end_two = m*high_bound + c
    start = [ high_bound,y_line_end_two]
    end = [low_bound,y_line_end_one]
    # plt.scatter(x,y)
    # plt.scatter(low_bound,y_line_end_two,c='red')
    # plt.scatter(high_bound,y_line_end_one,c='green')
    # plt.show()
    

    print('######################################## 2D locations plot ##############################################################')
   

    region = [np.min(lons),np.max(lons),np.min(lats),np.max(lats)] 
   
    dem_region = [np.min(lons)-0.4,np.max(lons)+0.4,np.min(lats)-0.4,np.max(lats)+0.4] 

    
    file_path_data = 'profile_data_meters.grd'
    file_path_model = 'profile_model_meters.grd'
    file_path_res = 'profile_res_meters.grd'
   
    pygmt.xyz2grd(x=lons.flatten(),y=lats.flatten(),z=los.flatten(),outgrid=file_path_data,region=region,spacing=(0.01,0.01))
    print('surface project')
    pygmt.xyz2grd(x=lons.flatten(),y=lats.flatten(),z=modLos_save.flatten(),outgrid=file_path_model,region=region,spacing=(0.01,0.01))
    pygmt.xyz2grd(x=lons.flatten(),y=lats.flatten(),z=residual.flatten(),outgrid=file_path_res,region=region,spacing=(0.01,0.01))
   
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
 
    vik = '/uolstore/Research/a/a285/homes/ee18jwc/code/colormaps/vik/vik.cpt'
    pygmt.makecpt(cmap='oleron',series=topo_cpt_series, continuous=True,output=topo_output_cpt,background=True) 
    pygmt.makecpt(cmap=vik,series=data_series, continuous=True,output=cmap_output_data,background=True) 
    if np.abs(min_data) > np.abs(max_data):
        range_limit = np.abs(min_data)
    else:
        range_limit = np.abs(max_data) 

    print([-range_limit,range_limit])
    pygmt.makecpt(series=[-range_limit, range_limit], cmap=vik,output=cmap_output_data)
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
    generate=(0.005),
    outfile='line_across.txt'
    )

    print(np.min(x), np.max(y))
    print(np.max(x), np.min(y))

    pygmt.grdtrack(
            grid = file_path_data, 
            points = 'line_across.txt',
            outfile="tmp_data_profile.txt",
            skiprows=False,
            newcolname='profile',
            crossprofile='0.75/0.005/0.001+v',
            stack="m+sstack.txt"
    )

    pygmt.grdtrack(
            grid = file_path_model, 
            points = 'line_across.txt',
            outfile="tmp_model_profile.txt",
            skiprows=False,
            newcolname='profile',
            crossprofile='0.75/0.005/0.001+v',
            stack="m+sstack_model.txt"
    )

    fig.plot(data='tmp_data_profile.txt', projection="M10", pen=0.01)
    # fig.plot(data='stack.txt',projection="M10", pen=0.01)
    # fig.plot(data='tmp_model_profile.txt',projection="M10", pen=0.01)
    fig.plot(data='line_across.txt',projection="M10", pen=1,fill='red')
    stack_median = [] 
    stack_upper = []
    stack_lower = [] 
    stack_distance = [] 

    with open("stack.txt", 'r') as f:
        lines = f.readlines()
    f.close()

    for ii,line in enumerate(lines):
        stack_median.append(float(line.split()[1]))
        stack_distance.append(float(line.split()[0]))
        stack_upper.append(float(line.split()[4]))
        stack_lower.append(float(line.split()[3]))

    stack_upper = np.array(stack_upper)
    stack_lower=np.array(stack_lower)
    stack_distance=np.array(stack_distance)
    stack_median=np.array(stack_median)
    stack_upper = stack_upper[~np.isnan(stack_median)]
    stack_lower= stack_lower[~np.isnan(stack_median)]
    stack_distance= stack_distance[~np.isnan(stack_median)]
    stack_median = stack_median[~np.isnan(stack_median)]

    stack_median_model = [] 
    stack_upper_model = []
    stack_lower_model = [] 
    stack_distance_model = [] 

    with open("stack_model.txt", 'r') as f:
        lines = f.readlines()
    f.close()

    for ii,line in enumerate(lines):
        stack_median_model.append(float(line.split()[1]))
        stack_distance_model.append(float(line.split()[0]))
        stack_upper_model.append(float(line.split()[4]))
        stack_lower_model.append(float(line.split()[3]))

    stack_upper_model = np.array(stack_upper_model)
    stack_lower_model=np.array(stack_lower_model)
    stack_distance_model=np.array(stack_distance_model)
    stack_median_model=np.array(stack_median_model)
    
    stack_upper_model = stack_upper_model[~np.isnan(stack_median_model)]
    stack_lower_model = stack_lower_model[~np.isnan(stack_median_model)]
    stack_distance_model = stack_distance_model[~np.isnan(stack_median_model)]
    stack_median_model = stack_median_model[~np.isnan(stack_median_model)]

    
    df_bound = pd.DataFrame(
    data={
        "x": stack_distance,
        "y": stack_median,
        "y_bound_low": stack_lower,
        "y_bound_upp": stack_upper,
        }
        )

    df_bound_model = pd.DataFrame(
    data={
        "x": stack_distance_model,
        "y": stack_median_model,
        "y_bound_low": stack_lower_model,
        "y_bound_upp": stack_upper_model,
        }
        )

    # print(max(stack_upper))
    # print(max(stack_upper_model))
    x_NP1_project = []
    y_NP1_project = []
    z_NP1_project = [] 
    distance_NP1_project = [] 

    with open("tmp_data_profile.txt", 'r') as f:
        lines = f.readlines()
    f.close()
    for line in lines:
        # print(line.split(' '))
        if line.startswith('>'):
            pass 
        else:
            x_NP1_project.append(float(line.split()[0]))
            y_NP1_project.append(float(line.split()[1]))
            z_NP1_project.append(float(line.split()[3]))
            distance_NP1_project.append(float(line.split()[2]))



    fig.text(x=x_NP1_project[-1], y=y_NP1_project[-1], text="B", font="8,Helvetica",offset="0.75c/0c")
    fig.text(x=x_NP1_project[0], y=y_NP1_project[0], text="A", font="8,Helvetica",offset="0.75c/0c")
  
    #### Model small panel 
      
    fig.shift_origin(xshift=11.5)
    fig.shift_origin(yshift=5.7)
    mod_grd_ds = pygmt.grdsample(grid=file_path_model,spacing='03s',registration='gridline',region=region)
    fig.grdimage(grid=mod_grd_ds,cmap=cmap_output_data,region=region,projection='M5c',shading=gradiant_terrain)
    fig.basemap(
      
       frame=["WStr","ya0.5f1"],
       region=region,
       map_scale="jBL+w10k+o0.5c/0.5c+f+lkm",
       projection='M5c',  
    )
    map_scale="jBL+w10k+o0.5c/0.5c+f+lkm"
   
    fig.shift_origin(yshift=-5.7)
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
        projection="X16.5/6",
        region=[np.min(stack_distance), np.max(stack_distance), np.nanmin(los), np.nanmax(los)],
        frame=['xafg100+l"Distance (Degrees)"', 'yafg50+l"Displacement in Line-of-Sight (m)"', "WSen"],
        # yshift=-7,
        # xshift=-11.5
        
    )

    print(df_bound)
    fig.plot(
    data=df_bound,
    fill="gray@50",
    # Add an outline around the envelope
    # Here, a dashed pen ("+p") with 0.5-points thickness and
    # "gray30" color is used
    close="+b+p0.1p,gray30,dashed",
    pen="0.1p,gray30",
    projection='X',
    label='Data in Line-of-Sight (m)'
    )
 

    fig.plot(
    data=df_bound_model,
    fill="pink@70",
    # Add an outline around the envelope
    # Here, a dashed pen ("+p") with 0.5-points thickness and
    # "gray30" color is used
    close="+b+p0.1p,pink,dashed",
    pen="0.1p,pink",
    projection='X',
    label='Model in Line-of-Sight (m)'
    
    )


    fig.plot(data='stack.txt',style='c0.02',fill='gray30')
    fig.plot(data='stack_model.txt',style='c0.02',fill='pink')

    fig.text(x=np.min(stack_distance)+0.01, y=-0.05, text="B", font="10,Helvetica")
    fig.text(x=np.max(stack_distance)-0.01, y=-0.05, text="A", font="10,Helvetica")
    fig.legend(position='JTL+jTL+o0.2c')

    filepath_out = filepath_results.split('/')[-1][:-4] 
    save_location = output_dir + '/' + filepath_out + '_profile_plot_fault'+str(select_fault)+ '.png'
    fig.savefig(save_location)
    fig.show()


# Plot the pair plot with pre-computed KDE values

def main(filepath_invert,filepath_res_los_mod,output_dir,burn_in,no_iterations,num_faults=1):
    generate_GBIS_plots(filepath_invert,burn_in,no_iterations,num_faults,output_dir)
    for ii in range(len(filepath_res_los_mod)):
        plot_mod_los_res(filepath_invert,filepath_res_los_mod[ii],output_dir,num_faults)
        # plot_location_and_profile(filepath_invert,filepath_res_los_mod[ii],output_dir,num_faults,select_fault=1)
    generate_summary(filepath_invert,burn_in,no_iterations,num_faults,output_dir)

if __name__== '__main__':

    # # # Load data
    # filepath = '/uolstore/Research/a/a285/homes/ee18jwc/projects/afghanistan/Final_products_checked/seperation_of_event_on_7th_purfecto_temp_best/invert_1_2_F_F_F/invert_1_2_F_F_F.mat'
    # filepath_results = '/uolstore/Research/a/a285/homes/ee18jwc/projects/afghanistan/Final_products_checked/seperation_of_event_on_7th_purfecto_temp_best/invert_1_2_F_F_F/Figures/res_los_modlos_lonlat_xy_013A_20230925_20231007.unw.mat'
    # filepath_results_two = '/uolstore/Research/a/a285/homes/ee18jwc/projects/afghanistan/Final_products_checked/seperation_of_event_on_7th_purfecto_temp_best/invert_1_2_F_F_F/Figures/res_los_modlos_lonlat_xy_020D_20230926_20231008.unw.mat'
    # output_dir = '/uolstore/Research/a/a285/homes/ee18jwc/projects/afghanistan/Final_products_checked/seperation_of_event_on_7th_purfecto_temp_best/python_output'
    # burn_in = int(4e4)
    # no_iterations = int(1e5)
    # no_models = 3e4
    # num_faults = 3
    # generate_GBIS_plots(filepath,burn_in,no_iterations,num_faults,output_dir)
    # plot_mod_los_res(filepath,filepath_results,output_dir,num_faults)
    # plot_mod_los_res(filepath,filepath_results_two,output_dir,num_faults)
    # generate_summary(filepath,burn_in,no_iterations,num_faults)

    # # Events on the 7th as a signle plane 
    # filepath = '/uolstore/Research/a/a285/homes/ee18jwc/projects/afghanistan/Final_products_checked/Afphgan_seq1_NP1_width_unlimited_THIS_ONE_USED/invert_1_2_F/invert_1_2_F.mat'
    # filepath_results = '/uolstore/Research/a/a285/homes/ee18jwc/projects/afghanistan/Final_products_checked/Afphgan_seq1_NP1_width_unlimited_THIS_ONE_USED/invert_1_2_F/Figures/res_los_modlos_lonlat_xy_013A_20230925_20231007.unw.mat'
    # filepath_results_two = '/uolstore/Research/a/a285/homes/ee18jwc/projects/afghanistan/Final_products_checked/Afphgan_seq1_NP1_width_unlimited_THIS_ONE_USED/invert_1_2_F/Figures/res_los_modlos_lonlat_xy_020D_20230926_20231008.unw.mat'
    # output_dir = '/uolstore/Research/a/a285/homes/ee18jwc/projects/afghanistan/Final_products_checked/Afphgan_seq1_NP1_width_unlimited_THIS_ONE_USED/python_output'
    # burn_in = int(5e4)
    # no_iterations = int(1e5)
    # no_models = 3e4
    # num_faults = 1
    # generate_GBIS_plots(filepath,burn_in,no_iterations,num_faults,output_dir)
    # plot_mod_los_res(filepath,filepath_results,output_dir,num_faults)
    # plot_mod_los_res(filepath,filepath_results_two,output_dir,num_faults)
    # plot_location_and_profile(filepath,filepath_results,output_dir,num_faults,select_fault=1)
    # plot_location_and_profile(filepath,filepath_results_two,output_dir,num_faults,select_fault=1)
    # generate_summary(filepath,burn_in,no_iterations,num_faults)

    # # EVENTS ON 15th and 11th single plane

    # filepath = '/uolstore/Research/a/a285/homes/ee18jwc/projects/afghanistan/Final_products_checked/Afphgan_seq2_NP2_newdata/invert_1_2_F/invert_1_2_F.mat'
    # filepath_results = '/uolstore/Research/a/a285/homes/ee18jwc/projects/afghanistan/Final_products_checked/Afphgan_seq2_NP2_newdata/invert_1_2_F/Figures/res_los_modlos_lonlat_xy_20231007_20231019_custom_mask.mat'
    # filepath_results_two = '/uolstore/Research/a/a285/homes/ee18jwc/projects/afghanistan/Final_products_checked/Afphgan_seq2_NP2_newdata/invert_1_2_F/Figures/res_los_modlos_lonlat_xy_20231008_20231020.unw.01.unwraperr_anomremoved.mat'
    # output_dir = '/uolstore/Research/a/a285/homes/ee18jwc/projects/afghanistan/Final_products_checked/Afphgan_seq2_NP2_newdata/python_output'
    # burn_in = int(5e4)
    # no_iterations = int(1e5)
    # no_models = 3e4
    # num_faults = 1
    # generate_GBIS_plots(filepath,burn_in,no_iterations,num_faults,output_dir)
    # plot_mod_los_res(filepath,filepath_results,output_dir,num_faults)
    # plot_mod_los_res(filepath,filepath_results_two,output_dir,num_faults)
    # plot_location_and_profile(filepath,filepath_results,output_dir,num_faults,select_fault=1)
    # plot_location_and_profile(filepath,filepath_results_two,output_dir,num_faults,select_fault=1)
    # generate_summary(filepath,burn_in,no_iterations,num_faults)


    # # Event on the 15th single ifgm 
    # filepath = '/uolstore/Research/a/a285/homes/ee18jwc/projects/afghanistan/Final_products_checked/seperation_of_event_on_15th_long_run_realsies/invert_1_F/invert_1_F.mat'#
    # filepath_results = '/uolstore/Research/a/a285/homes/ee18jwc/projects/afghanistan/Final_products_checked/seperation_of_event_on_15th_long_run_realsies/invert_1_F/Figures/res_los_modlos_lonlat_xy_15th_seperated_reunwrapped.mat'
    # output_dir = '/uolstore/Research/a/a285/homes/ee18jwc/projects/afghanistan/Final_products_checked/seperation_of_event_on_15th_long_run_realsies/python_output'
    # burn_in = int(3e4)
    # no_iterations = int(1e5)
    # no_models = 3e4
    # num_faults = 1
    # generate_GBIS_plots(filepath,burn_in,no_iterations,num_faults,output_dir)
    # plot_mod_los_res(filepath,filepath_results,output_dir,num_faults)
    # plot_location_and_profile(filepath,filepath_results,output_dir,num_faults,select_fault=1)
    # generate_summary(filepath,burn_in,no_iterations,num_faults)
   

    # # Event on the 11th subtraction for fault 
    # filepath = '/uolstore/Research/a/a285/homes/ee18jwc/projects/afghanistan/Final_products_checked/seperation_of_event_on_11th_long_run/invert_1_2_F/invert_1_2_F.mat'
    # filepath_results = '/uolstore/Research/a/a285/homes/ee18jwc/projects/afghanistan/seperation_attempts/seq_2/seperation_of_event_on_11th_long_run/invert_1_2_F/Figures/res_los_modlos_lonlat_xy_013A_matlab_heavilymasked_15th_removed_attempttwo.mat'
    # filepath_results_two = '/uolstore/Research/a/a285/homes/ee18jwc/projects/afghanistan/seperation_attempts/seq_2/seperation_of_event_on_11th_long_run/invert_1_2_F/Figures/res_los_modlos_lonlat_xy_020D_matlab_heavilymasked_15th_removed.mat'
    # output_dir = '/uolstore/Research/a/a285/homes/ee18jwc/projects/afghanistan/seperation_attempts/seq_2/seperation_of_event_on_11th_long_run/invert_1_2_F/python_output'
    # burn_in = int(3e5)
    # no_iterations = int(1e6)
    # no_models = 3e4
    # num_faults = 1
    # generate_GBIS_plots(filepath,burn_in,no_iterations,num_faults,output_dir)
    # plot_mod_los_res(filepath,filepath_results,output_dir,num_faults)
    # plot_mod_los_res(filepath,filepath_results_two,output_dir,num_faults)
    # plot_location_and_profile(filepath,filepath_results,output_dir,num_faults,select_fault=1)
    # plot_location_and_profile(filepath,filepath_results_two,output_dir,num_faults,select_fault=1)
    # generate_summary(filepath,burn_in,no_iterations,num_faults)

    # #     # Load data
    # # filepath = '/uolstore/Research/a/a285/homes/ee18jwc/projects/afghanistan/seperation_attempts/seq_2/seperation_of_event_on_11th_two_faults/invert_1_2_F_F/invert_1_2_F_F.mat'
    # # filepath_results = '/uolstore/Research/a/a285/homes/ee18jwc/projects/afghanistan/seperation_attempts/seq_2/seperation_of_event_on_11th_two_faults/invert_1_2_F_F/Figures/res_los_modlos_lonlat_xy_20231007_20231019_custom_mask.mat'
    # # output_dir = '/uolstore/Research/a/a285/homes/ee18jwc/projects/afghanistan/figures/test_folder'
    # # burn_in = int(4e4)
    # # no_iterations = int(1e5)
    # # no_models = 3e4
    # # num_faults = 2
    # # generate_GBIS_plots(filepath,burn_in,no_iterations,num_faults,output_dir)
    # # plot_mod_los_res(filepath,filepath_results,output_dir)


    # # 11th and 15th MODLED WITH TWO FAULTS 
    # filepath = '/uolstore/Research/a/a285/homes/ee18jwc/projects/afghanistan/Final_products_checked/seperation_of_event_on_11th_15th_two_faults/invert_1_2_F_F/invert_1_2_F_F.mat'
    # filepath_results = '/uolstore/Research/a/a285/homes/ee18jwc/projects/afghanistan/Final_products_checked/seperation_of_event_on_11th_15th_two_faults/invert_1_2_F_F/Figures/res_los_modlos_lonlat_xy_20231007_20231019_custom_mask.mat'
    # filepath_results_two = '/uolstore/Research/a/a285/homes/ee18jwc/projects/afghanistan/Final_products_checked/seperation_of_event_on_11th_15th_two_faults/invert_1_2_F_F/Figures/res_los_modlos_lonlat_xy_corrected_20231008_20231020.mat'
    # output_dir = '/uolstore/Research/a/a285/homes/ee18jwc/projects/afghanistan/Final_products_checked/seperation_of_event_on_11th_15th_two_faults/invert_1_2_F_F/Figures/python_output'
    # burn_in = int(3e5)
    # no_iterations = int(1e6)
    # # no_models = 3e6
    # num_faults = 2
    # generate_GBIS_plots(filepath,burn_in,no_iterations,num_faults,output_dir)
    # plot_mod_los_res(filepath,filepath_results,output_dir,num_faults)
    # plot_mod_los_res(filepath,filepath_results_two,output_dir,num_faults)
    # plot_location_and_profile(filepath,filepath_results,output_dir + '/profile_plot_one.png',num_faults,select_fault=1)
    # plot_location_and_profile(filepath,filepath_results_two,output_dir,num_faults,select_fault=1)
    # plot_location_and_profile(filepath,filepath_results,output_dir,num_faults,select_fault=2)
    # plot_location_and_profile(filepath,filepath_results_two,output_dir,num_faults,select_fault=2)
    # generate_summary(filepath,burn_in,no_iterations,num_faults)


    # # Two fault solution to the 7th 
    # filepath  ='/uolstore/Research/a/a285/homes/ee18jwc/projects/afghanistan/seperation_attempts/seq_1/seperation_of_event_on_7th/invert_1_2_F_F/invert_1_2_F_F.mat'
    # filepath_results = '/uolstore/Research/a/a285/homes/ee18jwc/projects/afghanistan/seperation_attempts/seq_1/seperation_of_event_on_7th/invert_1_2_F_F/Figures/res_los_modlos_lonlat_xy_013A_20230925_20231007.unw.mat'
    # filepath_results_two = '/uolstore/Research/a/a285/homes/ee18jwc/projects/afghanistan/seperation_attempts/seq_1/seperation_of_event_on_7th/invert_1_2_F_F/Figures/res_los_modlos_lonlat_xy_020D_20230926_20231008.unw.mat'
    # output_dir = '/uolstore/Research/a/a285/homes/ee18jwc/projects/afghanistan/seperation_attempts/seq_1/seperation_of_event_on_7th/invert_1_2_F_F/python_output'
    # burn_in = int(3e4)
    # no_iterations = int(1e5)
    # # no_models = 3e6
    # num_faults = 2
    # generate_GBIS_plots(filepath,burn_in,no_iterations,num_faults,output_dir)
    # plot_mod_los_res(filepath,filepath_results,output_dir,num_faults)
    # plot_mod_los_res(filepath,filepath_results_two,output_dir,num_faults)
    # plot_location_and_profile(filepath,filepath_results,output_dir + '/profile_plot_one_fault1.png',num_faults,select_fault=1)
    # plot_location_and_profile(filepath,filepath_results_two,output_dir + '/profile_plot_two_fault1.png',num_faults,select_fault=1)
    # plot_location_and_profile(filepath,filepath_results,output_dir + '/profile_plot_one_fautl2.png',num_faults,select_fault=2)
    # plot_location_and_profile(filepath,filepath_results_two,output_dir + '/profile_plot_two_fault2.png',num_faults,select_fault=2)
    # generate_summary(filepath,burn_in,no_iterations,num_faults)

    # # 3 faults on the 7th workingish 

    # filepath = '/uolstore/Research/a/a285/homes/ee18jwc/projects/afghanistan/seperation_attempts/seq_1/seperation_of_event_on_7th_workingish_3Faults/invert_1_2_F_F_F/invert_1_2_F_F_F.mat'
    # filepath_results = '/uolstore/Research/a/a285/homes/ee18jwc/projects/afghanistan/seperation_attempts/seq_1/seperation_of_event_on_7th_workingish_3Faults/invert_1_2_F_F_F/Figures/res_los_modlos_lonlat_xy_013A_20230925_20231007.unw.mat'
    # filepath_results_two = '/uolstore/Research/a/a285/homes/ee18jwc/projects/afghanistan/seperation_attempts/seq_1/seperation_of_event_on_7th_workingish_3Faults/invert_1_2_F_F_F/Figures/res_los_modlos_lonlat_xy_020D_20230926_20231008.unw.mat'
    # output_dir = '/uolstore/Research/a/a285/homes/ee18jwc/projects/afghanistan/seperation_attempts/seq_1/seperation_of_event_on_7th_workingish_3Faults/python_output'
    # burn_in = int(3e4)
    # no_iterations = int(1e5)
    # # no_models = 3e6
    # num_faults = 3
    # generate_GBIS_plots(filepath,burn_in,no_iterations,num_faults,output_dir)
    # plot_mod_los_res(filepath,filepath_results,output_dir,num_faults)
    # plot_mod_los_res(filepath,filepath_results_two,output_dir,num_faults)

    # # auto_test 
    # filepath_res_los_mod = ['/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us6000jk0t_NP1/invert_3_5_8_10_11_12_15_F/Figures/res_los_modlos_lonlat_xy_GEOC_072A_05090_131313_floatml_masked_GACOS_Corrected_clipped_signal_masked_QAed_20230108_20230201.ds_unw_Lon_Lat_Inc_Heading.GBIS.mat',
    #                         '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us6000jk0t_NP1/invert_3_5_8_10_11_12_15_F/Figures/res_los_modlos_lonlat_xy_GEOC_072A_05090_131313_floatml_masked_GACOS_Corrected_clipped_signal_masked_QAed_20230120_20230201.ds_unw_Lon_Lat_Inc_Heading.GBIS.mat',
    #                         '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us6000jk0t_NP1/invert_3_5_8_10_11_12_15_F/Figures/res_los_modlos_lonlat_xy_GEOC_072A_05090_131313_floatml_masked_GACOS_Corrected_clipped_signal_masked_QAed_20230120_20230213.ds_unw_Lon_Lat_Inc_Heading.GBIS.mat',
    #                         '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us6000jk0t_NP1/invert_3_5_8_10_11_12_15_F/Figures/res_los_modlos_lonlat_xy_GEOC_072A_05289_131313_floatml_masked_GACOS_Corrected_clipped_signal_masked_QAed_20230108_20230201.ds_unw_Lon_Lat_Inc_Heading.GBIS.mat',
    #                         '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us6000jk0t_NP1/invert_3_5_8_10_11_12_15_F/Figures/res_los_modlos_lonlat_xy_GEOC_072A_05289_131313_floatml_masked_GACOS_Corrected_clipped_signal_masked_QAed_20230120_20230201.ds_unw_Lon_Lat_Inc_Heading.GBIS.mat',
    #                         '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us6000jk0t_NP1/invert_3_5_8_10_11_12_15_F/Figures/res_los_modlos_lonlat_xy_GEOC_072A_05289_131313_floatml_masked_GACOS_Corrected_clipped_signal_masked_QAed_20230120_20230213.ds_unw_Lon_Lat_Inc_Heading.GBIS.mat',
    #                         '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us6000jk0t_NP1/invert_3_5_8_10_11_12_15_F/Figures/res_los_modlos_lonlat_xy_GEOC_079D_05210_131313_floatml_masked_GACOS_Corrected_clipped_signal_masked_QAed_20230121_20230214.ds_unw_Lon_Lat_Inc_Heading.GBIS.mat']
    # filepath_inv = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us6000jk0t_NP1/invert_3_5_8_10_11_12_15_F/invert_3_5_8_10_11_12_15_F.mat'
    # burn_in = int(1e5*0.3)
    # no_iterations = int(1e5)
    # num_faults=1
    # output_dir = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us6000jk0t_NP1/invert_3_5_8_10_11_12_15_F/Figures'
    # main(filepath_inv,filepath_res_los_mod,output_dir,burn_in,no_iterations,num_faults=1)




    filepath_res_los_mod = ['/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us7000he1h_NP1/invert_1_2_3_4_F/Figures/res_los_modlos_lonlat_xy_GEOC_128A_05969_131313_floatml_masked_GACOS_Corrected_clipped_signal_masked_QAed_20220423_20220622.ds_unw_Lon_Lat_Inc_Heading.GBIS.mat',
                            '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us7000he1h_NP1/invert_1_2_3_4_F/Figures/res_los_modlos_lonlat_xy_GEOC_128A_05969_131313_floatml_masked_GACOS_Corrected_clipped_signal_masked_QAed_20220505_20220622.ds_unw_Lon_Lat_Inc_Heading.GBIS.mat',
                            '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us7000he1h_NP1/invert_1_2_3_4_F/Figures/res_los_modlos_lonlat_xy_GEOC_128A_05969_131313_floatml_masked_GACOS_Corrected_clipped_signal_masked_QAed_20220517_20220622.ds_unw_Lon_Lat_Inc_Heading.GBIS.mat',
                            '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us7000he1h_NP1/invert_1_2_3_4_F/Figures/res_los_modlos_lonlat_xy_GEOC_128A_05969_131313_floatml_masked_GACOS_Corrected_clipped_signal_masked_QAed_20220529_20220622.ds_unw_Lon_Lat_Inc_Heading.GBIS.mat']



    filepath_inv = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us7000he1h_NP1/invert_1_2_3_4_F/invert_1_2_3_4_F.mat'
    burn_in = int(1e5*0.3)
    no_iterations = int(1e5)
    num_faults=1
    output_dir = '/uolstore/Research/a/a285/homes/ee18jwc/code/auto_inv/us7000he1h_NP1/invert_1_2_3_4_F/Figures'
    main(filepath_inv,filepath_res_los_mod,output_dir,burn_in,no_iterations,num_faults=1)
