import obspy 
import scrape_USGS as sUSGS
import data_ingestion as DI 
import preproc as pp
import time 

def run_auto_inv():

    test_event = sUSGS.USGS_event("us6000jk0t")
    #Decending test
    # test_event = sUSGS.USGS_event("us7000ki5u")
    test_block = DI.DataBlock(test_event)
    event_date_start = obspy.core.UTCDateTime(test_event.time_pos_depth['DateTime']) - (15*86400)
    event_date_end = obspy.core.UTCDateTime(test_event.time_pos_depth['DateTime']) + (15*86400)
    scale_factor_mag = 0.05 
    scale_factor_depth = 0.075

    scale_factor_clip_mag = 0.05
    scale_factor_clip_depth = 0.075


    print(event_date_start)
    print(event_date_end)


    # test_block.pull_frame_coseis()
    # geoc_path, gacos_path = test_block.pull_data_frame_dates(20220110,20220201,frame="100A_05036_121313")
    # ACENDING TEST SINGLE FRAME
    geoc_path,gacos_path = test_block.pull_data_frame_dates(20230108,20230201,frame="072A_05090_131313",single_ifgm=True)
    # DECENDING TEST SINGLE FRAME 
    # geoc_path,gacos_path = test_block.pull_data_frame_dates(20230716,20230728,frame="021D_05367_131313",single_ifgm=True)
    # All Coseismic 
    # geoc_path, gacos_path = test_block.pull_frame_coseis()
    print(geoc_path)
    geoc_ml_path = test_block.create_geoc_ml(geoc_path)
   
    DaN = pp.deformation_and_noise(test_event,test_block)

    # # Full mask, gacos, clip
    geoc_masked_path = DaN.coherence_mask(geoc_ml_path,0.1)
    # geoc_masked_path = geoc_ml_path
    try:
        geoc_gacos_corr_path = DaN.apply_gacos(geoc_masked_path,gacos_path)
    except: 
        geoc_gacos_corr_path = geoc_masked_path
        print("No GACOS availble for this frame")

    geoc_clipped_path = DaN.usgs_clip(geoc_gacos_corr_path,scale_factor_mag=scale_factor_clip_mag,scale_factor_depth=scale_factor_clip_depth)
    geoc_masked_signal = DaN.signal_mask(geoc_clipped_path,scale_factor_mag=scale_factor_mag,scale_factor_depth=scale_factor_depth)

    dirs_with_ifgms, meta_file_paths = test_block.get_path_names(geoc_masked_signal)
    print(geoc_masked_signal)
    print(dirs_with_ifgms)
    if isinstance(geoc_masked_signal,list):
         for ii in range(len(geoc_masked_signal)):
            for dir in dirs_with_ifgms[ii]:
                print(dir)
                try:
                    sill_semi, nugget_semi, range_semi = DaN.calc_semivariogram(geoc_masked_signal[ii],dir,signal_mask=True,mask=False,plot_semi=True,semi_mask_thresh=30.6,max_lag=150)  
                except:
                    pass
    else:
        for ii in range(len(dirs_with_ifgms)):
            sill_semi, nugget_semi, range_semi = DaN.calc_semivariogram(geoc_masked_signal,dirs_with_ifgms[ii],signal_mask=True,mask=False,plot_semi=True,semi_mask_thresh=30.6,max_lag=150)



    starttime = time.time()
    geoc_ds_path = DaN.nested_uniform_down_sample(geoc_masked_signal,1000,6000,nmpoints=None,scale_factor_mag=scale_factor_mag,scale_factor_depth=scale_factor_depth)
    endtime = time.time()
    print("Time elasped on downsampling = " + str(endtime-starttime))

