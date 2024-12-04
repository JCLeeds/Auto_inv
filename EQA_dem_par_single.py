    import os 
    import gdal 
    
    def make_EQA_DEM_PAR_for_tif(filepath)
        unw_tiffile = filepath
        geotiff = gdal.Open(unw_tiffile)
        width = geotiff.RasterXSize
        length = geotiff.RasterYSize
        lon_w_p, dlon, _, lat_n_p, _, dlat = geotiff.GetGeoTransform()
        ## lat lon are in pixel registration. dlat is negative
        lon_w_g = lon_w_p + dlon/2
        lat_n_g = lat_n_p + dlat/2
        dem_par = file_path + 'EQA.dem_par'

        if not os.path.exists(dempar):
            print('\nCreate EQA.dem_par', flush=True)

            text = ["Gamma DIFF&GEO DEM/MAP parameter file",
                "title: DEM", 
                "DEM_projection:     EQA",
                "data_format:        REAL*4",
                "DEM_hgt_offset:          0.00000",
                "DEM_scale:               1.00000",
                "width: {}".format(width), 
                "nlines: {}".format(length), 
                "corner_lat:     {}  decimal degrees".format(lat_n_g), 
                "corner_lon:    {}  decimal degrees".format(lon_w_g), 
                "post_lat: {} decimal degrees".format(dlat), 
                "post_lon: {} decimal degrees".format(dlon), 
                "", 
                "ellipsoid_name: WGS 84", 
                "ellipsoid_ra:        6378137.000   m",
                "ellipsoid_reciprocal_flattening:  298.2572236",
                "",
                "datum_name: WGS 1984",
                "datum_shift_dx:              0.000   m",
                "datum_shift_dy:              0.000   m",
                "datum_shift_dz:              0.000   m",
                "datum_scale_m:         0.00000e+00",
                "datum_rotation_alpha:  0.00000e+00   arc-sec",
                "datum_rotation_beta:   0.00000e+00   arc-sec",
                "datum_rotation_gamma:  0.00000e+00   arc-sec",
                "datum_country_list: Global Definition, WGS84, World\n"]
        
            with open(dempar, 'w') as f:
                f.write('\n'.join(text))
