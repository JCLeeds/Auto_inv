import LiCSBAS05op_clip_unw as clip  

def usgs_clip(geoc_ml_path,range_str):
        """
        Clips frame around USGS locations scaled inversly to depth and linearly with Mag 
        Needs changing to work in local domain not degrees 
        """
        #lon1/lon2/lat1/lat2
        # geoc_clipped_path = geoc_ml_path + "_clipped"
        range_str
        print(range_str)
        index_clip = True
        if isinstance(geoc_ml_path,list):
            geoc_clipped_path = []
            for ii in range(len(geoc_ml_path)):
                path = geoc_ml_path[ii] + "_clipped_index"
                print(path)
                clip.main(auto=[geoc_ml_path[ii],path,range_str],index_clip=True)
                geoc_clipped_path.append(path)
        else:
            geoc_clipped_path = geoc_ml_path + "_clipped_index"
            clip.main(auto=[geoc_ml_path,geoc_clipped_path,range_str],index_clip=True)
        return  geoc_clipped_path

if __name__ == "__main__":
    usgs_clip('/Users/jcondon/phd/code/auto_inv/us6000lfn5_insar_processing/GEOC_013A_05597_131313_floatml', '1568:2912/224:1344')
   