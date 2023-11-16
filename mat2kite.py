import logging
import glob
from os import path as op
from datetime import datetime, timedelta
import pyrocko.orthodrome as od

import numpy as num
from scipy import stats, interpolate, io

try:
    import h5py
except ImportError as e:
    raise e('Please install h5py library')

import matplotlib.pyplot as plt
from matplotlib import colors

from kite.scene import Scene, SceneConfig

log = logging.getLogger('mat2kite')

d2r = num.pi/180.
r2d = 180./num.pi


class DataStruct(dict):

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value


def _get_file(dirname, fname):
    fns = glob.glob(op.join(dirname, fname))
    if len(fns) == 0:
        raise ImportError('Cannot find %s file in %s' % (fname, dirname))
    if len(fns) > 1:
        raise ImportError('Found multiple files for %s: %s' %
                          (fname, ', '.join(fns)))
    fn = fns[0]

    log.debug('Found file %s', fn)
    return fn


def _read_mat(filename):
    try:
        mat = h5py.File(filename, 'r')
    except OSError:
        log.debug('using old scipy import for %s', filename)
        mat = io.loadmat(filename)
    return mat


def read_mat_data(filename, import_mv2=False, **kwargs):
    # TODO: Add old matlab import
    log.debug('Reading in Matlab files...')

    fn_data = kwargs.get(
        'fn_data', _get_file('.', filename))

    data_mat = _read_mat(fn_data)

    data = DataStruct()
    #data.ll_coords = num.asarray(data_mat['ll0'])
    data.ps_mean_v = num.asarray(data_mat['ph_disp']).ravel()

    geo_coords = num.asarray(data_mat['lonlat'])
    data.lons = geo_coords[:, 0]
    data.lats = geo_coords[:, 1]

    meter_coords = num.asarray(data_mat['lonlat_m'])
    data.lons_m = meter_coords[:,0]
    data.lats_m = meter_coords[:,1]
    print(data.lons)
    print(data.lats)

    days = num.asarray(data_mat['day'])
    data.tmin = timedelta(days=days.min() - 366.25) + datetime(1, 1, 1)
    data.tmax = timedelta(days=days.max() - 366.25) + datetime(1, 1, 1)

    if import_mv2:
        data.ps_mean_std = num.asarray(data_mat['std']).ravel()

    data.look_angles = num.asarray(data_mat['la']).ravel()

    heading = float(num.asarray(data_mat['heading']))
    if num.isnan(heading):
        raise ValueError('Heading information is missing!')

    data.heading = heading
    data.covariance = data_mat['cov']
    data.sill_range_nug = num.asarray(data_mat['sill_range_nug'])

    return data


def bin_ps_data(data, bins=(800, 800)):
    log.info('Binning matlab data...')
    bin_vels, edg_lat, edg_lon, _ = stats.binned_statistic_2d(data.lats, data.lons, data.ps_mean_v,statistic='mean', bins=bins)
    edg_la, _, _, _ = stats.binned_statistic_2d(data.lats, data.lons, data.look_angles,statistic='mean', bins=bins)

    if 'ps_mean_std' in data.keys():
        log.debug('Binning mean velocity variance...')
        bin_mean_std, _, _, _ = stats.binned_statistic_2d(
            data.lats, data.lons, data.ps_mean_std,
            statistic='mean', bins=bins)
        data.bin_mean_var = bin_mean_std**2  # We want variance

    data.bin_ps_mean_v = bin_vels
    data.bin_edg_lat = edg_lat
    data.bin_edg_lon = edg_lon
    data.bin_look_angles = edg_la

    return data


def mat2kite(filename='.', px_size=(1000, 1000), convert_m=True,
                import_var=False, **kwargs):
    '''Convert Matlab struture InSAR data to a Kite Scene, based on stamps2kite.py

    Wrote by Fei Liu, 06/09/2023, University of Leeds

    Usage: python mat2kite.py INPUTFILE.mat -r resolution1 resolution2
    Example: python mat2kite.py test.mat -r 1000 1000

    The input *.mat files should have corresponding variables:
    
    lonlat: the lonlat of the scene, should be a N*2 matrix, where N is number of pixels
    day(not clear if needed): the date of epochs, e.g., 737063 (corresponding to 3rd Jan 2018). 
        Should be a M*1 matrix, where M is the number of epochs
    ph_disp: the displacement, size of N*1, unit is mm,
    heading: the heading angle, a single number, unit is degree
    la: the looking angle, size of N*1, unit is rad

    std(optional, turn on by improt_var, not clear if needed): the standard deviation of measurement
    

    :returns: Kite Scene from the Matlab data
    :rtype: :class:`kite.Scene`
    '''
    data = read_mat_data(filename, import_mv2=import_var, **kwargs)
    log.info('Found a Matlab file at %s', op.abspath(filename))

    bbox = (data.lons.min(), data.lats.min(),
            data.lons.max(), data.lats.max())

    lengthN = od.distance_accurate50m(
        bbox[1], bbox[0],
        bbox[3], bbox[0])
    lengthE = od.distance_accurate50m(
        bbox[1], bbox[0],
        bbox[1], bbox[2])
    
    # lengthN = data.lats_m.max() - data.lats_m.min()
    # lengthE = data.lons_m.max() - data.lats_m.min()

    # sorted_unqiue_values,lats_index = num.unique(data.lats_m, return_index=True)
    # sorted_unique_values,lons_index = num.unique(data.lons_m, return_index=True)

    # # spacing_changes_lats = num.diff(data.lats_m)
    # # spacing_changes_lons = num.diff(data.lons_m) #uncomment if broken 

    spacing_changes_lats = num.diff(data.lats) #.astype(int) # get difference from a[i+1] - a[i]
    spacing_changes_lons = num.diff(data.lons) #.astype(int)
  
    spacing_changes_lats = num.unique(num.abs((spacing_changes_lats[num.nonzero(spacing_changes_lats)]))) # gives sorted unique values of diff 
    spacing_changes_lons = num.unique(num.abs((spacing_changes_lons[num.nonzero(spacing_changes_lons)]))) 
    print(spacing_changes_lats)
    print(spacing_changes_lons)
    samplespacingN_inner = spacing_changes_lats[0] # smallest unnique value from diff 
    samplespacingE_inner = spacing_changes_lons[0] # Inside sample_rate 
    samplespacingN_outer = spacing_changes_lats[1] # outside sample_rate 
    samplespacingE_outer = spacing_changes_lons[1]

    # samplespacingN =  samplespacingN_outer 
    # samplespacingE = samplespacingE_outer
    
    # # remove code block above if broken 

    
    # print(spacing_changes_lats)
    # print(spacing_changes_lons)
    
    # # samplespacingN = num.abs(num.min(spacing_changes_lats[num.nonzero(spacing_changes_lats)]))
    # # samplespacingE = num.abs(num.min(spacing_changes_lons[num.nonzero(spacing_changes_lons)]))
    
    # # samplespacingN = num.abs(min(spacing_changes_lats[num.nonzero(spacing_changes_lats)], key=abs))
    # # samplespacingE = num.abs(min(spacing_changes_lons[num.nonzero(spacing_changes_lons)], key=abs))
    


    # print(lats_index)
    # print(lons_index)

    # # samplespacingN = data.lats_m[lats_index[1]] - data.lats_m[lats_index[0]]
    # # samplespacingE = data.lons_m[lons_index[1]] - data.lons_m[lons_index[0]]

    # print(samplespacingE)
    # print(samplespacingN)
    # print(lengthE)
    # print(lengthN)
    


    # bins = (round(lengthN/samplespacingN),
    #         round(lengthE/samplespacingE))
    
    # print(bins)
    
    
    bins = (round(lengthE / px_size[0]),
            round(lengthN / px_size[1]))

    if convert_m:
        data.ps_mean_v /= 1e3

    if convert_m and import_var:
        data.ps_mean_std /= 1e3

    bin_ps_data(data, bins=bins)
    
    log.debug('Processing of LOS angles')
    # data.bin_theta = num.pi/2-data.bin_look_angles
    # Added by me check if correct 
    width = len(num.unique(data.lons))
    height = len(num.unique(data.lats))
    data.bin_theta = data.bin_look_angles
    data.theta = data.look_angles
    # data.ps_mean_v = num.reshape(data.ps_mean_v,(width,height))
    # data.theta = num.reshape(data.theta,(width,height))

    phi_angle = -data.heading * d2r + num.pi
    # phi_angle = data.heading*d2r
    if phi_angle > num.pi:
        phi_angle -= 2*num.pi

    # data.phi = num.full_like(data.theta, phi_angle)

    data.bin_phi = num.full_like(data.bin_theta, phi_angle)
    data.bin_phi[num.isnan(data.bin_theta)] = num.nan

    log.debug('Setting up the Kite Scene')
    config = SceneConfig()
    config.frame.llLat = data.bin_edg_lat.min()
    config.frame.llLon = data.bin_edg_lon.min()
    config.frame.dE = data.bin_edg_lon[1] - data.bin_edg_lon[0]
    config.frame.dN = data.bin_edg_lat[1] - data.bin_edg_lat[0]
    config.frame.spacing = 'degree' # change back to degree if doesnt work 

    scene_name = op.basename(op.abspath(filename))
    config.meta.scene_title = '%s (Matlab import)' % scene_name
    config.meta.scene_id = scene_name
    
    config.meta.time_master = data.tmin.timestamp()
    config.meta.time_slave = data.tmax.timestamp()


    config.quadtree.epsilon = 0.0009
    config.quadtree.nan_allowed = 1
    print(float(samplespacingN_outer))
    print(float(samplespacingN_inner))
    # config.quadtree.tile_size_max = float(samplespacingN_outer)
    config.quadtree.tile_size_max = float(samplespacingN_inner)
    config.quadtree.tile_size_min = float(samplespacingN_inner)

    #Attempt to add in covariance matrix
    config.covariance.covariance_matrix = data.covariance
    config.covariance.variance = num.max(data.covariance)
    # config.covariance.spacial_pairs = len(data.covariance)
    config.covariance.adaptive_subsampling = False
    config.covariance.noise_coord = [0.21, 0.86, 0.44, 0.25] # Recent change need to look into this
    config.covariance.plot = False
    config.covariance.weight_matrix = num.linalg.inv(data.covariance)


    # config.quadtree.
    # config.quadtree.tile_size_min = 0.01
    # print(data.sill_range_nug)
    # print(data.sill_range_nug[0])
    # print(len(data.sill_range_nug[0]))
    # config.covariance.model_coefficients = (float(data.sill_range_nug[0][0]),float(data.sill_range_nug[0][1]))
    # config.covariance.covariance_model = float(data.sill_range_nug[0][0]),float(data.sill_range_nug[0][1])
    
    print("################################### check here ##########################")
    print(num.shape(data.bin_ps_mean_v))
    print(num.shape(data.bin_theta))

    # print(config)

    scene = Scene(
        theta=data.bin_theta,
        phi=data.bin_phi,
        displacement=data.bin_ps_mean_v,
        config=config)
    
    # scene = Sce
    # scene.import_date(filename)
    
    
    # scene = Scene(theta=data.theta,
    #     phi=data.phi,
    #     displacement=data.ps_mean_v,
    #     config=config)
    

    if import_var:
        scene.displacement_px_var = data.bin_mean_var

    return scene


def plot_scatter_vel(lat, lon, vel):
    vmax = abs(max(vel.min(), vel.max()))
    norm = colors.Normalize(vmin=-vmax, vmax=vmax)

    plt.scatter(lat, lon, s=5, c=vel, cmap='RdYlBu', norm=norm)
    plt.show()


def main_command_line():
    import argparse

    parser = argparse.ArgumentParser(
        description='''Convert Matlab data into a Kite scene.''',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        'file', type=str,
        default='.',
        help='Matlab file name.')
    parser.add_argument(
        '--resolution', '-r', nargs=2, metavar=('mN', 'mE'),
        dest='resolution', type=int, default=(500, 500),
        help='pixel size of the output grid in North and East (meter).'
             'Default is 500 m by 500 m.')
    parser.add_argument(
        '--save', '-s', default=None, type=str, dest='save',
        help='filename to save the Kite scene to. If not given, the scene'
             ' will be opened in spool GUI.')
    parser.add_argument(
        '--force', '-f', default=False, action='store_true', dest='force',
        help='force overwrite of an existing scene.')
    parser.add_argument(
        '--keep-mm', action='store_true',
        default=False,
        help='keep mm/a and do not convert to m/a.')
    parser.add_argument(
        '--import-var', action='store_true', dest='import_var',
        default=True,
        help='import the variance from mv2.mat, which is added to Kite\'s'
             ' scene covariance matrix.')
    parser.add_argument(
        '-v', action='count',
        default=0,
        help='verbosity, add mutliple to increase verbosity.')

    args = parser.parse_args()

    log_level = logging.INFO - args.v * 10
    logging.basicConfig(level=log_level if log_level > 0 else 0)

    fn_save = args.save
    if args.save:
        for fn in (fn_save, fn_save + '.yml', fn_save + '.npz'):
            if op.exists(fn) and not args.force:
                raise UserWarning(
                    'File %s exists! Use --force to overwrite.' % fn_save)

    scene = mat2kite(filename=args.file, px_size=args.resolution,
                        convert_m=not args.keep_mm,
                        import_var=args.import_var)

    if fn_save:
        fn_save.rstrip('.yml')
        fn_save.rstrip('.npz')

        log.info('Saving Matlab scene to file %s[.yml/.npz]...', fn_save)
        scene.save(args.save)

    else:
        scene.spool()


    

def main(input_file,save_location,resolution):
    import argparse

    parser = argparse.ArgumentParser(
        description='''Convert Matlab data into a Kite scene.''',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        '--file','-i', type=str,
        default=input_file,
        help='Matlab file name.')
    parser.add_argument(
        '--resolution', '-r', nargs=2, metavar=('mN', 'mE'),
        dest='resolution', type=int, default=resolution,
        help='pixel size of the output grid in North and East (meter).'
             'Default is 500 m by 500 m.')
    parser.add_argument(
        '--save', '-s', default=save_location, type=str, dest='save',
        help='filename to save the Kite scene to. If not given, the scene'
             ' will be opened in spool GUI.')
    parser.add_argument(
        '--force', '-f', default=False, action='store_true', dest='force',
        help='force overwrite of an existing scene.')
    parser.add_argument(
        '--keep-mm', action='store_true',
        default=False,
        help='keep mm/a and do not convert to m/a.')
    parser.add_argument(
        '--import-var', action='store_true', dest='import_var',
        default=False,
        help='import the variance from mv2.mat, which is added to Kite\'s'
             ' scene covariance matrix.')
    parser.add_argument(
        '-v', action='count',
        default=0,
        help='verbosity, add mutliple to increase verbosity.')

    args = parser.parse_args()

    log_level = logging.INFO - args.v * 10
    logging.basicConfig(level=log_level if log_level > 0 else 0)

    fn_save = args.save
    if args.save:
        for fn in (fn_save, fn_save + '.yml', fn_save + '.npz'):
            if op.exists(fn) and not args.force:
                raise UserWarning(
                    'File %s exists! Use --force to overwrite.' % fn_save)

    scene = mat2kite(filename=args.file, px_size=args.resolution,
                        convert_m=not args.keep_mm,
                        import_var=args.import_var)

    if fn_save:
        fn_save.rstrip('.yml')
        fn_save.rstrip('.npz')

        log.info('Saving Matlab scene to file %s[.yml/.npz]...', fn_save)
        scene.save(args.save)

    else:
        scene.spool()



if __name__ == '__main__':
    main()
