import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# from scipy import signal
import time
from skimage.measure import block_reduce


# # don't think you'll need this now
# def downsample(data, kernel_size):
#     kernel = np.ones((kernel_size, kernel_size))
#     convolved = signal.convolve2d(data, kernel, mode='same')
#     downsampled = convolved[::kernel_size, ::kernel_size]
#     # some kind of summing error
#     return downsampled


def resample(data, size):
    return data[::size, ::size]


def circle(a, b, r, x, y):
    return (x - a) ** 2 + (y - b) ** 2 <= r ** 2


def resample(data, rate):
    return block_reduce(data, block_size=(rate, rate), func=np.nanmean,cval=np.nan)

def calc_rates(mask,data,ds_total_points):
    print(mask)
    points_inner = len(mask[mask==True])
    points_outer = len(mask[mask==False])

    print("Points inside === " + str(points_inner))
    print("points outside ==== " + str(points_outer))

    print("ds_aim is === " + str(ds_total_points))

    inside_downsample_rate = int(np.sqrt(int(points_inner/(ds_total_points*0.5))))
    outside_downsample_rate = int(np.sqrt(int(points_outer/(ds_total_points*0.5))))

    print(inside_downsample_rate)
    print(outside_downsample_rate)

    return inside_downsample_rate, outside_downsample_rate

def resample_all(data,Inc,Head, X, Y,cent,radius,ds_total_points,dlon,dlat,pixsp_a,pixsp_r,lon1,lat1):

    # ratio = outside_downsample_rate / inside_downsample_rate
    # inside_downsample_rate = 50
    # outside_downsample_rate = 200

    # np array with True if inside the circle and False if outside the circle
    mask = circle(cent[0], cent[1], radius, X, Y)
    inside_downsample_rate, outside_downsample_rate= calc_rates(mask,data,ds_total_points)



    # downsample your data for both rates. This is a very inexpensive step but it could be sped up further if required.
    data_inside_downsampled = resample(data, inside_downsample_rate)
    data_outside_downsampled = resample(data, outside_downsample_rate)
    inc_inside_downsampled = resample(Inc, inside_downsample_rate)
    inc_outside_downsampled = resample(Inc, outside_downsample_rate)
    head_inside_downsampled = resample(Head, inside_downsample_rate)
    head_outside_downsampled = resample(Head, outside_downsample_rate)
     


    # downsample your X and Y meshgrids
    X_inside_downsampled = resample(X, inside_downsample_rate)
    Y_inside_downsampled = resample(Y, inside_downsample_rate)

    X_outside_downsampled = resample(X, outside_downsample_rate)
    Y_outside_downsampled = resample(Y, outside_downsample_rate)


    # resample your mask
    mask_inside = resample(mask, inside_downsample_rate).astype(bool)
    mask_outside = resample(np.invert(mask), outside_downsample_rate).astype(bool)

    # need to do something here to prevent overlap if you want (at the moment there is a bit of overlap between
    # inside and outside parts)



    new_dlon_inside = dlon * inside_downsample_rate
    new_dlat_inside = dlat * inside_downsample_rate
    new_pixsp_a_inside = pixsp_a * inside_downsample_rate
    new_pixsp_r_inside = pixsp_r * inside_downsample_rate

    new_dlon_outside = dlon * outside_downsample_rate
    new_dlat_outside = dlat * outside_downsample_rate
    new_pixsp_a_outside = pixsp_a * outside_downsample_rate
    new_pixsp_r_outside = pixsp_r * outside_downsample_rate

    X_inside_downsampled_lon = lon1+ X_inside_downsampled*new_dlon_inside / new_pixsp_a_inside
    Y_inside_downsampled_lat = lat1+ Y_inside_downsampled*new_dlat_inside / new_pixsp_r_inside
    X_outside_downsampled_lon = lon1 + X_outside_downsampled*new_dlon_outside / new_pixsp_a_outside
    Y_outside_downsampled_lat = lat1 + Y_outside_downsampled*new_dlat_outside / new_pixsp_r_outside


    x_in_lon =  X_inside_downsampled_lon * mask_inside
    y_in_lat = Y_inside_downsampled_lat * mask_inside
    x_out_lon = X_outside_downsampled_lon * mask_outside
    y_out_lat = Y_outside_downsampled_lat * mask_outside

    x_in = X_inside_downsampled * mask_inside
    y_in = Y_inside_downsampled * mask_inside
    z_in = data_inside_downsampled * mask_inside
    inc_in = inc_inside_downsampled * mask_inside 
    head_in = head_inside_downsampled * mask_inside

    x_out = X_outside_downsampled * mask_outside
    y_out = Y_outside_downsampled * mask_outside
    z_out = data_outside_downsampled * mask_outside
    inc_out = inc_outside_downsampled * mask_outside 
    head_out = head_outside_downsampled * mask_outside

    # new_dlon_inside = dlon * inside_downsample_rate
    # new_dlat_inside = dlat * inside_downsample_rate
    # new_pixsp_a_inside = pixsp_a * inside_downsample_rate
    # new_pixsp_r_inside = pixsp_r * inside_downsample_rate

    # new_dlon_outside = dlon * outside_downsample_rate
    # new_dlat_outside = dlat * outside_downsample_rate
    # new_pixsp_a_outside = pixsp_a * outside_downsample_rate
    # new_pixsp_r_outside = pixsp_r * outside_downsample_rate

    # x_in_lon = lon1+ x_in*new_dlon_inside / new_pixsp_a_inside
    # y_in_lat = lat1+ y_in*new_dlat_inside / new_pixsp_r_inside
    # x_out_lon = lon1 + x_out*new_dlon_outside / new_pixsp_a_outside
    # y_out_lat = lat1 + y_out*new_dlat_outside / new_pixsp_r_outside

    

    # getting it into the format you wanted = rows of [x, y, z]
    inside_points = np.stack([x_in, y_in, z_in, inc_in, head_in,x_in_lon,y_in_lat]).reshape(7, -1).T
    outside_points = np.stack([x_out, y_out, z_out, inc_out, head_out,x_out_lon,y_out_lat]).reshape(7, -1).T

    # removing zeros 
    inside_points = inside_points[~np.all(inside_points == 0, axis=1)]
    outside_points = outside_points[~np.all(outside_points == 0, axis=1)]

    # Problem for future when using remove any it will remove the 0,0 point which is not what we want we only want the mask removed. 
    print(np.vstack([inside_points, outside_points]))

    return np.vstack([inside_points, outside_points])

# demo

if __name__ == '__main__':
    t = time.time()
    # points = resample_all(z_data, X, Y, inside_downsample, outside_downsample)
    t2 = time.time()
    print(f'took {1000 * (t2 - t):.4f} ms for {n**2} original points')

    plt.figure()
    plt.scatter(points[:, 0], points[:, 1])
    plt.show()