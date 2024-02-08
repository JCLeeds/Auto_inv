import obspy
from obspy.clients.fdsn import Client 
from obspy import UTCDateTime 


def pull_USGS_info(eventid):
    MTdict = {'moment':None,
                'magnitude':None ,
                'Depth_MT':None,
                'PercentDC':None,
                'Half Duration':None,
                'centroid_lat':None,
                'centroid_lon':None
                }
    strike_dip_rake = {"strike":0,
                        "dip":0, 
                        "rake":0}
    time_pos_depth = {"DateTime":None,
                        "Position":None,
                        "Depth":None}
    client = Client('USGS')
    event = client.get_events(eventid=eventid)[0]
    for x in event.magnitudes:
        if x['magnitude_type'] == 'Mww':
            MTdict['magnitude'] =  x.mag
    # for x in event.focal_mechanisms[0]:
    x = event.focal_mechanisms[0]
    for x in event.focal_mechanisms:
        print(x.resource_id)
        if 'mww' in str(x.resource_id):
            strike_dip_rake['strike'] = [x.nodal_planes.nodal_plane_1.strike,x.nodal_planes.nodal_plane_2.strike]
            strike_dip_rake['dip'] = [x.nodal_planes.nodal_plane_1.dip,x.nodal_planes.nodal_plane_2.dip] 
            strike_dip_rake['rake'] = [x.nodal_planes.nodal_plane_1.rake,x.nodal_planes.nodal_plane_2.rake] 
            MTdict['PercentDC'] = x.moment_tensor.double_couple
            MTdict['moment'] = x.moment_tensor.scalar_moment
            MTdict['type'] = 'mww'
            break 
        elif 'mwb' in str(x.resource_id):
            strike_dip_rake['strike'] = [x.nodal_planes.nodal_plane_1.strike,x.nodal_planes.nodal_plane_2.strike]
            strike_dip_rake['dip'] = [x.nodal_planes.nodal_plane_1.dip,x.nodal_planes.nodal_plane_2.dip] 
            strike_dip_rake['rake'] = [x.nodal_planes.nodal_plane_1.rake,x.nodal_planes.nodal_plane_2.rake] 
            MTdict['PercentDC'] = x.moment_tensor.double_couple
            MTdict['moment'] = x.moment_tensor.scalar_moment
            MTdict['type'] = 'mwb'
            break
    for x in event.origins:
        print(x)
        if 'mww' in str(x.resource_id) and x.depth_type == 'from moment tensor inversion' and MTdict['type'] == 'mww':
            MTdict['centroid_lat'] = x.latitude
            MTdict['centroid_lon'] = x.longitude
            MTdict['Depth_MT'] = x.depth/1000
            break
        elif 'mwb' in str(x.resource_id) and x.depth_type =='from moment tensor inversion' and MTdict['type'] == 'mwb':
            MTdict['centroid_lat'] = x.latitude
            MTdict['centroid_lon'] = x.longitude
            MTdict['Depth_MT'] = x.depth/1000
            break
    time_pos_depth['Depth'] = event.origins[0].depth/1000
    time_pos_depth['DateTime'] = UTCDateTime(event.origins[0].time)
    time_pos_depth['Position'] = [event.origins[0].latitude,event.origins[0].longitude]
    print(time_pos_depth)
    print(strike_dip_rake)
    print(MTdict)
    return  time_pos_depth, strike_dip_rake, MTdict




if __name__ == '__main__':
    pull_USGS_info('us6000bdq8')