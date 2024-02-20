import cv2
from geopy.distance import geodesic
# from shapely.geometry import Polygon
from pyproj import Transformer
from functools import partial
import numpy as np
import utils as u


#########################################################################

# UTM

# I
# NWE = 510314
# NWN = 7053625
# NEE = 510366
# NEN = 7053625
# SEE = 510366
# SEN = 7053535
# SWE = 510314
# SWN = 7053535

# F
NWE = 510229
NWN = 7053745
NEE = 510271
NEN = 7053745
SEE = 510271
SEN = 7053655
SWE = 510229
SWN = 7053655

# D
# NWE = 510329
# NWN = 7053745
# NEE = 510371
# NEN = 7053745
# SEE = 510371
# SEN = 7053655
# SWE = 510329
# SWN = 7053655

# H
# NWE = 510384
# NWN = 7053630
# NEE = 510426
# NEN = 7053630
# SEE = 510426
# SEN = 7053540
# SWE = 510384
# SWN = 7053540	

# J
# NWE = 510389
# NWN = 7053895
# NEE = 510431
# NEN = 7053895
# SEE = 510431
# SEN = 7053805
# SWE = 510389
# SWN = 7053805



filename = 'F.jpg'
site_utm = {'NW':(NWE, NWN), 'NE':(NEE, NEN), 'SE':(SEE, SEN),'SW':(SWE,SWN)}

#########################################################################


def draw_bounding_box(image_path, center_lat, center_lon, site_utm, meters_per_pixel):
    
    # Read the image from the file
    image = cv2.imread(image_path)
    
    # Convert UTM coordinates to latitude and longitude
    site_latlon = {}
    for key, value in site_utm.items():
        lon, lat = utm_to_latlon(value[0], value[1])
        lat *= -1 # adjust for southern hemisphere
        lat_delta = -1 #if lat - center_lat < 0 else 1
        lon_delta = -1 if lon - center_lon < 0 else 1
        site_latlon[key] = (lat, lon, lat_delta, lon_delta)

    for key, value in site_latlon.items():
        print(key, value)
    
    
    # Define the verticies of rectangle
    center = (center_lat, center_lon)   
    
    dx1 = int(site_latlon['SW'][3] * geodesic(center, (center_lat, site_latlon['SW'][1])).meters / meters_per_pixel)
    dy1 = int(site_latlon['SW'][2] * geodesic(center, (site_latlon['SW'][0], center_lon )).meters / meters_per_pixel)

    dx2 = int(site_latlon['NE'][3] * geodesic(center, (center_lat, site_latlon['NE'][1])).meters / meters_per_pixel)
    dy2 = int(site_latlon['NE'][2] * geodesic(center, (site_latlon['NE'][0], center_lon )).meters / meters_per_pixel)

    print(dx1, dx2, dy1, dy2)

    x_center, y_center = image.shape[1] // 2, image.shape[0] // 2  # Center of the image
    pt1 = (x_center + dx1, y_center + dy1)  # Top-left corner
    pt2 = (x_center + dx2, y_center + dy2)  # Bottom-right corner
    print('\n !!!!', pt1, pt2)

    

    # Reshape vertices array to match OpenCV's expectations
    # vertices = vertices.reshape((-1, 1, 2))

    # Draw the bounding box on the image
    color = (255, 255, 255)  # White color (or any other color you prefer)
    thickness = 2  # Thickness for the outline
    cv2.rectangle(image, pt2, pt1, color, thickness)
    
    # Display the image with the bounding box
    cv2.imwrite(filename, image)
    # cv2.imshow('Image with Bounding Box', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


#########################################################################

def utm_to_latlon(easting, northing, southern=True):
    # Define the UTM and WGS84 projections
    utm_zone = 33  # Adjust the UTM zone based on your data
    transformer = Transformer.from_crs(f"EPSG:326{utm_zone}", "EPSG:4326")
    if southern:
        northing = 10000000 - northing  # adjust for the southern hemisphere
    lon, lat = transformer.transform(easting, northing)
    return lat, lon

#########################################################################

# Setup
image_folder = 'images/202402/'
image_path = image_folder + '20240206.JPG'





# Preprocess image
metadata = u.get_metadata(image_path)
metadata = u.refine_metadata(metadata)

center_lat, center_lon = -26.63841236111111, 15.104567944444444  # Note the negative!
meters_per_pixel = u.compute_meters_per_pixel(metadata)


draw_bounding_box(image_path, center_lat, center_lon, site_utm, meters_per_pixel)
