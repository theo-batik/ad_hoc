import cv2
from geopy.distance import geodesic
# from shapely.geometry import Polygon
from pyproj import Transformer
# from functools import partial
# import numpy as np
import utils as u
import pandas as pd

#########################################################################


def draw_bounding_box(image, center_lat, center_lon, site_utm, meters_per_pixel, site):
    
    # Convert UTM coordinates to latitude and longitude
    site_latlon = {}
    for key, value in site_utm.items():
        lon, lat = utm_to_latlon(value[0], value[1])
        lat *= -1 # adjust for southern hemisphere
        lat_delta = -1 if lat - center_lat > 0 else 1
        lon_delta = -1 if lon - center_lon < 0 else 1
        site_latlon[key] = (lat, lon, lat_delta, lon_delta)
        # print(lon, lat)

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
    print('Rect. points', pt1, pt2)

    

    # Reshape vertices array to match OpenCV's expectations
    # vertices = vertices.reshape((-1, 1, 2))

    # Draw the bounding box on the image
    color = (255, 255, 255)  # White color (or any other color you prefer)
    thickness = 2  # Thickness for the outline
    cv2.rectangle(image, pt2, pt1, color, thickness)
    
    # Display the image with the bounding box
    # cv2.imwrite(filename, image)
    # cv2.imshow('Image with Bounding Box', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Define the font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Define the font scale
    font_scale = 1
    # Define the font color
    font_color = (255, 255, 255)  # White
    # Define the thickness of the text
    thickness = 2
    # Write the text on the image
    cv2.putText(image, site[6:], pt1, font, font_scale, font_color, thickness)


    return image


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

# print(metadata, '!!!!!!!')
center_lat = - metadata['GPSLatitude'] # Note the negative!
center_lon = metadata['GPSLongitude'] 

# -26.63841236111111, 15.104567944444444  
meters_per_pixel = u.compute_meters_per_pixel(metadata)

# Read it ops sites
df = pd.read_csv('data/ref_operational_sites.csv')
print(df.head())

# Read the image from the file
image = cv2.imread(image_path)

for index, row in df.iterrows():
    site = row['Tag']
    NWE = row['NWE']      
    NWN = row['NWN']     
    NEE = row['NEE']      
    NEN = row['NEN']     
    SEE = row['SEE']      
    SEN = row['SEN']     
    SWE = row['SWE']     
    SWN = row['SWN']
    site_utm = {'NW':(NWE, NWN), 'NE':(NEE, NEN), 'SE':(SEE, SEN),'SW':(SWE,SWN)}
    image = draw_bounding_box(image, center_lat, center_lon, site_utm, meters_per_pixel, site)

    # break
    # print(site_utm)


filename = 'All sites.jpg'
cv2.imwrite(filename, image)
# 


# 
