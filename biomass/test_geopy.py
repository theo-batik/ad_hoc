import cv2
from geopy.distance import geodesic
import utils as u



def draw_bounding_box(image_path, center_lat, center_lon, lat, lon, meters_per_pixel):
    # Read the image from the file
    image = cv2.imread(image_path)
    
    # Calculate the distance (in meters) between the center coordinate and the provided latitude and longitude
    distance = geodesic((center_lat, center_lon), (lat, lon)).meters
    
    # Convert the distance from meters to pixels
    distance_in_pixels = distance / meters_per_pixel

    # Determine the coordinates of the bounding box
    x_center, y_center = image.shape[1] // 2, image.shape[0] // 2  # Center of the image
    x1 = int(x_center - distance_in_pixels / 2)
    y1 = int(y_center - distance_in_pixels / 2)
    x2 = int(x_center + distance_in_pixels / 2)
    y2 = int(y_center + distance_in_pixels / 2)
    
    # Draw the bounding box on the image
    color = (255, 255, 255)  # White color (or any other color you prefer)
    thickness = 2  # Thickness for the outline
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    # Display the image with the bounding box
    cv2.imshow('Image with Bounding Box', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#########################################################################

# UTM
Easting	= 510340 
Northing = 7053580	
NWE = 510314
NWN = 7053625
NEE = 510366
NEN = 7053625
SEE = 510366
SEN = 7053535
SWE = 510314
SWN = 7053535







# Setup
image_folder = 'images/202402/'
image_path = image_folder + '20240206.JPG'


# Preprocess image
metadata = u.get_metadata(image_path)
metadata = u.refine_metadata(metadata)
print(metadata)

length = 0.000009 * 10
center_lat, center_lon = 26.63841236111111, 15.104567944444444  # Example center coordinate (New York City)
lat, lon = 26.63841236111111 + length, 15.104567944444444 + length  # Example target coordinate (Empire State Building)
meters_per_pixel = u.compute_meters_per_pixel(metadata)
draw_bounding_box(image_path, center_lat, center_lon, lat, lon, meters_per_pixel)
