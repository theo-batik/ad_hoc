################################################################################################################

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from datetime import datetime
from os import getenv
import numpy as np

################################################################################################################

# Load environment variables
IMAGE_SENSOR_WIDTH = float(getenv("IMAGE_SENSOR_WIDTH"))
CAMERA_FOCAL_LENGTH = float(getenv("CAMERA_FOCAL_LENGTH"))

################################################################################################################

# Metadata and meters-per-pixel calculation

def get_gps_data(exif_data):
    gps_info = {}
    for tag, value in exif_data.items():
        if TAGS.get(tag) == "GPSInfo":
            for key, val in value.items():
                if GPSTAGS.get(key):
                    gps_info[GPSTAGS[key]] = val
    return gps_info


def get_metadata(image_path):
    metadata = {}
    try:
        # Open the image
        with Image.open(image_path) as img:
            # Extract Exif data
            exif_data = img._getexif()

            if exif_data is not None:
                # Decode Exif data
                decoded_exif = {TAGS[key]: exif_data[key] for key in exif_data.keys() if key in TAGS and isinstance(exif_data[key], (str, int))}
                
                # Extract GPS information
                gps_info = get_gps_data(exif_data)
                
                # Store general metadata in dict.
                for key, value in decoded_exif.items():
                    metadata[key] = value
                
                # Store gps metadata in dict.
                for key, value in gps_info.items():
                    metadata[key] = value
                
                return metadata
            else:
                print("No Exif data found.")
    except Exception as e:
        print(f"Error: {e}")


def refine_metadata(metadata):
    
    refined_metadata = {}

    # Get Datetime
    datetime_string = metadata["DateTimeOriginal"]
    original_format = "%Y:%m:%d %H:%M:%S"
    new_format = "%Y-%m-%d"
    datetime_object = datetime.strptime(datetime_string, original_format)
    datetime_object = datetime_object.strftime(new_format)
    refined_metadata["DateTimeOriginal"] = datetime_object

    # Get altitude
    refined_metadata["GPSAltitude"] = metadata["GPSAltitude"]

    # Get latitude and longitude
    degrees_to_decimal = lambda x: float(x[0] + x[1]/60 + x[2]/(60*60))
    latitude = degrees_to_decimal( metadata["GPSLatitude"] )
    longitude = degrees_to_decimal( metadata["GPSLongitude"] )
    refined_metadata["GPSLatitude"] = latitude
    refined_metadata["GPSLongitude"] = longitude

    # Image dimensions
    refined_metadata["ImageLength"] = metadata["ImageLength"] 
    refined_metadata["ImageWidth"] = metadata["ImageWidth"] 

    return refined_metadata


def compute_meters_per_pixel(metadata):

    altitude = metadata["GPSAltitude"]
    image_width = metadata["ImageWidth"]

    meters_per_pixel = (IMAGE_SENSOR_WIDTH * altitude) / (CAMERA_FOCAL_LENGTH * image_width)

    return meters_per_pixel


def compute_canopy_area(path_to_binary_image, meters_per_pixel):
    image_array = np.array(Image.open(path_to_binary_image))
    canopy_area = np.sum(image_array == 0) * (meters_per_pixel**2) # sum(black pixel)*(pixel area)
    return canopy_area