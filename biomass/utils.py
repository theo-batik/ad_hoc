#-------------------------------------------------------------------------------------------

# Imports
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from datetime import datetime
from os import getenv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

#-------------------------------------------------------------------------------------------

# Load environment variables and set parameters

# Camera properties
IMAGE_SENSOR_WIDTH = float(getenv("IMAGE_SENSOR_WIDTH"))
CAMERA_FOCAL_LENGTH = float(getenv("CAMERA_FOCAL_LENGTH"))


# RGB Kelp Colour 
kelp_colour = (170/255, 182/255, 133/255)
blue_colour = (2/255, 91/255, 114/255)

# Reduction scale of drone image to coverage map
scale = int(getenv("GRID_BLOCK_PIXEL_LENGTH"))

# Harvest efficiency
HARVEST_EFFICIENCY = float(getenv('HARVEST_EFFICIENCY'))

IMAGE_SENSOR_WIDTH = float(getenv("IMAGE_SENSOR_WIDTH"))
CAMERA_FOCAL_LENGTH = float(getenv("CAMERA_FOCAL_LENGTH"))


#-------------------------------------------------------------------------------------------

# Basics

# load image
def load_image_as_array(image_path):
    image = Image.open(image_path)
    image_array = np.array(image)
    return image_array


#-------------------------------------------------------------------------------------------


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


#-------------------------------------------------------------------------------------------

# Canopy area and coverage maps

# Canopy area from binary image
def compute_canopy_area(path_to_binary_image, meters_per_pixel):
    image_array = load_image_as_array(path_to_binary_image)
    canopy_area = np.sum(image_array == 0) * (meters_per_pixel**2) # sum(black pixel)*(pixel area)
    return canopy_area


# Get coverage map
def produce_coverage_map_from_binary(path_to_image, scale=scale):
        ''' 
        Input: 
            Binary image
            Scaling factor
        Returns: 
            Grayscale image, reduced by X (scale), where each pixel value represents the 
            percentage coverage of the corresponding X-by-X region of the input image.
        '''
        image = load_image_as_array(path_to_image)

        # Get the dimensions of the binary image
        height = image.shape[0]
        width = image.shape[1]
        print('height', height, 'width', width)

        # Calculate the number of rows and columns in the reduced image
        reduced_rows = int(height / scale)
        reduced_cols = int(width / scale)
        region_area = scale * scale

        # Create an empty grayscale image for coverage with reduced dimensions
        coverage_map = np.zeros((reduced_rows, reduced_cols), dtype=np.uint8)

        print(scale, 'SCALE!')
        # Iterate over scale-by-scale regions in the binary image
        for row in range(0, reduced_rows*scale, scale): # Recontruct image dimensions from reduced rows to avoid rounding issues
            for col in range(0, reduced_cols*scale, scale):
                # Extract the scalexscale region
                region = image[row:row+scale, col:col+scale]
                # Calculate the ratio of pixels equal to zero in the region
                coverage_ratio = np.sum(region == 0) / region_area

                # Populate the corresponding pixel in the coverage image
                reduced_row = int(row / scale)
                reduced_col = int(col / scale)
                coverage_map[reduced_row, reduced_col] = np.array(coverage_ratio * 100, dtype=int)

        return coverage_map



def get_total_biomass_from_coverage_map(coverage_map, meters_per_pixel):
        '''
        Input: coverage_map
        Parameters: scale, average_meters_per_pixel_ratio
        Returns: total biomass of coverage area
        '''
        # Extract image dimensions (pixels)
        pixel_length = coverage_map.shape[0]
        pixel_width = coverage_map.shape[1]

        # Get grid block length
        grid_block_meters_length = scale * meters_per_pixel

        # Convert percentage coverage map to [total effective harvest efficiency] using standard harvest efficiency
        total_effective_harvest_efficiency = np.sum(coverage_map/100 * HARVEST_EFFICIENCY) / (pixel_length * pixel_width ) # (kg/m^2)
        print( '\tTotal effective harvest efficiency', round(total_effective_harvest_efficiency,2), 'kg/m^2' )

        # Total area of image
        total_area = ( pixel_length * grid_block_meters_length) * (pixel_width * grid_block_meters_length)
        print('\tTotal area:', round(total_area, 0), 'm^2')

        # Total biomass
        total_biomass = total_effective_harvest_efficiency * total_area
        total_biomass = total_biomass / 1000 # convert to tonnes
        print('\tTotal available biomass', round(total_biomass, 1), 'tonnes')
        return total_biomass


# Output coverage map
def output_coverage_map(image_array, name, date, meters_per_pixel, save=True, scale=scale):

        # Get grid block length
        grid_block_meters_length = scale * meters_per_pixel
        
        # Create new figure
        fig = plt.figure()

        # Define the custom colormap colors
        light_blue = (0.95, 0.98, 1.0)   # Lighter blue (i.e. Ocean) # (0.85, 0.95, 1.0) 

        # Create a custom colormap
        cmap_colors = [light_blue, kelp_colour] # (136/255, 160/255, 146/255) #  # (0.4, 0.22, 0.141)  # More brownish green (i.e. Kelp)
        custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', cmap_colors, N=256)

        # Set the extent to change the visual scale
        im = plt.imshow(image_array, cmap=custom_cmap, vmin=0, vmax=100) #, extent=extent)

        # Add labels for x and y axes
        plt.xlabel('Meters')
        plt.ylabel('Meters')
        # Set custom tick locations and labels for x-axis

        image_width = image_array.shape[0]
        image_height = image_array.shape[1]
        print('image_height', image_height)
        print('image_width', image_width)

        x_tick_positions = np.arange(0,  image_height + 1, 50) # int(image_width/20
        # x_tick_positions = np.linspace(0, image_width, 10)
        x_tick_labels = np.around(grid_block_meters_length * x_tick_positions, decimals=0).astype(int)
        plt.xticks(x_tick_positions, x_tick_labels)

        # Set custom tick locations and labels for y-axis
        y_tick_positions = np.arange(0, image_width + 1, 50)
        y_tick_labels = np.around(grid_block_meters_length * y_tick_positions, decimals=0).astype(int)
        plt.yticks(y_tick_positions, y_tick_labels)

        # print('y_tick_labels', y_tick_labels, 'x_tick_labels', x_tick_labels)
        
        # Scale & grid
        grid_block_meters_length_rounded = int(round(grid_block_meters_length, 0))
        plt.grid(True)
    
        # Add the colorbars
        # cbar1 = plt.colorbar(im, label=f'Percentage coverage (%)*', shrink=0.45)
        # cbar1.set_ticks([0, 20, 40, 60, 80, 100])
        # cbar1.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])

        # Manually set colorbar 2 ticks
        cbar2 = plt.colorbar(im, label=f'Biomass (kg)*', shrink=0.58 ) #, pad=0.05) # ax=plt.gca(),
        pixel_area = (grid_block_meters_length)**2
        max_biomass = HARVEST_EFFICIENCY * pixel_area # (kg/m^2)*(m^2)
        cbar2.set_ticks( [0, 20, 40, 60, 80, 100] )
        cbar2.set_ticklabels( np.around(np.linspace(0, max_biomass, 6), decimals=1) )
        
        # Get total biomass
        total_biomass = get_total_biomass_from_coverage_map(image_array, meters_per_pixel)

        plt.title( f'Kelp biomass from drone imaging: {date}', fontsize=12, loc='left', color=blue_colour, pad=5)
        footnote_1 = f'Total harvestable biomass: {round(total_biomass, 1)} (tonnes)'
        footnote_2 = f'*Per ${grid_block_meters_length_rounded}m^2$ (pixel) region based on corresponding proportion of visible canopy area\nand a harvesting efficiency of {HARVEST_EFFICIENCY} ($kg/m^2$).'
        plt.text(0, -0.20, footnote_1, ha='left', va='center', transform=plt.gca().transAxes, fontsize=8)
        plt.text(0, -0.27, footnote_2, ha='left', va='center', transform=plt.gca().transAxes, fontsize=8)

        # Layout
        plt.tight_layout(pad=3)

        # Save image
        if save:
            plt.savefig(name, dpi=500)