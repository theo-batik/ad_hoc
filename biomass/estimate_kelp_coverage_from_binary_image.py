################################################################################################################

# Imports
# from kelp_coverage_estimator import KelpCoverageEstimator
# import pandas as pd 
# from os import listdir
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from os import getenv
import utils as u

################################################################################################################

# Parameters and .env variables:

# RGB Kelp Colour 
kelp_colour = (170/255, 182/255, 133/255)
blue_colour = (2/255, 91/255, 114/255)

# Reduction scale of drone image to coverage map
scale = int(getenv("GRID_BLOCK_PIXEL_LENGTH"))

# Harvest efficiency
HARVEST_EFFICIENCY = float(getenv('HARVEST_EFFICIENCY'))

# Camera properties
IMAGE_SENSOR_WIDTH = float(getenv("IMAGE_SENSOR_WIDTH"))
CAMERA_FOCAL_LENGTH = float(getenv("CAMERA_FOCAL_LENGTH"))



image_path = "images/202401/20240123.JPG"
metadata = u.get_metadata(image_path)
metadata = u.refine_metadata(metadata)
meters_per_pixel = u.compute_meters_per_pixel(metadata)
grid_block_meters_length = scale * meters_per_pixel

# for key, value in get_metadata(image_path).items():
#     print(f"{key}: {value}")

################################################################################################################

# load image
def load_image_pillow(image_path):
    image = Image.open(image_path)
    image_array = np.array(image)
    return image_array

# Get coverage map
def produce_coverage_map(image, scale=scale):
        ''' 
        Input: 
            Binary image
            Scaling factor
        Returns: 
            Grayscale image, reduced by X (scale), where each pixel value represents the 
            percentage coverage of the corresponding X-by-X region of the input image.
        '''
    
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


def get_total_biomass_from_coverage_map(coverage_map):
        '''
        Input: coverage_map
        Parameters: scale, average_meters_per_pixel_ratio
        Returns: total biomass of coverage area
        '''
        # Extract image dimensions (pixels)
        pixel_length = coverage_map.shape[0]
        pixel_width = coverage_map.shape[1]

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
def output_coverage_map(image_array, name, date, save=True, scale=scale):

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
        total_biomass = get_total_biomass_from_coverage_map(image_array)

        plt.title( f'Kelp biomass from drone imaging: {date}', fontsize=12, loc='left', color=blue_colour)
        footnote_1 = f'Total harvestable biomass: {round(total_biomass, 1)} (tonnes)'
        footnote_2 = f'*Per ${grid_block_meters_length_rounded}m^2$ (pixel) region based on corresponding proportion of visible canopy area\nand a harvesting efficiency of {HARVEST_EFFICIENCY} ($kg/m^2$).'
        plt.text(0, -0.26, footnote_1, ha='left', va='center', transform=plt.gca().transAxes, fontsize=8)
        plt.text(0, -0.37, footnote_2, ha='left', va='center', transform=plt.gca().transAxes, fontsize=8)

        # Layout
        plt.tight_layout()

        # Save image
        if save:
            plt.savefig('images/202401/' + name, dpi=300) 


################################################################################################################

# Process image

binary_image = load_image_pillow('images/202401/20240123_binary.jpg')
coverage_map = produce_coverage_map(binary_image)

date = metadata["DateTimeOriginal"]
name = f'{date}-coverage-map_output'
output_coverage_map(coverage_map, name, date, save=True)




################################################################################################################



# Read in drone image metadata
# df = pd.read_csv('data/drone_image_metadata_202312.csv')
# latitude_array = df['latitude'].values
# longitude_array = df['longitude'].values
# meters_per_pixel_array = df['meters_per_pixel'].values

# # ad_hoc/biomass/estimate_kelp_coverage.py

# ################################################################################################################
# # Get a list of all the images in the folder
# # root = getcwd()
# image_folder = 'images/202312' 
# image_list = listdir(image_folder)
# jpg_drone_images = sorted(\
#                         [image for image in image_list if \
#                         image.lower().endswith('.jpg') and \
#                         not image.lower().startswith('output_')])

# ################################################################################################################
# # Create Kelp Coverage Estimator
# kce = KelpCoverageEstimator()

# ################################################################################################################
# # Get the percentage coverage area from the raw drone images
# image_list = []
# coverage_maps = []
# figures = []
# biomasses = []

# ################################################################################################################
# # Create iterator and loop to process each image
# import matplotlib.pyplot as plt
# iterator = iter( zip(jpg_drone_images, latitude_array, longitude_array, meters_per_pixel_array) )
# for jpg_drone_image, latitude, longitude, meter_per_pixel in iterator:
    
#     print(f'\nProcessing image: {jpg_drone_image}')
    
#     # Set image path
#     # path_to_image = 'images/' + jpg_drone_image

#     # Produce coverage map
#     image = kce.preprocess_image(image_folder + '/' + jpg_drone_image)
#     image = kce.produce_coverage_map(image)
#     # image = kce.get_total_biomass_from_coverage_map( image )
#     # image = kce.insert_covergage_map_into_enclosing_image(image, (latitude, longitude), meter_per_pixel)
#     # plt.imshow( image )
#     # plt.show()
#     kce.create_coverage_map_figure(image, jpg_drone_image, jpg_drone_image, save=True)

        

################################################################################################################
# Plot biomass bar chart
# date_strings = [s[0:-4] for s in jpg_drone_images]
# kce.create_bar_chart_for_total_biomasses(biomasses, date_strings, 'output_bar_chart_of_biomass_over_time')


################################################################################################################
# Create animation - see chatGPT


################################################################################################################
# Get the image-by-images changes

# change_maps = kce.produce_change_maps(coverage_maps)

# for i, dm in enumerate(change_maps):
#     kce.plot_difference_map(dm, name=str(i))

    


#############################################################################
# Create animations

# # Directory containing your .jpg files
# output_file = 'kelp_coverage_over_time.gif'

# # # Create a list of image files
# image_list = listdir(image_folder)
# # # print(image_list)
# output_image_paths = sorted(\
#                         [image for image in image_list if \
#                         image.lower().endswith('.jpg') and \
#                         image.lower().startswith('output_')])

# print(output_image_paths)

# # Create the animation
# with imageio.get_writer(output_file, duration=10) as writer:
#     for img_path in output_image_paths:
#         image = imageio.imread(image_folder + '/' + img_path)
#         writer.append_data(image)


