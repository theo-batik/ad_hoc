############################################################################################
'''' Imports '''

import cv2
import numpy as np
from os.path import join
from os import getcwd, getenv
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable


############################################################################################
''' Environment variables '''

# HSV filter
HSV_LOW = np.array([getenv("H_LOW"), getenv("S_LOW"), getenv("V_LOW")], dtype=np.uint8)
HSV_HIGH = np.array([getenv("H_HIGH"), getenv("S_HIGH"), getenv("V_HIGH")], dtype=np.uint8)

# Grayscale intensity filter
INTENSITY_LOW = int(getenv('INTENSITY_LOW'))
INTENSITY_HIGH = int(getenv('INTENSITY_HIGH'))

# Length of grid blocks overlayed (within which percentage coverage is computed)
scale = int(getenv('GRID_BLOCK_PIXEL_LENGTH'))
average_meters_per_pixel_ratio = 0.108048485 #0.07963395 # 0.081945478 #0.112424448 # CALCULATE FROM DRONE IMAGE METADATA - average because altitude changes (meters per pixel)
grid_block_meters_length = scale * average_meters_per_pixel_ratio

# Enclosing image dimensions
ENCLOSING_IMAGE_PIXEL_WIDTH = int(getenv('ENCLOSING_IMAGE_PIXEL_WIDTH'))
ENCLOSING_IMAGE_PIXEL_LENGTH = int(getenv('ENCLOSING_IMAGE_PIXEL_LENGTH'))
ENCLOSING_IMAGE_CENTRE_LON = float(getenv('ENCLOSING_IMAGE_CENTRE_LON'))
ENCLOSING_IMAGE_CENTRE_LAT = float(getenv('ENCLOSING_IMAGE_CENTRE_LAT'))
# OUTER_TO_INNER_PIXEL_DIST_COLUMN = int(getenv('OUTER_TO_INNER_PIXEL_DIST_COLUMN'))
# OUTER_TO_INNER_PIXEL_DIST_ROW = int(getenv('OUTER_TO_INNER_PIXEL_DIST_ROW'))

OUTER_TO_INNER_PIXEL_DIST_COLUMN = (ENCLOSING_IMAGE_PIXEL_LENGTH - 4000)/2
OUTER_TO_INNER_PIXEL_DIST_ROW = (ENCLOSING_IMAGE_PIXEL_WIDTH - 3000)/2
# print(OUTER_TO_INNER_PIXEL_DIST_COLUMN, OUTER_TO_INNER_PIXEL_DIST_ROW)

# RGB Kelp Colour 
kelp_colour = (170/255, 182/255, 133/255)
blue_colour = (2/255, 91/255, 114/255)

# Degrees per meter ratio
DEGREES_PER_METER = float(getenv("DEGREES_PER_METER"))

# Harvest efficiency
HARVEST_EFFICIENCY = float(getenv('HARVEST_EFFICIENCY'))


############################################################################################
'''Build the coverage estimator class'''

class KelpCoverageEstimator():


    def preprocess_image(self, path_to_image):
        
        # Load the image with openCV
        image = cv2.imread(path_to_image)

        # Convert the image to HSV color space for better filtering
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create a binary mask to extract regions where kelp is present
        mask = cv2.inRange(image, HSV_LOW, HSV_HIGH)

        # Apply mask to create single channel (grayscale) image
        image = cv2.bitwise_and(image, image, mask=mask)

        # Convert the BGR image to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to create a binary mask
        _, image = cv2.threshold(image, INTENSITY_LOW, INTENSITY_HIGH, cv2.THRESH_BINARY) # cv.ADAPTIVE_THRESH_GAUSSIAN_C

        # Define a kernel (structuring element) for morph. operations
        kernel = np.ones((3, 3), np.uint8)  # You can adjust the kernel size as needed

        # Perform erosion
        image = cv2.erode(image, kernel, iterations=2)

        # Perform dilation
        image = cv2.dilate(image, kernel, iterations=2)


        return image


    def produce_coverage_map(self, image):
        ''' 
        Input: 
            Binary image
            Scaling factor
        Returns: 
            Grayscale image, reduced by X (scale), where each pixel value represents the 
            percentage coverage of the corresponding X-by-X region of the input image.
        '''
    
        # Get the dimensions of the binary image
        
        # image = image[130:2130, 0:4000] # temp fix for 202312!!
        height, width = image.shape
        # Calculate the number of rows and columns in the reduced image
        reduced_rows = int(height / scale)
        reduced_cols = int(width / scale)
        region_area = scale * scale

        # Create an empty grayscale image for coverage with reduced dimensions
        coverage_map = np.zeros((reduced_rows, reduced_cols), dtype=np.uint8)

        # Debug
        print('scale',scale)
        print('width', width, 'height', height)
        print('reduced_cols',reduced_cols)
        print('reduced_rows',reduced_rows)

        # Iterate over scale-by-scale regions in the binary image
        for row in range(0, reduced_rows*scale, scale): # Recontruct image dimensions from reduced rows to avoid rounding issues
            for col in range(0, reduced_cols*scale, scale):
                # Extract the scalexscale region
                region = image[row:row+scale, col:col+scale]
                # Calculate the ratio of pixels equal to zero in the region
                ratio = np.sum(region == 0) / region_area

                # Populate the corresponding pixel in the coverage image
                reduced_row = int(row / scale)
                reduced_col = int(col / scale)
                coverage_map[reduced_row, reduced_col] = int(ratio * 100)

        return coverage_map
    

    def find_bottom_left_coordinate(image_shape, centre, meter_per_pixel):
        '''
        Input: 
            Centre coordinate of image (latitude, longitude)
            Meter to pixel ratio
        Returns: 
            The (latitude, longitude) of the bottom left corner of the image, 
            used to translate to UTM coordinates
            '''
        Delta_x = int(image_shape[0]/2)        
        Delta_y = int(image_shape[1]/2)
        pass


    def plot_difference_map(self, difference_map, name, save=True):
        # Plot the coverage difference map with a color bar
        plt.figure()

        # Define a custom colormap from red to green
        cmap_colors = [(1, 0, 0), (0, 1, 0)]  # Red to green
        custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', cmap_colors, N=50)

        # Find min and max difference
        min_diff = np.min(difference_map) * 100
        max_diff = np.max(difference_map) * 100
        print('min_diff', min_diff)
        print('max_diff', max_diff)

        # Axes
        plt.xlabel('Meters')
        plt.ylabel('Meters')
    
        extent = [0, 496, 0, 360]
        plt.imshow(difference_map, cmap=custom_cmap, vmin=min_diff, vmax=max_diff, extent=extent)
        plt.colorbar(label='Relative difference (%)')
        grid_block_meters_length_rounded = int(round(grid_block_meters_length, 0))
        plt.title(f'Biomass changes per ${grid_block_meters_length_rounded}m^2$ region') # Link to grid block length 
        
        if save:
            plt.savefig('images/' + 'output_' + name + '.jpg', dpi=1500)
        


    def create_bar_chart_for_total_biomasses(self, biomasses, dates, name, save=True):
        # Convert date strings to datetime objects
        dates = [date[4:6] + '-' + date[0:4] for date in dates]
        dates = [datetime.strptime(date, "%m-%Y") for date in dates]
        dates = [date.strftime("%B %Y") for date in dates]  # Format the datetime object as "Month Year"

        # Create a bar graph
        fig, ax = plt.subplots()
        bars = ax.bar(dates, biomasses, color=kelp_colour)

        # Customize the plot
        plt.title('Kelp Biomass Estimation: total available by month', fontsize=12, loc='left', color=blue_colour, pad=5)
        plt.xlabel('Month')
        plt.ylabel('Biomass (tonnes)')
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability

        # Add values on top of the bars
        for bar, biomass in zip(bars, biomasses):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f'{biomass:.2f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()  # Adjust layout for better appearance

        # Save image
        name = name + '.jpg'
        if save:
            plt.savefig('images/' + name, dpi=1500)


    def create_coverage_map_figure(self, image_array, name, date, save=False):

        # Create new figure
        fig = plt.figure()

        # Define the custom colormap colors
        light_blue = (0.95, 0.98, 1.0)   # Lighter blue (i.e. Ocean) # (0.85, 0.95, 1.0) 

        # Create a custom colormap
        cmap_colors = [light_blue, kelp_colour] # (136/255, 160/255, 146/255) #  # (0.4, 0.22, 0.141)  # More brownish green (i.e. Kelp)
        custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', cmap_colors, N=256)

        # Set the extent to change the visual scale
        extent = [0, 496, 0, 360] # Automate extent calculation
        im = plt.imshow(image_array, cmap=custom_cmap, vmin=0, vmax=100, extent=extent)

        # Add labels for x and y axes
        plt.xlabel('Meters')
        plt.ylabel('Meters')
        
        # Scale & grid
        grid_block_meters_length_rounded = int(round(grid_block_meters_length, 0))
        plt.grid(True)
    
        # Add the colorbars
        cbar1 = plt.colorbar(im, label=f'Percentage coverage (%)*', shrink=0.6)
        cbar1.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
        cbar2 = plt.colorbar(im, ax=plt.gca(), label=f'Biomass (kg)*', shrink=0.6, pad=0.05)

        # Manually set colorbar 2 ticks
        pixel_area = (scale * average_meters_per_pixel_ratio)**2
        max_biomass = HARVEST_EFFICIENCY * pixel_area
        cbar2.set_ticklabels( np.around(np.linspace(0, max_biomass, 50), decimals=1) )
        
        # Date
        date = date[4:6] + '-' + date[0:4]
        date = datetime.strptime(date, "%m-%Y")
        date = date.strftime("%B %Y") # Format the datetime object as "Month Year"
        # date = '2023-12-28'
        
        # Get total biomass
        total_biomass = self.get_total_biomass_from_coverage_map(image_array)

        plt.title( f'Kelp Biomass Estimation - {date}', fontsize=12, loc='left', color=blue_colour, pad=5)
        footnote_1 = f'Total biomass available: {round(total_biomass, 5)} (tonnes)'
        footnote_2 = f'*Per ${grid_block_meters_length_rounded}m^2$ (pixel) region based on corresponding proportion of visible canopy area\n \
            and a harvesting efficiency of {HARVEST_EFFICIENCY} ($kg/m^2$).'
        plt.text(0, -0.26, footnote_1, ha='left', va='center', transform=plt.gca().transAxes, fontsize=8)
        plt.text(0, -0.34, footnote_2, ha='left', va='center', transform=plt.gca().transAxes, fontsize=8)


        # Save image
        if save:
            plt.savefig('images/' + name, dpi=1500) 
        
        return fig


    def get_distance_between_centres(self, inner_centre_lat, inner_centre_lon):
        '''
        Input: 
            spatial co-ordinates of inner image centre as tuple (longitude, latitude)
        Returns: 
            spatial distance between inner image centre and enclosing image centre
        '''
        Delta_lat = inner_centre_lat - ENCLOSING_IMAGE_CENTRE_LAT
        Delta_lon = inner_centre_lon - ENCLOSING_IMAGE_CENTRE_LON

        return Delta_lat, Delta_lon 


    def convert_length_from_degrees_to_pixels(self, degree_length, meters_per_pixel):
        ''' 
        Input: 
            length in degrees (approximated as Euclidean distance)
            meters-per-pixel ratio of the ORIGINAL image
            # scale factor (how many pixels of cover map image correspond to 1 pixel in original)
        Returns: 
            Approx. length in number of pixels
        '''
        pixel_length = degree_length / DEGREES_PER_METER / meters_per_pixel
        return pixel_length
    

    def convert_length_from_pixels_to_degrees(self, pixel_length, meters_per_pixel):
        ''' 
        Input: 
            length in pixels
            meters-per-pixel ratio of the ORIGINAL image
            # scale factor (how many pixels of cover map image correspond to 1 pixel in original)
        Returns: 
            Approx. length in degrees
        '''
        degree_length = pixel_length * DEGREES_PER_METER * meters_per_pixel
        return degree_length


    def get_total_biomass_from_coverage_map(self, coverage_map):
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
    

    def insert_covergage_map_into_enclosing_image(self, coverage_map, centre, meters_per_pixel):
        '''
        Input:
            Coverage map image
            Centre co-ordinate (longitude, latitude) of coverage map
            Meters-per-pixel ration of coverage map 
        Returns:
            Image with the coverage map placed within the enclosing map at the correct point in space
        '''

        # Define blank enclosing image
        enclosing_image =  np.zeros((int(ENCLOSING_IMAGE_PIXEL_WIDTH/scale), int(ENCLOSING_IMAGE_PIXEL_LENGTH/scale)), np.uint8)
        
        # self.save_coverage_map(enclosing_image, 'before')

        # print(2*'\n')
        # print('coverage_map shape', coverage_map.shape)
        # print('enclosing image shape', enclosing_image.shape)
        # print(centre)
        
        # Compute distance between centres and 
        Delta_lat, Delta_lon = self.get_distance_between_centres( centre[0], centre[1] )

        # print( 'Delta_lon, Delta_lat', Delta_lon, Delta_lat )
    
        # Compute insert indices
        Delta_pixel_columns = self.convert_length_from_degrees_to_pixels(Delta_lon, meters_per_pixel)
        Delta_pixel_rows = self.convert_length_from_degrees_to_pixels(Delta_lat, meters_per_pixel)

        # print('Delta_pixel_columns', Delta_pixel_columns )
        # print('Delta_pixel_rows', Delta_pixel_rows)

        # print('OUTER_TO_INNER_PIXEL_DIST_COLUMN', OUTER_TO_INNER_PIXEL_DIST_COLUMN)
        # print('OUTER_TO_INNER_PIXEL_DIST_ROW', OUTER_TO_INNER_PIXEL_DIST_ROW)
            
        insert_at_column = int((OUTER_TO_INNER_PIXEL_DIST_COLUMN + Delta_pixel_columns)/scale)
        insert_at_row = int((OUTER_TO_INNER_PIXEL_DIST_ROW + Delta_pixel_rows)/scale)
        
        enclosing_image[insert_at_row:insert_at_row + coverage_map.shape[0], \
                        insert_at_column:insert_at_column + coverage_map.shape[1]] = coverage_map

        return enclosing_image


    def produce_change_maps(self, image_arrays_list):

        # Initialize the list to store the figures
        changes = []

        # Loop through each pair of consecutive images
        for i in range(1, len(image_arrays_list)):
            # Get images from the list
            image1 = image_arrays_list[i - 1]
            image2 = image_arrays_list[i]

            # Ensure images have the same shape
            min_shape = [min(image1.shape[i], image2.shape[i]) for i in range(2)]
            image1 = image1[:min_shape[0], :min_shape[1]]
            image2 = image2[:min_shape[0], :min_shape[1]]

            # Compute pixel-wise absolute difference
            change_map = np.divide( np.subtract(image1, image2), np.where(image1 != 0, image1, np.nan))

            # Append the figure to the list
            changes.append(change_map)

        return changes
        


        



    



