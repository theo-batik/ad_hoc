############################################################################################
'''' Imports '''

import cv2
import numpy as np
from os.path import join
from os import getcwd, getenv
import matplotlib.pyplot as plt

############################################################################################
''' Environment variables '''

# HSV filter
HSV_LOW = np.array([getenv("H_LOW"), getenv("S_LOW"), getenv("V_LOW")], dtype=np.uint8)
HSV_HIGH = np.array([getenv("H_HIGH"), getenv("S_HIGH"), getenv("V_HIGH")], dtype=np.uint8)

# Grayscale intensity filter
INTENSITY_LOW = int(getenv('INTENSITY_LOW'))
INTENSITY_HIGH = int(getenv('INTENSITY_HIGH'))

# Length of grid blocks overlayed (within which percentage coverage is computed)
scale = int(getenv('GRID_BLOCK_LENGTH'))

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


# Degrees per meter ratio
DEGREES_PER_METER = float(getenv("DEGREES_PER_METER"))

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


    # def save_cv2_image(self, image, name):
    #     cwd = getcwd()
    #     output_image_name = name + '.jpg'
    #     path_to_output_image = join(cwd, 'images', output_image_name)
    #     jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, 100]
    #     cv2.imwrite(path_to_output_image, image, jpeg_params)


    def produce_coverage_map(self, image):
        ''' 
        Input: binary image, scaling factor
        Returns: grayscale image, reduced by X (scale), where each pixel value represents the 
        percentage coverage of the corresponding X-by-X region of the input image
        '''
    
        # Get the dimensions of the binary image
        height, width = image.shape

        # Calculate the number of rows and columns in the reduced image
        reduced_rows = height // scale
        reduced_cols = width // scale
        region_area = scale * scale

        # Create an empty grayscale image for coverage with reduced dimensions
        coverage_map = np.zeros((reduced_rows, reduced_cols), dtype=np.uint8)

        # Iterate over scale-by-scale regions in the binary image
        for row in range(0, height, scale):
            for col in range(0, width, scale):
                # Extract the scalexscale region
                region = image[row:row+scale, col:col+scale]
                # Calculate the ratio of pixels equal to zero in the region
                ratio = np.sum(region == 0) / region_area

                # Populate the corresponding pixel in the coverage image
                reduced_row = row // scale
                reduced_col = col // scale
                coverage_map[reduced_row, reduced_col] = int(ratio * 100)

        return coverage_map
    

    def plot_coverage_map(self, coverage_map):
        # Plot the coverage map with a color bar
        plt.figure()
        plt.imshow(coverage_map, cmap='viridis', vmin=0, vmax=100)
        plt.colorbar(label='Coverage Percentage')
        plt.title('Biomass and Percentage Kelp Coverage per [] region')
        plt.show()
        # plt.savefig('images/' + 'output_' + name)


    def save_figure(self, array, name, date):

        # Plot the coverage map with a color bar
        fig = plt.figure()

        # Set the extent to change the visual scale
        extent = [0, 496, 0, 360]
        plt.imshow(array, cmap='viridis', vmin=0, vmax=100, extent=extent)

        # plt.annotate('', xy=(array.shape[1] - 50, array.shape[0] - 50), xytext=(array.shape[1] - 50 - 35, array.shape[0] - 50),
        #              arrowprops=dict(facecolor='white', edgecolor='white', arrowstyle='-', linewidth=1))

        # Add labels for x and y axes
        plt.xlabel('Meters')
        plt.ylabel('Meters')
    
        plt.colorbar(label='Coverage (%)', shrink=0.8)

        average_meters_per_pixel_ratio = 0.112424448 # CALCULATE FROM DRONE IMAGE METADATA - average because altitude changes

        grid_block_length = int(round(scale * average_meters_per_pixel_ratio, 0))
        date_formated = date[4:6] + '-' + date[0:4]
        plt.title( f'{date_formated}')
        plt.suptitle(f'Kelp percentage coverage per {grid_block_length}$m^2$ region', y=0.90, fontsize=16, color='darkblue')
        plt.grid(True)

        # Save image
        plt.savefig('images/' + 'output_' + name, dpi=300) 



    def get_distance_between_centres(self, inner_centre_lat, inner_centre_lon):
        '''
        Input: spatial co-ordinates of inner image centre as tuple (longitude, latitude)
        Returns: spatial distance between inner image centre and enclosing image centre
        '''
        Delta_lat = inner_centre_lat - ENCLOSING_IMAGE_CENTRE_LAT
        Delta_lon = inner_centre_lon - ENCLOSING_IMAGE_CENTRE_LON

        return Delta_lat, Delta_lon 


    def convert_length_from_degrees_to_pixels(self, degree_length, meters_per_pixel):
        ''' 
        Input: 
            length in degrees (approximated as Euclidean distance)
            meters-per-pixel ratio of the ORIGINAL image
            scale factor (how many pixels of cover map image correspond to 1 pixel in original)
        Returns: 
            Approx. lengh in number of pixels
        '''
        pixel_length = degree_length / DEGREES_PER_METER / meters_per_pixel

        return pixel_length


        # 1) Get distance spatial between centre points ( Del lon, Del lat) 
    # 2) Convert to pixel distance -> (Del pix_y, Del pix_x)
    # 3) Add (+600, )
    # 4) Insert inner image at (Del pix_y, Del pix_x) 


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


    def compute_coverage_differences(self, image_list):
        differences = np.e

        


        



    



