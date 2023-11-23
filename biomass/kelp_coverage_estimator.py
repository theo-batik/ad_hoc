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


############################################################################################
'''Build the coverage estimator class'''

class KelpCoverageEstimator():


    # def __init__(self):
        
        # self.hsv_low = HSV_LOW
        # self.hsv_high = HSV_HIGH
        # self.intensity_low = INTENSITY_LOW
        # self.intensity_high = INTENSITY_HIGH
        # self.grid_block_length = GRID_BLOCK_LENGTH


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


    def save_image(self, image, name):
        cwd = getcwd()
        output_image_name = name + '.jpg'
        path_to_output_image = join(cwd, 'images', output_image_name)
        cv2.imwrite(path_to_output_image, image)


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
        plt.imshow(coverage_map, cmap='viridis', vmin=0, vmax=100)
        plt.colorbar(label='Coverage Percentage')
        plt.title('Forestry Map: percentage coverage by [] area')
        plt.show()

    



