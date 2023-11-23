############################################################################################
# Imports

import cv2
import numpy as np
from os.path import join
from os import getcwd
import matplotlib.pyplot as plt

############################################################################################
# Environment variables

# Set the HSV color range for kelp (adjust as needed)
H_LOW, S_LOW, V_LOW = 0, 0, 110
H_HIGH, S_HIGH, V_HIGH = 255, 255, 255
THRESHOLD = 1
# Harvest efficiency (kg/m2)
# 5,0592533

############################################################################################
# Functions

def convert_to_hsv(image_path):
    '''
    Input: String of path to image
    Returns: cv2 image in HSV format
    '''

    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to HSV color space for better color filtering
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    return hsv_image


def apply_hsv_filter(hsv_image):
    '''
    Input: cv2 image in HSV format
    Returns: filtered cv2 image in HSV format  
    '''

    # Define the lower and upper bounds of the kelp color in HSV
    lower_bound = np.array([H_LOW, S_LOW, V_LOW])
    upper_bound = np.array([H_HIGH, S_HIGH, V_HIGH])

    # Create a binary mask for the kelp color
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # Apply mask to create single channel image
    hsv_filtered_image = cv2.bitwise_and(hsv_image,hsv_image,mask=mask)

    return hsv_filtered_image


def make_gray(image):
    '''
    Input: cv2 image in HSV colour format
    Returns: cv2 grayscale image
    '''
    # Convert the BGR image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return gray


def apply_intensity_filter(gray_image, threshold):
    # Apply thresholding to create a binary mask
    _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)

    return binary_image


def produce_coverage_map(binary_image, scale):
    ''' 
    Input: binary image, scaling factor
    Returns: grayscale image, reduced by X (scale), where each pixel value represents the 
    percentage coverage of the corresponding X by X region of the input image
    '''
    
    # Get the dimensions of the binary image
    height, width = binary_image.shape

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
            region = binary_image[row:row+scale, col:col+scale]

            # Calculate the ratio of pixels equal to zero in the region
            ratio = np.sum(region == 0) / region_area

            # Populate the corresponding pixel in the coverage image
            reduced_row = row // scale
            reduced_col = col // scale
            coverage_map[reduced_row, reduced_col] = int(ratio * 100)

    return coverage_map


def plot_coverage_map(coverage_map):
    # Plot the coverage map with a color bar
    plt.imshow(coverage_map, cmap='viridis', vmin=0, vmax=100)
    plt.colorbar(label='Coverage Percentage')
    plt.title('Forestry Map: percentage coverage by [] area')
    plt.show()


# def write_image(image):
#     cwd = getcwd()
#     output_image_name = 'output.jpg'
#     path_to_output_image = join(cwd, 'images', output_image_name)
#     cv2.imwrite(path_to_output_image, image)


def main():

    # Path to the drone image
    image_path = 'images/input.jpg'

    # Convert BGR to HSV image
    hsv_image = convert_to_hsv(image_path)
    print(hsv_image)

    # Filter by HSV values
    hsv_filtered_image = apply_hsv_filter(hsv_image)

    # Convert to grayscale image
    gray_image = make_gray(hsv_filtered_image)

    # Apply intensity filter base on threshold value
    binary_image = apply_intensity_filter(gray_image, threshold=THRESHOLD)

    coverage_map = produce_coverage_map(binary_image, scale=10)

    print(coverage_map)
    print(coverage_map.shape)
    print(type(coverage_map))

    # Save images
    plot_coverage_map(coverage_map)
    # write_image(coverage_map)

if __name__ == "__main__":
    main()
