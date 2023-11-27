################################################################################################################
# Imports
from kelp_coverage_estimator import KelpCoverageEstimator
import pandas as pd 
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from IPython.display import HTML
from os import listdir
# import matplotlib
# import os
import imageio
import numpy as np

################################################################################################################
# Read in drone image metadata
df = pd.read_csv('data/drone_image_metadata.csv')
latitude_array = df['latitude'].values
longitude_array = df['longitude'].values
meters_per_pixel_array = df['meters_per_pixel'].values

################################################################################################################
# Get a list of all the images in the folder
# root = getcwd()
image_folder = 'images' 
image_list = listdir(image_folder)
jpg_drone_images = sorted(\
                        [image for image in image_list if \
                        image.lower().endswith('.jpg') and \
                        not image.lower().startswith('output_')])

################################################################################################################
# Create Kelp Coverage Estimator
kce = KelpCoverageEstimator()

################################################################################################################
# Get the percentage coverage area from the raw drone images
image_list = []
coverage_maps = []
figures = []
biomasses = []

################################################################################################################
# Create iterator and loop to process each image
iterator = iter( zip(jpg_drone_images, latitude_array, longitude_array, meters_per_pixel_array) )
for jpg_drone_image, latitude, longitude, meter_per_pixel in iterator:
    
    print(f'\nProcessing image: {jpg_drone_image}')
    
    # Set image path
    # path_to_image = 'images/' + jpg_drone_image

    # Produce coverage map
    image = kce.preprocess_image(image_folder + '/' + jpg_drone_image)
    image = kce.produce_coverage_map(image)
    image = kce.insert_covergage_map_into_enclosing_image(image, (latitude, longitude), meter_per_pixel)

    # Get total biomass
    biomasses.append(kce.get_total_biomass_from_coverage_map(image))

    figures.append( kce.create_coverage_map_figure(image, 'output_coverage_map_' + jpg_drone_image, jpg_drone_image, save=True ) )
    coverage_maps.append(image)
        

################################################################################################################
# Plot biomass bar chart
date_strings = [s[0:-4] for s in jpg_drone_images]
kce.create_bar_chart_for_total_biomasses(biomasses, date_strings, 'output_bar_chart_of_biomass_over_time')


################################################################################################################
# Create animation - see chatGPT


################################################################################################################
# Get the image-by-images changes

difference_maps = kce.get_coverage_differences(coverage_maps)

for dm in difference_maps:
    print(dm.shape)
    print('max', dm.max)
    print('min', dm.min)

    kce.plot_difference_map(dm)
    


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


