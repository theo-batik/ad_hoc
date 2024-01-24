################################################################################################################
# Imports
from kelp_coverage_estimator import KelpCoverageEstimator
import pandas as pd 
from os import listdir

################################################################################################################
# Read in drone image metadata
df = pd.read_csv('data/drone_image_metadata_202312.csv')
latitude_array = df['latitude'].values
longitude_array = df['longitude'].values
meters_per_pixel_array = df['meters_per_pixel'].values

# ad_hoc/biomass/estimate_kelp_coverage.py

################################################################################################################
# Get a list of all the images in the folder
# root = getcwd()
image_folder = 'images/202312' 
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
import matplotlib.pyplot as plt
iterator = iter( zip(jpg_drone_images, latitude_array, longitude_array, meters_per_pixel_array) )
for jpg_drone_image, latitude, longitude, meter_per_pixel in iterator:
    
    print(f'\nProcessing image: {jpg_drone_image}')
    
    # Set image path
    # path_to_image = 'images/' + jpg_drone_image

    # Produce coverage map
    image = kce.preprocess_image(image_folder + '/' + jpg_drone_image)
    image = kce.produce_coverage_map(image)
    # image = kce.get_total_biomass_from_coverage_map( image )
    # image = kce.insert_covergage_map_into_enclosing_image(image, (latitude, longitude), meter_per_pixel)
    # plt.imshow( image )
    # plt.show()
    kce.create_coverage_map_figure(image, jpg_drone_image, jpg_drone_image, save=True)

        

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


