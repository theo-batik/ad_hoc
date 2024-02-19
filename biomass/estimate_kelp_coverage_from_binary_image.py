# Imports
import utils as u

# biomasses = []
# dates = []

# Preprocess image
image_path = "images/202402/DJI_0143.JPG"
metadata = u.get_metadata(image_path)
metadata = u.refine_metadata(metadata)
meters_per_pixel = u.compute_meters_per_pixel(metadata)

# Process image

# Produce coverage map
coverage_map = u.produce_coverage_map_from_binary(image_path[0:-4]+ '_binary.jpg')

# Output coverage map
date = metadata["DateTimeOriginal"]
name = f'{date}-coverage-map-output'
folder = 'images/202402/'
u.output_coverage_map(coverage_map, folder + name, date, meters_per_pixel, save=True)



# ################################################################################################################
# # Get a list of all the images in the folder
# # root = getcwd()
# image_folder = 'images/202312' 
# image_list = listdir(image_folder)
# jpg_drone_images = sorted(\
#                         [image for image in image_list if \
#                         image.lower().endswith('.jpg') and \
#                         not image.lower().startswith('output_')])



