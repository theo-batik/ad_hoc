# ------------------------------------------------------------------------------

# Imports
import utils as u
import pandas as pd

# Controls & parameters
display = True
image_format = '.JPG'

# Paths
path_to_calibration = 'calibration/input/data/calibration_2024_02_01.csv'
path_to_images = 'calibration/input/images/'
output_path = 'calibration/output/'

# Load data into pandas dataframe
df = pd.read_csv(path_to_calibration)

if display:
    print( '\nCalibration data loaded:')
    print( df.head(5) )

# ------------------------------------------------------------------------------

if display:
    print('\nPROCESSING IMAGES:')

for i, (image_before, image_after) in enumerate(zip( df['image_before_name'].values, df['image_after_name'].values)):
    
    # Check for match
    if not df.at[i, 'match']:
        continue

    if display:
        print('\t', i, '->  Before,', image_before, '| After,', image_after)
    
    # Set paths to raw images
    path_to_image_before = path_to_images + image_before
    path_to_image_after =  path_to_images + image_after

    # Compute meters_per_pixel ratio -> before
    metadata_before = u.get_metadata(path_to_image_before + image_format)
    metadata_before = u.refine_metadata(metadata_before)
    meters_per_pixel_before = u.compute_meters_per_pixel(metadata_before)
    df.at[i, 'meters_per_pixel_before'] = meters_per_pixel_before

    # Compute meters_per_pixel ratio -> after
    metadata_after = u.get_metadata(path_to_image_after + image_format)
    metadata_after = u.refine_metadata(metadata_after)
    meters_per_pixel_after = u.compute_meters_per_pixel(metadata_after)
    df.at[i, 'meters_per_pixel_after'] = meters_per_pixel_after

    if display:
        print(f'\t   ->  Meters-per-pixel ratios: {round(meters_per_pixel_before, 3)}, {round(meters_per_pixel_after, 3)}')

    # Compute canopy area before
    path_to_binary_cropped_image_before = path_to_image_before + '_binary_cropped' + image_format.lower()
    canopy_area_before = u.compute_canopy_area(path_to_binary_cropped_image_before, meters_per_pixel_before)
    df.at[i, 'canopy_area_before'] = canopy_area_before

    # Compute canopy area after
    path_to_binary_cropped_image_after = path_to_image_after + '_binary_cropped' + image_format.lower()
    canopy_area_after = u.compute_canopy_area(path_to_binary_cropped_image_after, meters_per_pixel_after)
    df.at[i, 'canopy_area_after'] = canopy_area_after

    if display:
        print('\t   ->  Canopy areas')

    # Compute canopy area harvested, as difference between canopy areas (i.e. not plot area)
    canopy_area_harvested = canopy_area_before - canopy_area_after
    df.at[i, 'canopy_area_harvested'] = canopy_area_harvested

    if display:
        print(f'\t   ->  Canopy area harvested: {int(canopy_area_harvested)} (m^2)')

    # Compute the biomass density of the area harvested (i.e. "harvest efficiency")
    biomass_density = df.at[i, 'biomass_measured'] / canopy_area_harvested
    df.at[i, 'biomass_density'] = biomass_density
    
    if display:
        print(f'\t   ->  biomass density: {round(biomass_density, 1)} (kg/m^2)')



# ------------------------------------------------------------------------------

# Output results:
    # Write fully populated calibration file to .csv (with timestamped name)
    # Write average canopy biomass density to .csv alongside timestamp

now = u.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
output_file = output_path + 'calibration_output_' + now + '.csv'
df.to_csv(output_file, index=True)
