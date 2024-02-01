# ------------------------------------------------------------------------------

# Imports

import utils
import pandas as pd
from os import getenv

# ------------------------------------------------------------------------------

# Controls
display = True

# Load environment variables
IMAGE_SENSOR_WIDTH = float(getenv("IMAGE_SENSOR_WIDTH"))
CAMERA_FOCAL_LENGTH = float(getenv("CAMERA_FOCAL_LENGTH"))

# Paths
path_to_calibration = 'calibration/input/data/calibration_2024_02_01.csv'
path_to_images = 'calibration/input/images/'

# ------------------------------------------------------------------------------

# Load data into pandas dataframe
df = pd.read_csv(path_to_calibration)

if display:
    print( '\nCalibration data loaded:')
    print( df.head(5) )

# ------------------------------------------------------------------------------

drone_images_before = df['image_before_name'].values
print( type(drone_images_before ))
# Compute meter-to-pixel ratio for each raw drone image -> before
    # Extract metadata
        # Altitude
    # Compute the meters_per_pixel ratio, r
    # Insert r into column

for image_before, image_after in zip( df['image_before_name'].values, df['image_after_name'].values):
    print( image_before, image_after)

    


# ------------------------------------------------------------------------------

# Compute canopy area for each cropped binary image
    # canopy_area = SUM( black pixels ) * (r**2)

# ------------------------------------------------------------------------------

# Compute difference in area (i.e. estimate of canopy area harvested) for each before/after pair
    # Delta_A = A_before - A_after ~ area_harvested


# ------------------------------------------------------------------------------

# Compute the biomass density of the area harvested (i.e. "harvested efficiency")
    # biomass_density = biomass_measured / area_harvested

# ------------------------------------------------------------------------------

# Calculate the average canopy biomass density (harvest efficiency)

# ------------------------------------------------------------------------------

# Output results:
    # Write fully populated calibration file to .csv (with timestamped name)
    # Write average canopy biomass density to .csv alongside timestamp
