# ------------------------------------------------------------------------------

# SETUP 

# Imports
import pandas as pd

# Controls & parameters
display = True

# Paths
path_to_calibration_output = 'calibration/output/calibration_output_2024-02-07 12:53:29.csv'
output_path = 'validation/output/'

# Load data into pandas dataframe
df = pd.read_csv(path_to_calibration_output)

# ------------------------------------------------------------------------------

# COMPUTE VALIDATION METRICS

# Compute harvest efficiency mean and standard deviation
biomass_density_semi_predicted = df['biomass_density_semi_predicted']
harvest_efficiency_mean = biomass_density_semi_predicted.mean()
harvest_efficiency_std = biomass_density_semi_predicted.std()

# Compute biomass and biomass density difference
df['biomass_density_difference'] = harvest_efficiency_mean - biomass_density_semi_predicted 
df['biomass_predicted'] = harvest_efficiency_mean * df['canopy_area_harvested']
df['biomass_difference'] = df['biomass_predicted'] - df['biomass_measured']

# Compute the mean absolute errors (MAE) for biomass and biomass density
biomass_MAE = df['biomass_difference'].abs().mean()
biomass_density_MAE = df['biomass_density_difference'].abs().mean()

# Fetch calibration datetime and sample size
calibration_datetime = df['calibration_datetime'].iloc[0]
sample_size = (df['match'] == 1).sum()

# ------------------------------------------------------------------------------

# OUTPUT

if display:
    print(f'Datetime of calibration: {calibration_datetime}')
    print(f'Harvest efficiency mean (semi-predicted): {harvest_efficiency_mean}')
    print(f'Harvest efficiency std (semi-predicted): {harvest_efficiency_std}')
    print(f'Biomass MAE: {round(biomass_MAE, 2)}')
    print(f'Biomass density MAE: {round(biomass_density_MAE, 2)}')
    print(f'Sample size: {sample_size}')

# Create dateframe for validation output 
validation_output_data = {
    'calibration_datetime': [calibration_datetime],
    'harvest_efficiency_mean': [harvest_efficiency_mean],
    'harvest_efficiency_std': [harvest_efficiency_std],
    'biomass_MAE': [biomass_MAE],
    'biomass_density_MAE': [biomass_density_MAE],
    'sample_size': [sample_size]
}
df_validation = pd.DataFrame(validation_output_data)

# Append dateframe to existing validation output file
output_file = output_path + 'validation_output.csv'
df_validation.to_csv(output_file, mode='a', header=False, index=False)

