# Imports
import pandas as pd
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt


# Load data
df = pd.read_csv('stats/data/BACI_data_hanna_nutrients.csv')

# General cleaning
df = df[df['NO'].str.strip() != ""] # Remove rows where df['NO'] is just a space " "
df['NO'] = df['NO'].astype('float')
df.loc[df['NO']>1, 'NO'] /= 100
df.loc[df['NO2']>1, 'NO2'] /= 100
df['Depth'] = pd.to_numeric(df['Depth'], errors='coerce') # Convert rv to numeric, forcing errors to NaN
# df = df[df['Depth'].str.strip() != ""] # Remove rows where df['NO'] is just a space " "
df = df.dropna(subset=['Depth'])
df = df[ df['Depth'] < 30 ]

# Prepare dataset
rv = 'NO' # response variable 
df = df[['Site', 'Time', 'Site_type', 'Depth', rv]]
df = df[df['Site']!='NA-PS-OFF'] # Remove pilot site (not valid control)
df[rv] = pd.to_numeric(df[rv], errors='coerce') # Convert rv to numeric, forcing errors to NaN
df = df.dropna(subset=[rv]) # Drop rows with NaN values in rv after conversion
df['Time'] = df['Time'].astype('category') # Convert to categorical variables
df['Site_type'] = df['Site_type'].astype('category')
print(df.head())

# Plot distribution of response variable
print('\n', df[['Site_type', rv]].groupby('Site_type').mean())
print('\n', df[['Time', rv]].groupby('Time').mean())

# Plot distribution of control vs impact
plt.figure(2)
plt.hist(df[ df['Site_type']=='Control' ][rv], bins=100, label='Control')  # Adjust the number of bins as needed
plt.hist(df[ df['Site_type']=='Impact' ][rv], bins=100, label='Impact')  # Adjust the number of bins as needed
plt.xlabel(rv)
plt.ylabel('Frequency')
plt.title(f'Histogram of {rv}')
plt.legend()
plt.show()

# Plot distribution of before vs after
plt.figure(3)
plt.hist(df[ df['Time']=='Before' ][rv], bins=100, label='Before')
plt.hist(df[ df['Time']=='After' ][rv], bins=100, label='After')
plt.xlabel(rv)
plt.ylabel('Frequency')
plt.title(f'Histogram of {rv}')
plt.legend()
plt.show()


# Define model and get result
model = smf.mixedlm(f'{rv} ~ Site_type * Time', df, groups=df['Site'])
result = model.fit(full_output=True) # maxiter=1000,


# Summary of the model
print(result.summary())


# Visualize data

# Create an interaction plot with custom error bars
plt.figure(1)
sns.pointplot(
    x='Time', 
    y=rv, 
    hue='Site_type',
    order=['Before', 'After'],
    data=df, 
    errorbar=('ci', 95),
    err_kws={'linewidth': 1.5},           # Increase the width of the error bars
    capsize=0.05,            # Add caps to the error bars
    # err_kws={'color': 'black'}  # Change the color of the error bars to black
)
plt.title(f'BACI Analysis: Concentration of {rv} Over Time')
plt.ylabel(f'{rv} Concentration')
plt.xlabel('Time')
plt.show()

