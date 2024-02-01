import pandas as pd

# File path to the Parquet file
parquet_file_path = 'data/Operationalsites-0.parquet'

# Read Parquet file into a Pandas DataFrame
df = pd.read_parquet(parquet_file_path, engine='pyarrow')

# Display the DataFrame
print("First few rows of the DataFrame:")
print(list(df))   


print(df)