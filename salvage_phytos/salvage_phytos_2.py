import os
import json 
import pandas as pd 
import numpy as np

# Define the directory path
directory = 'salvage_phytos/data_2'

# Define the column names
columns = [
    'sampling_site_id', 
    'submission_date_1', 
    'submission_date_2', 
    'depth', 
    'genus', 
    'species', 
    'abundance'
]

# Create an empty DataFrame with these columns
df = pd.DataFrame(columns=columns)

# Loop through all files in the directory
for i, filename in enumerate(os.listdir(directory)):
    # Check if the file ends with .json
    if filename.endswith('.json'):
        # Construct the full file path
        
        file_path = os.path.join(directory, filename)

        # Open and read the JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)


        phyto = True if data['listpicker_8'][0] == 'Phytoplankton' else False
        if phyto:
            # print(json.dumps(data, indent=4)) 
            print('\nCapturing Phytos:')

            date_1 = data['datepicker_4'][:10]
            date_2 = data['datepicker_7'][:10]
            print('Date', date_1, date_2)

            siteID = data['lookuplistpicker_4'][0]# ['selectedNames'][0]
            print('siteID', siteID)

            depth = data.get('lookuplistpicker_17', ' ')[0] # if len(data['lookuplistpicker_17']) > 0 else ''
            print('depth', depth)

            occurences = data['subform_2']
            for oc in occurences:

                genus = oc.get('lookuplistpicker_10', '')[0] if len(oc['lookuplistpicker_10']) > 0 else ''
                species = oc.get('lookuplistpicker_11', '')[0] if len(oc['lookuplistpicker_11']) > 0 else ''
                abundance = oc.get('numeric_1', '') 

                print(genus, species, abundance)
                # print(oc['lookuplistpicker_10'][0])

                new_row = pd.DataFrame(
                    {
                    'sampling_site_id': [siteID], 
                    'submission_date_1': [date_1], 
                    'submission_date_2': [date_2], 
                    'depth': [depth], 
                    'genus': [genus], 
                    'species': [species], 
                    'abundance': [abundance]
                    }
                )

                df = pd.concat([df, new_row], ignore_index=True)


df.to_csv('salvage_phytos/phyto_data_2.csv', index=False)
