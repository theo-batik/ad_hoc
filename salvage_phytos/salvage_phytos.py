from azure.storage.blob import BlobServiceClient ##, BlobClient, ContainerClient
import io
import pandas as pd
import os
import json


def retrieve_field_keys(all_columns_from_mapping, page):

    for section in page['section']:
        for field in section['field']:
            if 'subform' in field['fieldKey']:
                all_columns_from_mapping[field['fieldKey']] = field['fieldName']
                all_columns_from_mapping = retrieve_field_keys(all_columns_from_mapping, field['subForm'])
                
            else:
                all_columns_from_mapping[field['fieldKey']] = field['fieldName']


def list_blobs(blob_conn_str, container_name, prefix, include=None):

    blob_service_client = BlobServiceClient.from_connection_string(blob_conn_str)
    container_client = blob_service_client.get_container_client(container_name)
    blobs = container_client.list_blobs(include=include)
    blob_names = [x.name for x in blobs]

    return [x for x in blob_names if x.startswith(prefix)]


def download_blob(blob_conn_str, containername, blobname):
    blob_service_client = BlobServiceClient.from_connection_string(blob_conn_str)
    blob_client = blob_service_client.get_blob_client(container=containername, blob=blobname)
    content = blob_client.download_blob().readall()
    return  io.BytesIO(content)


def open_json(bytes_data):
    
    string_data = bytes_data.getvalue().decode('utf-8')
    return json.loads(string_data)

# List ecology data blobs
connect_str = os.getenv('KELPBLUE_CONNECTION_STRING')
container_name = 'fastfield-forms'
blobs = list_blobs(connect_str, container_name,  prefix='marine-ecology/data/', include=None)

# import random
# random.shuffle(blobs)


# Define the column names
columns = [
    'sampling_site_id', 
    'submission_date_1', 
    'submission_date_2', 
    'depth', 
    'genus', 
    'species', 
    'common_name', 
    'abundance'
]

# Create an empty DataFrame with these columns
df = pd.DataFrame(columns=columns)



for i, blob in enumerate(blobs):

    # Get date
    date_string = blob.split('/')[2]
    month = int(date_string[5:7]) 
    year = int(date_string[0:4]) # 2023-09-13
    
    day = int(date_string[9:12])
    # print(day)


    if (year == 2024 and month >= 2 ) : #and day >= 19):
        
        print('month', month)
        print('year', year)
    
        print('\nBlob: ', blob)
        blob_data = download_blob(connect_str, container_name, blob)

        # blob_mapping_name = blob.split('/')[0] + '/mapping/' +  '_'.join(blob.split('/')[-1].split('_')[:-1])+ '.json'
        # print('Mapping: ', blob_mapping_name)
        # try:
        #     blob_mapping = download_blob(connect_str, container_name, blob_mapping_name)
        # except:
        #     continue

        file_data = open_json(blob_data)
        file_data['listpicker_8']
        print(file_data)
        # file_mapping = open_json(blob_mapping)
        file_data.pop('formMetaData')
        file_data.pop('workflowData')

        if file_data['listpicker_8'][0] == 'Phytoplankton':
            
            print('Blob for Phytoplankton captured')
            # EVENT DATA
            submission_date_1 = file_data.get( 'datepicker_4' , '')
            submission_date_2 = file_data.get( 'datepicker_7' , '')
            sampling_site_id = file_data.get( 'lookuplistpicker_4' , ' ' )[0]
            depth = file_data.get( 'lookuplistpicker_17', ' ')[0]

            # OCCURRENCE DATA
            occurrences = file_data['subform_2'] # list of dict's
            for occ in occurrences:
                # print(occ)
                genus = occ.get( 'lookuplistpicker_10', ' ')[0] if len(occ['lookuplistpicker_10']) > 0 else ''
                species = occ.get( 'lookuplistpicker_11', ' ')[0] if len(occ['lookuplistpicker_11']) > 0 else ''
                common_name = occ.get( 'lookuplistpicker_12', ' ')[0] if len(occ['lookuplistpicker_12']) > 0 else ''
                abundance = occ.get( 'numeric_1', '')
                print(genus, species, common_name, abundance)
                new_row = pd.DataFrame(
                    {
                    'sampling_site_id': [sampling_site_id], 
                    'submission_date_1': [submission_date_1], 
                    'submission_date_2': [submission_date_2], 
                    'depth': [depth], 
                    'genus': [genus], 
                    'species': [species], 
                    'common_name': [common_name], 
                    'abundance': [abundance]
                    }
                )
                df = pd.concat([df, new_row], ignore_index=True)


            # Write JSON MAPPING and DATA
            # with open(f'salvage_phytos/data/data_{i}', 'w') as file:
            #     json.dump(file_data, file, indent=4)
            # with open(f'data/mapping_{i}.json', 'w') as file:
            #     json.dump(file_mapping, file, indent=4)

        else:
            continue

    else:
        continue

    # if i == 6:
    #     break

df.to_csv('salvage_phytos/phyto_data_2.csv', index=False)
# print(connect_str)


# blob_service_client = BlobServiceClient.from_connection_string(connect_str)

# # Get a client to interact with a specific container and blob

# blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

# # Download blob content
# with open("./downloaded_blob.txt", "wb") as download_file:
#     download_file.write(blob_client.download_blob().readall())
