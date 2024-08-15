#!/bin/bash

# List of AphiaIDs
aphia_ids=(
  145199
  221035
  707542
  212636
  159037
  231434
  212636
  127405
  799
  1078
  135220
  852
  101361
  138228
  130123
  1518612
  328803
  883
  830
  149151
  196815
  149092
  149152
  148985
  149236
  100694
  213781
  156859
  334314
  157566
  136141
)

# Loop through each AphiaID and make the GET request
for aphia_id in "${aphia_ids[@]}"
do
  echo "Fetching data for AphiaID: $aphia_id"
  response=$(curl -s -X GET "https://api.obis.org/v3/taxon/$aphia_id" -H "accept: */*")
  scientific_name=$(echo $response | jq -r '.results[0].scientificName')
  echo "Scientific Name: $scientific_name"
  echo -e "\n" # Add a newline for better readability
done

