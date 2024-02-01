// Use DBML to define your database structure
// Docs: https://dbml.dbdiagram.io/docs

Table EventCore {
    eventID varchar [primary key]
    parentEventID varchar
    eventDate datetime
    samplingProtocol varchar
    sampleSizeValue float
    sampleSizeUnit varchar
    year integer
    month integer
    day integer
    location varchar
    locationID varchar
    decimalLongitude float
    decimalLatitude float
    eventRemarks varchar
    minimumDepthInMeters float
    maximumDepthInMeters float
    habitat varchar
    continent varchar
    waterBody varchar
    country varchar
    countryCode varchar(2)
    municipality varchar
    localicy varchar
    institutionID varchar
    institutionCode varchar 
  }
  
  Table OccurrenceExtension {
    eventID integer [primary key]
    username varchar
    role varchar
    created_at timestamp
  }
  
 
  
  Ref: EventCore.eventID > OccurrenceExtension.eventID // many-to-one
  
    