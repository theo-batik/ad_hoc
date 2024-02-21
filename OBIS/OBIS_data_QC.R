#------------------------------------------------
# SETUP
#------------------------------------------------

library(obistools)
library(Hmisc)

# Base Directory
base_dir <- "//wsl.localhost/Ubuntu/home/theo/kelp_blue/ad_hoc/OBIS/data/"

# Load data 
event <- read.csv( paste0(base_dir, "6-1-events.csv") )
occur <- read.csv(paste0(base_dir, "6-1-occurrences.csv"))

#------------------------------------------------
# OBIS TOOLS
#------------------------------------------------

# Match taxa
worms <- match_taxa(unique(occur$scientificName))
# occur<-merge(occur, worms, by='scientificName')
# colnames(occur)[colnames(occur)] == "scientificNameID.y]" = "scientificNameID"

# Check fields
cef <- check_fields(event)
cof<- check_fields(occur)

# Check locations
plot_map(event)
plot_map_leaflet(event)

# Check depth
cd <-check_depth(event, report=T, depthmargin = 15) 

# Check eventDate format
ced <- check_eventdate(event)

# Check eventID's
ceid <- check_eventids(event)

# Check eventID matches to extension tables
cee <- check_extension_eventids(event, occur)

#------------------------------------------------
# Hmisc
#------------------------------------------------

describe_events <- describe(event)
print(describe_events)
