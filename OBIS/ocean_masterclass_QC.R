# install.packages("Hmisc")
remotes::install_github("iobis/obistools", dependencies = TRUE)

# library(obistools)
library(Hmisc)

base_dir <- "/home/theo/kelp_blue/ad_hoc/OBIS/data/"
events <- read.csv(paste0(base_dir, "6-1-events.csv"))
occurrences <- read.csv(paste0(base_dir, "6-1-occurrences.csv"))

desc <- describe(events)
desc
