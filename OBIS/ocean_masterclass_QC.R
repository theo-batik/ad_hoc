# install.packages("Hmisc")
remotes::install_github("iobis/obistools", dependencies = TRUE)

install.packages("devtools")
library(devtools)
devtools::install_github("iobis/obistools")
install.packages("renv")
renv::init()


# library(obistools)
# library(Hmisc)

# base_dir <- "/home/theo/kelp_blue/ad_hoc/OBIS/data/"
# events <- read.csv(paste0(base_dir, "6-1-events.csv"))
# occurrences <- read.csv(paste0(base_dir, "6-1-occurrences.csv"))

# describe_events <- describe(events)
# print(describe_events)


