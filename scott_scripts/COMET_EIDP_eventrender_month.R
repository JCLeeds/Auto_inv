renv::load('C:/Users/earcwat/Desktop/COMET_LICSAR')
library(rmarkdown)
library(data.table)
library(tidyverse)
library(DT)
library(knitr)
library(kableExtra)
library(lubridate)

usgs_quakes <- read.csv2("http://gws-access.ceda.ac.uk/public/nceo_geohazards/LiCSAR_products/EQ/eqs.csv", dec = ".", fileEncoding="UTF-8-BOM") 
#usgs_quakes <- read.csv2("C:/Users/Scott/Desktop/eq_test.csv", dec = ".")#If running locally 
#usgs_quakes <- filter(usgs_quakes, USGS_ID == "us7000e54r") #Run for specific quake


usgs_quakes$time <- as.POSIXct(usgs_quakes$time) 
usgs_quakes <-arrange(usgs_quakes, desc(time))
usgs_quakes <- dplyr::rename(usgs_quakes, USGS.ID = USGS_ID, Date.Time.UTC = time, Latitude = lat, Longitude = lon, Depth.km = depth, Magnitude = magnitude, LiCSAR.Data = link, Location = location)
#Change column order for the shiny table output
usgs_quakes <- setcolorder(usgs_quakes, c("Location", "Date.Time.UTC", "Magnitude", "Depth.km", "USGS.ID", "Latitude", "Longitude", "LiCSAR.Data"))


#Filter by date
usgs_quakes <- filter(usgs_quakes, Date.Time.UTC >= today() - days(31))
#usgs_quakes <- subset(usgs_quakes,  Date.Time.UTC > "2019-12-31" &  Date.Time.UTC < "2020-08-31")


#Remove Milan's 'dummy' eqs
usgs_quakes <- filter(usgs_quakes, Magnitude >= 5)

#Remove eqs causing script to fail if bug cannot be identified
problem_events <- c('us6000k9rh', 'us6000k9mb')
usgs_quakes<- filter(usgs_quakes, !USGS.ID %in% problem_events)


for (id in unique(usgs_quakes$USGS.ID)){
  subgroup <- usgs_quakes[usgs_quakes$USGS.ID == id,]
  render("C:/Users/earcwat/Desktop/COMET_LICSAR/EIDP/outputs/COMET_earthquake_responder_html_template.rmd",output_file = paste0(id, '.html'))
  }

