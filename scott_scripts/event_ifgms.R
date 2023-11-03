library(data.table)
library(tidyverse)
#library(DT)
library(rvest)
library(dplyr)
#ßlibrary(RCurl)
library(httr)

################################################################################################################
## Set event ID
args <- commandArgs(trailingOnly = TRUE)
id <- args[1]


################################################################################################################



##Function to scrape links from the LiCSAR html pages
scrapelinks <- function(url){
  # Create an html document from the url
  webpage <- xml2::read_html(url)
  # Extract the URLs
  url_ <- webpage %>%
    rvest::html_nodes("a") %>%
    rvest::html_attr("href")
  # Extract the link text
  frame_ <- webpage %>%
    rvest::html_nodes("a") %>%
    rvest::html_text()
  return(data.frame(frame = frame_, url =  url_))
}


##Scrape the html
url <- paste("http://gws-access.ceda.ac.uk/public/nceo_geohazards/LiCSAR_products/EQ/", id,  ".html", sep = "")
scraped_links <- scrapelinks(url)

## Create data frame from scraped info
scraped_links_df <- data.frame(ID = id, Links = scraped_links$frame)
scraped_links_df <- scraped_links_df [- grep("metadata", scraped_links_df$Links),]
scraped_links_df <- scraped_links_df [- grep("km", scraped_links_df$Links),]
scraped_links_df <- scraped_links_df[- grep("USGS", scraped_links_df$Links),] %>% separate(Links, c("frame", "dates"), sep = ": ")
scraped_links_df$frame <- as.factor(scraped_links_df$frame)
scraped_links_df$direction <- str_sub(scraped_links_df$frame, 4,4)
scraped_links_df$track <- str_sub(scraped_links_df$frame, 0,3)
scraped_links_df$ifgm_url <- paste0("https://gws-access.jasmin.ac.uk/public/nceo_geohazards/LiCSAR_products/", scraped_links_df$track,"/", scraped_links_df$frame,"/","interferograms/",scraped_links_df$dates,"/",scraped_links_df$dates,".geo.diff_pha.tif") #to get a hyperlink

## Write out CSV file
write.csv(scraped_links_df,args[2], row.names=F)


