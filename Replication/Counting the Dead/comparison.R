### Comparing death counts in ACLED and WWAD ###
#############################################
rm(list=ls())
library(ggplot2)
library(dplyr)
library(countrycode)
library(lubridate)
library(zoo)
library(reshape2)

setwd('~/Blog/data/ACLED v WWAD/')

acled = read.csv('acled.csv')
wwad = read.csv('pitf_atrocities.csv')

acled = select(acled, EVENT_DATE, YEAR, EVENT_TYPE, INTER2, COUNTRY, ADMIN1, ADMIN2, LOCATION, FATALITIES)
acled = filter(acled, INTER2==7, FATALITIES>=5, YEAR>=1997, YEAR<=2012, FATALITIES<=1400)
acled$iso = countrycode(acled$COUNTRY, "country.name", "iso3c")
acled$month = month(as.Date(acled$EVENT_DATE, "%d %b %Y"))
acled = select(acled, iso, YEAR, month, FATALITIES)
colnames(acled) = c('iso','year','month','acled.deaths')

#wwad = filter(wwad, Event.Type=="Incident")
wwad$Country[wwad$Country=='SUD'] = 'SDN'
wwad$Country[wwad$Country=='Nigeria'] = 'NGA'
wwad$Country[wwad$Country=="South Sudan "]="SSD"
wwad$Country[wwad$Country=="South Sudan"]="SSD"
wwad$Country[wwad$Country=="Somalia"]="SOM"
wwad$Country[wwad$Country=="SOM "]="SOM"
wwad = select(wwad, Country, Start.Year, Start.Month, Deaths.Number)
wwad$region = countrycode(wwad$Country, "iso3c", "continent")
wwad$Deaths.Number[wwad$Deaths.Number=="dozens"]=24
wwad$deaths = as.numeric(as.character(wwad$Deaths.Number))

wwad = filter(wwad, region == 'Africa', Start.Year>=1997, Start.Year<=2012, deaths<=5000)
wwad$Deaths.Number = wwad$region = NULL

colnames(wwad) = c('iso','year','month','wwad.deaths')
wwad = na.omit(wwad)

countries = unique(acled$iso)
years = seq(1997, 2012, 1)
months = seq(1, 12, 1)
grid = expand.grid(countries, years, months)
colnames(grid) = c('iso','year','month')

wwad$wwad.deaths[wwad$wwad.deaths == NA] = 0 
monthly = group_by(wwad, iso, year, month)
wwad = summarise(monthly, wwad.deaths = sum(wwad.deaths))

acled$acled.deaths = as.numeric(acled$acled.deaths)
monthly = group_by(acled, iso, year, month)
acled = summarise(monthly, acled.deaths = sum(acled.deaths))

data = merge(grid, wwad, by=c('iso','year','month'), all.x=TRUE)
data = merge(data, acled, by=c('iso','year','month'), all.x=TRUE)
data$wwad.deaths[is.na(data$wwad.deaths)] = 0
data$acled.deaths[is.na(data$acled.deaths)] = 0
data = transform(data, Date = as.Date(paste(year, month, 1, sep = "-")))


data_long = melt(data, id=c("Date", "iso","year","month"))
ggplot(data_long, aes(Date, value, colour=variable)) + 
        geom_line(alpha=.5) +
        ggtitle("Violence in Africa") +
        ylab("Deaths") 

congo = filter(data, iso=='COD')
congo_long = melt(congo, id=c("Date", "iso","year","month"))
ggplot(congo_long, aes(Date, value, colour=variable)) + 
        geom_line(alpha=.5) +
        ggtitle("Violence in the DRC") +
        ylab("Deaths")
