install.packages('nflverse')
library(nflverse)
library(tidyverse)

#games from 1990-2022 data
games<-read.csv('games.csv')
view(games)

#Team DVOA from 2011-21, used for projecting team overall strength (Defense-adjusted Value Over Average,
#compares a team's performance to a league baseline based on situation in order to determine value over average)
#sourced from footballoutsiders.com, the creators of DVOA
DVOA<-read.csv('DVOA_2011_2021.csv')
view(DVOA)
#changing LAR to LA to fit with games.csv abbreviations
DVOA$Team[DVOA$Team=='LAR']<-'LA'

#see a breakdown of the columns
summary(games)

#remove observations where no gambling odds were recorded, over_odds were chosen to be used for is.na
#because columns with na for betting data all were missing this column
games_nona<-games[!is.na(games$over_odds),]
view(games_nona)

#selecting all games except for the current season because it hasn't concluded
games_nona2<-games_nona[games_nona$season!=2022,]
view(games_nona2)

#games with no wind recorded were all indoor/Dome, therefore the wind is 0
games_nona2$wind[is.na(games_nona2$wind)]<-0

#add outdoor column
games_nona3<-games_nona2 %>% mutate(outdoor = ifelse(roof=='outdoors',1,0))
view(games_nona3)

#add grass column to distinguish if field was turf or not
games_nona4<-games_nona3 %>% mutate(grass = ifelse(surface=='grass'|surface=='grass ',1,0))
view(games_nona4)

#add playoff column, if game_type is REG then it is not a playoff game
games_nona4<-games_nona4 %>% mutate(playoff = ifelse(game_type=='REG',0,1))

#Limiting data to just the 2011 to 2021 season due to NAs in certain columns
games_nona5<-games_nona4[games_nona4$season>2010,]
view(games_nona5)

#joining the DVOA table for home team DVOA ranks and total DVOA
games_nona6<-games_nona5 %>% left_join(DVOA, by=c("season"="Season", "home_team"="Team"))
view(games_nona6)

#renaming DVOA columns to represent that its the home team's DVOA
games_nona6<-games_nona6 %>% rename(home_DVOA_Rank=Total.DVOA.Rank,home_DVOA=Total.DVOA)

#joining the DVOA table for away team DVOA ranks and total DVOA
games_nona6<-games_nona6 %>% left_join(DVOA, by=c("season"="Season", "away_team"="Team"))

#renaming DVOA columns to represent that its the home team's DVOA
games_nona6<-games_nona6 %>% rename(away_DVOA_Rank=Total.DVOA.Rank,away_DVOA=Total.DVOA)

#add home_win column that will be used for predictions. result is (home score - away score) 
#So if result is greater than 0, the home team won the game
games_nona7<-games_nona6 %>% mutate(home_win = ifelse(result>0,1,0))
view(games_nona7)

#after noticing the NA's for temperature in domed stadiums, I did research to find that the average temperature 
#for domed stadiums on any given week is ~70 degress with most teams hovering between 68-72 during the game
#based in the data from nflweather.com. So I replaced the NA's with 70 since it was the mean temp for domed stadiums
games_nona8<-games_nona7
games_nona8$temp[is.na(games_nona8$temp)]<-70
view(games_nona8)

#add variables to determine if home/away team is playing the game following their bye week
games_nona9<-games_nona8 %>% mutate(home_afterbye = ifelse(home_rest==(week+1),1,0))
games_nona9<-games_nona9 %>% mutate(away_afterbye = ifelse(away_rest==(week+1),1,0))

#add variable to signify a tie
games_nona10<-games_nona9 %>% mutate(tie = ifelse(result==0,1,0))

#split gametime into gametime_hour and gametime_minute
games_nona11<-games_nona10 %>% separate(gametime,sep=':', into = c('gametime_hour','gametime_minute'), remove = F)
view(games_nona11)

#removing unnecessary columns that are only used for relational data in other tables from nflverse
games_limited<- select(games_nona11,-c(old_game_id,gsis,nfl_detail_id,pff,away_qb_name,home_qb_name,away_coach,home_coach))
view(games_limited)

#verifying to make sure there are no NAs in any column
colSums(is.na(games_limited))

#cleaned file to upload to S3 bucket
write.csv(games_limited,'NFL_2011_21.csv')



