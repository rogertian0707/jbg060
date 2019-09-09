# read.csv("../data/combined_data/combined_csv50.csv")
install.packages("highfrequency")

library(ggplot2)
library(highfrequency)
data <- read.csv("../data/combined_data/combined_csv50.csv",dec = ",")

ggplot(data, aes(TimeStamp, Value)) + geom_line() + scale_x_date(date_labels  = "%d-%m-%Y %H:%M:%S") + xlab("") + ylab("Value")
