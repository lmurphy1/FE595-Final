# FE595-Final

FE 595 Project – Group 1

The group wanted to create a website that analyzes the relationship between the sentiment of news headlines and stock returns.

We created a webscraper data to gather headlines and dates from Seeking Alpha. The webscraper relies on archived versions of the Seeking Alpha page for TSLA in order to continuously access old headlines. The script pulls the headlines dynamically, so the page takes about a minute to load. This is contained in the file webscraper.py

We performed the sentiment analysis on the data we webscraped from Seeking Alpha and then we also downloaded data regarding Tesla from Yahoo Finance. Furthermore, we studied different aspects regarding the data obtained from the sentiment analysis and the data regarding Tesla taken from Yahoo Finance. This can be found in analysis.py.

With the sentiment scores and TSLA returns, we created various machine learning models and calculated their accuracy.

We developed the flask app to run sentiment analysis and display the output of model accuracies in the html file.

We setup up a AWS EC2 instance and deployed our code.

The url to the output of our code is http://ec2-18-223-187-91.us-east-2.compute.amazonaws.com:1234/
