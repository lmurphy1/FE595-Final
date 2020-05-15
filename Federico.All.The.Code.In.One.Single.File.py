import nltk

nltk.download("vader_lexicon")
from datetime import date
from datetime import timedelta
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer as vader_sentiment
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yfinance_reader
from sklearn import preprocessing
from sklearn.preprocessing import scale
from bs4 import BeautifulSoup
import requests
import pandas as pd
import re


def update_sentiment_value(input_df_sentiment):

    input_df_sentiment["Score"] = 0.0
    for i, row in input_df_sentiment.iterrows():
        curr_score = float(
            vader_sentiment().polarity_scores(row["Headlines"])["compound"]
        )
        input_df_sentiment.at[i, "Score"] = curr_score
        input_df_sentiment.at[i, "Dates"] = datetime.strptime(
            str(row["Dates"]), "%m/%d/%Y"
        ).date()
    return input_df_sentiment


def get_mean_score_by_date(input_df_sentiment):

    df_sentiment_mean_by_day = input_df_sentiment.groupby(["Dates"]).mean()
    return df_sentiment_mean_by_day


def load_TSLA_Yahoo_Data(input_data_file):

    df_TSLA_data = pd.read_csv(input_data_file)
    df_TSLA_data["Date"] = df_TSLA_data["Date"].astype("datetime64[ns]")
    df_TSLA_data.set_index("Date", inplace=True)
    df_TSLA_data["Returns"] = (
        df_TSLA_data["Adj Close"] / df_TSLA_data["Adj Close"].shift(1) - 1
    )
    return df_TSLA_data


def load_TSLA_by_yfinance_Data(ticker_name):

    yfinance_reader.pdr_override()
    TSLA_Yahoo_data = pdr.get_data_yahoo(
        ticker_name, start="2015-11-15", end=date.today()
    )
    TSLA_Yahoo_data["Returns"] = (
        TSLA_Yahoo_data["Adj Close"] / TSLA_Yahoo_data["Adj Close"].shift(1) - 1
    )
    return TSLA_Yahoo_data


def compare_sentiment_returns(df_sentiment, df_yahoo, shift_value):

    print(df_sentiment["Score"].head)
    df_sentiment["Score"] = df_sentiment.shift(shift_value)
    print(df_sentiment["Score"].head)
    merged_df = pd.merge(
        df_yahoo[["Returns"]],
        df_sentiment[["Score"]],
        left_index=True,
        right_index=True,
        how="left",
    )
    return merged_df


def normalize_dataset(merged_dataset):

    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    scaled_df = scaler.fit_transform(merged_dataset)
    scaled_df = pd.DataFrame(scaled_df, columns=["Score", "Returns"])
    return scaled_df


# New Webscraper - Start


def create_url(ticker, closest_timestamp):

    """
    Creates url string for Seeking Alpha symbol page from Web Archive
    :param ticker: stock ticker
    :param closest_timestamp: date from web archive
    :return: url string to scrape
    """
    return f"https://web.archive.org/web/{closest_timestamp}/http://seekingalpha.com/symbol/{ticker}"


def extract_source(url):

    """
    Get request for webpage source code
    :param url: webpage url to scrape
    :return: Result of get request
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36"
    }
    source = requests.get(url, headers=headers).text
    return source


def format_date(x, curr_date):

    """
    Takes dates from Seeking Alpha and formats them to %m/%d/%Y
    :param x: format of date from Seeking Alpha
    :param curr_date: Date the page was archived
    :return:
    """
    x = re.search("(?<=\•).+?(?=\•)", x)[0]  # get date in between two bullets
    if x.startswith("Today"):
        ret = curr_date
    elif x.startswith("Yesterday"):
        ret = curr_date - timedelta(days=1)
    else:
        try:  # remove edge cases
            if "May" in x:  # may is not abbreviated so date has no "." after month
                match = re.search(r"(\D{3}\ \d{1,2})$", x)
                if match is not None:  # the article is from this year
                    x = (
                        match[0] + f", {curr_date.year}"
                    )  # change it to be year from other loop
                ret = datetime.strptime(x, "%b %d, %Y")
            else:
                match = re.search(r"(\D{3}\.\ \d{1,2})$", x)
                if match is not None:  # the article is from this year
                    x = (
                        match[0] + f", {curr_date.year}"
                    )  # change it to be year from other loop
                ret = datetime.strptime(x, "%b. %d, %Y")
        except:
            ret = None
    if ret is not None:
        ret = ret.strftime("%m/%d/%Y")
    return ret


def bs_get(soup):

    """
    Get headlines and dates from 1 Seeking Alpha Page
    :param soup: BeautifulSoup object of archived Seeking Alpha page
    :return: pandas dataframe of headlines and dates from 1 Seeking Alpha page
    """
    d = re.search("(?<=(FILE ARCHIVED ON )).+?(?=( AND RETRIEVED))", str(soup))[0]
    archive_date = datetime.strptime(d, "%H:%M:%S %b %d, %Y")
    try:
        container = soup.find_all("div", class_="content_block_investment_views")[
            0
        ].find_all("li", class_="symbol_item")
    except:
        container = []
    headlines = []
    dates = []
    for con in container:
        headline = con.find("div", class_="symbol_article").a.get_text()
        headlines.append(headline)
        article_d = con.find("div", class_="date_on_by").text
        article_d = format_date(article_d, archive_date)
        dates.append(article_d)
    df = pd.DataFrame({"Headlines": headlines, "Dates": dates})
    return df


def get_headlines(ticker="TSLA", num_pages=52):

    """
    Get headlines for a given stock ticker from Seeking Alpha.
    Scrapes Web Archive page for Seeking Alpha starting today, going back
    two weeks num_pages times
    :param ticker: stock ticker
    :param num_pages: number of pages to scrape
    :return: pandas dataframe of headlines and date for the ticker
    """
    data = pd.DataFrame()
    current_date = date.today()
    for i in range(num_pages):
        s = extract_source(create_url(ticker, current_date.strftime("%Y%m%d")))
        current_date = current_date - timedelta(days=14)
        soup = BeautifulSoup(s, "html.parser")
        data = data.append(bs_get(soup))
    data = data.dropna().drop_duplicates().reset_index(drop=True)
    return data


# New Webscraper - End


if __name__ == "__main__":

    df_grabbed_sentiment = get_headlines("TSLA", 10)
    print(df_grabbed_sentiment)
    df_grabbed_sentiment = update_sentiment_value(df_grabbed_sentiment)
    TSLA_Yahoo_data = load_TSLA_by_yfinance_Data("TSLA")
    df_mean_score = get_mean_score_by_date(df_grabbed_sentiment)
    for shift_value in range(1, 5):
        merged_df = compare_sentiment_returns(
            df_mean_score, TSLA_Yahoo_data, shift_value
        )
        merged_df = merged_df.fillna(merged_df.mean())
        merged_df.plot(x="Score", y="Returns", style="o")
        plt.title("Returns vs. Score")
        plt.xlabel("Score")
        plt.ylabel("Returns")
        print(merged_df["Returns"].corr(merged_df["Score"]))
        normalized_merged_dataset = normalize_dataset(merged_df)
        print(normalized_merged_dataset)
        normalized_merged_dataset.plot(x="Score", y="Returns", style="o")
        plt.title("Returns vs. Score")
        plt.xlabel("Score")
        plt.ylabel("Returns")
        plt.show()


# For Sanket: I already prepared for you the data that you can take into consideration for your part. Please do not
#             delete the last "for loop" or the other parts of the analysis. I think that you can start your part
#             from the following data (you just need to remove "#"):

# merged_df = compare_sentiment_returns(df_mean_score, TSLA_Yahoo_data, 1)
# merged_df = merged_df.fillna(merged_df.mean())
