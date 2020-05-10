from bs4 import BeautifulSoup
import requests
import pandas as pd
from datetime import date, timedelta, datetime
import re


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


if __name__ == "__main__":
    print(get_headlines("AAPL", 10))
