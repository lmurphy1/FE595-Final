import nltk

nltk.download("vader_lexicon")
import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer as vader_sentiment
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yfinance_reader
from sklearn import preprocessing


def update_sentiment_value(input_df_sentiment):

    input_df_sentiment["Score"] = 0.0
    for i, row in input_df_sentiment.iterrows():
        curr_score = float(
            vader_sentiment().polarity_scores(row["Headlines"])["compound"]
        )
        input_df_sentiment.at[i, "Score"] = curr_score
        input_df_sentiment.at[i, "Dates"] = datetime.datetime.strptime(
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
        ticker_name, start="2018-11-15", end="2020-05-12"
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


if __name__ == "__main__":

    input_datafile_csv = "new_data.csv"
    df_grabbed_sentiment = pd.read_csv(input_datafile_csv)
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
        print(merged_df["Returns"].corr(merged_df["Score"]))
        normalized_merged_dataset = normalize_dataset(merged_df)
        print(normalized_merged_dataset)
        normalized_merged_dataset.plot(x="Score", y="Returns", style="o")


# For Sanket: I already prepared for you the data that you can take into consideration for your part. Please do not
#             delete the last "for loop" or the other parts of the analysis. I think that you can start your part
#             from the following data (you just need to remove "#"):

# merged_df = compare_sentiment_returns(df_mean_score, TSLA_Yahoo_data, 1)
# merged_df = merged_df.fillna(merged_df.mean())
