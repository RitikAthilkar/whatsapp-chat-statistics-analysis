import re
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

sentiments = SentimentIntensityAnalyzer()

def preprocess(data):
    pattern = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'
    message = re.split(pattern, data)[1:]
    date = re.findall(pattern, data)
    df = pd.DataFrame({'Date_time': date, 'User_message': message})
    df['Date_time'] = pd.to_datetime(df['Date_time'], format='%d/%m/%y, %H:%M - ')
    name = []
    message = []
    for i in df['User_message']:
        seprate = re.split('([\w\W]+?):\s', i)
        if seprate[1:]:
            name.append(seprate[1])
            message.append(" ".join(seprate[2:]))
        else:
            name.append('group_notification')
            message.append(seprate[0])

    df['Name'] = name
    df['message'] = message
    df = df.drop(['User_message'], axis=1)
    df['Days'] = df['Date_time'].dt.day
    df['Months'] = df['Date_time'].dt.month
    df['Year'] = df['Date_time'].dt.year
    df['Hour'] = df['Date_time'].dt.hour
    df['day_name'] = df['Date_time'].dt.day_name()
    df['month_name'] = df['Date_time'].dt.month_name()
    df['Minutes'] = df['Date_time'].dt.minute

    period = []
    for hour in df[['day_name', 'Hour']]['Hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))

    df['period'] = period

    df["positive"] = [sentiments.polarity_scores(i)["pos"] for i in df["message"]]
    df["negative"] = [sentiments.polarity_scores(i)["neg"] for i in df["message"]]
    df["neutral"] = [sentiments.polarity_scores(i)["neu"] for i in df["message"]]

    x = sum(df["positive"])
    y = sum(df["negative"])
    z = sum(df["neutral"])

    return  df