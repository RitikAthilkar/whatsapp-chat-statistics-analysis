from urlextract import URLExtract
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# import emoji
sentiments = SentimentIntensityAnalyzer()
extract = URLExtract()
def fetch_stats(selected_user,df):

# fetching total msg and word
    if selected_user=="Group Analysis":
        num_message= df.shape[0]
        word = []
        for i in df['message']:
            word.extend(i.split())
        return num_message,len(word)
    else:
        new_df=df[df['Name'] == selected_user]
        num_message = new_df.shape[0]
        word=[]
        for i in new_df['message']:
            word.extend(i.split())

        return num_message,len(word)

#fetching total media sent
def media_count(df):
    num_media = df[df['message'] == '<Media omitted>\n'].shape[0]
    return num_media
#fetching total link sent:
num_link=[]
def link_count(df):
     for i in df['message']:
         num_link.extend(extract.find_urls(i))
     return len(num_link)

#fetching total members
def total_member(df):
    df = df[df['Name'] != 'group_notification']
    num_member = df['Name'].unique()
    return len(num_member)

#fetching most busy days
def most_busy(df):
    x = df['Name'].value_counts().head()
    new_df = round((df['Name'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'Name': 'percentage'})
    new_df = new_df.sort_values(by='percentage', ascending=False)
    return  x,new_df

#creating  word cloud
def creat_wordclouds(selected_user,df):
    if selected_user!='Group Analysis':
        df=df[df['Name']==selected_user]

    f = open('hinglish.txt', 'r')
    unwanted = f.read()

    df = df[df['message'] != '<Media omitted>\n']
    df = df[df['message'] != 'group_notification']
    df = df[df['message'] != 'deleted message']
    df = df[df['message'] != 'message deleted']
    def remove_stop_words(message):
        words = []
        for word in message.lower().split():
                if word not in unwanted:
                    words.append(word)
        return " ".join(words)

    wc = WordCloud(width=500,height=500,min_font_size=15,background_color='black')
    df['message']=df['message'].apply(remove_stop_words)
    img_wc = wc.generate(df['message'].str.cat(sep=" "))
    return img_wc

#fetching most common word used
def common_word(selected_user,df):
    if selected_user != 'Group Analysis':
        df = df[df['Name'] == selected_user]
    f = open('hinglish.txt','r')
    unwanted = f.read()

    df = df[df['message'] != '<Media omitted>\n']
    df = df[df['message'] != 'group_notification']
    df = df[df['message'] != 'deleted']
    df = df[df['message'] != 'message']

    words = []
    for i in df['message']:
        for word in i.lower().split():
            if word not in unwanted:
                words.append(word)
    com_words =pd.DataFrame(Counter(words).most_common(20))
    com_words = com_words.sort_values(by=1, ascending=True)
    return  com_words


#fetching most active time
def most_active_time(selected_user,df):

    if selected_user !='Group Analysis':
        df = df[df['Name'] == selected_user]

    timeline = df.groupby(['Year', 'month_name']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(str(timeline['month_name'][i]) + "-" + str(timeline['Year'][i]))

    timeline['time'] = time

    return timeline

#fetching daily chat analysis
def daily_chat(selected_user,df):
      if selected_user != 'Group Analysis':
         df = df[df['Name'] == selected_user]

         daily = df.groupby(['Days']).count()['message'].reset_index()
         return daily


def daily_timeline(selected_user,df):
    if selected_user != 'Group Analysis':
        df = df[df['Name'] == selected_user]

    daily_timeline = df.groupby('Days').count()['message'].reset_index()

    return daily_timeline
def busy_days(selected_user,df):
    if selected_user != 'Group Analysis':
        df = df[df['Name'] == selected_user]

    return df['day_name'].value_counts()
def busy_months(selected_user,df):
    if selected_user != 'Group Analysis':
        df = df[df['Name'] == selected_user]

    return df['month_name'].value_counts()
def activity_heatmap(selected_user,df):

    if selected_user != 'Group Analysis':
        df = df[df['Name'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap

#sentiment analysis
def sentiment_analysis(selected_user, df):
    if selected_user != 'Group Analysis':
        df = df[df['Name'] == selected_user]
    df["positive"] = [sentiments.polarity_scores(i)["pos"] for i in df["message"]]
    df["negative"] = [sentiments.polarity_scores(i)["neg"] for i in df["message"]]
    df["neutral"] = [sentiments.polarity_scores(i)["neu"] for i in df["message"]]

    x = sum(df["positive"])
    y = sum(df["negative"])
    z = sum(df["neutral"])

    sentiment = []

    def score(a, b, c):
        if (a > b) and (a > c):
            sentiment.append('Positive')
        if (b > a) and (b > c):
            sentiment.append('Negative')
        if (c > a) and (c > b):
            sentiment.append('Neutral')
        return sentiment
    score(x, y, z)
    return  " ".join(sentiment)








