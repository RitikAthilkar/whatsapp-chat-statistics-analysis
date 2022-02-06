import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import processor, function

# creating side bar

st.sidebar.title("Whatsapp Chat Analyzer")
uploaded_file = st.sidebar.file_uploader("Choose a file")

# making menu for uploading file

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = processor.preprocess(data)
    # st.dataframe(df)

    name_list = df['Name'].unique().tolist()
    name_list.sort()
    name_list.insert(0, "Group Analysis")
    name_list.remove('group_notification')
    #
    # date_list = date['time'].tolist()
    # date_list.sort()
    # date_list.insert(0, "Group Analysis")
    # date_list.remove('group_notification')

    # creating name menu

    selected_user = st.sidebar.selectbox("select wrt", name_list)
    # selected_date = st.sidebar.selectbox("select date", date_list)

    if st.sidebar.button("Analyse"):
        st.title("General Statistics")
        num_messages, word = function.fetch_stats(selected_user, df)
        num_media=function.media_count(df)
        num_link = function.link_count(df)
        num_member = function.total_member(df)
        sentiment = function.sentiment_analysis(selected_user, df)

        col1,col2,col3,col4,col5,col6= st.columns(6)

        with col1:
            st.header("Total Member")
            st.title(num_member)
        with col2:
            st.header("Total message")
            st.title(num_messages)
        with col3:
            st.header("Total words")
            st.title(word)
        with col4:
            st.header("Total media")
            st.title(num_media)
        with col5:
            st.header("Total  links")
            st.title(num_link)
        with col6:
            st.header("sentiment analysis")
            st.title(sentiment)
# most Active members

        st.title("Most Active Members")
        if selected_user=="Group Analysis":
            x,new_df = function.most_busy(df)
            fig,ax = plt.subplots()
            col1, col2 = st.columns(2)
            with col1:
               ax = plt.bar(x.index,x.values)
               plt.xticks(rotation='vertical')
               st.pyplot(fig)

            with col2:
               st.dataframe(new_df)
        else:
            st.text("Only for group analysis")

# heatmap
        st.title("Weekly Activity Map")
        user_heatmap = function.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

# daily chat analysis

        st.title("Daily chat analysis")

        daily_timeline = function.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.bar(daily_timeline['Days'], daily_timeline['message'], color='#B03A2E', edgecolor='#922B21')
        st.pyplot(fig)

# most active chat time
        st.title("Chat Active Time Analysis")
        col1, col2 = st.columns(2)
        with col1:
           busy_day = function.busy_days(selected_user, df)
           fig, ax = plt.subplots()
           ax.bar(busy_day.index, busy_day.values,
                    color=['#6C3483', '#76448A', '#884EA0', '#9B59B6', '#AF7AC5', '#D7BDE2', '#D2B4DE'])
           plt.xticks(rotation='vertical')
           st.pyplot(fig)

        with col2:
           busy_month = function.busy_months(selected_user, df)
           fig, ax = plt.subplots()
           ax.bar(busy_month.index, busy_month.values,
                   color=['#7B241C', '#78281F', '#512E5F', '#4A235A', '#154360', '#1B4F72', '#0E6251', '#0B5345',
                          '#145A32', '#7D6608', '#7E5109', '#784212'])
           plt.xticks(rotation='vertical')
           st.pyplot(fig)
            # most active time

        st.title("Active Time Analysis")
        timeline = function.most_active_time(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # most common word
        st.title("Most common words used")
        com_words = function.common_word(selected_user, df)
        fig, ax = plt.subplots()
        ax = plt.barh(com_words[0], com_words[1])
        # plt.xticks(rotation='vertical')
        st.pyplot(fig)


        # word cloud

        st.title("Word cloud")
        img_wc= function.creat_wordclouds(selected_user,df)
        fig,ax= plt.subplots()
        ax.imshow(img_wc)
        st.pyplot(fig)

        # sentiment detetction

