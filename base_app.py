"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd

#vizualization libraries
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

from preprocess import preprocess
from preprocess import comment


# Vectorizer
news_vectorizer = open("resources/models/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file



# Load your raw data
raw = pd.read_csv("resources/data/train.csv")

# The main function where we will build the actual app
def main():
    """Tweet Classifier App with Streamlit """

    # Creates a main title and subheader on your page -
    # these are static across all pages
    #st.title("Tweet Classifier")
    #st.subheader("Climate change tweet classification")

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = [ "Information","Exploratory Data Analysis","Prediction"]

    selection = st.sidebar.selectbox("Choose Option", options)

    # Building out the "Information" page
    if selection == "Information":
        # st.info("General Information")
    	# You can read a markdown file from supporting resources folder

    	st.markdown("""
        ### What is this?
        This is a web application that classifies sentiment of posts made on twitter.

        """)
        # image = Image.open('images/twitter_sentiment_analysis.png')
        # st.image(image, use_column_width=True)

    	image = Image.open('images/twitter_sentiment_analysis.png')

    	st.image(image, use_column_width=True)

    	st.markdown("""
        ### Get Started!

        ### _Step 1:_  `Perform EDA on raw dataset` \n
            - Choose EDA option on the side bar
            - Look at properties of the dataset
            - Draw insights from visuals (Histogram, WordCloud)

        ### _Step 2:_ ` Make Predictions` \n

            - Choose prediction option on the side bar
            - choose to classify a single tweet or multiple tweets in a CSV
            - Type the text or upload csv file depending on above choice
            - Choose your classification model
            - click on `classify` to perfom the classification

        #### Note: Dataset format \n
            - Dataset must have the same structure as:
            -----------------------------------------------------------
            |message                          |tweetid|
            -----------------------------------------------------------
            |global warming is a serious issue| 1200  |

            - arrangement of the columns is not important, as long as the names are the same

        """)

    	#st.subheader("Raw Twitter data and label")
    	#if st.checkbox('Show raw data'): # data is hidden if box is unchecked
    	#	st.write(raw[['sentiment', 'message']]) # will write the df to the page

        #building out the sentiment analysis page



    if selection == "Exploratory Data Analysis":
        image = Image.open('images/eda-header.png')
        st.image(image, use_column_width=True)

        st.info("""
            Let us start by understanding our data set by perfoming the EDA.
            This will help us make educated assumptions on our data before any predictions and
            guide us to correctly interpret the results.

        """)
        #st.subheader("Exploratory Data Analysis")
        input_data = st.file_uploader("Begin by uploading a csv file", type="csv")
        if input_data:
            if input_data:
                input_data = pd.read_csv(input_data)

                dim = (15.0, 10.0)
                fig, ax = plt.subplots(figsize=dim)

                Visual_choice = ["Dataset properties","Histogram","Pie chart", "Word Cloud"]

                choice = st.selectbox("Choose Visual",Visual_choice)
                if choice == "Dataset properties":

                    st.write("Size of the dataset is", len(input_data),fontsize=30)

                    st.dataframe(pd.DataFrame({'Column name':input_data.columns,
                    'column type':input_data.dtypes,'Missing Data':input_data.isnull().sum()}))

                elif choice =="Histogram":

                    #input_data['sentiment'].plot.hist()
                    #plt.ylabel('count')
                    #st.pyplot()

                    # Setup chart size



                    # Create color palette
                    cmrmap = sns.color_palette('YlGn')
                    sns.set_palette(cmrmap)

                    # Connect data to chart
                    sns.countplot(x='sentiment', data=input_data, order=[-1, 0, 1, 2])

                    # Create labels
                    plt.title('Distribution of Sentiments in the Dataset', fontsize=30)
                    plt.ylabel('Count of Posts',fontsize=30)
                    plt.xlabel('Sentiment Value',fontsize=30)

                    plt.xticks(fontsize=12)
                    plt.yticks(fontsize=12)
                    st.pyplot()

                elif choice == "Pie Chart":
                    input_data['sentiment'].plot.hist()
                    st.pyplot()

                elif choice == "Word Cloud":
                    st.write("Patience, this will take some time")
                    text = ''
                    for tweet in input_data['message']:
                        text =text+ " "+ tweet

                    wordcloud = WordCloud(max_words=50,background_color='white').generate(text)
                    # Display the generated image:
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    st.pyplot()





        # Building out the predication page
    if selection == "Prediction":
        #st.title('Classify tweet text')
        image = Image.open('images/project-name.png')
        st.image(image, use_column_width=True)
        task_list = ['classify single tweet', 'Classify csv file']
        task = st.selectbox("Choose classification task", task_list)
        if task == 'classify single tweet':
        #st.info("Classify a single tweet")
        # Creating a text box for user input
            tweet_text = st.text_area("Enter tweet in the box and press classify","Type Here")
            model_list = ['Logistic Regression','Naive Bayes']
            model_choice = st.selectbox("Select Model",model_list)

            model_dict = {"Logistic Regression":'Logistic_regression.pkl'}
            model = model_dict[model_choice]

            if st.button("Classify"):
            	# Transforming user input with vectorizer
            	vect_text = preprocess(tweet_text,task='single')
            	# Load your .pkl file with the model of your choice + make predictions
            	# Try loading in multiple models to give the user a choice
            	predictor = joblib.load(open(os.path.join("resources/models/"+model),"rb"))
            	prediction = predictor.predict(vect_text)

            	# When model has successfully run, will print prediction
            	# You can use a dictionary or similar structure to make this output
            	# more human interpretable.
            	st.success(comment(prediction))


        else:
            #st.info(__doc__)

            input_data = st.file_uploader("Upload", type="csv")
            if input_data:
                input_data = pd.read_csv(input_data)

                if st.checkbox('Show your data'): #Hide data if box is unchecked

                    st.write(input_data.head(5))

                        #st.write('no file was aploaded')
                model_list = ['Logistic Regression','Naive Bayes']
                model_choice = st.selectbox("Select Model",model_list)

                model_dict = {"Logistic Regression":'Logistic_regression.pkl'}
                model = model_dict[model_choice]

                if st.button("Classify File"):
                    #transforming text
                    text_data = preprocess(input_data, task='csv')
                    classifier = joblib.load(open(os.path.join("resources/models/"+model),"rb"))

                    results = classifier.predict(text_data)
                    input_data['results'] = results

                    visuals = ["Histogram","Pie chart"]
                    Visual_choice = st.selectbox("Visualize your results",visuals)

                    dim = (15.0, 10.0)
                    fig, ax = plt.subplots(figsize=dim)
                    if Visual_choice == "Histogram":

                        cmrmap = sns.color_palette('YlGn')
                        sns.set_palette(cmrmap)

                        # Connect data to chart
                        sns.countplot(x='results', data=input_data, order=[-1, 0, 1, 2])

                        # Create labels
                        plt.title('Distribution of Sentiments in the prediction results', fontsize=30)
                        plt.ylabel('Count of Predictions',fontsize=30)
                        plt.xlabel('Sentiment Value',fontsize=30)

                        plt.xticks(fontsize=12)
                        plt.yticks(fontsize=12)
                        st.pyplot()

                    if Visual_choice == "Pie Chart":

                        # Pie chart, where the slices will be ordered and plotted counter-clockwise:

                        index = input_data['sentiment'].value_counts().index
                        index_dic = {-1:'anti',0:'neutral',1:'pro',2:'news'}

                        labels = index_dic[index[0]], index_dic[index[1]], index_dic[index[2]], index_dic[index[3]]
                        sizes = [round(number*100/len(input_data['sentiment']),2) for number in list(input_data['sentiment'].value_counts())]
                        explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

                        fig1, ax1 = plt.subplots()
                        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                                shadow=True, startangle=90)
                        ax1.axis('equal');  # Equal aspect ratio ensures that pie is drawn as a circle.
                        st.pyplot()


                    if st.button("Save as csv"):
                        csv = results.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="results:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
                        st.markdown(href)


     # Creating a text box for user input








# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
	main()
