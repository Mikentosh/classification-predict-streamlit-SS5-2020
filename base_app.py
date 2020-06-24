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
    	#st.info("General Information")
    	# You can read a markdown file from supporting resources folder
    	st.markdown("""
        The aim of this web application is to perform climate change  sentiment classification.
        This will assist market researchers to know how their products may be received by the
        general public in relation with climate change

        """)

    	st.subheader("Raw Twitter data and label")
    	if st.checkbox('Show raw data'): # data is hidden if box is unchecked
    		st.write(raw[['sentiment', 'message']]) # will write the df to the page

        #building out the sentiment analysis page



    if selection == "Exploratory Data Analysis":
        image = Image.open('images/eda-header.png')
        st.image(image, use_column_width=True)

        st.info('I am the goat')
        #st.subheader("Exploratory Data Analysis")
        input_data = st.file_uploader("Upload", type="csv")
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
            	vect_text = tweet_cv.transform([tweet_text]).toarray()
            	# Load your .pkl file with the model of your choice + make predictions
            	# Try loading in multiple models to give the user a choice
            	predictor = joblib.load(open(os.path.join("resources/models/"+model),"rb"))
            	prediction = predictor.predict(vect_text)

            	# When model has successfully run, will print prediction
            	# You can use a dictionary or similar structure to make this output
            	# more human interpretable.
            	st.success("Text Categorized as: {}".format(prediction))


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
                    text_data = tweet_cv.transform(input_data['message']).toarray()
                    classifier = joblib.load(open(os.path.join("resources/models/"+model),"rb"))

                    results = classifier.predict(text_data)
                    st.write(results)

                    if st.button("Save as csv"):
                        csv = results.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="results:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
                        st.markdown(href, unsafe_allow_html=True)


     # Creating a text box for user input








# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
	main()
