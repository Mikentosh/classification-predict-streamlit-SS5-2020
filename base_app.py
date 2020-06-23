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

#pyplot
import matplotlib.pyplot as plt

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
    """Tweet Classifier App with Streamlit """

    # Creates a main title and subheader on your page -
    # these are static across all pages
    #st.title("Tweet Classifier")
    #st.subheader("Climate change tweet classification")

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Prediction", "Information", "Sentiment Analysis","Exploratory Data Analysis"]
    selection = st.sidebar.selectbox("Choose Option", options)

    # Building out the "Information" page
    if selection == "Information":
    	st.info("General Information")
    	# You can read a markdown file from supporting resources folder
    	st.markdown("resources/info.md")

    	st.subheader("Raw Twitter data and label")
    	if st.checkbox('Show raw data'): # data is hidden if box is unchecked
    		st.write(raw[['sentiment', 'message']]) # will write the df to the page

        #building out the sentiment analysis page
    if selection == "Sentiment Analysis":
    	st.info("General Information")
    	# You can read a markdown file from supporting resources folder
    	st.markdown("Some information here")

    	st.subheader("Raw Twitter data and label")
    	if st.checkbox('Show raw data'): # data is hidden if box is unchecked
    		st.write(raw[['sentiment', 'message']]) # will write the df to the page



    if selection == "Exploratory Data Analysis":
        st.subheader("Exploratory Data Analysis")
        input_data = st.file_uploader("Upload", type="csv")
        if input_data:
            input_data = pd.read_csv(input_data)

            if st.button("Histogram"):

                input_data['sentiment'].plot.hist()
                plt.ylabel('count')
                st.pyplot()

            if st.button("Pie Chart"):
                input_data['sentiment'].plot.hist()
                st.pyplot()

            if st.button("Word Cloud"):
                input_data['sentiment'].plot.hist()
                st.pyplot()





        # Building out the predication page
    if selection == "Prediction":
        st.info("Classify a single tweet")
        # Creating a text box for user input
        tweet_text = st.text_area("Enter tweet in the box and press classify","Type Here")

        if st.button("Classify"):
        	# Transforming user input with vectorizer
        	vect_text = tweet_cv.transform([tweet_text]).toarray()
        	# Load your .pkl file with the model of your choice + make predictions
        	# Try loading in multiple models to give the user a choice
        	predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
        	prediction = predictor.predict(vect_text)

        	# When model has successfully run, will print prediction
        	# You can use a dictionary or similar structure to make this output
        	# more human interpretable.
        	st.success("Text Categorized as: {}".format(prediction))

        st.info("Classify tweets in a csv file")
        #st.info(__doc__)

        input_data = st.file_uploader("Upload", type="csv")
        if input_data:
            input_data = pd.read_csv(input_data)

            if st.checkbox('Show your data'): #Hide data if box is unchecked

                st.write(input_data.head(5))

                    #st.write('no file was aploaded')

            if st.button("Classify File"):
                #transforming text
                text_data = tweet_cv.transform(input_data['message']).toarray()
                classifier = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))

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
