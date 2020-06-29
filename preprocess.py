


def preprocess(df):
    import scipy
    import joblib,os
    import re
    # Data dependencies
    import pandas as pd
    import string
    def make_lower(text):

        """
        This function takes a string of text as input. It makes the strings lowercase and returns the modified text.
        """

        # return lowercase
        return text.lower()


    def clean_url(text):

        """
        This function takes a string of text as input. It replaces the url with with the string 'web-url
        from the text string and returns the modified text string.
        """

        # define RegEx pattern for a url link
        pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'

        # define replacement word
        subs_url = r'web-url'

        # replace urls with 'web-url'
        return re.sub(pattern_url, subs_url, text)


    def remove_punctuation_numbers(text):

        """
        This function takes a string of text as input. It removes punctuation and numbers from the text string
        and returns the modified text string.
        """

        # define punctuation and numbers
        punc_numbers = string.punctuation + '0123456789'


        # return the modeified text
        return ' '.join([l for l in text.split() if l not in punc_numbers])

    def clean(tweet):

        """
        This function takes some words that are contracted and expands them
        """
        tweet = re.sub(r"he's", "he is", tweet)
        tweet = re.sub(r"there's", "there is", tweet)
        tweet = re.sub(r"We're", "We are", tweet)
        tweet = re.sub(r"That's", "That is", tweet)
        tweet = re.sub(r"won't", "will not", tweet)
        tweet = re.sub(r"they're", "they are", tweet)
        tweet = re.sub(r"Can't", "Cannot", tweet)
        tweet = re.sub(r"wasn't", "was not", tweet)
        tweet = re.sub(r"aren't", "are not", tweet)
        tweet = re.sub(r"isn't", "is not", tweet)
        tweet = re.sub(r"What's", "What is", tweet)
        tweet = re.sub(r"i'd", "I would", tweet)
        tweet = re.sub(r"should've", "should have", tweet)
        tweet = re.sub(r"where's", "where is", tweet)
        tweet = re.sub(r"we'd", "we would", tweet)
        tweet = re.sub(r"i'll", "I will", tweet)
        tweet = re.sub(r"weren't", "were not", tweet)
        tweet = re.sub(r"They're", "They are", tweet)
        tweet = re.sub(r"let's", "let us", tweet)
        tweet = re.sub(r"it's", "it is", tweet)
        tweet = re.sub(r"can't", "cannot", tweet)
        tweet = re.sub(r"don't", "do not", tweet)
        tweet = re.sub(r"you're", "you are", tweet)
        tweet = re.sub(r"i've", "I have", tweet)
        tweet = re.sub(r"that's", "that is", tweet)
        tweet = re.sub(r"i'll", "I will", tweet)
        tweet = re.sub(r"doesn't", "does not", tweet)
        tweet = re.sub(r"i'd", "I would", tweet)
        tweet = re.sub(r"didn't", "did not", tweet)
        tweet = re.sub(r"ain't", "am not", tweet)
        tweet = re.sub(r"you'll", "you will", tweet)
        tweet = re.sub(r"I've", "I have", tweet)
        tweet = re.sub(r"Don't", "do not", tweet)
        tweet = re.sub(r"I'll", "I will", tweet)
        tweet = re.sub(r"I'd", "I would", tweet)
        tweet = re.sub(r"Let's", "Let us", tweet)
        tweet = re.sub(r"you'd", "You would", tweet)
        tweet = re.sub(r"It's", "It is", tweet)
        tweet = re.sub(r"Ain't", "am not", tweet)
        tweet = re.sub(r"Haven't", "Have not", tweet)
        tweet = re.sub(r"Could've", "Could have", tweet)
        tweet = re.sub(r"youve", "you have", tweet)
        tweet = re.sub(r"haven't", "have not", tweet)
        tweet = re.sub(r"hasn't", "has not", tweet)
        tweet = re.sub(r"There's", "There is", tweet)
        tweet = re.sub(r"He's", "He is", tweet)
        tweet = re.sub(r"It's", "It is", tweet)
        tweet = re.sub(r"You're", "You are", tweet)
        tweet = re.sub(r"I'M", "I am", tweet)
        tweet = re.sub(r"shouldn't", "should not", tweet)
        tweet = re.sub(r"wouldn't", "would not", tweet)
        tweet = re.sub(r"i'm", "I am", tweet)
        tweet = re.sub(r"I'm", "I am", tweet)
        tweet = re.sub(r"Isn't", "is not", tweet)
        tweet = re.sub(r"Here's", "Here is", tweet)
        tweet = re.sub(r"you've", "you have", tweet)
        tweet = re.sub(r"we're", "we are", tweet)
        tweet = re.sub(r"what's", "what is", tweet)
        tweet = re.sub(r"couldn't", "could not", tweet)
        tweet = re.sub(r"we've", "we have", tweet)
        tweet = re.sub(r"who's", "who is", tweet)
        tweet = re.sub(r"y'all", "you all", tweet)
        tweet = re.sub(r"would've", "would have", tweet)
        tweet = re.sub(r"it'll", "it will", tweet)
        tweet = re.sub(r"we'll", "we will", tweet)
        tweet = re.sub(r"We've", "We have", tweet)
        tweet = re.sub(r"he'll", "he will", tweet)
        tweet = re.sub(r"Y'all", "You all", tweet)
        tweet = re.sub(r"Weren't", "Were not", tweet)
        tweet = re.sub(r"Didn't", "Did not", tweet)
        tweet = re.sub(r"they'll", "they will", tweet)
        tweet = re.sub(r"they'd", "they would", tweet)
        tweet = re.sub(r"DON'T", "DO NOT", tweet)
        tweet = re.sub(r"they've", "they have", tweet)

        return tweet


    def correct_spelling(text):

        """
        This function takes a string of text as input. It corrects the spelling by applying textblob's correction
        method and returns the modified text string.
        """

        # instantiate TextBlob object
        blob = TextBlob(text)

        # correct spelling and return modified string
        return str(blob.correct())






    train_df = df.copy()
    train_df['message'] = df['message'].apply(clean_url)


    # Remove punctuation
    train_df['message'] = train_df['message'].apply(remove_punctuation_numbers)



    # Make lower case
    train_df['message'] = train_df['message'].apply(make_lower)


    ## Remove rt
    train_df['message'] = train_df['message'].replace(to_replace = r'rt', value = '', regex = True)
    ##cleaning function
    train_df['message'] = train_df['message'].apply(clean)

    lemmatizer = open("resources/models/lemmatizer.pkl","rb")
    lemmatizer = joblib.load(lemmatizer)
    train_df['lemma'] = [' '.join([lemmatizer.lemmatize(word) for word in text.split(' ')])for text in train_df['message']]


    vectorizer = open("resources/models/tf_vectorizer.pkl","rb")
    vectorizer = joblib.load(vectorizer)

    count_vectorizer = open("resources/models/count_vectorizer.pkl","rb")
    count_vectorizer = joblib.load(count_vectorizer)



    vectorized=vectorizer.transform(train_df['lemma'])


    X = train_df['lemma']
    count_vectorized=count_vectorizer.transform(train_df['lemma'])

    X = scipy.sparse.hstack([vectorized, count_vectorized])



    return X
