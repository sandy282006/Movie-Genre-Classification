import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load the dataset
data = pd.read_csv('data/movie_plots.csv')

# Clean text data
def clean_text(text):
    text = re.sub(r'\W', ' ', text)             # Remove special characters
    text = text.lower()                         # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)            # Remove extra spaces
    return text

data['cleaned_plot'] = data['plot'].apply(clean_text)

# Tokenize and remove stopwords
stop_words = set(stopwords.words('english'))
data['tokens'] = data['cleaned_plot'].apply(lambda x: [word for word in word_tokenize(x) if word not in stop_words])

# Save preprocessed data
data.to_csv('data/preprocessed_movie_plots.csv', index=False)