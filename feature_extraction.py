from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Load preprocessed data
data = pd.read_csv('data/preprocessed_movie_plots.csv')

# Convert tokens back to cleaned text (if needed)
data['cleaned_text'] = data['tokens'].apply(lambda x: ' '.join(eval(x)))

# Create TF-IDF features
tfidf = TfidfVectorizer(max_features=5000)  # Use top 5000 words
X = tfidf.fit_transform(data['cleaned_text'])

# Save the features and target
y = data['genre']
pd.DataFrame(X.toarray(), columns=tfidf.get_feature_names_out()).to_csv('data/features.csv', index=False)
y.to_csv('data/target.csv', index=False)