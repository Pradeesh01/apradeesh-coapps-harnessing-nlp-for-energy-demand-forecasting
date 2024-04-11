#import necessary libraries
Import pandas as pd
Import numpy as np
Import nltk From nltk.tokenize 
Import word_tokenize From nltk.corpus 
Import stopwords From nltk.stem 
Import WordNetLemmatizer From nltk.probability 
Import FreqDistFrom sklearn.model_selection 
Import train_test_split From sklearn.feature_extraction.text 
Import TfidfVectorizer From sklearn.ensemble 
import RandomForestRegressor From sklearn.metricsImport mea
n_squared_error
# Download NLTK resources
Nltk.download(‘punkt’)
Nltk.download(‘stopwords’)
Nltk.download(‘wordnet’)
# Define functions for text preprocessing
Def preprocess_text(text):
Tokens = word_tokenize(text.lower())
Tokens = [word for word in tokens if word.isalnum()]
Tokens = [word for word in tokens if word not in 
stopwords.words(‘english’)]Lemmatizer = WordNetLemmatizer()
Tokens = [lemmatizer.lemmatize(word) for word in tokens]
Return ‘ ‘.join(tokens)
# Load and preprocess textual data
# Replace ‘data.csv’ with your dataset file path containing 
text data
Data = pd.read_csv(‘data.csv’)
Data[‘preprocessed_text’] = 
data[‘text’].apply(preprocess_text)
# Split data into training and testing sets
X_train, X_test, y_train, y_test = 
train_test_split(data[‘preprocessed_text’], 
data[‘energy_demand’], test_size=0.2, random_state=42)# Convert text data into numerical features using TF-IDF 
vectorization
Tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
# Train machine learning model
Model = RandomForestRegressor()
Model.fit(X_train_tfidf, y_train)
# Evaluate model
Y_pred = model.predict(X_test_tfidf)
Mse = mean_squared_error(y_test, y_pred)
Print(“Mean Squared Error:”, mse)