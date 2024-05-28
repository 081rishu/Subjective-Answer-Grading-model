#!/usr/bin/env python
# coding: utf-8

# ## Adding necessary imports

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ## Loading the dataset

# In[2]:


df = pd.read_csv("merged_students_responses.csv")


# In[6]:


df.head()


# In[7]:


df.shape


# In[8]:


df.info()


# In[9]:


df.describe()


# In[10]:


df.isnull().sum()


# ## dropping rows with null values

# In[11]:


df.dropna(subset =["Score"], inplace = True)
df.isnull().sum()


# ## Dropping useless columns

# In[12]:


df.drop(['Row', 'StudentsResponse3', 'Question'], axis = 1, inplace = True)


# In[13]:


df.head()


# #### Distribution of Score

# In[14]:


import seaborn as sns

# Plot box plot
plt.figure(figsize=(8, 6))
sns.boxplot(x='Score', data=df, color='skyblue')
plt.xlabel('Score')
plt.title('Distribution of Scores')
plt.grid(True)
plt.show()


# ## NLP preprocessing

# ### converting response text to lower case

# In[15]:


df['Response Text'] = df['Response Text'].apply(lambda x: x.lower())
df.head()


# ### removing stop words

# In[16]:


import nltk
nltk.download('stopwords')


# In[17]:


from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_tokens)
df['Response Text'] = df['Response Text'].apply(remove_stopwords)
print(df.head())


# ### Lemmatization

# In[18]:


from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')


# In[19]:


nltk.download('omw-1.4')


# In[20]:


lemmatizer = WordNetLemmatizer()
def lemmatize_text(text):
    tokens = text.split()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(lemmatized_tokens)
df['Response Text'] = df['Response Text'].apply(lemmatize_text)
print(df.head())


# ### Removing Special Characters

# In[21]:


import re

def remove_special_characters(text):
    pattern = r'[^a-zA-Z0-9\s]'
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

df['Response Text'] = df['Response Text'].apply(remove_special_characters)

df.head()


# ### Tokenization

# In[22]:


from nltk.tokenize import word_tokenize


# In[23]:


nltk.download('punkt')


# In[24]:


def tokenize_text(text):
    tokens = word_tokenize(text)
    return tokens


# In[25]:


df['Tokenized Text'] = df['Response Text'].apply(tokenize_text)
df.head()


# ### word cloud

# In[26]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Joining the tokenized text into a single string
preprocessed_text_str = ' '.join([' '.join(tokens) for tokens in df['Tokenized Text']])

# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(preprocessed_text_str)

# Plot word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Word Cloud of Preprocessed Text')
plt.axis('off')
plt.show()



from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Assuming you have a DataFrame `df` with tokenized text
# Joining the tokenized text into a single string
preprocessed_text_str = ' '.join([' '.join(tokens) for tokens in df['Tokenized Text']])

# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(preprocessed_text_str)

# Plot word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Word Cloud of Preprocessed Text')
plt.axis('off')

# Save the word cloud visualization as an image
wordcloud.to_file('wordcloud.png')
plt.show()

# Print the path of the saved image file
print('wordcloud.png') 

import subprocess

# Assuming df is your DataFrame
# Save DataFrame to JSON file
df.to_json('data.json')

# Execute load_model.py and pass the path of the JSON file as input
subprocess.run(['python', 'load_model.ipynb', 'data.json'])



# ### Vectorization

# In[27]:


# here we can use tf-idf or bag of words


# In[28]:


from sklearn.feature_extraction.text import CountVectorizer


# In[29]:


tokenized_texts_as_strings = df['Tokenized Text'].apply(lambda tokens: ' '.join(tokens))
vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(tokenized_texts_as_strings)
X_bow_array = X_bow.toarray()
df['Tokenized Text'] = list(X_bow_array)


# In[30]:


df.head()


# In[31]:


# lets checkout the change we have done 
bow_df = pd.DataFrame(X_bow.toarray(), columns=vectorizer.get_feature_names_out())
bow_df.head()


# ### Word Frequency Distribution

# In[32]:


import numpy as np
import matplotlib.pyplot as plt

# Assuming X_bow_array is your bag-of-words representation after vectorization
word_freq = np.sum(X_bow_array, axis=0)

# Assuming vectorizer.vocabulary_ contains the mapping of words to indices
# If not, you can use vectorizer.get_feature_names_out() to get the list of feature names
feature_names = vectorizer.get_feature_names_out()

# Create a dictionary mapping feature names to their frequencies
word_freq_dict = dict(zip(feature_names, word_freq))

# Sort the dictionary by frequency in descending order
sorted_word_freq_dict = dict(sorted(word_freq_dict.items(), key=lambda item: item[1], reverse=True))

# Plot the top N words by frequency
N = 20  # Change this value as needed
top_words = list(sorted_word_freq_dict.keys())[:N]
top_freqs = list(sorted_word_freq_dict.values())[:N]

plt.figure(figsize=(10, 6))
plt.barh(top_words, top_freqs)
plt.xlabel('Frequency')
plt.ylabel('Word')
plt.title('Word Frequency Distribution')
plt.gca().invert_yaxis()  # Invert y-axis to display most frequent words at the top
plt.show()


# ## feature enginnering

# In[33]:


# using TF-IDF


# In[34]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[35]:


tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_bow)


# In[36]:


print(X_tfidf)


# ### word importance

# In[37]:


import numpy as np
import matplotlib.pyplot as plt

# Assuming X_tfidf is your TF-IDF representation after transformation
idf_values = tfidf_transformer.idf_
feature_names = vectorizer.get_feature_names_out()
idf_dict = dict(zip(feature_names, idf_values))
sorted_idf_dict = dict(sorted(idf_dict.items(), key=lambda item: item[1]))
top_words = list(sorted_idf_dict.keys())[-N:]
top_idfs = list(sorted_idf_dict.values())[-N:]

plt.figure(figsize=(10, 6))
plt.barh(top_words, top_idfs)
plt.xlabel('IDF Value')
plt.ylabel('Word')
plt.title('Top 20 Word Importance based on IDF')
plt.gca().invert_yaxis() 
plt.show()


# ## splitting the data

# In[38]:


from sklearn.model_selection import train_test_split
X = X_tfidf
y = df['Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ### distribution of the target variable in your training and testing sets 

# In[39]:


import matplotlib.pyplot as plt

# Visualize the distribution of target variable in training set
plt.figure(figsize=(8, 6))
plt.hist(y_train, bins=20, color='blue', alpha=0.7, label='Training Set')
plt.title('Distribution of Score in Training Set')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Visualize the distribution of target variable in testing set
plt.figure(figsize=(8, 6))
plt.hist(y_test, bins=20, color='green', alpha=0.7, label='Testing Set')
plt.title('Distribution of Score in Testing Set')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# ## Model Building

# ### Model Selection

# In[40]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Define candidate models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC()
}

# Dictionary to store mean accuracy scores
mean_scores = {}

# Perform cross-validation and evaluate each model
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    mean_scores[name] = scores.mean()
    print(f"{name}: Mean Accuracy - {scores.mean():.4f}, Std Dev - {scores.std():.4f}")

# Select the best model based on performance
best_model_name = max(mean_scores, key=mean_scores.get)
best_model = models[best_model_name]
print(f"\nBest Model: {best_model_name}")

# Optionally, train the best model on the entire training set
best_model.fit(X_train, y_train)

# Evaluate the best model on the test set
test_accuracy = accuracy_score(y_test, best_model.predict(X_test))
print(f"Test Accuracy of the Best Model: {test_accuracy:.4f}")


# ### Difference in models

# In[41]:


import matplotlib.pyplot as plt

# Visualize mean accuracy scores of different models with percentage labels
plt.figure(figsize=(10, 6))
bars = plt.bar(mean_scores.keys(), mean_scores.values(), color='skyblue')

# Add percentage labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2%}', ha='center', va='bottom')

plt.xlabel('Models')
plt.ylabel('Mean Accuracy')
plt.title('Mean Accuracy of Different Models')
plt.xticks(rotation=45)
plt.show()


# ### Applying Hyperparameter tuning 

# #### hyperparameter tuning on random forest

# In[44]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
param_grid = {
    'n_estimators': [50, 100, 200],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Maximum depth of the trees
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required at each leaf node
}

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Instantiate GridSearchCV
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy')

# Perform grid search to find the best hyperparameters
grid_search.fit(X_train, y_train)

# Print the best hyperparameters found
print("Best Hyperparameters:", grid_search.best_params_)

# Get the best model
best_rf_model = grid_search.best_estimator_

# Evaluate the best model on the test set
test_accuracy = best_rf_model.score(X_test, y_test)
print("Test Accuracy of the Best Model:", test_accuracy)


# In[45]:


# Extract the mean cross-validated scores and hyperparameters from the grid search results
results = grid_search.cv_results_
params = results['params']
mean_scores = results['mean_test_score']

# Create a DataFrame to store the results
df_results = pd.DataFrame(params)
df_results['Mean Accuracy'] = mean_scores

# Reshape the DataFrame to have hyperparameters as columns
df_pivot = df_results.pivot_table(index='max_depth', columns=['min_samples_leaf', 'min_samples_split', 'n_estimators'], values='Mean Accuracy')

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df_pivot, annot=True, cmap='viridis', fmt=".4f")
plt.title('Mean Accuracy Heatmap for Random Forest Hyperparameters')
plt.xlabel('Number of Estimators')
plt.ylabel('Max Depth')
plt.show()


# In[46]:


# hpm tuning on svm 
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Define the parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'kernel': ['linear', 'rbf', 'poly'],  # Kernel type
    'gamma': ['scale', 'auto'],  # Kernel coefficient
}

# Create an SVM classifier
svm_classifier = SVC(random_state=42)

# Instantiate GridSearchCV
grid_search = GridSearchCV(estimator=svm_classifier, param_grid=param_grid, cv=5, scoring='accuracy')

# Perform grid search to find the best hyperparameters
grid_search.fit(X_train, y_train)

# Print the best hyperparameters found
print("Best Hyperparameters:", grid_search.best_params_)

# Get the best model
best_svm_model = grid_search.best_estimator_

# Evaluate the best model on the test set
test_accuracy = best_svm_model.score(X_test, y_test)
print("Test Accuracy of the Best Model:", test_accuracy)


# In[47]:


import matplotlib.pyplot as plt
import seaborn as sns

# Extracting grid search results
results = grid_search.cv_results_
params = results['params']
mean_test_scores = results['mean_test_score']

# Reshape mean test scores to create a grid
mean_test_scores_grid = mean_test_scores.reshape(len(param_grid['C']), len(param_grid['kernel']), len(param_grid['gamma']))

# Create a heatmap to visualize the mean test scores
plt.figure(figsize=(12, 8))
sns.heatmap(mean_test_scores_grid.mean(axis=0), annot=True, fmt='.4f', cmap='YlGnBu', 
            xticklabels=param_grid['gamma'], yticklabels=param_grid['kernel'])
plt.xlabel('Gamma')
plt.ylabel('Kernel')
plt.title('Mean Test Accuracy of SVM with Different Hyperparameters')
plt.show()


# In[48]:


# hprprmtr tuning for logistic rgrsn
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# Define the parameter grid
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength
    'penalty': ['l1', 'l2'],  # Penalty type (L1 or L2 regularization)
    'solver': ['liblinear', 'saga']  # Solver algorithm
}

# Create a Logistic Regression classifier
logreg_classifier = LogisticRegression(random_state=42)

# Instantiate GridSearchCV
grid_search = GridSearchCV(estimator=logreg_classifier, param_grid=param_grid, cv=5, scoring='accuracy')

# Perform grid search to find the best hyperparameters
grid_search.fit(X_train, y_train)

# Print the best hyperparameters found
print("Best Hyperparameters:", grid_search.best_params_)

# Get the best model
best_logreg_model = grid_search.best_estimator_

# Evaluate the best model on the test set
test_accuracy = best_logreg_model.score(X_test, y_test)
print("Test Accuracy of the Best Model:", test_accuracy)


# In[49]:


import matplotlib.pyplot as plt
import numpy as np

# Extract the mean cross-validated accuracy scores from the grid search results
results = grid_search.cv_results_
mean_test_scores = results['mean_test_score']

# Create an array to represent the combinations of hyperparameters
param_combinations = [f"C={c}, Penalty={p}, Solver={s}" 
                      for c in param_grid['C'] 
                      for p in param_grid['penalty'] 
                      for s in param_grid['solver']]

# Create a bar plot
plt.figure(figsize=(12, 6))
plt.barh(param_combinations, mean_test_scores, color='skyblue')
plt.xlabel('Mean Test Score')
plt.ylabel('Hyperparameter Combinations')
plt.title('Mean Test Scores for Logistic Regression Hyperparameters')
plt.gca().invert_yaxis()  # Invert y-axis to have the highest score on top
plt.show()


# In[53]:


# Save the trained model to disk
from joblib import dump
dump(best_logreg_model, 'logistic_regression_model.joblib')


# In[ ]:




