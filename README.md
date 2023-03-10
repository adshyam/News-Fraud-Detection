News Fraud Detection

This is a project that aims to detect fraud or fake news articles from real ones. The project uses machine learning algorithms such as Multinomial Naïve Bayes, Bernoulli Naive Bayes and Bidirectional LSTM to analyze and classify news articles based on their authenticity. In this repository, we have compared the performance of all three models.

Requirements Used libraries: • Pandas, Numpy, Scikit-learn, Keras, and NLTK Python libraries.

Dataset:
• https://www.kaggle.com/competitions/fake-news/data?select=train.csv 
. https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Steps used:

• Preprocessed the dataset by removing stop words, stemming, and applying CountVectorizer to convert a collection of text documents into a matrix of token counts (i.e., Bag of Words) with ngrams range (1,3). • Trained the machine learning models (Multinomial Naïve Bayes and Bidirectional LSTM) using the preprocessed dataset. • Also applied Grid Search CV to see the performance of models with 5 folds CV. Evaluated the performance of the models using various metrics such as accuracy, precision, recall, f1 score and Confusion Matrix. • Compared the performance of both models.

Multinomial Naïve Bayes: Used Naive Bayes classifier with the Multinomial Naive Bayes algorithm from scikit-learn, and then evaluated the performance of the classifier on a test set using accuracy and a confusion matrix.

Bernoulli Naïve Bayes: Used Naive Bayes classifier with the Bernoulli Naive Bayes algorithm from scikit-learn, and then evaluated the performance of the classifier on a test set using accuracy and a confusion matrix.

Bidirectional LSTM: Used the one_hot function from the Keras preprocessing.text module to convert each word in a list of text documents into a unique integer, known as a "one-hot encoding". This encoding is a binary vector where all elements are zero except for the element corresponding to the index of the word in the vocabulary, which is set to 1. Again, used sequential model with sigmoid activation and Adam optimizer with 2 layers: first LSTM layer which processed the sequence of word vectors and second final dense layer which produces a single output that predicts the class label. The loss function used for training is binary-loss entropy. Also, calculated the confusion matrix, accuracy, and classification report.

Results

The accuracy obtained from Multinomial Naïve Bayes, Bernoulli Naive Bayes and  Bidirectional LSTM were compared. 

Conclusion

This project can help in identifying fake news articles and combating the spread of misinformation. By using machine learning algorithms, we can classify news articles based on their authenticity and help people make informed decisions. The comparison of both models helps us to understand which model performs better for this task.

