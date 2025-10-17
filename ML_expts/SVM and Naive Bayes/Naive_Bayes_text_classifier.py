# we use multinomial naive bayes here => means multiple possible outcomes (like binomial is only 2 possible outcomes)

# here we use sparse word count features from newsgroup dataset in sklearn (sparse = feature values mostly zero)
# contains 20k newsgroup documents spread across 20 different newsgroups
# e.g. of word count => doc 1 = "cat sat", out of vocab "hat rat cat sat and mat we have feature vector [0 0 1 1 0] for word count
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
data = fetch_20newsgroups()
print(data.target_names)  # these are all categories for docs in dataset

# is a large dataset, so we will only select 4 categories and download a training and test set for each
categories = ['talk.religion.misc', 'soc.religion.christian',
              'sci.space', 'comp.graphics']
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)
# can see what ac actual doc looks like from within a category now
print(train.data[3])

# now to use the data for ML need to convert content into vector of numbers
# use TF-IDF vectorizer which converts info to numeric vector by comparing no times word appears in doc with number of docs word appears in
# use the vectorizer and multinomial bayes in pipeline to create our model architecture (which is blank as we need to fit data still)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
NB_text_model = make_pipeline(TfidfVectorizer(), MultinomialNB())  # steps is how we prepare data for ML and memory is the ML tool we use to create model

# now we apply model to our data
NB_text_model.fit(train.data, train.target)  # (our (X_train, y_train)
predicted_labels = NB_text_model.predict(test.data)

# now can evaluate model with confusion matrix to see if our predicted labels align with true labels for words in test set
from sklearn.metrics import confusion_matrix
def print_confusion_matrix():
    mat = confusion_matrix(test.target, predicted_labels)  # true vs predicted
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=train.target_names, yticklabels=train.target_names)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.show()

# print_confusion_matrix()
# above confusion matrix shows that this simple classifier is able to distinguish between the topics of the texts we chose in our categories (is able to fairly accurately distinguish between religious and space talk for e.g.)
# can see one are of confusion for the machine is separating talk between religion and christianity which makes sense

# now we can use model to predict a classifier for any input string we give, for e.g. lets test if it can classify the below
def predict_category(input_str, train_set, model):
    prediction = model.predict([input_str])
    return train_set.target_names[prediction[0]]  # our label

s = 'launching to the ISS in one hour'
out_category = predict_category(s, train, NB_text_model)
print(s, "=> gives category: ", out_category)
s1 = 'islam is different from hinduism due to its monotheistic basis'
out_category1 = predict_category(s1, train, NB_text_model)
print(s1, "=> gives category: ", out_category1)
s2 = 'the screen resolution does not allow for high quality streaming'
out_category2 = predict_category(s2, train, NB_text_model)
print(s2, "=> gives category: ", out_category2)
