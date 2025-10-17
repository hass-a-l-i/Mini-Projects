# can use SVM for facial recognition
# will use labelled faces in Wild dataset in sklearn = collection of many famous peoples labelled faces

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
import seaborn as sns
faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)  # 1348 images of 62 by 47 pixel size

# let's see what some of the faces look like
def check_faces():
    fig, ax = plt.subplots(3, 5)
    for i, axi in enumerate(ax.flat):
        axi.imshow(faces.images[i], cmap='bone')
        axi.set(xticks=[], yticks=[],
                xlabel=faces.target_names[faces.target[i]])
    plt.show()
# check_faces()

# as we have 62x47 pixels, using each pixel as feature is too much - so we use principal component analysis to extract 150 fundamental features to feed to SVM classifier

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA  # principle component analysis
# note PCA is basically finding eigenvalues of covariance matrix and sorting by highest as eigenvalues of covar matrix are directions of axes where most variance (most info is)
# corresponding eigenvectors are principal components
# feature vector then constructed our of eigen vectors and use this as our new axis for data
pca = PCA(n_components=150, whiten=True, random_state=42)  # whitening makes data uncorrelated and gives unit (identical) variance for all features => helps easier modelling and is applicable here as want to regularize data
svc = SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca, svc)  # combines our SVC model and PCA for us to extract features

# as we want to test we need to split into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(faces.data, faces.target,
                                                    test_size=0.3, random_state=42)

# we now use cross validation to find the best combination of parameters
# adjust C (margin hardness) and gamma (size of radial basis function) then we can find best model

from sklearn.model_selection import GridSearchCV
param_grid = {'svc__C': [1, 5, 10, 50],
              'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}  # this is our grid which we search and fit as parameters for gamma and C to find best => chosen via tuning
grid = GridSearchCV(model, param_grid)

grid.fit(X_train, y_train)
print(grid.best_params_)
# here we find the best parameter (optimal values) are in middle of grid (gridsearchcv finds the best parameters from set of parameters in a grid)
# we found C = 1 and gamma = 0.001 were best of grid above

# now with these values we use our best values and build model then test on test data
model = grid.best_estimator_  # best estimator is the stored output of above
y_fit = model.predict(X_test)

# plot predictions vs actuals
def test():
    fig, ax = plt.subplots(4, 6)
    for i, axi in enumerate(ax.flat):
        axi.imshow(X_test[i].reshape(62, 47), cmap='bone')
        axi.set(xticks=[], yticks=[])
        axi.set_ylabel(faces.target_names[y_fit[i]].split()[-1],
                       color='black' if y_fit[i] == y_test[i] else 'red')
    fig.suptitle('Predicted Names; Incorrect Labels in Red', size=14)
    plt.show()
# test()

# only 3 misslabelled - can get clearer picture with classification report which gives list of metrics for each class
from sklearn.metrics import classification_report
print(classification_report(y_test, y_fit,
                            target_names=faces.target_names))

# alternatively to visualise, can make confusion matrix
from sklearn.metrics import confusion_matrix
def print_confusion_matrix():
    mat = confusion_matrix(y_test, y_fit)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,  # note .T is transpose from numpy
                xticklabels=faces.target_names,
                yticklabels=faces.target_names)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.show()

print_confusion_matrix()

