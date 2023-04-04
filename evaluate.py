import pickle
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import numpy as np

filename = 'data.csv'

'''
# Consider a scenario where you determine that the best classifier is KNN, and
# after a hyperparameter search determine the best hyperparameters are n_neighbors =1 and 
# metric=cosine. You can train the model with those hyperparameters and train it as follows:
#
# classifier = KNeighborsClassifier(n_neighbors=1, metric='cosine')
# classifier.fit(train_X, train_Y)
# with open('best_out.scikitmodel', 'wb') as file:
#     pickle.dump(classifier, file)
#
#
# This script will load that classifier and have it predict on data specified by filename
# You also need to perform any preprocessing in this script. Hyperparameter tuning should
# not be performed here, just use the best ones you found in other scripts.

# I will change the filename and click run to generate the test set score used for your grade
# ensure it works correctly by trying it with 'data.csv' 
'''

# 1) load the data and perform any preprocessing
# you must load data as a data matrix X, and labels matrix Y
data = pd.read_csv(filename)

# One hot encoding for categorical features
data = pd.get_dummies(data, columns=['type_of_meal_plan', 'room_type_reserved', 'market_segment_type'])

# Clean data
X = data.drop(['booking_status'], axis=1)
X = X.drop(columns=['Booking_ID'])
Y = data['booking_status']
Y = Y.map({'Canceled': 1, 'Not_Canceled': 0})

# # Get counts for report
# counts = Y.value_counts()
# print("1s: ", counts[1])
# print("0s: ", counts[0])

# Split data into test and training sets (Read online to hardcode random state
# to 42 to have consistency in the data split)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# scaler = StandardScaler()
# features = scaler.fit_transform(X.head(7500))
#
# # PCA
# pca = PCA()
# pca_model = pca.fit_transform(features)
# print(pca.explained_variance_ratio_)
#
# # t-SNE
# tsne = TSNE(n_components=2, perplexity=30, random_state=0)
# tsne_model = tsne.fit_transform(features)
#
# # Visualize results
# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#
# axs[0].scatter(pca_model[:, 0], pca_model[:, 1], c=Y.head(7500))
# axs[0].set_title('PCA Visualization')
# axs[1].scatter(tsne_model[:, 0], tsne_model[:, 1], c=Y.head(7500))
# axs[1].set_title('t-SNE Visualization')
# plt.show()
#
# Classifier dictionary
# classifiers = {
#     'Majority Class Baseline': None,
#     'KNN': KNeighborsClassifier(),
#     'Decision Tree': DecisionTreeClassifier(),
#     'Logistic Regression': SGDClassifier(loss='log', penalty='l2', max_iter=1000),
#     'Logistic Polynomial Regression': SGDClassifier(loss='log', penalty='l2', max_iter=1000),
#     'Neural Network': MLPClassifier(),
#     'Linear SVM': SGDClassifier(loss='hinge', penalty='l2', max_iter=1000),
#     'RBF SVM': SVC()
# }
#
# # Parameter dictionary
# parameters = {
#     'KNN': {'n_neighbors': list(range(1, 10))},
#     'Decision Tree': {'max_depth': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]},
#     'Logistic Regression': {'alpha': [0.0001, 0.001, 0.01, 0.1, 1]},
#     'Logistic Polynomial Regression': {'alpha': [0.0001, 0.001, 0.01, 0.1, 1]},
#     'Neural Network': {'hidden_layer_sizes': [(5,), (10,), (5, 5), (10, 10)], 'alpha': [0.0001, 0.001, 0.01, 0.1, 1]},
#     'Linear SVM': {'alpha': [0.0001, 0.001, 0.01, 0.1, 1]},
#     'RBF SVM': {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}
# }
#
# best_accuracy = 0
# best_model = None
# # Train and evaluate using 5-fold cross-validation
# for classifier_name, classifier in classifiers.items():
#     print(classifier_name)
#     if classifier_name == 'Majority Class Baseline':
#         # Create a majority class baseline classifier
#         count = np.bincount(y_train)
#         majority = np.argmax(count)
#         baseline = DummyClassifier(strategy='constant', constant=majority)
#         # Evaluate MCB
#         scores = cross_val_score(baseline, X_train, y_train, cv=5)
#         print("Test accuracy: ", scores)
#
#     elif classifier_name == 'Logistic Polynomial Regression':
#         # Didn't quite understand how to use the Polynomial Features here
#         poly = PolynomialFeatures(degree=2)
#         parameter = parameters[classifier_name]
#         model = GridSearchCV(classifier, parameter, cv=5, n_jobs=-1)
#         model.fit(poly.fit_transform(X_train), y_train)
#         # Evaluate the best model on the test set
#         y_hat = model.predict(poly.transform(X_test))
#         accuracy = accuracy_score(y_test, y_hat)
#         print('Best hyperparameters:', model.best_params_)
#         print('Test accuracy:', accuracy)
#         if accuracy > best_accuracy:
#             best_accuracy = accuracy
#             best_model = model
#     else:
#         # Perform a hyperparameter search for each classifier using 5-fold cross-validation
#         parameter = parameters[classifier_name]
#         model = GridSearchCV(classifier, parameter, cv=5)
#         model.fit(X_train, y_train)
#
#         # Evaluate on the test set
#         y_hat = model.predict(X_test)
#         accuracy = accuracy_score(y_test, y_hat)
#         print('Best hyperparameters:', model.best_params_)
#         print('Test accuracy:', accuracy)
#         if accuracy > best_accuracy:
#             best_accuracy = accuracy
#             best_model = model
#
# model_filename = 'best_out.scikitmodel'
# pickle.dump(best_model, open(model_filename, 'wb'))
# 2) load your model and report accuracy (you shouldn't have to change this part)
# Sample code to load your saved model:
with open('best_out.scikitmodel', 'rb') as file:
    loaded_classifier = pickle.load(file)
predictions = loaded_classifier.predict(X)
accuracy = accuracy_score(Y, predictions)
print("accuracy = ", accuracy)
