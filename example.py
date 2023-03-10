# here is an example of how you can use scikit learn classifiers
#  In this case, we are creating and training a knn classifier
#  on the training data, then evaluating it on the training data
# It is missing code to load the data and split it into training
# and test sets (If that is what you end up doing)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
classifier.fit(train_X, train_Y)
predictions = classifier.predict(test_X)
accuracy = sklearn.metrics.accuracy_score(test_Y, predictions)
print("KNN accuracy = ", accuracy)