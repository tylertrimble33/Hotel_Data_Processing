import pickle
import sklearn.metrics

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

#1) load the data and perform any preprocessing
# you must load data as a data matrix X, and labels matrix Y


#2) load your model and report accuracy (you shouldn't have to change this part)
# Sample code to load your saved model:
with open('best_out.scikitmodel', 'rb') as file:
    loaded_classifier = pickle.load(file)
predictions = loaded_classifier.predict(X)
accuracy = sklearn.metrics.accuracy_score(Y, predictions)
print("accuracy = ", accuracy)


