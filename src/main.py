from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
import numpy

labelFileName = "activeLabels.txt"
dataFileName = "data_updated.txt"

with open(dataFileName) as f:
    content = f.readlines()

# Remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content]

labels = [x.split(",")[-1] for x in content]
data = [','.join(x.split(",")[0:-1]) for x in content]

labelDict = {'transport': 0, 'walking': 1, 'airplane': 2}


def tokenize(txt):
    return txt.split(",")


vec = CountVectorizer(tokenizer=tokenize)
transformed_data = vec.fit_transform(data).toarray()
# data_to_train = vec.fit_transform(data_to_train).toarray()

transformed_labels = []
for lbl in labels:
    output_row = list([0]*3)
    output_row[labelDict.get(lbl)] = 1
    transformed_labels.append(output_row)

data_train, data_test, lbl_train, lbl_test = train_test_split(data, labels, test_size=0.2, random_state=1)

# Linear SVM
# clf = svm.LinearSVC()
#
# clf.fit(data_train,lbl_train)
# svm_classifier = clf.predict(data_test)
# equality_check = [a == b for (a,b) in zip(svm_classifier,lbl_test)]
# acc = sum(equality_check) / len(equality_check)
# print('Linear SVM Accuracy: ', acc)

#############################
# arr = numpy.zeros((10, 10))

# Parameter observation
# C = 1
# coeff = 0.7
# for i in range(1, 6):
#     print('Degree: ', i)
#     gamma = 0.1
#     for j in range(10):
#         clf = svm.SVC(decision_function_shape='ovo', C=C, kernel='poly', degree=i, gamma=gamma, coef0=coeff)
#
#         scores = cross_val_score(clf, data_to_train, labels_to_train, cv=5)
#         # print(scores)
#         print('Gamma: ', gamma, '  Cross Val. Score: ', scores.mean())
#         gamma = gamma + 0.1
#
#     print('')
#    # C = C - 0.1
#     gamma = gamma * 10
#     print('\n')

#################
# Linear SVM
# clf = svm.SVC(kernel='linear', max_iter=10000)
# scores = cross_val_score(clf, data_to_train, labels_to_train, cv=10)
# print("Linear SVM scores (5000 rows data, 10 fold cross val.): ")
# print(scores)
# print("Mean score: ", scores.mean())
# print('')
# scores = cross_val_score(clf, data, labels, cv=10)
# print("Linear SVM scores (whole data, 10 fold cross val.): ")
# print(scores)
# print("Mean score: ", scores.mean())

##################

# for i in range(10):
#     for j in range(10):
#         print(arr[i-1][j-1], end=' ')
#     print('\n')

# clf = svm.SVC(decision_function_shape='ovo', C=0.1, kernel='rbf', degree=3, gamma=1.0, coef0=0.0)
#
# clf.fit(data_train,lbl_train)
# svm_classifier = clf.predict(data_test)
# equality_check = [a == b for (a,b) in zip(svm_classifier,lbl_test)]
# acc = sum(equality_check) / len(equality_check)
# print('SVM with ovo Accuracy: ', acc)

# scores = cross_val_score(clf, data, labels, cv=5)
# print(scores)
# print(scores.mean())
# T = [L[i] for i in indices]
