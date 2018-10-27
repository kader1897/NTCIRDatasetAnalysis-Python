from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import normalize
from sklearn import svm
import numpy


dataFileName = "data_updated.txt"

with open(dataFileName) as f:
    content = f.readlines()

# Remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content]

labels = [x.split(",")[-1] for x in content]
data = [','.join(x.split(",")[0:-1]) for x in content]

labelDict = {'transport': 1, 'walking': 2, 'airplane': 3}


def tokenize(txt):
    return txt.split(",")


vec = CountVectorizer(tokenizer=tokenize)
transformed_data = vec.fit_transform(data).toarray()
normalized_data = normalize(transformed_data, norm='l1')
# data_to_train = vec.fit_transform(data_to_train).toarray()

transformed_labels = list(map(lambda x: labelDict.get(x), labels))

data_train, data_test, lbl_train, lbl_test = train_test_split(normalized_data, transformed_labels,
                                                              test_size=0.2, random_state=1, shuffle=True)

# # Linear SVM
# print()
# clf = svm.LinearSVC()
# scores = cross_val_score(clf, data_train, lbl_train, cv=5)
# print("Linear SVM")
# print("Cross validation scores: ", scores)
# print("Average score: ", numpy.mean(scores))
#
# clf.fit(data_train, lbl_train)
# prediction = clf.predict(data_test)
# print("Test accuracy: ", accuracy_score(lbl_test, prediction))

# # SVM with Rbf function
# print()
# print("SVM with Rbf kernel")
# print("Gamma: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0")
# CBest = 1.0
# gammaBest = 0.1
# bestScore = 0.0
# C = 1.0
# for i in range(1, 6):
#     print("C: ", C)
#     gamma = 0.1
#     scoreList = [];
#     for j in range(0,10):
#         clf = svm.SVC(C = C,kernel='rbf',gamma = gamma, coef0=0.0)
#         scores = cross_val_score(clf, data_train, lbl_train, cv=5)
#         meanValue = numpy.mean(scores)
#         # print(" ", meanValue, end=' ')
#         scoreList.append(meanValue)
#         if meanValue > bestScore:
#             bestScore = meanValue
#             CBest = C
#             gammaBest = gamma
#         gamma = gamma + 0.1
#     print(scoreList)
#     C = C - 0.1
#
# clf = svm.SVC(C = CBest, kernel='rbf', gamma=gammaBest)
# clf.fit(data_train, lbl_train)
# prediction = clf.predict(data_test)
# print()
# print("***Evaluation on Test Set***")
# print("C Value: ", CBest, " Gamma Value: ", gammaBest)
# print("Test accuracy: ", accuracy_score(lbl_test, prediction))
#
# # SVM with Sigmoid function
# print()
# print("SVM with Sigmoid kernel")
# print("r: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10")
# CBest = 1.0
# gammaBest = 0.1
# coefBest = 0
# bestScore = 0.0
# C = 1.0
# for i in range(1, 6):
#     print("C: ", C)
#     gamma = 0.1
#     scoreList = numpy.zeros((10, 11))
#     for j in range(0,10):
#         print("Gamma: ", gamma, end=' ')
#         coeff = 0
#         for k in range(0,11):
#             clf = svm.SVC(C = C,kernel='sigmoid',gamma = gamma, coef0=coeff, max_iter=5000)
#             scores = cross_val_score(clf, data_train, lbl_train, cv=5)
#             meanValue = numpy.mean(scores)
#             # print(" ", meanValue, end=' ')
#             scoreList[j][k] = meanValue
#             if meanValue > bestScore:
#                 bestScore = meanValue
#                 CBest = C
#                 gammaBest = gamma
#                 coefBest = coeff
#             coeff = coeff + 1
#         gamma = gamma + 0.1
#     print(scoreList)
#     C = C - 0.1
#
# clf = svm.SVC(C = CBest, kernel='sigmoid', gamma=gammaBest, coef0=coefBest)
# clf.fit(data_train, lbl_train)
# prediction = clf.predict(data_test)
# print()
# print("***Evaluation on Test Set***")
# print("C Value: ", CBest, " Gamma Value: ", gammaBest, " Coefficient: ", coefBest)
# print("Test accuracy: ", accuracy_score(lbl_test, prediction))

# SVM with Polynomial function
print()
for d in (1, 2, 3, 4, 5):
    print("SVM with degree ", d, " polynomial kernel")
    print("r: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10")
    CBest = 1.0
    gammaBest = 0.1
    coefBest = 0
    bestScore = 0.0
    C = 1.0
    for i in range(1, 6):
        print("C: ", C)
        gamma = 0.1
        scoreList = numpy.zeros((10,11))
        for j in range(0,10):
            print("Gamma: ", gamma, end=' ')
            coeff = 0
            for k in range(0,11):
                clf = svm.SVC(C = C,kernel='poly',gamma = gamma, coef0=coeff, degree=d)
                scores = cross_val_score(clf, data_train, lbl_train, cv=5)
                meanValue = numpy.mean(scores)
                # print(" ", meanValue, end=' ')
                scoreList[j][k] = meanValue
                if meanValue > bestScore:
                    bestScore = meanValue
                    CBest = C
                    gammaBest = gamma
                    coefBest = coeff
                coeff = coeff + 1
            gamma = gamma + 0.1
        print(scoreList)
        C = C - 0.1

    clf = svm.SVC(C = CBest, kernel='poly', gamma=gammaBest, coef0=coefBest, degree=d)
    clf.fit(data_train, lbl_train)
    prediction = clf.predict(data_test)
    print()
    print("***Evaluation on Test Set***")
    print("C Value: ", CBest, " Gamma Value: ", gammaBest, " Coefficient: ", coefBest)
    print("Test accuracy: ", accuracy_score(lbl_test, prediction))