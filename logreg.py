import cv2

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV

GENERATE_DIGIT_IMAGE = True

df = pd.read_csv("./mnist.csv")

labels = df.iloc[:, 0]
digits = df.iloc[:, 1:]

ndp = np.asarray(digits.iloc[1]).reshape(28, 28).astype('float32')
npdf = digits.to_numpy()
npdf = npdf.reshape(42000, 28, 28).astype('float32')

new_dig = []
for i in range(42000):
    new_dig.append(cv2.resize(npdf[i], (14, 14)))

new_dig = np.asarray(new_dig).reshape(42000, 196).astype('float32')

train_set = np.asarray(new_dig[:5000])
test_set = np.asarray(new_dig[5000:])
y_train = labels[:5000]
y_test = labels[5000:]

regression_model = linear_model.LogisticRegressionCV(tol=0.01, penalty='l1', solver='saga', n_jobs=5)
regression_model_gs = GridSearchCV(regression_model, param_grid={"Cs": [1, 3, 5, 7, 9, 10]}, cv=10)
regression_model_gs.fit(train_set, y_train)
print("tuned hyerparameters :(best parameters) ", regression_model_gs.best_params_)
print("accuracy :", regression_model_gs.best_score_)
digit_prediction = regression_model_gs.predict(test_set)

#print("train acc:", regression_model_gs.score(train_set, y_train))
#print("test acc:", regression_model_gs.score(test_set, y_test))
#print(confusion_matrix(y_test, digit_prediction))
#print(regression_model_gs.get_params())

print(regression_model_gs.cv_results_)
