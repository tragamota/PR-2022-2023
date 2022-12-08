import numpy as np
import pandas as pd

from statsmodels.stats.contingency_tables import mcnemar

lr_results = pd.read_csv('lr_prediction.csv')
svm_results = pd.read_csv('svm_prediction.csv')
svm_results = svm_results.drop(['label'], axis=1)
svm_results.columns = ['svm']

total_prediction = pd.concat([lr_results, svm_results], axis=1)
total_prediction.columns = ["label", "lr", 'svm']

lr_results = np.zeros(len(total_prediction))
svm_results = np.zeros(len(total_prediction))

for i, value in total_prediction.iterrows():
    true_label = value[0]
    lr = value[1]
    svm = value[2]

    lr_results[i] = lr == true_label
    svm_results[i] = svm == true_label


df = pd.concat([pd.Series(lr_results), pd.Series(svm_results)], axis=1)
df = df.astype(bool)

results = [[0, 0], [0, 0]]

for i, value in df.iterrows():
    svm = bool(value[0])
    lr = bool(value[1])

    print(f"SVM {svm}")

    if svm is True and lr is True:
        results[0][0] += 1
    elif svm is True and lr is False:
        results[0][1] += 1
    elif svm is False and lr is True:
        results[1][0] += 1
    else:
        results[1][1] += 1

print(results)

stat = mcnemar(results, exact=False)

print(stat)
