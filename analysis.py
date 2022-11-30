import os
import cv2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

GENERATE_DIGIT_IMAGE = True

df = pd.read_csv("./mnist.csv")

print(df.describe())

labels = df.iloc[:, 0]
digits = df.iloc[:, 1:]

labels_unique = labels.unique()

if GENERATE_DIGIT_IMAGE:
    if not os.path.exists('digit_images'):
        os.makedirs('digit_images')

        for label in labels_unique:
            os.makedirs(f'digit_images/{str(label)}')

    for index, row in df.iterrows():
        label = row[0]
        pixels = row[1:].to_numpy().reshape(28, 28)

        cv2.imwrite(f'digit_images/{str(label)}/{str(index)}.png', pixels)

# Baseline majority class

majority_label = labels.mode()[0]

correct_labeled_digit = 0
for index, row in df.iterrows():
    true_digit_label = row[0]

    if true_digit_label == majority_label:
        correct_labeled_digit += 1


print(f"Baseline majority class has an accuracy of {correct_labeled_digit / len(df) * 100} %")

# Baseline mean squared error

digit_baseline = np.array([digits.loc[11721].to_numpy(), digits.loc[4076].to_numpy(), digits.loc[247].to_numpy(),
                          digits.loc[232].to_numpy(), digits.loc[258].to_numpy(), digits.loc[2232].to_numpy(),
                          digits.loc[3668].to_numpy(), digits.loc[4916].to_numpy(), digits.loc[4819].to_numpy(),
                          digits.loc[984].to_numpy()])

correct_labeled_digit = 0
predict_labels = np.zeros(len(df))

for index, row in df.iterrows():
    true_digit_label = row[0]
    pixels = row[1:].to_numpy()

    error_results = np.zeros(len(labels_unique))

    for label_index in range(len(labels_unique)):
        squares_error = np.subtract(digit_baseline[label_index], pixels) ** 2
        mean_square_error = np.sum(squares_error) / len(digit_baseline[label_index])
        error_results[label_index] = np.sqrt(mean_square_error)

    predict = np.argmin(error_results)
    predict_labels[index] = predict

    if predict == true_digit_label:
        correct_labeled_digit += 1


print(f"Baseline Mean squared error has an accuracy of {correct_labeled_digit / len(df) * 100} %")
print(confusion_matrix(labels, predict_labels))

# Create pixel occurance image of all digits (which pixels are unused and which are used)

pixel_means = np.zeros(784)
useless_pixels = []

for pixel_index in range(len(pixel_means)):
    pixel_mean = np.mean(digits.iloc[:, pixel_index])

    if pixel_mean == 0:
        useless_pixels.append(pixel_index + 1)

    pixel_means[pixel_index] = pixel_mean

cv2.imwrite('average_mean_of_all_digits.png', pixel_means.reshape(28, 28))

plt.imshow(pixel_means.reshape(28, 28))
plt.show()

# label distribution

plt.hist(labels)
plt.show()

# Task 2 / 3 Ink

ink = np.array([np.sum(row) for _, row in digits.iterrows()])
ink_mean = [np.mean(ink[labels == i]) for i in range(len(labels_unique))]
ink_std = [np.std(ink[labels == i]) for i in range(len(labels_unique))]

digits_ink = [ink[labels == i] for i in range(len(labels_unique))]

print(ink)
print(ink_mean)
print(ink_std)

plt.boxplot(digits_ink)
plt.show()

ink_scaled = preprocessing.scale(ink).reshape(-1, 1)

regression_model = linear_model.LogisticRegressionCV()
regression_model.fit(ink_scaled, labels)
digit_prediction = regression_model.predict(ink_scaled)

print(regression_model.score(ink_scaled, labels))
print(confusion_matrix(labels, digit_prediction))

# New feature from the digits (perimeter)

perimeter = np.zeros(len(df))

for digit_index in range(len(digits)):
    img = digits.loc[digit_index].to_numpy().reshape(28, 28).astype(np.uint8)

    ret, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    perimeter_total = 0
    for contour in contours:
        perimeter_total += cv2.arcLength(contour, True)

    perimeter[digit_index] = perimeter_total


perimeter_mean = [np.mean(perimeter[labels == i]) for i in range(len(labels_unique))]
perimeter_std = [np.std(perimeter[labels == i]) for i in range(len(labels_unique))]
digits_perimeter = [perimeter[labels == i] for i in range(len(labels_unique))]

print(perimeter_mean)
print(perimeter_std)

plt.boxplot(digits_perimeter)
plt.show()

perimeter_scaled = preprocessing.scale(perimeter).reshape(-1, 1)

regression_model = linear_model.LogisticRegressionCV()
regression_model.fit(perimeter_scaled, labels)
digit_prediction = regression_model.predict(perimeter_scaled)

print(regression_model.score(perimeter_scaled, labels))
print(confusion_matrix(labels, digit_prediction))

# Combine the ink and perimeter feature together for fit a multinomial logistic regression

combine_ink_perimeter = pd.DataFrame(ink_scaled)
combine_ink_perimeter[1] = pd.DataFrame(perimeter_scaled)
combine_ink_perimeter.columns = ['ink', 'perimeter']

regression_model = linear_model.LogisticRegressionCV()
regression_model.fit(combine_ink_perimeter, labels)
digit_prediction = regression_model.predict(combine_ink_perimeter)

print(regression_model.score(combine_ink_perimeter, labels))
print(confusion_matrix(labels, digit_prediction))


# Task 5 multinomial logit model (LASSO)

df = df.drop(useless_pixels)
df = shuffle(df)

train_df = df.iloc[:5000, :]
test_df = df.iloc[5000:, :]

train_X = train_df.iloc[:, 1:]
train_Y = train_df.iloc[:, 0]

test_X = test_df.iloc[:, 1:]
test_Y = test_df.iloc[:, 0]

train_X = preprocessing.scale(train_X)
test_X = preprocessing.scale(test_X)


