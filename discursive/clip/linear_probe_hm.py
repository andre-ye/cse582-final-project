'''
Load original and fine-tuned CLIP embeddings and compare performance in predicting features.
'''

EMBEDDINGS_ORIG = '/home/andre/discursive/clip/embeddings/hm_clip_embeddings_orig.jsonl'
EMBEDDINGS_FT = '/home/andre/discursive/clip/embeddings/hm_clip_embeddings_v6-e9.jsonl'

LABELS = '/home/andre/discursive/data/hatefulmemes/train.jsonl'

import pandas as pd
import numpy as np
import sklearn
import json
from tqdm import tqdm

# Load embeddings into a list of dictionaries with keys "img", "embed_orig", and "embed_ft"
# make sure to index 0 in embeddings as they are nested in a list
# each embeddings jsonl file contains a key 'img' and 'features'
EMBEDDINGS = {}
temp_embeds_orig, temp_embeds_ft = {}, {} # key: img, value: embedding

with open(EMBEDDINGS_ORIG, 'r') as f:
    for line in f:
        data = json.loads(line)
        temp_embeds_orig[data['img']] = data['features'][0]

with open(EMBEDDINGS_FT, 'r') as f:
    for line in f:
        data = json.loads(line)
        temp_embeds_ft[data['img']] = data['features'][0]

for img in temp_embeds_orig:
    EMBEDDINGS[img] = {
        'embed_orig': temp_embeds_orig[img],
        'embed_ft': temp_embeds_ft[img]
    }

# Load labels into a DataFrame
LABELS = pd.read_json(LABELS, lines=True)

# select only columns in LABELS_MORAL which are img_name or end in "mean"
# LABELS_MORAL = LABELS_MORAL[[col for col in LABELS_MORAL.columns if col == 'img_name' or 'mean' in col]]

# select only images that have embeddings
# LABELS_MORAL = LABELS_MORAL[LABELS_MORAL['img_name'].apply(lambda x: x + '.jpg').isin(EMBEDDINGS.keys())]
# LABELS_NATURAL = LABELS_NATURAL[LABELS_NATURAL['img_name'].apply(lambda x: x + '.jpg').isin(EMBEDDINGS.keys())]

'''
labels have a column called 'img_name' which does not include the file extension ('.jpg') which the embeddings have 
all other columns are prediction targets
labels_moral is a regression task for all columns
labels_natural is a classification task for all columns
'''

# define train split with fixed seed for both tasks
from sklearn.model_selection import train_test_split
moral_train, moral_test = train_test_split(LABELS, test_size=0.2, random_state=42)
natural_train, natural_test = train_test_split(LABELS, test_size=0.2, random_state=42)


'''
moral task
1. create train and test sets
2. train linear regression model on train
3. print error for each label on test
'''

X_train_orig = [EMBEDDINGS[img]['embed_orig'] for img in moral_train['img']]
X_train_ft = [EMBEDDINGS[img]['embed_ft'] for img in moral_train['img']]
X_test_orig = [EMBEDDINGS[img]['embed_orig'] for img in moral_test['img']]
X_test_ft = [EMBEDDINGS[img]['embed_ft'] for img in moral_test['img']]

y_train = moral_train['label']
y_test = moral_test['label']


'''
Run logistic regression
'''

# from sklearn.linear_model import LogisticRegression as reg_model
from sklearn.neural_network import MLPClassifier as reg_model
from sklearn.metrics import accuracy_score

params = {

    # mlpclassifier
    'hidden_layer_sizes': (8, 8),
    'max_iter': 1000
    
    # logistic regression
    # 'max_iter': 1000
}

model_orig = reg_model(**params).fit(X_train_orig, y_train)
model_ft = reg_model(**params).fit(X_train_ft, y_train)

y_pred_orig = model_orig.predict(X_test_orig)
y_pred_ft = model_ft.predict(X_test_ft)
y_pred_orig_train = model_orig.predict(X_train_orig)
y_pred_ft_train = model_ft.predict(X_train_ft)

acc_orig = accuracy_score(y_test, y_pred_orig)
acc_ft = accuracy_score(y_test, y_pred_ft)
acc_orig_train = accuracy_score(y_train, y_pred_orig_train)
acc_ft_train = accuracy_score(y_train, y_pred_ft_train)

print(f'Accuracy original: {acc_orig} [train: {acc_orig_train}]')
print(f'Accuracy fine-tuned: {acc_ft} [train: {acc_ft_train}]')



'''
Excess
'''

# # standardize on X_train_orig and X_train_ft, apply to X_test_orig and X_test_ft
# from sklearn.preprocessing import StandardScaler
# scaler_orig = StandardScaler().fit(X_train_orig)
# scaler_ft = StandardScaler().fit(X_train_ft)
# X_train_orig = scaler_orig.transform(X_train_orig)
# X_train_ft = scaler_ft.transform(X_train_ft)
# X_test_orig = scaler_orig.transform(X_test_orig)
# X_test_ft = scaler_ft.transform(X_test_ft)

# # standardize on y_train and apply to y_test
# scaler_y = StandardScaler().fit(y_train)
# y_train = scaler_y.transform(y_train)
# y_test = scaler_y.transform(y_test)

# from sklearn.linear_model import LinearRegression as reg_model
# # from sklearn.linear_model import Lasso as reg_model
# # from sklearn.kernel_ridge import KernelRidge as reg_model
# from sklearn.metrics import mean_squared_error

# params = {

#     # lasso
#     # 'alpha': 0.01

#     # kernel reg
#     # 'kernel': 'rbf',

# }

# # feature wise results
# for i, feature in enumerate(LABELS_MORAL.columns[1:]):

#     model_orig = reg_model(**params).fit(X_train_orig, y_train[:,i])
#     model_ft = reg_model(**params).fit(X_train_ft, y_train[:,i])

#     y_pred_orig = model_orig.predict(X_test_orig)
#     y_pred_ft = model_ft.predict(X_test_ft)

#     mse_orig = mean_squared_error(y_test[:,i], y_pred_orig)
#     mse_ft = mean_squared_error(y_test[:,i], y_pred_ft)

#     mse_train_orig = mean_squared_error(y_train[:,i], model_orig.predict(X_train_orig))
#     mse_train_ft = mean_squared_error(y_train[:,i], model_ft.predict(X_train_ft))

#     # print mses w. 4 digits precision and feature up to 20 characters
#     print()
#     print(f'{feature:20} | orig [test: {mse_orig:.3f}] [train: {mse_train_orig:.3f}] | ft [test: {mse_ft:.3f}] [train: {mse_train_ft:.3f}]')

# model_orig = reg_model(**params).fit(X_train_orig, y_train)
# model_ft = reg_model(**params).fit(X_train_ft, y_train)

# y_pred_orig = model_orig.predict(X_test_orig)
# y_pred_ft = model_ft.predict(X_test_ft)

# mse_orig = mean_squared_error(y_test, y_pred_orig)
# mse_ft = mean_squared_error(y_test, y_pred_ft)

# print(f'MSE original: {mse_orig}')
# print(f'MSE fine-tuned: {mse_ft}')

# '''
# natural task
# 1. create train and test sets
# 2. train logistic regression model on train, which allows for prediction of multiple targets
# 3. print accuracy for each label on test
# '''

# X_train_orig = [EMBEDDINGS[img + '.jpg']['embed_orig'] for img in natural_train['img_name']]
# X_train_ft = [EMBEDDINGS[img + '.jpg']['embed_ft'] for img in natural_train['img_name']]
# X_test_orig = [EMBEDDINGS[img + '.jpg']['embed_orig'] for img in natural_test['img_name']]
# X_test_ft = [EMBEDDINGS[img + '.jpg']['embed_ft'] for img in natural_test['img_name']]

# y_train = natural_train.drop('img_name', axis=1)
# y_test = natural_test.drop('img_name', axis=1)

# from sklearn.linear_model import LogisticRegression
# from sklearn.multioutput import MultiOutputClassifier
# from sklearn.metrics import accuracy_score

# model_orig = MultiOutputClassifier(LogisticRegression(max_iter=1000)).fit(X_train_orig, y_train)
# model_ft = MultiOutputClassifier(LogisticRegression(max_iter=1000)).fit(X_train_ft, y_train)

# y_pred_orig = model_orig.predict(X_test_orig)
# y_pred_ft = model_ft.predict(X_test_ft)

# acc_orig = accuracy_score(y_test, y_pred_orig)
# acc_ft = accuracy_score(y_test, y_pred_ft)

# print(f'Accuracy original: {acc_orig}')
# print(f'Accuracy fine-tuned: {acc_ft}')

