import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
from sklearn import linear_model


plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

train.SalePrice.describe()

target = np.log(train.SalePrice)
print ("Skew is:", target.skew())
plt.hist(target, color='blue')
plt.show()

numeric_features = train.select_dtypes(include=[np.number])
numeric_features.dtypes


corr = numeric_features.corr()




train.OverallQual.unique()


quality_pivot = train.pivot_table(index='OverallQual',
                                  values='SalePrice', aggfunc=np.median)




categoricals = train.select_dtypes(exclude=[np.number])
categoricals.describe()

print ("Original: \n")
print (train.Street.value_counts(), "\n")

train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
test['enc_street'] = pd.get_dummies(train.Street, drop_first=True)

print ('Encoded: \n')
print (train.enc_street.value_counts())

submission = pd.DataFrame()
submission['Id'] = test.Id


data = train.select_dtypes(include=[np.number]).interpolate().dropna()

y = np.log(train.SalePrice)
X = data.drop(['SalePrice', 'Id'], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, random_state=42, test_size=.33)
feats = test.select_dtypes(
        include=[np.number]).drop(['Id'], axis=1).interpolate()
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)

predictions = model.predict(feats)


final_predictions = np.exp(predictions)

print ("Original predictions are: \n", predictions[:5], "\n")
print ("Final predictions are: \n", final_predictions[:5])

submission['SalePrice'] = final_predictions
submission.head()


submission.to_csv('submission1.csv', index=False)
