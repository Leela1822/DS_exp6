# DS_exp6
# Ex-06-Feature-Transformation
# AIM
To read the given data and perform Feature Transformation process and save the data to a file.

# EXPLANATION
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# ALGORITHM
STEP 1: Read the given Data

STEP 2: Clean the Data Set using Data Cleaning Process

STEP 3: Apply Feature Transformation techniques to all the features of the data set

STEP 4: Print the transformed features

# PROGRAM
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer

df=pd.read_csv("data_trans.csv")
df

sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.HighlyNegativeSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.ModeratePositiveSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.ModerateNegativeSkew,fit=True,line='45')
plt.show()

df['HighlyPositiveSkew']=np.log(df.HighlyPositiveSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['HighlyNegativeSkew']=np.log(df.HighlyNegativeSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['ModeratePositiveSkew_1'], parameters=stats.yeojohnson(df.ModeratePositiveSkew)
sm.qqplot(df.ModeratePositiveSkew_1,fit=True,line='45')
plt.show()

df['ModerateNegativeSkew_1'], parameters=stats.yeojohnson(df.ModerateNegativeSkew)
sm.qqplot(df.ModerateNegativeSkew_1,fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['ModerateNegativeSkew']]))
sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt= QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2']=pd.DataFrame(qt.fit_transform(df[['ModerateNegativeSkew']]))

sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

df2=df.copy()

df2['HighlyPositiveSkew']= 1/df2.HighlyPositiveSkew
sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')

plt.show()
```
# OUTPUT:
![image](https://github.com/Leela1822/DS_exp6/assets/106167639/10b462b1-cd40-4fbc-8155-69ef77ea3921)


![image](https://github.com/Leela1822/DS_exp6/assets/106167639/5ca87a69-7bee-4058-b8ca-9c1ff713b940)


![image](https://github.com/Leela1822/DS_exp6/assets/106167639/101014ec-fb3f-457a-be7a-e94b693536c0)


![image](https://github.com/Leela1822/DS_exp6/assets/106167639/2ce3959f-a7fb-4ea4-9a89-29a6fd442fd6)


![image](https://github.com/Leela1822/DS_exp6/assets/106167639/96758c56-6e92-4aa5-83ba-454720cec6a3)


![image](https://github.com/Leela1822/DS_exp6/assets/106167639/dee626c0-3049-4f56-9fb8-42f96eddc0ae)


![image](https://github.com/Leela1822/DS_exp6/assets/106167639/f6744b06-1385-433b-ba4d-effb58f84b22)


![image](https://github.com/Leela1822/DS_exp6/assets/106167639/6fa5edf0-aa64-4284-b611-6ee1d312db01)


![image](https://github.com/Leela1822/DS_exp6/assets/106167639/03c6e826-3e9d-4ca2-b342-3ba0e832ca3f)


![image](https://github.com/Leela1822/DS_exp6/assets/106167639/de45f0b3-33c1-4c47-a0d4-696370483a12)


# RESULT:
Thus feature transformation is done for the given dataset.
