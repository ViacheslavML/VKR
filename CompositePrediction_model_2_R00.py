#!/usr/bin/env python
# coding: utf-8

# ### 1. ИМПОРТ БИБЛИОТЕК

# In[1]:




import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn import tree
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn.metrics import explained_variance_score
import tensorflow as tf

from tensorflow import keras
from keras.models import Sequential 
from keras.layers import Dense
from keras import models
from keras import layers


# ### 1.1 Объединение данных

# In[2]:


x_bp = pd.read_excel('X_bp.xlsx', index_col = 0)
x_nup = pd.read_excel('X_nup.xlsx', index_col = 0)


# In[3]:


df_sum = x_bp.merge(x_nup, left_index = True, right_index = True, how = 'inner')


# In[4]:


df_sum.info()


# In[5]:


col = len(df_sum.index)
inform_col = []
for i in df_sum.columns:
    c = df_sum[i].value_counts(dropna = False)
    pr = (c / col).iloc[0]
    if pr > 0.95:
        inform_col.append(i)
        print(c)
        print()
print('done')


# ### 1.2 Анализ пропусков данных

# In[6]:


df_sum.isnull().sum()


# In[7]:


df_sum = df_sum.dropna(axis = 0)
df_sum.describe()


# In[8]:


skaler = MinMaxScaler()
df_sum_normal = pd.DataFrame(skaler.fit_transform(df_sum), columns = df_sum.columns, index = df_sum.index)


# ### 1.3 Нормализация данных

# In[9]:


df_sum_normal.describe()


# In[10]:


target = df_sum_normal ['Модуль упругости при растяжении, ГПа']
train = df_sum_normal [['модуль упругости, ГПа', 'Количество отвердителя, м.%', 'Содержание эпоксидных групп,%_2',                       'Температура вспышки, С_2', 'Поверхностная плотность, г/м2', 'Потребление смолы, г/м2',                       'Угол нашивки, град', 'Шаг нашивки', 'Плотность нашивки']]


# In[11]:


Xtrain, Xtest, Ytrain, Ytest = train_test_split(train, target, test_size = 0.3)


# ### 2. АНАЛИЗ ДАННЫХ. ПРИМЕНЕНИЕ НАБОРА МЕТОДОВ

# In[12]:


# ЛИНЕЙНАЯ РЕГРЕССИЯ
test_set = [] 
test_set_R = []
for i in range(100):
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(train, target, test_size = 0.3)
    lin_reg_mod = LinearRegression()
    lin_reg_mod.fit(Xtrain, Ytrain)
    prediction = lin_reg_mod.predict(Xtest)
    test_set.append(np.sqrt(mean_squared_error(Ytest, prediction)))
    test_set_R.append(r2_score(Ytest, prediction))
print(np.median(test_set))
print(np.median(test_set_R))


# In[13]:


# ЛИНЕЙНАЯ РЕГРЕССИЯ
cv = ShuffleSplit(n_splits = 10, test_size = 0.3, random_state=0)
results = cross_validate(estimator=LinearRegression(), X = train, y = target,
                cv = cv, scoring=('neg_mean_absolute_error','neg_mean_squared_error', 'r2', ))
print('Средняя абсолютная ошибка = ', results['test_neg_mean_absolute_error'].mean())
print('Средняя квадратичная ошибка = ', results['test_neg_mean_squared_error'].mean())
print('Коэффициент детерминации = ', results['test_r2'].mean()) 


# In[14]:


# РЕГРЕССИЯ МЕТОДОМ ОПОРНЫХ ВЕКТОРОВ
cv = ShuffleSplit(n_splits = 10, test_size = 0.3, random_state=0)
results = cross_validate(estimator=svm.SVR(), X = train, y = target,
                cv = cv, scoring=('neg_mean_absolute_error','neg_mean_squared_error', 'r2', ))
print('Средняя абсолютная ошибка = ', results['test_neg_mean_absolute_error'].mean())
print('Средняя квадратичная ошибка = ', results['test_neg_mean_squared_error'].mean())
print('Коэффициент детерминации = ', results['test_r2'].mean()) 
results = 0


# In[15]:


# МЕТОД СТОХАСТИЧЕСКИХ ГРАДИЕНТОВ
cv = ShuffleSplit(n_splits = 10, test_size = 0.3, random_state=0)
reg = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3))
results = cross_validate(reg, X = train, y = target,
                cv = cv, scoring=('neg_mean_absolute_error','neg_mean_squared_error', 'r2', ))
print('Средняя абсолютная ошибка = ', results['test_neg_mean_absolute_error'].mean())
print('Средняя квадратичная ошибка = ', results['test_neg_mean_squared_error'].mean())
print('Коэффициент детерминации = ', results['test_r2'].mean()) 


# In[16]:


# МЕТОД ДЕРЕВЬЕВ
cv = ShuffleSplit(n_splits = 10, test_size = 0.3, random_state=0)
reg = tree.DecisionTreeRegressor()
results = cross_validate(reg, X = train, y = target,
                cv = cv, scoring=('neg_mean_absolute_error','neg_mean_squared_error', 'r2', ))
print('Средняя абсолютная ошибка = ', results['test_neg_mean_absolute_error'].mean())
print('Средняя квадратичная ошибка = ', results['test_neg_mean_squared_error'].mean())
print('Коэффициент детерминации = ', results['test_r2'].mean()) 


# In[17]:


# МЕТОД СЛУЧАЙНЫХ ДЕРЕВЬЕВ
cv = ShuffleSplit(n_splits = 10, test_size = 0.3, random_state=0)
reg = RandomForestRegressor(max_depth=2, random_state=0)
results = cross_validate(reg, X = train, y = target,
                cv = cv, scoring=('neg_mean_absolute_error','neg_mean_squared_error', 'r2', ))
print('Средняя абсолютная ошибка = ', results['test_neg_mean_absolute_error'].mean())
print('Средняя квадратичная ошибка = ', results['test_neg_mean_squared_error'].mean())
print('Коэффициент детерминации = ', results['test_r2'].mean()) 


# In[18]:


# МЕТОД LASSO
cv = ShuffleSplit(n_splits = 10, test_size = 0.3, random_state=0)
reg = LassoCV()
results = cross_validate(reg, X = train, y = target,
                cv = cv, scoring=('neg_mean_absolute_error','neg_mean_squared_error', 'r2', ))
print('Средняя абсолютная ошибка = ', results['test_neg_mean_absolute_error'].mean())
print('Средняя квадратичная ошибка = ', results['test_neg_mean_squared_error'].mean())
print('Коэффициент детерминации = ', results['test_r2'].mean()) 


# In[19]:


# МЕТОД ГРЕБНЕВАЯ РЕГРЕССИЯ
cv = ShuffleSplit(n_splits = 10, test_size = 0.3, random_state=0)
reg = RidgeCV()
results = cross_validate(reg, X = train, y = target,
                cv = cv, scoring=('neg_mean_absolute_error','neg_mean_squared_error', 'r2', ))
print('Средняя абсолютная ошибка = ', results['test_neg_mean_absolute_error'].mean())
print('Средняя квадратичная ошибка = ', results['test_neg_mean_squared_error'].mean())
print('Коэффициент детерминации = ', results['test_r2'].mean()) 


# In[20]:


# МЕТОД ElasticNet 
cv = ShuffleSplit(n_splits = 10, test_size = 0.3, random_state=0)
reg = ElasticNetCV()
results = cross_validate(reg, X = train, y = target,
                cv = cv, scoring=('neg_mean_absolute_error','neg_mean_squared_error', 'r2', ))
print('Средняя абсолютная ошибка = ', results['test_neg_mean_absolute_error'].mean())
print('Средняя квадратичная ошибка = ', results['test_neg_mean_squared_error'].mean())
print('Коэффициент детерминации = ', results['test_r2'].mean()) 


# In[21]:


# РЕГРЕССИЯ МЕТОДОМ ОПОРНЫХ ВЕКТОРОВ
cv = ShuffleSplit(n_splits = 10, test_size = 0.3, random_state=0)
results = cross_validate(estimator=svm.SVR(), X = train, y = target,
                cv = cv, scoring=('neg_mean_absolute_error','neg_mean_squared_error', 'r2', ))
print('Средняя абсолютная ошибка = ', results['test_neg_mean_absolute_error'].mean())
print('Средняя квадратичная ошибка = ', results['test_neg_mean_squared_error'].mean())
print('Коэффициент детерминации = ', results['test_r2'].mean()) 
results = 0


# In[22]:


# МЕТОД СТОХАСТИЧЕСКИХ ГРАДИЕНТОВ
cv = ShuffleSplit(n_splits = 10, test_size = 0.3, random_state=0)
reg = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3))
results = cross_validate(reg, X = train, y = target,
                cv = cv, scoring=('neg_mean_absolute_error','neg_mean_squared_error', 'r2', ))
print('Средняя абсолютная ошибка = ', results['test_neg_mean_absolute_error'].mean())
print('Средняя квадратичная ошибка = ', results['test_neg_mean_squared_error'].mean())
print('Коэффициент детерминации = ', results['test_r2'].mean()) 


# ### 3. ПОСТРОЕНИЕ МОДЕЛИ

# In[23]:


target = df_sum_normal ['Соотношение матрица-наполнитель']
train = df_sum_normal.drop(['Прочность при растяжении, МПа', 'Соотношение матрица-наполнитель',                           'Модуль упругости при растяжении, ГПа' ], axis = 1)


# In[24]:


Xtrain, Xtest, Ytrain, Ytest = train_test_split(train, target, test_size = 0.3)


# In[25]:


Xtrain.shape


# In[26]:


massiv_optimizer = ['SGD', 'RMSprop', 'Adam', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl']
massiv_activation = ['relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh', 'selu', 'elu', 'exponential']
results = []
results_mae = []
results_mse = []
for i in range(len(massiv_optimizer)):
    for j in range(len(massiv_activation)):
        for z in range(len(massiv_activation)):
            model = Sequential([
                Dense(9, activation = massiv_activation [j], input_dim = 10),
                Dense(5, activation = massiv_activation [z]),
                Dense(1),
            ])

            model.compile(optimizer = massiv_optimizer[i], loss='mse', metrics=['mae']) 
            model.summary()
            model.fit(Xtrain, Ytrain, batch_size=32, epochs=10)
            model.evaluate(Xtest, Ytest) 
            cash = str(massiv_optimizer[i])+ '; ' + massiv_activation [j] + '; '+ massiv_activation [z] + '; '+  str(model.evaluate(Xtest, Ytest)[0])+ '; '+  str(model.evaluate(Xtest, Ytest)[1]) + '\n' 
            results.append(cash)
            results_mae.append(model.evaluate(Xtest, Ytest)[1])
            results_mse.append(model.evaluate(Xtest, Ytest)[0])           


# In[27]:


for i in range(len(results)):
    print (results[i])


# ### 3.1 Расчет метрик MAE, MSE

# In[28]:


results_mae


# In[29]:


results_mse


# In[30]:


min(results_mae)


# In[31]:


min(results_mse)


# In[32]:


results_mae.index(min(results_mae))


# In[33]:


results_mse.index(min(results_mse))


# In[34]:


results[results_mae.index(min(results_mae))]


# In[35]:


results[results_mse.index(min(results_mse))]


# In[36]:


plt.title('Гистограмма')
plt.ylabel("значение")
sns.histplot(data = results_mse)
plt.xlim([0, 0.2])
plt.ylim([0,500])


# In[37]:


plt.title('Гистограмма')
plt.ylabel("значение")
sns.histplot(data = results_mae)
plt.xlim([0.1, 0.6])
plt.ylim([0,400])


# In[38]:


massiv_optimizer = ['RMSprop']
massiv_activation = ['relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh', 'selu', 'elu', 'exponential']
results = []
results_mae = []
results_mse = []
for i in range(len(massiv_optimizer)):
    for j in range(len(massiv_activation)):
        for z in range(len(massiv_activation)):
            model = Sequential([
                Dense(9, activation = massiv_activation [j], input_dim = 10),
                Dense(5, activation = massiv_activation [z]),
                Dense(1),
            ])

            model.compile(optimizer = massiv_optimizer[i], loss='mse', metrics=['mae']) 
            model.summary()
            model.fit(Xtrain, Ytrain, batch_size=32, epochs=10)
            model.evaluate(Xtest, Ytest) 
            cash = str(massiv_optimizer[i])+ '; ' + massiv_activation [j] + '; '+ massiv_activation [z] + '; '+  str(model.evaluate(Xtest, Ytest)[0])+ '; '+  str(model.evaluate(Xtest, Ytest)[1]) + '\n' 
            results.append(cash)
            results_mae.append(model.evaluate(Xtest, Ytest)[1])
            results_mse.append(model.evaluate(Xtest, Ytest)[0]) 


# In[39]:


min(results_mae)


# In[40]:


min(results_mse)


# In[41]:


results_mae.index(min(results_mae))


# In[42]:


results_mse.index(min(results_mse))


# In[43]:


results[results_mae.index(min(results_mae))]


# In[44]:


results[results_mse.index(min(results_mse))]


# ### 3.2 Оценка модели

# In[45]:


model = Sequential([
    Dense(9, activation = 'softsign', input_dim = 10),
    Dense(9, activation='softsign'),
    Dense(1, activation='softsign'),
    
])

model.compile(optimizer='RMSprop', 
              loss='mse', 
              metrics=['mae']) 

model.summary()
model.fit(Xtrain, Ytrain, batch_size=32, epochs=10)
model.evaluate(Xtest, Ytest) 


# In[ ]:




