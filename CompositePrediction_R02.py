#!/usr/bin/env python
# coding: utf-8

# In[39]:


# УСТАНОВКА БИБЛИОТЕК
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn
import seaborn as sns


# In[40]:


from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from tensorflow import keras 
from tensorflow.keras.layers import Dense 
from tensorflow.keras.layers import Flatten 
from pandas import read_excel, DataFrame, Series


# ### 1.  ОБЪЕДИНЕНИЕ ДАННЫХ. ФОРМИРОВАНИЕ  DATASET

# ### 1.1 Чтение данных 

# In[41]:


df1 = pd.read_excel('X_bp.xlsx', index_col = 0)
df1


# In[42]:


df2 = pd.read_excel('X_nup.xlsx', index_col = 0)
df2


# ### 1.2 Объдинение данных 

# In[43]:


df = df1.merge(df2, left_index = True, right_index = True, how = 'inner')
df


# In[44]:


# ИНФОРМАЦИЯ О ПОЛУЧЕННОМ DATASET
df.info()


# ### 1.3 Описательная статистика

# In[45]:


df.describe()


# In[46]:


# ПРОВЕРКА НАЛИЧИЯ ДУБЛИКАТОВ
df.duplicated().sum()


# ### 2. ВИЗУАЛИЗАЦИЯ

# In[47]:


df.columns


# ### 2.1 Визуальная оценка величины и характера разброса данных. Гистограммы 

# In[48]:


for col in df.columns:
    plt.figure(figsize=(15, 5))
    plt.title("Гистограмма" + str(col))
    sns.histplot(data=df[col])
    plt.show


# ### 2.2 Визуальная оценка величины и характера разброса данных. "Ящик с усами" 

# In[49]:


for col in df.columns:   
    sns.boxplot(x = col, data = df)
    plt.title(col)
    plt.show()


# Значения "Угол нашивки" не имеют выбросов. Прочие значения необходимо очистить от выбросов

# In[50]:


# ВНЕСЕНИЕ НАЗВАНИЙ СТОЛБЦОВ С ВЫБРОСАМИ (за исключением "Угол нашивки")
columns_drop = ["Соотношение матрица-наполнитель","Плотность, кг/м3","модуль упругости, ГПа","Количество отвердителя, м.%",
         "Содержание эпоксидных групп,%_2","Температура вспышки, С_2","Поверхностная плотность, г/м2",
         "Модуль упругости при растяжении, ГПа","Прочность при растяжении, МПа","Потребление смолы, г/м2",
                 "Шаг нашивки","Плотность нашивки"]


# In[51]:


#ПО ВСЕМ СТОЛБЦАМ С ВЫБРОСАМИ ВЫПОЛНЯЕТСЯ ЗАМЕНА ВЫБРОСОВ НА ПУСТЫЕ ЗНАЧЕНИЯ
for x in columns_drop:
    q75,q25 = np.percentile(df.loc[:,x],[75,25])
    intr_qr = q75-q25
    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)
    df.loc[df[x]<min,x] = np.nan
    df.loc[df[x]>max,x] = np.nan


# In[52]:


# КОЛИЧЕСТВО ВЫБРОСОВ ПО КАЖДОМУ СТОЛБЦУ
df.isnull().sum()


# In[53]:


# ИСКЛЮЧЕНИЕ СТРОК С ВЫБРОСАМИ (ПУСТЫЕ ЗНАЧЕНИЯ)
df_withoutdrop = df.dropna(axis=0)


# In[54]:


# ПОСТРОЕНИЕ ДИАГРАММ "ЯЩИК С УСАМИ"
for col in df.columns:   
    sns.boxplot(x = col, data = df_withoutdrop)
    plt.title(col)
    plt.show()


# ### 2.3 Визуальная оценка величины и характера разброса данных. Попарная корреляция

# In[55]:


from pandas.plotting import scatter_matrix
scatter_matrix(df, alpha=0.2, figsize=(16, 16), diagonal='kde');


# In[56]:


cols = df.columns
p = sns.PairGrid(df[cols]) # ПРИМЕНЕНИЕ МЕТОДА PAIRGRID ДЛЯ СОЗДАНИЯ ПАРНЫХ ГРАФИКОВ 
p.map(sns.scatterplot) 


# In[57]:


from pandas.plotting import scatter_matrix
scatter_matrix(df_withoutdrop, alpha=0.2, figsize=(16, 16), diagonal='kde');


# ### 2.4 Визуальная оценка величины и характера разброса данных. Тепловая карта

# In[58]:


# КОРРЕЛЯЦИЯ 
matrix = np.triu(df.corr())
f, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(df.corr(), vmax=0.3, center= 0, cmap= 'coolwarm', linecolor='black', mask = matrix)
plt.show ()


# In[59]:


# ОТОБРАЖЕНИЕ КОРРЕЛЯЦИИ С ЧИСЛОВЫМИ ЗНАЧЕНИЯМИ
matrix = np.triu(df.corr())
f, ax = plt.subplots(figsize = (15, 15))
sns.heatmap(df.corr(),annot=True, vmax=0.2, center= 0, cmap= 'coolwarm', linecolor='black', mask = matrix, linewidths = 1)


# In[60]:


from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# ### 2.5 Нормализация данных

# In[61]:


minmaxscalar = preprocessing.MinMaxScaler()
col = df_withoutdrop.columns
result = minmaxscalar.fit_transform(df_withoutdrop)
dfmm = pd.DataFrame(result, columns=col)
dfmm


# ### 2.6 Описательная статистика

# In[62]:


dfmm.describe()


#   ### 2.7 Определение корреляций между параметрами

# In[63]:


dfmm[df_withoutdrop.columns].corr()


# ### 3. ПОСТРОЕНИЕ НЕЙРОСЕТИ

#   ### 3.1 Импорт методов 

# In[71]:


from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Activation, Dropout
from numpy.random import seed
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


# In[72]:


# ОПРЕДЕЛЕНИЕ ВХОДНЫХ И ВЫХОДНЫХ ПЕРЕМЕННЫХ
inputcol = ['Плотность, кг/м3',
       'модуль упругости, ГПа', 'Количество отвердителя, м.%',
       'Содержание эпоксидных групп,%_2', 'Температура вспышки, С_2',
       'Поверхностная плотность, г/м2', 'Потребление смолы, г/м2',
       'Угол нашивки, град', 'Шаг нашивки', 'Плотность нашивки']
result = ['Соотношение матрица-наполнитель', 'Модуль упругости при растяжении, ГПа',
       'Прочность при растяжении, МПа']
X_train = dfmm[inputcol]
y_train = dfmm[result]


# In[73]:


# ИНФОРМАЦИЯ О ВХОДНЫХ И ВЫХОДНЫХ ПЕРЕМЕННЫХ
X_train.info()
y_train.info()


# In[74]:


# ПОДГОТОВКА ОБУЧАЮЩЕЙ И ТЕСТОВОЙ ВЫБОРОК (СООТНОШЕНИЕ 70 НА 30)
Xtrn, Xtest, Ytrn, Ytest = train_test_split(X_train, y_train, test_size=0.3)


# In[75]:


# ОПРЕДЕЛЕНИЕ ФУНКЦИИ create_model С ПАРАМЕТРАМИ lyrs, act, opt, dr
def create_model(lyrs=[128], act='relu', opt='Nadam', dr=0.0):
    
# СОЗДАНИЕ ЭКЗЕМПЛЯРА КЛАССА SEQUENTIAL (ПОСЛЕДОВАТЕЛЬНАЯ АРХИТЕКТУРА НЕЙРОСЕТИ)
    model = Sequential()
    
# СОЗДАНИЕ ПЕРВОГО СКРЫТОГО СЛОЯ
    model.add(Dense(lyrs[0], input_dim=X_train.shape[1], activation=act))
    
# СОЗДАНИЕ ДОПОЛНИТЕЛЬНЫХ СКРЫТЫХ СЛОЕВ
    for i in range(1,len(lyrs)):
        model.add(Dense(lyrs[i], activation=act))
    
# МЕТОД РЕГУЛЯРИЗАЦИИ ДЛЯ УМЕНЬШЕНИЯ ПЕРЕОБУЧЕНИЯ СЕТИ
    model.add(Dropout(dr))
    
# СОЗДАНИЕ ВЫХОДНОГО СЛОЯ
    model.add(Dense(3, activation='sigmoid'))

# НАСТРОЙКА ОБУЧЕНИЯ МОДЕЛИ ЧЕРЕЗ МЕТОД compile 
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
 
    return model


# In[76]:


# СОЗДАНИЕ БАЗОВОГО ЭКЗЕМПЛЯРА МОДЕЛИ
model=create_model()

#  ПОКАЗАТЬ АРХИТЕКТУРУ МОДЕЛИ
model.summary()


# In[77]:


# ОБУЧЕНИЕ МОДЕЛИ
model.fit(X_train, y_train,batch_size=30, epochs=15, validation_split=0.3)


# In[78]:


model.compile(optimizer='adam',
loss='categorical_crossentropy',
metrics=['accuracy'])


# In[80]:


model.evaluate(X_train, y_train)

print(X_train.columns)
print(y_train.columns)

model.save('./new_model/')

import joblib

a = df_withoutdrop[inputcol]
scaler = MinMaxScaler()
result = scaler.fit_transform(a)

joblib.dump(scaler, 'scaler.pkl')


# In[ ]:




