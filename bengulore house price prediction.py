#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
matplotlib.rcParams["figure.figsize"]=(20,10)


# In[2]:


df=pd.read_csv('D:\\data files\\Bengaluru_House_Data.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


#df.groupby('area_type')['area_type'].agg('count')
df['area_type'].unique()


# In[7]:


df2=df.drop(['area_type','society','balcony','availability'],axis='columns')
df2.shape


# In[8]:


df2.head()


# In[9]:


df2.isnull().sum()


# In[10]:


df3=df2.dropna()


# In[11]:


df3.shape


# In[13]:


df3['size'].unique()


# In[13]:


df3.groupby('size')['size'].agg('count')


# In[14]:


df3['bhk']=df3['size'].apply(lambda x:int(x.split(' ')[0]))
df3.bhk.unique()


# In[15]:


df3.head()


# In[16]:


df3['bhk'].unique()


# In[17]:


df3[df3.bhk>20]


# In[18]:


df3.total_sqft.unique()


# In[19]:


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[21]:


df3[~df3['total_sqft'].apply(is_float)].head()


# In[22]:


def convert_sqft_to_num(x):
    token=x.split('-')
    if len(token)==2:
        return(float(token[0])+float(token[1]))/2
    try:
        return float(x)
    except:
        return None


# In[22]:


convert_sqft_to_num('2133')


# In[23]:


convert_sqft_to_num('2354-3452')


# In[24]:


convert_sqft_to_num('34.4556 Meters')#no output means not working we need to do something


# In[23]:


df4=df3.copy()
df4['total_sqft']=df4.total_sqft.apply(convert_sqft_to_num)
df4 = df4[df4.total_sqft.notnull()]
df4.head(2)


# In[24]:


df4.head()


# In[25]:


df4.loc[30]


# # feature engineering

# In[26]:


df5=df4.copy()


# In[27]:


df5['price_per_sqft']=df5['price']*100000/df5['total_sqft']


# In[28]:


df5.head()


# In[30]:


df5_stats = df5['price_per_sqft'].describe()
df5_stats


# In[31]:


df5.to_csv("bhp.csv",index=False)


# In[29]:


len(df5.location.unique())


# In[32]:


df5.location=df5.location.apply(lambda x:x.strip())
location_stats=df5['location'].value_counts(ascending=False)
location_stats


# In[33]:


len(location_stats[location_stats>10])


# In[34]:


len(location_stats)


# In[36]:


len(location_stats[location_stats<=10])


# In[37]:


len(df5.location.unique())


# In[39]:


location_stats_less_than_ten=location_stats[location_stats<=10]
location_stats_less_than_ten


# In[40]:


df5.location=df5.location.apply(lambda x:'other' if x in location_stats_less_than_ten else x)
len(df5.location.unique())


# In[41]:


df5.head(10)


# # outliers removing

# In[42]:


df5[df5.total_sqft/df5.bhk<300].head()


# In[43]:


df5.shape


# In[44]:


df6=df5[~(df5.total_sqft/df5.bhk<300)]
df6.shape


# In[45]:


df6.shape


# In[50]:





# In[46]:


#df5.shape


# In[47]:


#df6=df5[~(df5.total_sqft/df5.bhk<300)]
#df6.shape


# In[48]:


df6.price_per_sqft.describe()


# In[50]:


def remove_ppt_outliers(df):
    df_out=pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m=np.mean(subdf.price_per_sqft)
        st=np.std(subdf.price_per_sqft)
        redused_df=subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft <= (m+st))]
        df_out=pd.concat([df_out,redused_df],ignore_index=True)
    return df_out
df7=remove_ppt_outliers(df6)
df7.shape


# In[51]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
def plot_scatter_chart(df,location):
    bhk2=df[(df.location==location) & (df.bhk==2)]
    bhk3=df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize']=(15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price_per_sqft,color='blue',label='2 bhk',s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price_per_sqft,color='green',label='3 bhk',s=50)
    plt.xlabel('total squre feet area')
    plt.ylabel('price per squrefeet')
    plt.title('location')
    plt.legend
plot_scatter_chart(df7,"Hebbal")#Hebbal, White field


# In[56]:


{
    '1':{
        'mean':4000,
        'std':2000,
        'count':34
    },
    '2':{
        'mean':4300,
        'std':2300,
        'count':22
    },
}


# In[88]:


#import numpy as np
def remove_bhk_outliers(df):
    exclude_indices=np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats= {}
        for bhk,bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk]={
                'mean':np.mean(bhk_df.price_per_sqft),
                'std':np.std(bhk_df.price_per_sqft),
                'count':bhk_df.shape[0]
            }
        for bhk,bhk_df in location_df.groupby('bhk'):
            stats=bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices=np.append(exclude_indices,bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df8=remove_bhk_outliers(df7)
df8.shape


# In[89]:


def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df8 = remove_bhk_outliers(df7)
# df8 = df7.copy()
df8.shape


# In[112]:


plot_scatter_chart(df8,"Rajaji Nagar")


# In[113]:


import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)
plt.hist(df8.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")


# In[90]:





# In[91]:


df8.bath.unique()


# In[114]:


plt.hist(df8.bath,rwidth=0.8)
plt.xlabel("Number of bathrooms")
plt.ylabel("Count")


# In[116]:


df8[df8.bath>10]


# In[117]:


df8[df8.bath>df8.bhk+2]


# In[118]:


df9=df8[df8.bath<df8.bhk+2]


# In[119]:


df9.shape


# In[120]:


df10=df9.drop(['size','price_per_sqft'],axis='columns')
df10.head(3)


# # model building
# 

# In[98]:


dummies=pd.get_dummies(df10.location)
dummies.head(3)


# In[121]:


#df11=pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')

df11 = pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
df11.head()


# In[123]:


df12=df11.drop('location',axis='columns')
df12.head()


# In[124]:


df12.shape


# In[140]:


X=df12.drop(['price'],axis='columns')
X.shape


# In[141]:


y=df12.price  # depending variable
y.shape


# In[142]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)


# # Linear Regression

# In[143]:


from sklearn.linear_model import LinearRegression
LR_model=LinearRegression()
LR_model.fit(x_train,y_train)
LR_model.score(x_test,y_test)


# In[144]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
cv=ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
cross_val_score(LinearRegression(),x,y,cv=cv)


# In[145]:


from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model_using_gridsearchcv(X,y)


# In[151]:


def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return LR_model.predict([x])[0]


# In[ ]:



   


# In[152]:


predict_price('Indira Nagar',1000,2,2)


# In[153]:


from sklearn.tree import DecisionTreeRegressor
dt_model=DecisionTreeRegressor()
dt_model.fit(x_train,y_train)
dt_model.score(x_test,y_test)


# In[155]:


#decition tree
def predict_price1(location,total_sqft,bath,bhk):
    loc_index=np.where(X.columns==location)[0][0]
    x = np.zeros(len(X.columns))
    x[0]=total_sqft
    x[1]=bath
    x[2]=bhk
    if loc_index>=0:
        x[loc_index]=1
    return dt_model.predict([x])[0]


# In[156]:


predict_price1('Indira Nagar',1000,2,2)


# In[ ]:





# In[ ]:




