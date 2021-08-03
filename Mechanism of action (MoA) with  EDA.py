#!/usr/bin/env python
# coding: utf-8

# # Mechanism of action Dataset Exploratory Data Analysis(EDA)

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.metrics import log_loss
from sklearn import metrics


# # READING DATA :

# In[2]:


# Reading  testdata datasets
test_data = pd.read_csv("test_features.csv")
test_data


# In[3]:


# Reading  traindata datasets
train_data = pd.read_csv("train_features.csv")
train_data


# In[4]:


#Target variables
# Reading  targetdata datasets
target_data = pd.read_csv("train_targets_scored.csv")
target_data


# # EXPLORATORY DATA ANALYSIS :
#     
# Features:
# sig_id - It is the drug ID for each drug
# 
# cp_type - Represents two categories, trt_cp is the one which has mechanism of action and ctl_vehicle is simply placebo and has no mechanism of action
# 
# cp_time - Represents three categories of duration of treatment i.e 24, 48, 72 hours.
# 
# cp_dose - Represents two categories ie High dose and low dose 
# 
# g - Signifies gene expression data
# 
# c - Signifies cell viablity data

# In[5]:


#Exploratory data analysis
train_data.shape


# In[6]:


train_data.describe()


# In[7]:


test_data.shape


# In[8]:


test_data.describe()


# In[9]:


target_data.shape


# In[10]:


target_data.describe()


# In[11]:


# Missing values in both test and train data

missing_value_train =print(train_data.isnull().sum().sum())
missing_value_test = print(test_data.isnull().sum().sum())
missing_value_target = print(target_data.isnull().sum().sum())


# There are no missing values in the dataset.

# # Train features:
#  
# There are three categorical features in train data i.e cp_type, cp_time, cp_dose
# 

# In[5]:


# cp_type
train_data.cp_type.astype("category")
plt.figure(figsize=(5,5))
sns.set_palette("tab20")
sns.countplot(train_data['cp_type'], color='dodgerblue')
plt.title(" Samples with MoA (trt_cp) and sample without MoA (ctl_vehicle) ", fontsize=12, weight='bold')
plt.show()


# As we can see that cp_type is imbalanced whith high number of trt_cp, which is expected since it has MoA.As we know ctl_ vehcicle has no MoA(placebo), so it will of no use to us and we will discard it from our dataset for our further analysis

# In[97]:


# remove ctl_vehicle from train data
train_data= train_data.drop(train_data[train_data.cp_type == "ctl_vehicle"].index)
train_data


# In[6]:


#cp_time
train_data.cp_time.astype("category")
plt.figure(figsize=(5,5))
sns.set_palette("tab20")
sns.countplot(train_data['cp_time'], color='palegreen')
plt.title(" Duration of treatment ", fontsize=12, weight='bold')
plt.show()

# Treatment durations are pretty balanced for our dataset


# In[7]:


#cp_dose
train_data.cp_dose.astype("category")
plt.figure(figsize=(5,5))
sns.set_palette("tab20")
sns.countplot(train_data['cp_dose'], color='salmon')
plt.title("Dose type high or low ", fontsize=12, weight='bold')
plt.show()
#Dose type are pretty balanced and evenly distributed for our dataset


# # Distribution of Genes :

# In[14]:


gene = [g for g in train_data.columns if g.startswith("g-")]
plt.figure(figsize=(16,22))
plt.title("Distribution og Gene", fontsize=12, weight='bold')
sns.set_style('darkgrid')

gene_distribution =np.random.choice(len(gene),16)
for i,col in enumerate(gene_distribution):
    plt.subplot(4,4,i+1)
    plt.suptitle("Gene Data Distribution", fontsize=16)
    plt.ylabel('Count',fontsize=10)
    plt.xlabel('Distribution range',fontsize=10)
    plt.hist(train_data.loc[:,gene[col]],bins=100,color="purple")
    plt.title(gene[col])


# From above data we can see that the range for -10 to 10. Negative value indicates gene supression and positive value indicates gene expression

# # Distribution of cell viablity:

# In[16]:


cells = [c for c in train_data.columns if c.startswith("c-")]
plt.figure(figsize=(16,22))
sns.set_style('darkgrid')
gene_distribution =np.random.choice(len(gene),16)
for i,col in enumerate(gene_distribution):
    plt.subplot(4,4,i+1)
    plt.suptitle("Cell Viablity Data Distribution", fontsize=16)
    plt.ylabel('Count',fontsize=10)
    plt.xlabel('Distribution range',fontsize=10)
    plt.hist(train_data.loc[:,gene[col]],bins=100,color="coral")
    plt.title(gene[col])


# From above data we can see that the range for -10 to 10. Negative value indicates dead cells and positive value indicates living cells

# # Effect of cp_time and cp_dose on gene and cell viablity data

# In[102]:


# Effect of cp_time on  random cell viablity
plt.figure(figsize = (12,12))

plt.subplot(2,3,1)
c_1 = train_data[["c-1","cp_time"]]
c_1_24 = c_1[c_1["cp_time"]==24]
sns.histplot(c_1_24["c-1"],color = "red")
plt.title("C-1 Vs CP TIME = 24")

plt.subplot(2,3,2)
c_1_48 = c_1[c_1["cp_time"]==48]
sns.histplot(c_1_48["c-1"],color = "blue")
plt.title("C-1 Vs CP TIME = 48")


plt.subplot(2,3,3)
c_1_72 = c_1[c_1["cp_time"]==72]
sns.histplot(c_1_72["c-1"],color = "green")
plt.title("C-1 Vs CP TIME = 72")

plt.subplot(2,3,4)
c_2 = train_data[["c-50","cp_time"]]
c_2_24 = c_2[c_2["cp_time"]==24]
sns.histplot(c_2_24["c-50"],color = "red")
plt.title("C-2 Vs CP TIME = 24")

plt.subplot(2,3,5)
c_2_48 = c_2[c_2["cp_time"]==48]
sns.histplot(c_2_48["c-50"],color = "blue")
plt.title("C-2 Vs CP TIME = 48")


plt.subplot(2,3,6)
c_2_72 = c_2[c_2["cp_time"]==72]
sns.histplot(c_2_72["c-50"],color = "green")
plt.title("C-2 Vs CP TIME = 72")

plt.suptitle("Cell Viability Vs CP Time")

plt.show()


# We can say from above graph that there is an increase in negative cell viablity as the time duration increases.

# In[103]:


# Effect of cp_dose on  randomcell viablity
plt.figure(figsize = (12,12))

plt.subplot(2,2,1)
c_1 = train_data[["c-1","cp_dose"]]
c_1_D1 = c_1[c_1["cp_dose"]=="D1"]
sns.histplot(c_1_D1["c-1"],color = "red")
plt.title("C-1 Vs CP DOSE = D1")

plt.subplot(2,2,2)
c_1_D2 = c_1[c_1["cp_dose"]=="D2"]
sns.histplot(c_1_D2["c-1"],color = "green")
plt.title("C-1 Vs CP DOSE = D2")

plt.subplot(2,2,3)
c_2 = train_data[["c-50","cp_dose"]]
c_2_D2 = c_2[c_2["cp_dose"]=="D1"]
sns.histplot(c_2_D2["c-50"],color = "red")
plt.title("C-2 Vs CP DOSE = D1")

plt.subplot(2,2,4)
c_2_D2 = c_2[c_2["cp_dose"]=="D2"]
sns.histplot(c_2_D2["c-50"],color = "green")
plt.title("C-2 Vs CP DOSE = D2")

plt.suptitle("Cell Viability Vs CP DOSE")

plt.show()


# Higher dose has marginally more negative cell viablity than lower dose.

# In[20]:


## Effect of cp_time on  random Gene data
plt.figure(figsize = (12,12))

plt.subplot(2,3,1)
g_1 = train_data[["g-1","cp_time"]]
g_1_24 = g_1[g_1["cp_time"]==24]
sns.histplot(g_1_24["g-1"],color = "red")
plt.title("G-1 Vs CP TIME = 24")

plt.subplot(2,3,2)
g_1_48 = g_1[g_1["cp_time"]==48]
sns.histplot(g_1_48["g-1"],color = "blue")
plt.title("G-1 Vs CP TIME = 48")


plt.subplot(2,3,3)
g_1_72 = g_1[g_1["cp_time"]==72]
sns.histplot(g_1_72["g-1"],color = "green")
plt.title("G-1 Vs CP TIME = 72")

plt.subplot(2,3,4)
g_2 = train_data[["g-300","cp_time"]]
g_2_24 = g_2[g_2["cp_time"]==24]
sns.histplot(g_2_24["g-300"],color = "red")
plt.title("G-2 Vs CP TIME = 24")

plt.subplot(2,3,5)
g_2_48 = g_2[g_2["cp_time"]==48]
sns.histplot(g_2_48["g-300"],color = "blue")
plt.title("G-2 Vs CP TIME = 48")


plt.subplot(2,3,6)
g_2_72 = g_2[g_2["cp_time"]==72]
sns.histplot(g_2_72["g-300"],color = "green")
plt.title("G-2 Vs CP TIME = 72")

plt.suptitle("Gene Features Vs CP Time")

plt.show()


# We can say from above graph that there is an increase in negative Gene expression as the time duration increases. Which was also true in cell viablity case.

# In[21]:


# Effect of cp_dose on random Gene data
plt.figure(figsize = (12,12))

plt.subplot(2,2,1)
g_1 = train_data[["g-1","cp_dose"]]
g_1_D1 = g_1[g_1["cp_dose"]=="D1"]
sns.histplot(g_1_D1["g-1"],color = "red")
plt.title("G-1 Vs CP DOSE = D1")

plt.subplot(2,2,2)
g_1_D2 = g_1[g_1["cp_dose"]=="D2"]
sns.histplot(g_1_D2["g-1"],color = "blue")
plt.title("G-1 Vs CP DOSE = D2")

plt.subplot(2,2,3)
g_2 = train_data[["g-300","cp_dose"]]
g_2_D1 = g_2[g_2["cp_dose"]=="D1"]
sns.histplot(g_2_D1["g-300"],color = "red")
plt.title("G-2 Vs CP DOSE = D1")

plt.subplot(2,2,4)
g_2_D2 = g_2[g_2["cp_dose"]=="D2"]
sns.histplot(g_2_D2["g-300"],color = "blue")
plt.title("G-2 Vs CP DOSE = D2")

plt.suptitle("Cell Viability Vs CP DOSE")

plt.show()


# Higher dose shows more gene expression on both negative side and positive side.

# # Correlations

# In[137]:


# Correlations between genes

gene = train_data.iloc[1:3624 ,4:771]
plt.figure(figsize=(15,6))
sns.heatmap(gene.corr(),cmap='coolwarm',alpha=0.75)
plt.title('Correlation: Gene Data', fontsize=15, weight='bold')
plt.show()


# In[138]:


def diagonal_pairs(df):
    drop_pairs = set()
    x = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            drop_pairs.add((x[i], x[j]))
    return drop_pairs
def correlations(df,n=5):
    top_corr = df.corr().abs().unstack()
    labels_to_drop = diagonal_pairs(df)
    top_corr = top_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return top_corr[0:n]

diagonal_pairs(gene)
print(correlations(gene, 10))


# In[139]:


#Correlation between cells
cell = train_data.iloc[1:3624 ,776:876]
plt.figure(figsize=(15,6))
sns.heatmap(cell.corr(),cmap='twilight',alpha=0.75)
plt.title('Correlation: Cell Viablity Data', fontsize=15, weight='bold')
plt.show()


# In[140]:


diagonal_pairs(cell)
print(correlations(cell, 10))


# Both Gene data and cell viablity data shows good correlation  within them, but we can see from above that there is higher degree of correlation between Cell viablity data than Gene data.

# # Target variable
# 
# We have 207 MoA through which drugs are classified. A single drug can have more than one mechanism of action. 

# In[13]:


# Plotting distribution of target variables
plt.figure(figsize=(8,8))
sns.countplot(target_data.sum(axis=1))
plt.xlabel('Labels',fontsize=15)
plt.title(" Count Plot for Scored targets distribution", fontsize=16, weight='bold')
plt.show()


# From the above graph we can see that most of the drugs have atleast one target site. Although some drugs have more than one target site, its very negligeble compared to "0" and "1". Also some targets are repeated more often than others. We can have a look at them in our further analysis.

# In[142]:


# Sorting targets
x_axis = list(target_data.columns.values)
sig_id_values  = x_axis[1:]
len(sig_id_values) , sig_id_values[:4]
count_of_target = target_data.iloc[:,1:].sum().values
len(count_of_target) , count_of_target[:3]
dct =dict(zip(sig_id_values, count_of_target))
sorted_dict = dict( sorted(dct.items(), key=lambda i: i[1], reverse=True))

#TOP 20 TARGETS OCCURING MORE FREQUNTLY WITH THEIR COUNTS
list(sorted_dict.items())[:20]
plt.figure(figsize=(19,10))
plt.barh(list(sorted_dict.keys())[:20], list(sorted_dict.values())[:20] )
plt.title('COUNT OF TARGET VS TARGET FEATUES')
plt.xlabel('COUNT OF TARGET ')
plt.ylabel('TARGET FEATUES')
plt.show()


# We can conclude from above graph that we have two target features which have more common occurence i.e nkfb_inhibitor and proteasome_inhibitor.

# In[143]:


#LEAST 20 TARGETS OCCURING MORE FREQUNTLY WITH THEIR COUNTS
plt.figure(figsize=(19,10))
plt.barh(list(sorted_dict.keys())[-20:], list(sorted_dict.values())[-20:],alpha=0.8 )
plt.title('COUNT OF TARGET VS TARGET FEATUES')
plt.xlabel('COUNT OF TARGET ')
plt.ylabel('TARGET FEATUES')
plt.show()


# We can conclude from above graph that we have two target features which have only one occurence i.e erbb2_inhibitor and atp_sensitive_pottasium_channel_antagonist.
# 
# Outliers would not be removed from the dataset since they are important observations, and give meaningful insights to the dataset.

# # Correlations

# In[144]:


#Correlations within target variables
new_target_data = target_data.drop(["sig_id"], axis=1)
correlation_target= new_target_data.corr()
plt.figure(figsize= (10,10))
sns.heatmap(correlation_target, square= True,cmap=plt.cm.BuPu)
plt.title('Correlation: Target Data', fontsize=15, weight='bold')
plt.show()


# There is very less correlations among the target variables.

# In[146]:


diagonal_pairs(correlation_target)
print(correlations(correlation_target, 10))


# There are very few highly correlated pairs in target dataset.

# # Correlatrion between Training data features and Target  data features

# In[24]:


# Converting cp_time and cp_dose to categorical variable
train_data = train_data[train_data['cp_type']!='ctl_vehicle']
dummy_data = pd.get_dummies(train_data, columns=["cp_type","cp_time","cp_dose"])

# Merge two data frames of train_data and target_data
new_train_data = dummy_data.merge(target_data, on = "sig_id")


# Removed sig_id from above dataframe
mixed_correlation = new_train_data.drop(["sig_id"], axis=1)


# In[16]:


#correlation between train data and target features
get_ipython().run_line_magic('matplotlib', 'inline')
correlation_data= mixed_correlation.corr()
correlation_data_new =correlation_data.iloc[:875,875:]
plt.figure(figsize= (13,8))
plt.hist(correlation_data_new)
plt.ylabel('Count',fontsize=15)
plt.xlabel('Correlation',fontsize=15)
plt.title('Correlation: Training features and Target features', fontsize=16, weight='bold')
plt.show()


# The above plot  gives us  representation of correlation between Train data features and target features. We have some highly positive correlation and some highly negative correlation. Most of the data is weekly correlated, but we can sort some correlation among them in our further analysis.

# In[32]:


# Highly correlated Train features and target features
sort_corr = correlation_data_new.abs().unstack()
sort_corr.sort_values(ascending = False).drop_duplicates()[:40]


# In[33]:


# least correlated Train features and target features
sort_corr = correlation_data_new.abs().unstack()
sort_corr.sort_values(ascending = True).drop_duplicates()[:40]


# From the above analysis for correlation between train features and target features we can conclude that there Gene expression and cell viablity data can be used to predict target site of an drug with unknown mechanism of action.
# 

# # Test data
# 
# There are three categorical features in test data i.e cp_type, cp_time, cp_dose

# In[34]:


# cp_type
test_data.cp_type.astype("category")
plt.figure(figsize=(5,5))
sns.set_palette("tab20")
sns.countplot(test_data['cp_type'], color='tan')
plt.title("Test Data: Samples with MoA (trt_cp) and sample without MoA (ctl_vehicle) ", fontsize=12, weight='bold')
plt.show()


# As we can see that cp_type is imbalanced whith high number of trt_cp, which is expected since it has MoA.
#  As we know ctl_ vehcicle has no MoA, so it will of no use to us and we will discard it from our dataset for our further analysis

# In[5]:


# remove ctl_vehicle from train data
test_data= test_data.drop(test_data[test_data.cp_type == "ctl_vehicle"].index)
test_data_dummie = pd.get_dummies(test_data, columns=["cp_type","cp_time","cp_dose"])
test_data_dummie


# In[6]:


#cp_time
test_data.cp_time.astype("category")
plt.figure(figsize=(5,5))
sns.set_palette("tab20")
sns.countplot(test_data['cp_time'], color='crimson')
plt.title("Test Data : Duration of treatment ", fontsize=12, weight='bold')
plt.show()
# Treatment durations are pretty balanced for our dataset


# In[7]:


test_data.cp_dose.astype("category")
plt.figure(figsize=(5,5))
sns.set_palette("tab20")
sns.countplot(test_data['cp_dose'], color='olive')
plt.title("Test Data : Dose type high or low ", fontsize=12, weight='bold')
plt.show()
#Dose type are pretty balanced and evenly distributed for our dataset


# # Distribution of Gene in Test data

# In[19]:


gene_test = [g for g in test_data.columns if g.startswith("g-")]
plt.figure(figsize=(16,22))
sns.set_style('darkgrid')
gene_distribution_test =np.random.choice(len(gene_test),16)
for i,col in enumerate(gene_distribution_test):
    plt.subplot(4,4,i+1)
    plt.suptitle("Gene Data Distribution", fontsize=16)
    plt.ylabel('Count',fontsize=10)
    plt.xlabel('Distribution range',fontsize=10)
    plt.hist(test_data.loc[:,gene_test[col]],bins=100,color="darkcyan")
    plt.title(gene_test[col])


# From above data we can see that the range for -10 to 10. Negative value indicates gene supression and positive value indicates gene expression

# # Distribution of cell viablity in Test data

# In[18]:


cell_test = [c for c in test_data.columns if c.startswith("c-")]
plt.figure(figsize=(16,22))
sns.set_style('darkgrid')
cell_distribution_test =np.random.choice(len(cell_test),16)
for i,col in enumerate(cell_distribution_test):
    plt.subplot(4,4,i+1)
    plt.suptitle("Cell Viablity Data Distribution", fontsize=16)
    plt.ylabel('Count',fontsize=10)
    plt.xlabel('Distribution range',fontsize=10)
    plt.hist(test_data.loc[:,cell_test[col]],bins=100,color="crimson")
    plt.title(cell_test[col])


# From above data we can see that the range for -10 to 10. Negative value indicates dead cells and positive value indicates living cells.
# 
# Thus we can say that both test data and train data has gene data and cell viablity data within the range of -10 to 10.

# # Correlation within Gene data and Cellvibality data

# In[119]:


#Correlation within Gene data
gene_test = test_data.iloc[1:3624 ,4:771]
diagonal_pairs(gene_test)
print(correlations(gene_test, 10))


# In[42]:


#Correlation within Cell viablity data
cell_test = test_data.iloc[1:3624 ,776:876]
diagonal_pairs(cell_test)
print(correlations(cell_test, 10))


# From the above correlation we can conclude that both train and test data for Gene expression and cell viablity shows similar correlation

# # Converting multilabelled data into binary data for analysis

# In[30]:


# Converting multilabelled data into bivariant data for analysis

new_target_data= new_train_data.iloc[:,879:]
new_target_data['y'] =  new_target_data.iloc[:,:].sum(axis=1)
new_target_data.shape


# In[33]:


plt.figure(figsize=(5,5))
sns.countplot(new_target_data["y"])

plt.title("The Scored targets distribution for Multi-labelled Data", fontsize=16, weight='bold')
plt.ylabel('Count',fontsize=10)
plt.xlabel('Number of target sites',fontsize=10)


# In[35]:


new_target_data.loc[new_target_data['y'] > 0, 'y' ]=1


# In[36]:


plt.figure(figsize=(5,5))
sns.countplot(new_target_data["y"])
plt.ylabel('Count',fontsize=10)
plt.xlabel('Number of target sites',fontsize=10)

plt.title("The Scored targets distribution for Binary Data", fontsize=16, weight='bold');


# Multi labelled data is converted to Binary data with catogeries of 0 represents no mechanism of action and 1 represents mechaniosm of action at 1 or more site.
