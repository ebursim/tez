import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,KFold

path="/home/simsek/workspace/EDUCATION/Thesis/test"
useful_cols=["CoordID","Latitude","Longitude","Date(yyyy-MM-dd)","B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B11","B12","quality_scene_classification"]
feature_bands=["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B11","B12"]
#feature_bands=["B3","B4","B5","B8A","B12"]
feature_variables_sc=pd.DataFrame()
feature_variables_nosc=pd.DataFrame()
feature_variables_sc=pd.DataFrame()
cloudy_pixel_count=0

for filename in os.listdir(path):
#    print(filename)
    if filename.endswith(".csv"):
        if "_snow" in filename:
            df=pd.read_csv(os.path.join(path,filename), skiprows=range(6), delimiter="\t")
            df=df[useful_cols]
            cloudless_pixels=df.query("quality_scene_classification!=7 and quality_scene_classification!=8 and quality_scene_classification!=9 and quality_scene_classification!=10")
            if(cloudless_pixels.empty==False):
                cloudless_pixels=cloudless_pixels[feature_bands]
                feature_variables_sc=feature_variables_sc.append(cloudless_pixels)
            else:
                cloudy_pixel_count+=1

for filename in os.listdir(path):
    #print(filename)
    if filename.endswith(".csv"):
        if "NOsnow" in filename:
            nosnow_df=pd.read_csv(os.path.join(path,filename), skiprows=range(6), delimiter="\t")
            nosnow_df=nosnow_df[useful_cols]
            cloudless_pixels=nosnow_df.query("quality_scene_classification!=7 and quality_scene_classification!=8 and quality_scene_classification!=9 and quality_scene_classification!=10")
            if(cloudless_pixels.empty==False):
                cloudless_pixels=cloudless_pixels[feature_bands]
                feature_variables_nosc=feature_variables_nosc.append(cloudless_pixels)
            else:
                cloudy_pixel_count+=1

sc=np.repeat(1,feature_variables_sc.shape[0])
nosc=np.repeat(0,feature_variables_nosc.shape[0])
feature_variables_sc.insert(loc=len(feature_bands),column="SnowCover",value=sc)
feature_variables_nosc.insert(loc=len(feature_bands),column="SnowCover",value=nosc)
feature_variables=feature_variables_sc.append(feature_variables_nosc)

##### Discretization ########
feature_variables.B1=pd.cut(feature_variables.B1,10,labels=[1,2,3,4,5,6,7,8,9,10])
feature_variables.B2=pd.cut(feature_variables.B2,10,labels=[1,2,3,4,5,6,7,8,9,10])
feature_variables.B3=pd.cut(feature_variables.B3,10,labels=[1,2,3,4,5,6,7,8,9,10])
feature_variables.B4=pd.cut(feature_variables.B4,10,labels=[1,2,3,4,5,6,7,8,9,10])
feature_variables.B5=pd.cut(feature_variables.B5,10,labels=[1,2,3,4,5,6,7,8,9,10])
feature_variables.B6=pd.cut(feature_variables.B6,10,labels=[1,2,3,4,5,6,7,8,9,10])
feature_variables.B7=pd.cut(feature_variables.B7,10,labels=[1,2,3,4,5,6,7,8,9,10])
feature_variables.B8=pd.cut(feature_variables.B8,10,labels=[1,2,3,4,5,6,7,8,9,10])
feature_variables.B8A=pd.cut(feature_variables.B8A,10,labels=[1,2,3,4,5,6,7,8,9,10])
feature_variables.B9=pd.cut(feature_variables.B9,10,labels=[1,2,3,4,5,6,7,8,9,10])
feature_variables.B11=pd.cut(feature_variables.B11,10,labels=[1,2,3,4,5,6,7,8,9,10])
feature_variables.B12=pd.cut(feature_variables.B12,10,labels=[1,2,3,4,5,6,7,8,9,10])
##########################

feature_variables=feature_variables.dropna()
feature_variables=feature_variables.reset_index(drop=True)
#feature_variables.to_csv("deneme.csv", sep="\t",index=False)    disabled temporarily

# train, test = train_test_split(feature_variables, test_size = 0.2)
# # Resetting indexes after splits
# train = train.reset_index(drop=True)
# test = test.reset_index(drop=True)


def likelihood(train):
    ### Laplace Smoothing #####
    N_snow = len(train[train.SnowCover==1]) + 10.0
    N_nosnow=len(train[train.SnowCover==0]) + 10.0
    snow_likelihood = np.zeros((10,len(feature_bands)))
    nosnow_likelihood = np.zeros((10,len(feature_bands)))
    #### Calculate likelihood ####
    j=0
    for column in train[train.SnowCover==1].drop("SnowCover", axis=1):
        for i in range(10):
            count=train[train.SnowCover==1].drop(["SnowCover"], axis=1).groupby([column]).size()[i+1]
            snow_likelihood[i,j]=(count+1.0)/N_snow
        j+=1
    j=0
    for column in train[train.SnowCover==0].drop("SnowCover", axis=1):
        for i in range(10):
            count=train[train.SnowCover==0].drop(["SnowCover"], axis=1).groupby([column]).size()[i+1]
            nosnow_likelihood[i,j]=(count+1.0)/N_nosnow
        j+=1
    return np.log(snow_likelihood), np.log(nosnow_likelihood)


## predictions
def predict(test_data, snow_likelihood, nosnow_likelihood, class_prior):
    test_data = test_data.drop(["SnowCover"], axis=1)
    pred = np.zeros(len(test_data))
    k = 0
    for row in test_data.itertuples():
        pos_snow = 0.0
        pos_nosnow = 0.0
        for i in range(len(feature_bands)):
            pos_snow += snow_likelihood[:,i][row[1:13][i]-1]
            pos_nosnow += nosnow_likelihood[:,i][row[1:13][i]-1]
        if(pos_snow+class_prior[1]>pos_nosnow+class_prior[0]):
            pred[k] = 1
        k+=1
    return pred


#pred = predict(test, train_lk_snow, train_lk_nosnow)
# #### calculate accuracy of the classification
# count = 0.0
# for i in range(len(test)):
#     if test.SnowCover[i] == pred[i]:
#         count +=1.0
#
# print(count/len(test))

### Cross Validation #########################################
subsets=10
feature_variables = feature_variables.sample(frac=1).reset_index(drop=True)
kf = KFold(subsets)
kf.get_n_splits(feature_variables)
# cv_data = list(kf.split(data))
k = 0
pred = np.zeros((subsets, len(feature_variables)/subsets))
count = np.zeros(subsets)
acc = np.zeros(subsets)


for train_index, test_index in kf.split(feature_variables):
    # Calculating class prior probabilities
    N_snow = len(feature_variables.iloc[train_index][feature_variables.SnowCover==1]) + 10.0
    N_nosnow=len(feature_variables.iloc[train_index][feature_variables.SnowCover==0]) + 10.0
    class_prior = np.zeros(2)
    class_prior[0] = np.log(N_nosnow / (N_snow + N_nosnow))
    class_prior[1] = np.log(N_snow / (N_snow + N_nosnow))
    likelihood_snow,likelihood_nosnow = likelihood(feature_variables.iloc[train_index])
    pred[k,:] = (predict(feature_variables.iloc[test_index], likelihood_snow, likelihood_nosnow, class_prior))
    count[k] = sum(feature_variables.SnowCover.iloc[test_index]==pred[k,:])
    acc[k] = count[k]/len(feature_variables.iloc[test_index])
    k+=1


print(np.mean(acc))

##################################
# sc=np.repeat(1,feature_variables.shape[0])
# feature_variables.insert(loc=12,column="SnowCover",value=sc)
