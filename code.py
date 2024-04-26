#!/usr/bin/env python
# coding: utf-8

# In[1]:


from nilearn import masking
from nilearn import maskers
from joblib import Parallel,delayed
import numpy as np 
from glob import glob
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import random
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import xgboost as xgb
import pickle
import shap
import scipy.sparse
import tempfile
import json
import pandas as pd
import spacy
from collections import Counter


# In[2]:


#Construction Mask
masks = glob(r"/media/sunjc/program/tzq/tpl-MNI152NLin2009cAsym/*.nii") #Mask provided by the data set
mask = masking.intersect_masks(masks,threshold=1)   


# In[4]:


func = maskers.NiftiMasker(mask,smoothing_fwhm=6,standardize = True,detrend = True)
func.fit()


# In[ ]:


#10071 voxel number obtained by ISC method
volume_10071 = np.load(r"/media/sunjc/program/tzq/signal/volume_10071.npy")
#Abbreviated names of the nine stimulus tasks
story_name = ["21styear","lucy","milky-ori","milky-vod","mouth","notintact","pieman","slum","tunnel"]


# In[ ]:


#Extraction of fMRI data
def para_fun_story(i):
    return func.transform_single_imgs(total_story[i])
story_data = {}
for name in story_name:
    total_story = glob(r"/media/sunjc/program/tzq/afni-nosmooth/story/"+name+"/*.nii") #Data set provided by fMRI data
    tmp_data = Parallel(n_jobs = -1)(delayed(para_fun_story)(i) for i in range(len(total_story)))
    story_data[name] = (np.sum(tmp_data,axis = 0)/len(tmp_data))[:,volume_10071]


# In[ ]:


#Find 3671 non-repeating words
story_csv = glob(r"/media/sunjc/program/tzq/align_csv/*.csv")#The csv file provided with the dataset, written with the word and time information
story_word_dic = {}
for story in story_csv:
    tmp = []
    tmp_csv = pd.read_csv(story)
    for i in range(len(tmp_csv)):
        if tmp_csv.loc[i,"a"] == "<unk>" or str(tmp_csv.loc[i,"a"])=="nan":
            tmp.append(tmp_csv.loc[i,"A"])
        else:
            tmp.append(tmp_csv.loc[i,"a"])
    story_word_dic[story[21:-4]] = tmp
uni_word = []
for name in story_name:
    for word in story_word_dic[name]:
        if word in uni_word:
            pass
        else:
            uni_word.append(word)


# In[ ]:


#Constructing HRF
time_length = 30.0
frame_times = np.linspace(0, time_length, 301)
onset, amplitude, duration = 0.0, 1.0, 1.0
exp_condition = np.array((onset, duration, amplitude)).reshape(3, 1)
oversampling = 16
signal, _labels = compute_regressor(
        exp_condition,
        "spm",
        frame_times,
        con_id="main",
        fir_delays = [5.5],
        oversampling=oversampling,
    )


# In[ ]:


#Convolution with HRF by the time course of the stimulus to construct the sample set
story_signal_points = {"21styear":2249,"lucy":368,"milky-ori":297,"milky-vod":297,"mouth":475,"notintact":397,"pieman":299,"slum":1202,"tunnel":1038}
total_story_hrf = {}
for story,name in zip(story_csv,story_name):
    tmp_story_hrf = []
    tmp_uni_word = np.zeros((3671,story_signal_points[name]*15))
    tmp_csv = pd.read_csv(story)
    for i in range(len(tmp_csv)):
        if tmp_csv.loc[i,"end"]>0:
            if tmp_csv.loc[i,"a"] == "<unk>" or str(tmp_csv.loc[i,"a"])=="nan":
                tmp_word = tmp_csv.loc[i,"A"]
            else:
                tmp_word = tmp_csv.loc[i,"a"]
            tmp_endtime = tmp_csv.loc[i,"end"]*100
            if (tmp_endtime%10) > 4.8:
                end_time = tmp_endtime//10 + 1
            else:
                end_time = tmp_endtime//10
        tmp_uni_word[uni_word.index(tmp_word)][int(end_time)] = 1
    for z in tmp_uni_word:
        tmp_story_hrf.append(np.convolve(z,signal.flatten()))
    total_story_hrf[name] = tmp_story_hrf


# In[ ]:


#Extraction of low-level stimulus characteristics
story_jsons = glob(r"/media/sunjc/program/tzq/align_json/*json")#The json file provided with the dataset, written with information about the phoneme levels
total_phone = []
for story_json in story_jsons:
    with open (story_json,"r") as load_f:
        tmp_json = json.load(load_f)
    for word in tmp_json["words"]:
        if word["case"] =="success":
            for phone in word["phones"]:
                total_phone.append(phone["phone"])
uni_phone = np.unique(total_phone)


total_word_rate = {}
total_phone_rate = {}
total_phone_categories = {}
for story,story_json,name in zip(story_csv,story_jsons,story_name):
    with open (story_json,"r") as load_f:
        tmp_json = json.load(load_f)
    tmp_csv = pd.read_csv(story)
    tmp_word_rate = np.zeros(len(story_volume_number[name])*15)
    tmp_phone_rate = np.zeros(len(story_volume_number[name])*15)
    tmp_phone_categories = np.zeros((len(story_volume_number[name])*15,114))
    for i in range(len(tmp_csv)):
        if tmp_csv.loc[i,"start"]>0:
            tmp_word_rate[int(tmp_csv.loc[i,"start"]*10+0.5)]+=1
            if tmp_json["words"][i]["case"] == "success":
                tmp_phone_rate[int(tmp_csv.loc[i,"start"]*10+0.5)]+=len(tmp_json["words"][i]["phones"])
                for j in range(len(tmp_json["words"][i]["phones"])):
                    tmp_phone_cate = np.where(uni_phone==tmp_json["words"][i]["phones"][j]["phone"])[0]
                    tmp_phone_categories[int(tmp_csv.loc[i,"start"]*10+0.5)][tmp_phone_cate]+=1
    
    total_word_rate[name] = tmp_word_rate
    total_phone_rate[name] =  tmp_phone_rate
    total_phone_categories[name] = tmp_phone_categories
    
#Convolution with HRF
total_word_rate_hrf = {}
for name in story_name:
    total_word_rate_hrf[name] = np.convolve(total_word_rate[name],signal.flatten())
    
total_phone_rate_hrf = {}
for name in story_name:
    total_phone_rate_hrf[name] = np.convolve(total_phone_rate[name],signal.flatten())
    
total_phone_categories_hrf = {}
for name in story_name:
    tmp_cate = []
    for i in total_phone_categories[name].T:
        tmp_cate.append(np.convolve(i,signal.flatten()))
    total_phone_categories_hrf[name] = tmp_cate


# In[ ]:


#Build the dataset for modeling
story_point_section = {"21styear":[18,2242],"lucy":[4,354],"milky-ori":[18,286],"milky-vod":[18,286],"mouth":[19,467],"notintact":[23,382],"pieman":[14,291],"slum":[21,1196],"tunnel":[4,1015]}
total_x_ = []
total_y = []
for name in story_name:
    tmp_1 = np.array(total_story_hrf[name])
    tmp_2 = story_data[name]
    tmp_word_rate = total_word_rate_hrf[name]
    tmp_phone_rate = total_phone_rate_hrf[name]
    tmp_phone_categories = np.array(total_phone_categories_hrf[name])
    for time in np.array(range(story_point_section[name][0],story_point_section[name][1]))*15:
        tmp_train = tmp_1[:,time+15]
        tmp_train = list(tmp_train)
        tmp_train.extend([tmp_word_rate[time+15]])
        tmp_train.extend([tmp_phone_rate[time+15]])
        tmp_train.extend(tmp_phone_categories[:,time+15])
        total_x_.append(tmp_train) 
        total_y.append(tmp_2[int(time/15),:])
scaler = StandardScaler()
total_x = scaler.fit_transform(total_x_)


# In[ ]:


#Counting syntactic structures that occur more than 300 times
nlp = spacy.load("en_core_web_trf")
total_json = []
for story_json in story_jsons:
    with open (story_json,"r") as load_f:
        tmp_json = json.load(load_f)
    total_json.append(nlp(tmp_json["transcript"]))
    
tmp = []
for doc in total_json:
    for token in doc:
        tmp.append(token.dep_)
over300_dep = []
for key in Counter(tmp):
    if Counter(tmp)[key]>300:
        over300_dep.append([key,Counter(tmp)[key]])


# In[ ]:


# Identify word pairs with syntactic structural relationships
total_dep = {}
for key in over300_dep:
    tmp = []
    for doc in total_json:
        for token in doc:
            if token.dep_== key:
                if token.text[0]!="'" and token.head.text[0]!="'":
                    tmp.append([token.text,token.head.text])
    total_dep[key]=tmp


# In[ ]:


#Statistical word frequency
stats_fre_mat = np.zeros((3671,3671))
for sample in total_x_:
    sits = np.where(sample!=0)[0]
    for i in sits:
        for j in sits:
            stats_fre_mat[i][j]+=0.5
            stats_fre_mat[j][i]+=0.5


# In[ ]:


#Training XGBoost model
X_train,X_valid,y_train,y_valid = train_test_split(total_x,train_y,test_size = 0.1,random_state = 42)
XGB_10071 = {}
for i in range(10071):
    clf = xgb.XGBRegressor(gamma=0.001,max_depth=7,min_child_weight=2,subsample=1,n_estimators=1000,colsample_bytree=0.7,
                          learning_rate=0.1,reg_alpha=0.01,reg_lambda=0.1,random_state=42,tree_method = "gpu_hist",gpu_id=0)
    clf.fit(X_train,y_train[:,i])
    XGB_10071[volume_10071[i]] = clf
    
#Acquisition of interaction matrix by SHAP interpretation model
total_sparse_mat={}
for key in XGB_10071.keys():
    model = XGB_10071[key]
    model.set_params(**{"gpu_id":"0","predictor":"gpu_predictor","n_jobs":"1"})
    with tempfile.TemporaryFile() as dump_file:
        pickle.dump(model,dump_file)
        dump_file.seek(0)
        clf = pickle.load(dump_file) 
    explainer = shap.TreeExplainer(clf)
    shap_interaction_values = abs(explainer.shap_interaction_values(total_x[0].reshape(1,-1)))[0][:3671,:3671]
    for i in range(6379):
        tmp = abs(explainer.shap_interaction_values(total_x[i+1].reshape(1,-1)))[0][:3671,:3671]
        shap_interaction_values = shap_interaction_values+tmp       
    sparse_matrix = scipy.sparse.csc_matrix(shap_interaction_values)
    total_sparse_mat[key] = sparse_matrix


# In[ ]:


#Eliminate two types of invalid syntactic structures
deps_ = list(total_dep.keys())
deps_.remove("punct")
deps_.remove("ROOT")
stats_fre_mat[np.where(stats_fre_mat==0)] = 1e10

#Identify word pairs with repetitive syntactic structural relationships for subsequent screening
story_dep_repeat = {}
for dep in deps_:
    tmp_dep = []
    for word in total_dep[dep]:
        stats = -1
        for key in deps_:
            for word2 in total_dep[key]:
                if len(set(word)&set(word2))==2:
                    stats+=1
        tmp_dep.append(stats)
    story_dep_repeat[dep]=tmp_dep

total_dep_repeat = []
for key in story_dep_repeat.keys():
    total_dep_repeat.extend(np.array(total_dep[key])[np.where(np.array(story_dep_repeat[key])>0)[0]])

tmp_filter = []
for words in total_dep_repeat:
    a = 0
    for tmp in tmp_filter:
        if len(set(words)&set(tmp)) == 2:
            a+=1
            break
    if a==0:
        tmp_filter.append(words)

total_wordpair_filter = []
for words in tmp_filter:
    total_wordpair_filter.append(str(words[0])+str(words[1]))


# In[ ]:


#Statistical response strength of each voxel to 20 syntactic structures
stats_dep_val = {}
for key in total_sparse_mat.keys():
    tmp_mat = total_sparse_mat[key].toarray()
    tmp_mat = tmp_mat/stats_fre_mat
    tmp = []
    for dep in deps_:
        sum_ = 0
        ave_ = 0
        for pair in total_dep[dep]:
            tmp_f1 = np.array(pair[0]+pair[1])
            tmp_f2 = np.array(pair[1]+pair[0])
            if (tmp_f1==total_wordpair_filter).any() or (tmp_f2==total_wordpair_filter).any():
                pass
            else:               
                if pair[0] in uni_word and pair[1] in uni_word:
                    idx0 = int(np.where(uni_word==pair[0])[0])
                    idx1 = int(np.where(uni_word==pair[1])[0])
                    if idx0!=idx1:
                        sum_=sum_+tmp_mat[idx0][idx1]+tmp_mat[idx1][idx0]
                        ave_+=1
                    else:
                        pass
        tmp.append((sum_/ave_)/np.sum(tmp_mat))
    stats_dep_val[key] = tmp   


# In[ ]:





# In[ ]:





# In[ ]:


#Construct the corresponding syntactic network for each voxel, find the nodes and connected edges of the network
stats_net = []
for key in total_sparse_mat.keys():
    tmp_mat = total_sparse_mat[key].toarray()
    tmp_mat = tmp_mat/stats_fre_mat
    tmp_word = []
    tmp = []
    for dep in deps_:
        for pair in total_dep[dep]:
            tmp_f1 = np.array(pair[0]+pair[1])
            tmp_f2 = np.array(pair[1]+pair[0])
            if (tmp_f1==total_wordpair_filter).any() or (tmp_f2==total_wordpair_filter).any():
                pass
            else: 
                if pair[0] in uni_word and pair[1] in uni_word:
                    idx0 = int(np.where(uni_word==pair[0])[0])
                    idx1 = int(np.where(uni_word==pair[1])[0])
                    if idx0!=idx1:
                        if tmp_mat[idx0][idx1]+tmp_mat[idx1][idx0] != 0:
                            tmp_word.append(pair[0])
                            tmp_word.append(pair[1])
                            tmp.append([idx0,idx1,tmp_mat[idx0][idx1]+tmp_mat[idx1][idx0]])
    tmp_word = np.unique(tmp_word)
    stats_net.append([tmp_word,tmp])


# In[ ]:


#Convert the corresponding syntactic network of voxels into a 3-D vector by the graph2vec method
net_8048_3d = np.load(r"/media/sunjc/program/tzq/net_8048_3d.npy",allow_pickle = True)#shape is (8048, 3)


# In[ ]:


#Corresponding to the 0-255 scale of RGB
scaler = MinMaxScaler()
minmax_embedding = scaler.fit_transform(tmp)
minmax_embedding = minmax_embedding*255


# In[ ]:


#Build 64 category centroids
sits = []
for i in range(0,255,80):
    for j in range(0,255,80):
        for k in range(0,255,80):
            sits.append([i,j,k])
sits = np.array(sits)


# In[ ]:


#Calculate the distances of all voxels to different classification centroids
tmp_total = []
for i in sits:
    tmp = []
    for j in minmax_embedding:
        tmp.append(np.linalg.norm(i-j))
    tmp_total.append(tmp)
tmp_total = np.array(tmp_total)


# In[ ]:


#The voxels included in each category are counted separately
class12 = np.zeros([13,8048])
label = 1
for i in range(64):
    if len(np.where(tmp_total[i]<45)[0])>50:
        for j in np.where(tmp_total[i]<45)[0]:
            class12[label][j] = label
        label+=1



tmp = np.load(r"/media/sunjc/program/tzq/volume_8048.npy",allow_pickle = True) #8048 voxels filtered by P value
sits_8048 = []
for i in tmp:
    sits_8048.append(int(np.where(volume_10071==i)[0]))
    
total_y = total_y[:,sits_8048]  #shape is (6380, 8048)


# In[ ]:


#The voxels of the same category were averaged and treated as signals of that category
total_y_class12 = []
for i in range(12):
    tmp = np.where(class12[i+1] == i+1)[0]
    total_y_class12.append(np.sum(total_y[:,tmp],axis = 1)/len(tmp))
total_y_class12 = np.array(total_y_class12).T #shape is (6380, 12)


# In[ ]:


#Train the XGBoost model separately for each category
X_train,X_valid,y_train,y_valid = train_test_split(total_x,total_y_class12,test_size = 0.1,random_state = 42)
tmp_dict = []
for i in range(12):
    clf = xgb.XGBRegressor(gamma=0.001,max_depth=7,min_child_weight=2,subsample=1,n_estimators=1000,colsample_bytree=0.7,
                          learning_rate=0.1,reg_alpha=0.01,reg_lambda=0.1,random_state=42,tree_method = "gpu_hist",gpu_id=0)
    clf.fit(X_train,y_train[:,i])
    tmp_dict.append(clf)
    
#The interaction matrix corresponding to each category is obtained by SHAP interpretation    
class12_sparse_mat=[]
for key in range(12):
    model = tmp_dict[key]
    model.set_params(**{"gpu_id":"1","predictor":"gpu_predictor","n_jobs":"1"})
    with tempfile.TemporaryFile() as dump_file:
        pickle.dump(model,dump_file)
        dump_file.seek(0)
        clf = pickle.load(dump_file) 
    explainer = shap.TreeExplainer(clf)
    shap_interaction_values = abs(explainer.shap_interaction_values(total_x[0].reshape(1,-1)))[0][:3671,:3671]
    for i in range(6379):
        tmp = abs(explainer.shap_interaction_values(total_x[i+1].reshape(1,-1)))[0][:3671,:3671]
        shap_interaction_values = shap_interaction_values+tmp  
    sparse_matrix = scipy.sparse.csc_matrix(shap_interaction_values)
    class12_sparse_mat.append(sparse_matrix)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




