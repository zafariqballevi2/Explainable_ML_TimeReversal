from ast import Load
from captum.attr import IntegratedGradients, GradientShap, NoiseTunnel, Occlusion
from matplotlib.colors import LinearSegmentedColormap
from captum.attr import visualization as viz
from matplotlib.backends.backend_pdf import PdfPages
import torchvision.transforms as T
import sys
sys.path.append("..")
import pickle
from pathlib import Path
import numpy as np
import torch
import mat73
from wholeMILC import NatureOneCNN
from lstm_attn import subjLSTM
from utils import get_argparser
from All_Architecture import combinedModel
from torch.utils.data import DataLoader, TensorDataset
import os
from tqdm import tqdm
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.patches as patches
import warnings
from scipy.stats import wasserstein_distance
from enum import Enum
from typing import Any, Iterable, List, Optional, Tuple, Union
import numpy as np
from matplotlib import cm, colors, pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.pyplot import axis, figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
import csv
from numpy import ndarray
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn import preprocessing
#parser = get_argparser()
#args = parser.parse_args()
#print("JOBID: ", args.jobid)
#Path to store extracted attributes
if torch.cuda.is_available():
        cudaID = str(torch.cuda.current_device())
        device = torch.device("cuda:" + cudaID)
else:
        device = torch.device("cpu")
print("Device: ",device)


jobss = []
file1 = open(os.path.join("/data/users4/ziqbal5/abc/MILC", 'output2.txt'), 'r+')
lines = file1.readlines()
lines = [line.replace(' ', '') for line in lines]

start = '/data/users4/ziqbal5/abc/MILC/Data_BSNIP/'

for line in lines:
    jobss.append(str(line.rstrip('\n')))
#jobss.sort()



#print(len(jobss))
n_col = len(jobss) #4  # (*2)
Ad = []
for i in range(n_col):
    for dirpath, dirnames, filenames in os.walk(start):
        for dirname in dirnames:
            if dirname.startswith(jobss[i]):
                filename = os.path.join(dirpath, dirname)
                Ad.append(filename)

#print(Ad)

def stitch_windows(saliency, components, nw, wsize, tp, ws): 
    stiched_saliency = np.zeros((saliency.shape[0], components, nw * wsize))
    print('shape of saliency maps before stiching: ', saliency.shape)
    
    for i in range(saliency.shape[0]):
        for j in range(saliency.shape[1]):
            stiched_saliency[i, :, j * wsize:j * wsize + wsize] = saliency[i, j, :, :]
            stiched_saliency[i, :, j * wsize:j * wsize + wsize] = saliency[i, j, :, :]

    saliency = stiched_saliency
    #print('Saliency: ', saliency.shape)
    
   
    # if ws == 15:
    #     for i in range(saliency.shape[0]):
    #         for j in range(components):
    #             ii,kk = 0,0
    #             for k in range(52):  #if window size is 20 and tp are 1040 then there should be 52 nonoverlapping windows
    #                 avg_saliency[i,j, ii:ii+20] = saliency[i,j, kk:kk+20]
    #                 ii = ii+20
    #                 kk = kk+15

    if ws == wsize:
        avg_saliency = np.zeros((saliency.shape[0], components, tp))
        avg_saliency = saliency
    else:
        print("Windows are overlapped. Look at stitch_windows function again")
 
    print('Stiched_saliency', avg_saliency.shape)
    return avg_saliency

def parameters(Add):



    ind_ws = Add.index('ws')
    ind_wsize = Add.index('wsize')
    ind_nw = Add.index('nw')
    ind_tp = Add.index('tp')
    ind_seed = Add.index('seed')
    ind_samples = Add.index('samp')

    ws = int(Add[ind_ws+3:ind_wsize-1])
    wsize = int(Add[ind_wsize+6:ind_nw-1])
    nw = int(Add[ind_nw+3:ind_seed-1])
    tp = int(Add[ind_tp+3:ind_samples-1])
    dataset = Add[47 : ind_ws-1]
    return dataset, ws, wsize, nw, tp

def normalize(a):
    return np.asarray(a)/np.sum(a)

def emd(a, b):
    assert len(a) == len(b), "Sequences should be same length!"
    a = normalize(a)
    b = normalize(b)
    #return wasserstein_distance(a,b)
    return np.sum(np.abs(np.cumsum(a - b)))

def spikiness(a):
    #template = np.ones_like(a)
    template = np.array([1.0/len(a)]*len(a))
    return emd(a, template)    
    
# print(Ad[0])



# len_ad = int(len(Ad)/2)
# Ad_ptr = Ad[0:len_ad]
# Ad_nptr = Ad[len_ad:len_ad+len_ad]
# for i in range(len_ad):
#     ptr = Ad_ptr[i]
#     nptr = Ad_nptr[i]



#     print(ptr)
# goodbreak

Acc_ptr, Acc_nptr = [], []
Auc_ptr, Auc_nptr = [], []
indexx = 0

for i in range(n_col):
    
    #print(i)

    with open(os.path.join(Ad[i], 'ACC_AUC.csv'), newline = '') as csvfile:
        a = csv.reader(csvfile, delimiter=',')
        for row in a:
            pass


    import re
    row[1] = re.sub(r'[\[\]]', '', str(row[1]))
    row[2] = re.sub(r'[\[\]]', '', str(row[2]))  



    Acc = float(row[1])*100
    Auc = float(row[2])*100

    Acc_ptr.append(Acc)
    Auc_ptr.append(Auc)
    indexx = indexx + 1
    #print(Ad[i])
    print(Auc, jobss[i])
    if indexx == 10:
        #print("len", len(Acc_ptr))

        #print(Ad[i])
        overall_acc_ptr = sum(Acc_ptr)/len(Acc_ptr)
        overall_auc_ptr = sum(Auc_ptr)/len(Auc_ptr)
        print("Acc and AUC: ",overall_acc_ptr, overall_auc_ptr)
        Acc_ptr = []
        Auc_ptr = []
        indexx = 0


for i in range(n_col):

    with open(os.path.join(Ad[i], 'ACC_AUC.csv'), newline = '') as csvfile:
        a = csv.reader(csvfile, delimiter=',')
        for row in a:
            pass

        

    Acc = round(float(row[0])*100)
    Auc = round(float(row[1])*100)
    if i<=(n_col/2-1):
        Acc_nptr.append(Acc)
        Auc_nptr.append(Auc)
    else:
        Acc_ptr.append(Acc)
        Auc_ptr.append(Auc) 


overall_acc_nptr = round(sum(Acc_nptr)/len(Acc_nptr))
overall_acc_ptr = round(sum(Acc_ptr)/len(Acc_ptr))
overall_auc_nptr = round(sum(Auc_nptr)/len(Auc_nptr))
overall_auc_ptr = round(sum(Auc_ptr)/len(Auc_ptr))



fig, axs = plt.subplots(2)
fig.suptitle('Accuracy and AUC plots for 10 folds')
axs[0].plot(Acc_nptr, label = 'Acc_nptr')
axs[0].plot(Acc_ptr,label = 'Acc_ptr')
axs[0].title.set_text('Overall_Acc_nptr: ' + str(overall_acc_nptr) + ' :: OVerall_Acc_ptr: ' + str(overall_acc_ptr))
axs[0].legend(prop={'size': 8})
axs[1].plot(Auc_nptr,label = 'Auc_nptr')
axs[1].plot(Auc_ptr,label = 'Auc_ptr')
axs[1].title.set_text('Overall_Auc_nptr: ' + str(overall_auc_nptr) + ' :: OVerall_Auc_ptr: ' + str(overall_auc_ptr))
axs[1].legend(prop={'size': 8})
#plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig(Ad[0]+"/AccAUC.png") 
plt.close()   


# print(type(float(row[0])))



sal = []
All_sal_F = []
All_sal_R = []

All_spikiness = []

# for i in range(n_col):
#     with open(os.path.join(Ad[i], 'test_data.pickle'),"rb") as infile:
#         test_data = pickle.load(infile)
#     print(Ad[i])
#     print(test_data[0][0][0][5:10])
    



for i in range(n_col):

    with open(os.path.join(Ad[i], 'all_saliencies_S.pickle'),"rb") as infile:
        temp = pickle.load(infile)
    print(temp.shape)
    with open(os.path.join(Ad[i], 'test_labels.pickle'),"rb") as infile:
        test_labels = pickle.load(infile)


        print("Path: ", os.path.join(Ad[i], 'all_saliencies_S.pickle'))
        dataset, ws, wsize, nw, tp = parameters(Ad[i])
        components =  53
        print("Parameters(ws, wsize, nw, comp, tp, dataset): ", ws, wsize, nw, components, tp, dataset)
        if ws == tp:
            pass
        else:
            temp = stitch_windows(temp, components, nw, wsize, tp, ws)
        

        sal.append(temp)


    #print(len(sal), sal[0].shape, print(test_labels.shape))
    
    print(test_labels)
    HC = torch.where(test_labels == 0)[0]
    Pat = torch.where(test_labels == 1)[0]

    print("HC or F: ", len(HC), " :: Patients or R: ", len(Pat))

   

    totaltest = len(sal[i]) #300

    All_sal_F.append(sal[i][HC])
    All_sal_R.append(sal[i][Pat])

    # All_sal_F.append(sal[i][0:int(totaltest/2)])
    # All_sal_R.append(sal[i][int(totaltest/2):totaltest])
    
#array_f = np.zeros((len(All_sal_F), All_sal_F[0].shape[0],All_sal_F[0].shape[1],All_sal_F[0].shape[2]))


#for i in range(len(All_sal_F)):
    
    
print("In total ", len(All_sal_F), All_sal_F[0].shape, All_sal_R[0].shape)




Li_HC_nptr = []
Li_HC_ptr = []
Li_Pat_nptr = []
Li_Pat_ptr = []
threshold = .95
half= int(len(All_sal_F)/2)

for ii in range(len(All_sal_F)):
    rep_salF = All_sal_F[ii]
    for kk in range(len(rep_salF)):
        tempS = rep_salF[kk]
        tempS = abs(tempS)
        tempS = (tempS - np.min(tempS))/np.ptp(tempS) #Noramlized [0,1]
        #tempD = 2.*(tempD - np.min(tempD))/np.ptp(tempD)-1 #Normalized [-1,1]
        sarray = tempS.reshape(1, tempS.shape[0] * tempS.shape[1])
        sarray = np.sort(sarray)
        index = int((tempS.shape[0]* tempS.shape[1])*(threshold))        
        thresholdValue = sarray[0][index]
        maskedArray = np.where(tempS<thresholdValue, 0, 1)
        matrix = np.multiply(tempS, maskedArray)
        matrix = np.sum(matrix, axis = 0)

        if ii < half:
            Li_HC_nptr.append(matrix)
        else:
            Li_HC_ptr.append(matrix)

for ii in range(len(All_sal_R)):
    rep_salR = All_sal_R[ii]
    for kk in range(len(rep_salR)):
        #print(ii, kk)
        tempS = rep_salR[kk]
        tempS = abs(tempS)
        tempS = (tempS - np.min(tempS))/np.ptp(tempS) #Noramlized [0,1]
        #tempD = 2.*(tempD - np.min(tempD))/np.ptp(tempD)-1 #Normalized [-1,1]
        sarray = tempS.reshape(1, tempS.shape[0] * tempS.shape[1])
        sarray = np.sort(sarray)
        index = int((tempS.shape[0]* tempS.shape[1])*(threshold))
        thresholdValue = sarray[0][index]
        maskedArray = np.where(tempS<thresholdValue, 0, 1)
        matrix = np.multiply(tempS, maskedArray)
        matrix = np.sum(matrix, axis = 0)

        if ii < half:
            Li_Pat_nptr.append(matrix)
        else:
            Li_Pat_ptr.append(matrix)

print("HC [nptr:ptr] :",len(Li_HC_nptr), len(Li_HC_ptr), Li_HC_nptr[0].shape)
print("Pat [nptr:ptr] :",len(Li_Pat_nptr), len(Li_Pat_ptr))
# with open('bsnip_npt.pickle', "wb") as outfile:
#     pickle.dump(Li_Pat_nptr, outfile)
# with open('bsnip_ptr.pickle', "wb") as outfile:
#     pickle.dump(Li_Pat_ptr, outfile)

    

# #ags,bgs = Li_Pat_ptr[3], Li_Pat_ptr[1]
# #ags,bgs = Li_Pat_ptr[18], Li_Pat_ptr[1]
# ags,bgs = Li_Pat_ptr[0], Li_Pat_ptr[1]
# ags_res, bgs_res = spikiness(ags), spikiness(bgs)
# print(ags_res,bgs_res)
# fig, ax = plt.subplots(nrows=2, ncols=1)
# plt.subplots_adjust(hspace=0.3)
# ax[0].plot(ags, color = 'red')
# ax[0].set_title(str(ags_res))
# ax[1].plot(bgs, color = 'red')
# ax[1].set_title(str(bgs_res))
# custom_ylim = (0, 15)
# plt.setp(ax, ylim=custom_ylim)
# plt.savefig('abcdefgs.png')

#####
# HControlsnpt = []
# HControlspt = []
# for i in range(len(Li_HC_nptr)):
#     t1,t2 = spikiness(Li_HC_nptr[i]), spikiness(Li_HC_ptr[i])
#     HControlsnpt.append(t1)
#     HControlspt.append(t2)

# Patontrolsnpt = []
# Patontrolspt = []
# for i in range(len(Li_Pat_nptr)):
#     t1,t2 = spikiness(Li_Pat_nptr[i]), spikiness(Li_Pat_ptr[i])
#     Patontrolsnpt.append(t1)
#     Patontrolspt.append(t2)

# print(Patontrolsnpt)
# print(Patontrolspt)  

# print(HControlsnpt)
# print(HControlspt)

# ddd
####


rows, columns = 50, 4
fig, axs = plt.subplots(rows, columns, figsize=(50, 50))
k1,k2,k3,k4 = 0,0,0,0
np.set_printoptions(threshold=sys.maxsize)
# print(Li_HC_nptr[0][0:10])
# print(Li_HC_nptr[1][0:10])

def plott(data, abc):
    axs[i,j].plot(data, color = abc)
    axs[i,j].set_ylim(0, 10)
    axs[i,j].spines['top'].set_visible(False)
    axs[i,j].spines['right'].set_visible(False)
    axs[i,j].set_xticks([])
    axs[i,j].set_yticks([])
    #axs[i,j].spines['bottom'].set_visible(False)
    axs[i,j].spines['left'].set_visible(False)
    if i == 0 and j == 0:
        axs[i,j].set_title("NPT_HC", fontsize = 40)
    if i == 0 and j == 1:
        axs[i,j].set_title("NPT_Patients", fontsize = 40)
    if i == 0 and j == 2:
        axs[i,j].set_title("PTR_HC", fontsize = 40)
    if i == 0 and j == 3:
        axs[i,j].set_title("PTR_Patients", fontsize = 40)



for i in range(rows):
    for j in range(columns):
        if j == 0:
            plott(Li_HC_nptr[k1], 'blue')
            k1 = k1+1
        if j == 1:
            plott(Li_Pat_nptr[k2], 'red')
            k2 = k2+1
        if j == 2:
            plott(Li_HC_ptr[k3], 'blue')
            k3 = k3+1
        if j == 3:
            plott(Li_Pat_ptr[k4], 'red')
            k4 = k4+1
plt.savefig(Ad[0]+"/Spikiness.png")
plt.close()
ddd

length = []

# print(len(All_sal_F), All_sal_F[0].shape, len(Li_HC_nptr), Li_HC_nptr[0].shape)
# print(len(Li_HC_nptr), len(Li_HC_ptr),len(Li_Pat_nptr),len(Li_Pat_ptr))
# addi = Li_HC_nptr+Li_HC_ptr+Li_Pat_nptr+Li_Pat_ptr
# print(len(addi))
# spk = []
# for kgbd in range(len(addi)):
#     temp = addi[kgbd]
#     temp = temp.flatten()
#     spikes = spikiness(temp)
#     print(kgbd, spikes)
#     spk.append(spikes)

#print(len(spk))
#print(spk)


#len(All_sal_F) is the number of jobs
for ii in range(len(All_sal_F)):
    dataset, ws, wsize, nw, tp = parameters(Ad[ii])
    rep_salF = abs(All_sal_F[ii])
    rep_salR = abs(All_sal_R[ii])
    #print(rep_salF.shape, rep_salR.shape)
    #rep_salF[rep_salF<0] = 0
    #rep_salR[rep_salR<0] = 0
    
    All_spk_HC = []
    All_spk_Patients = []

    print("length: ", len(rep_salF), len(rep_salR))
    length.append(int(len(rep_salF)+len(rep_salR)))
    for k in range(len(rep_salF)):
        HC = rep_salF[k]
        HC = HC.flatten()
        spk_HC = spikiness(HC)
        All_spk_HC.append(spk_HC)

    for k in range(len(rep_salR)):
        Patients = rep_salR[k]
        Patients = Patients.flatten()
        spk_Patients = spikiness(Patients)
        All_spk_Patients.append(spk_Patients)
        
    print(All_spk_HC)
    print(All_spk_Patients)
    All_spikiness.append(All_spk_HC)
    All_spikiness.append(All_spk_Patients)


training_type = []
health_condition = []


for i in range(len(All_spikiness)):
    if i%2 == 0:
        t1= ['HC']*len(All_spikiness[i])
        health_condition.append(t1)
    else:
        t1= ['Patients']*len(All_spikiness[i])
        health_condition.append(t1)

    if i < int(len(All_spikiness)/2):
        t2= ['NPT']*len(All_spikiness[i])
        training_type.append(t2)
    elif i >= int(len(All_spikiness)/2):
        t2= ['PTR']*len(All_spikiness[i])
        training_type.append(t2)


flat_training_type = []
flat_health_condition = []
falt_All_spikiness = []

for xs in health_condition:
    for x in xs:
        flat_health_condition.append(x)

for xs in training_type:
    for x in xs:
        flat_training_type.append(x)  

for xs in All_spikiness:
    for x in xs:
        falt_All_spikiness.append(x)


df = pd.DataFrame({
    'EMD': falt_All_spikiness,
    'Condition' : flat_health_condition,
    'Trtype' : flat_training_type
})

#print(df.to_string())
#print(df.head())


#df_Pat_NPT = df[(df['Condition']=='Patients') & (df['Trtype']=='NPT') ]['EMD']
#print(df_Pat_NPT.to_string(index = False))

# df_Pat_PTR = df[(df['Condition']=='Patients') & (df['Trtype']=='PTR') ]['EMD']
# print(df_Pat_PTR.to_string(index = False))
# kkccc


sns.set_theme(style="whitegrid")
ax = sns.boxplot(x="Condition", y="EMD", hue="Trtype",
            data=df, palette="Set3", width=0.14, showfliers = False).set(title=dataset, xlabel = 'Subjects')
#plt.tight_layout()
print("Combined figure saved here :", Ad[0])
plt.savefig(Ad[0]+"/EMD_overall.png")
plt.close()

dcdc

df_length = int(len(df))
df_half = int(len(df)/2)
#print(df_length)
df_nptr = df[0:df_half]
df_ptr = df[df_half::]
#print(df_F)
#print('abc')
#print(df_R)
#print(len(df_F), len(df_R))

n_subj = int(len(df)/n_col)
#print(n_subj, len(df_nptr))

k = 0
for i in range(int(n_col/2)):
    temp_F = df_nptr[k:k+length[i]]
    temp_R = df_ptr[k:k+length[i]]
    k = k+ length[i]

    temp_concat = pd.concat([temp_F, temp_R], axis = 0)
    #print(temp_concat)
    #print(Ad[i])
    #print(temp_concat.to_string())
    ax = sns.boxplot(x="Condition", y="EMD", hue="Trtype",
            data=temp_concat, palette="Set3", width=0.14, showfliers = False).set(title=dataset, xlabel = 'Training Type')
    #plt.tight_layout()
    print("figure saved here :", Ad[i])
    plt.savefig(Ad[i]+"/EMD.png")
    plt.close()



    

