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


cap_viz = False   #Do you want to use captum's visualization method?
EMD = False
n_pages = 1
fnc = False #If you want to calculate FNC or Saliency maps
mean_C = False #if True, will calculte mean of components.
thr = .90   # .85 means 85 perc of the region is masked while calculating FNC, hence using 15% of the salient data
if fnc == False:
    onepager = 12
    nr = 'SM_M'
else:
    onepager = 12 #Number of rows in one page while visualizing results
    nr = 'SM_C_'+str(thr)


jobss = []
file1 = open(os.path.join("/data/users4/ziqbal5/abc/MILC", 'output2.txt'), 'r+')
lines = file1.readlines()
lines = [line.replace(' ', '') for line in lines]

start = '/data/users4/ziqbal5/abc/MILC/Data_old/'

for line in lines:
    jobss.append(str(line.rstrip('\n')))
jobss.sort()

#print(len(jobss))
n_col = len(jobss) #4  # (*2)
Ad = []
for i in range(n_col):
    for dirpath, dirnames, filenames in os.walk(start):
        for dirname in dirnames:
            if dirname.startswith(jobss[i]):
                filename = os.path.join(dirpath, dirname)
                Ad.append(filename)

class ImageVisualizationMethod(Enum):
    heat_map = 1
    blended_heat_map = 2
    original_image = 3
    masked_image = 4
    alpha_scaling = 5



class VisualizeSign(Enum):
    positive = 1
    absolute_value = 2
    negative = 3
    all = 4


def _prepare_image(attr_visual: ndarray):
    return np.clip(attr_visual.astype(int), 0, 255)


def _normalize_scale(attr: ndarray, scale_factor: float):
    #print("ab: ", attr.shape)
    #print(scale_factor)
    assert scale_factor != 0, "Cannot normalize by scale factor = 0"
    if abs(scale_factor) < 1e-5:
        warnings.warn(
            "Attempting to normalize by value approximately 0, visualized results"
            "may be misleading. This likely means that attribution values are all"
            "close to 0."
        )
    attr_norm = attr / scale_factor
    return np.clip(attr_norm, -1, 1)
    


def _cumulative_sum_threshold(values: ndarray, percentile: Union[int, float]):
    # given values should be non-negative
    assert percentile >= 0 and percentile <= 100, (
        "Percentile for thresholding must be " "between 0 and 100 inclusive."
    )
    sorted_vals = np.sort(values.flatten())
    cum_sums = np.cumsum(sorted_vals)
    threshold_id = np.where(cum_sums >= cum_sums[-1] * 0.01 * percentile)[0][0]
    return sorted_vals[threshold_id]


def _normalize_attr(
    attr: ndarray,
    sign: str,
    outlier_perc: Union[int, float] = 2,
    reduction_axis: Optional[int] = None,
):
    attr_combined = attr
    #print('a', attr_combined.shape)
    #print("rax", reduction_axis)
    if reduction_axis is not None:
        attr_combined = np.sum(attr, axis=reduction_axis)
    #print('a', attr_combined.shape)
    # Choose appropriate signed values and rescale, removing given outlier percentage.
    if VisualizeSign[sign] == VisualizeSign.all:
        threshold = _cumulative_sum_threshold(np.abs(attr_combined), 100 - outlier_perc)
    elif VisualizeSign[sign] == VisualizeSign.positive:
        attr_combined = (attr_combined > 0) * attr_combined
        threshold = _cumulative_sum_threshold(attr_combined, 100 - outlier_perc)
    elif VisualizeSign[sign] == VisualizeSign.negative:
        attr_combined = (attr_combined < 0) * attr_combined
        threshold = -1 * _cumulative_sum_threshold(
            np.abs(attr_combined), 100 - outlier_perc
        )
    elif VisualizeSign[sign] == VisualizeSign.absolute_value:
        attr_combined = np.abs(attr_combined)
        threshold = _cumulative_sum_threshold(attr_combined, 100 - outlier_perc)
    else:
        raise AssertionError("Visualize Sign type is not valid.")
    return _normalize_scale(attr_combined, threshold)


def visualize_image_attr(
    attr: ndarray,
    original_image: Union[None, ndarray] = None,
    method: str = "heat_map",
    sign: str = "absolute_value",
    plt_fig_axis: Union[None, Tuple[figure, axis]] = None,
    outlier_perc: Union[int, float] = 2,
    cmap: Union[None, str] = None,
    alpha_overlay: float = 0.5,
    show_colorbar: bool = False,
    title: Union[None, str] = None,
    fig_size: Tuple[int, int] = (6, 6),
    use_pyplot: bool = True,
):

    heat_map = None

    norm_attr = _normalize_attr(attr, sign, outlier_perc, reduction_axis=2)



    return norm_attr, norm_attr, norm_attr

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
    # elif ws == 10 or ws == 100:
    #     print(saliency[:, :, 0:ws].shape)
    #     avg_saliency[:, :, 0:ws] = saliency[:, :, 0:ws]


    #     for j in range(nw-1):
    #         a = saliency[:, :, wsize*j+wsize:wsize*j+wsize]
    #         b = saliency[:, :, wsize*(j+1):wsize*(j+1)+ws]
    #         avg_saliency[:, :, ws*j+ws:ws*j+wsize] = (a + b)/2

    #     avg_saliency[:, :, tp-ws:tp] = saliency[:, :, nw*wsize-ws:nw*wsize]
    #     #print("gggg", avg_saliency.shape)

    # elif ws == 20 and wsize == 40:
    #     #print(saliency[:, :, 0:20].shape)
    #     avg_saliency[:, :, 0:20] = saliency[:, :, 0:20]


    #     for j in range(nw-1):
    #         a = saliency[:, :, 40*j+20:40*j+40]
    #         b = saliency[:, :, 40*(j+1):40*(j+1)+20]
    #         avg_saliency[:, :, 20*j+20:20*j+40] = (a + b)/2

    #     avg_saliency[:, :, tp-20:tp] = saliency[:, :, nw*wsize-20:nw*wsize]
    #     #print("gggg", avg_saliency.shape)

    # else:
    #     for i in range(saliency.shape[0]):
    #         for j in range(tp):
    #             L = []
    #             if j < 20:
    #                 index = j
    #             else:
    #                 index = 19 + (j - 19) * 20

    #             L.append(index)

    #             s = saliency[i, :, index]
    #             count = 1
    #             block = 1
    #             iteration = min(19, j)
    #             for k in range(0, iteration, 1):
    #                 if index + block * 20 - (k + 1) < nw * wsize:
    #                     s = s + saliency[i, :, index + block * 20 - (k + 1)]
    #                     L.append(index + block * 20 - (k + 1))
    #                     count = count + 1
    #                     block = block + 1
    #                 else:
    #                     break
    #             avg_saliency[i, :, j] = s/count
    #             # print('Count =', count, ' and Indices =', L)
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

def visualization(hdf,page):
    
    
    sal = []
    LCP = []
    All_Loss_F = []
    All_Loss_R = []
    All_conf_F = []
    All_conf_R = []
    All_pred_F = []
    All_pred_R = []
    All_sal_F = []
    All_sal_R = []
    pred_L = []
    test_L = []
    data = []
    All_data_F = []
    All_data_R = []
    All_test_LF = []
    All_test_LR = []

    
    
    #correctly order indices corresponding to brain networks
    network = [
        [1, 2, 3, 4, 5],
        [6, 7],
        [8, 9, 10, 11, 12, 13, 14, 15, 16],
        [17, 18, 19, 20, 21, 22, 23, 24, 25],
        [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42], 
        [43, 44, 45, 46, 47, 48, 49],
        [50, 51, 52, 53]]
    

    for i in range(n_col):
        with open(os.path.join(Ad[i], 'all_saliencies_S.pickle'),"rb") as infile:
            temp = pickle.load(infile)


            print("Path: ", os.path.join(Ad[i], 'all_saliencies_S.pickle'))
            dataset, ws, wsize, nw, tp = parameters(Ad[i])
            components =  53
            print("Parameters(ws, wsize, nw, comp, tp, dataset): ", ws, wsize, nw, components, tp, dataset)
            if ws == tp:
                pass
            else:
                temp = stitch_windows(temp, components, nw, wsize, tp, ws)
         


            sal.append(temp)
            

             
        with open(os.path.join(Ad[i], 'LCP.pickle'), "rb") as infile:
            temp = pickle.load(infile)
            LCP.append(temp)
            pred_L.append(temp[2])
        #print(LCP[0][0])
        
       

        with open(os.path.join(Ad[i], 'test_labels.pickle'), "rb") as infile:
            test_labels = pickle.load(infile)
            
        test_L.append(test_labels)
        
        with open(os.path.join(Ad[i], 'test_data.pickle'), "rb") as infile:
                data.append(pickle.load(infile))

        #test_labels = torch.from_numpy(test_labels)
        HC = np.where(test_labels == 0)[0]
        Pat = np.where(test_labels == 1)[0]
        print(len(HC), len(Pat))
        
 
       
        All_sal_F.append(sal[i][HC])
        All_sal_R.append(sal[i][Pat])
        All_data_F.append(data[i][HC])
        All_data_R.append(data[i][Pat])
        
        print(All_data_F[0].shape, All_data_R[0].shape)
        print("Forward vs Reverse time points: Should be reversely same")
        print(All_data_F[i][0][0][10])
        print(All_data_R[i][0][0][10])



        print(len(All_sal_F), All_data_F[0].shape, len(All_data_F))
        #dataf = np.squeeze(All_data_F[0])
        #dataf = dataf[10]
        #datar = np.squeeze(All_data_R[0])
        #datar = datar[10]

        # salf = All_sal_F[0]
        # salf[salf<0] = 0
        # salr = All_sal_R[0]
        # salr[salr<0] = 0

        # plt.figure(figsize = (50,10))
        # plt.xticks([])
        # plt.yticks([])
        # plt.imshow(salf, cmap = 'Reds', interpolation='nearest', aspect = 'auto')
        # plt.savefig('aaa.png')
        # plt.imshow(salr, cmap = 'Reds', interpolation='nearest', aspect = 'auto')
        # plt.savefig('bbb.png')
        # plt.close()

        
        All_Loss_F.append(LCP[i][0][HC])   #This 0 is to choose loss, 1 is for confidence, 2 is for pred values.
        All_Loss_R.append(LCP[i][0][Pat])

        # All_conf_F.append(LCP[i][1][HC])
        # All_conf_R.append(LCP[i][1][Pat])
                
        All_pred_F.append(LCP[i][2][HC])
        All_pred_R.append(LCP[i][2][Pat])

        All_test_LF.append(test_L[i][HC])
        All_test_LR.append(test_L[i][Pat])
         

    #When to select all the SMs
    for ii in range(len(All_sal_F)):
        #print('Ad', Ad[ii])
        dataset, ws, wsize, nw, tp = parameters(Ad[ii])
        rep_salF = All_sal_F[ii]
        rep_salR = All_sal_R[ii]
            
        #rep_salF = abs(rep_salF)
        #rep_salR = abs(rep_salR)
        rep_salF[rep_salF<0] = 0
        rep_salR[rep_salR<0] = 0
       

        Loss_F = All_Loss_F[ii]
        Loss_R = All_Loss_R[ii]
        # Conf_F = All_conf_F[ii]
        # Conf_R = All_conf_R[ii]
        Pred_F = All_pred_F[ii]
        Pred_R = All_pred_R[ii]
        label_F = All_test_LF[ii]
        label_R = All_test_LR[ii]


        if dataset == 'FBIRN' or dataset == 'COBRE' or dataset == 'BISNP':
            print(len(rep_salF), len(rep_salR))
            min_val = min(len(rep_salF), len(rep_salR))
            
            ind9 = 0
            ind90 = 0
            f1,r1 = 0,0
            fig = plt.figure(figsize=(50, 50))
            fig.suptitle(str(dataset) + ' Saliency Maps', fontsize = 50)
        
            import math
            rows = 3
            columns = math.ceil((min_val*2)/rows)
            #print('col', columns)
            
           
            outer = gridspec.GridSpec(columns+3, rows, wspace=0.3, hspace=0.2)
            
            print('mininum value: ', min_val)
            for i in range(columns*rows+3):
                    ax = plt.Subplot(fig, outer[i])
                    if i < min_val:
                        
                        if ind90 >= min_val:
                            continue
                        if label_F[ind9] == Pred_F[ind9]:
                            ax.spines[['top', 'right', 'left', 'bottom']].set_edgecolor('green')
                            ax.spines[['top', 'right','left', 'bottom']].set_linewidth(3)
                        else:
                            ax.spines[['top', 'right', 'left', 'bottom']].set_edgecolor('red')
                            ax.spines[['top', 'right','left', 'bottom']].set_linewidth(3)
                        ind9 = ind9+1
        

                        ax.set_ylabel('HC   ', fontsize=30, rotation = 0)
                        
                        ax.imshow(rep_salF[f1], cmap = 'Reds', interpolation='nearest', aspect = 'auto')
                
                        f1= f1+1

                    elif i == min_val or i == min_val+1 or i == min_val+2:
                        pass
        
                    elif i >= min_val+1:
                        #print('outside')
                        if ind90 >= min_val:
                            continue
                        if label_R[ind90] == Pred_R[ind90]:
                            ax.spines[['top', 'right', 'left', 'bottom']].set_edgecolor('green')
                            ax.spines[['top', 'right','left', 'bottom']].set_linewidth(3)
                        else:
                            ax.spines[['top', 'right', 'left', 'bottom']].set_edgecolor('red')
                            ax.spines[['top', 'right','left', 'bottom']].set_linewidth(3)
                        ind90 = ind90+1
                        ax.set_ylabel('SZ   ', fontsize=30, rotation = 0)
                    
                        ax.imshow(rep_salR[r1], cmap = 'Reds', interpolation='nearest', aspect = 'auto')
                        r1= r1+1


            
                    ax.set_xticks([])
                    ax.set_yticks([])
                    fig.add_subplot(ax)
            print("Figure saved: ", Ad[ii])
            plt.savefig(Ad[ii]+"/Rep_mapsdownstream.png")
            plt.close()
    
        else:
            #Start: Code for submodular pick
            # Data = rep_salF
            # arr_W = np.zeros((len(Data), tp*53))
            # for i in range(len(Data)):
            #     temp = Data[i]
            #     temp = temp.flatten()
            #     arr_W[i] = temp
            # arr_W[np.isnan(arr_W)] = 0

            # V = []
            # W = arr_W
            # importance = np.sum(abs(W), axis=0)**.5  
            # remaining_indices = set(range(len(W)))
            
            # for _ in range(20):
                
            #     best = 0
            #     best_ind = None
            #     current = 0
            #     for i in remaining_indices:
            #         current = np.dot(
            #                 (np.sum(abs(W)[V + [i]], axis=0) > 0), importance
            #                 )   
            #         if current >= best:
            #             best = current
            #             best_ind = i
            #     V.append(best_ind)
            #     remaining_indices -= {best_ind}
            # print("V1", V)


            # Data = rep_salR
            # arr_W = np.zeros((len(Data), tp*53)) #25970
            # for i in range(len(Data)):
            #     temp = Data[i]
            #     temp = temp.flatten()
            #     arr_W[i] = temp
            # arr_W[np.isnan(arr_W)] = 0
            
            # V2 = []
            # W = arr_W
            # importance = np.sum(abs(W), axis=0)**.5
            # remaining_indices = set(range(len(W)))
            
            # for _ in range(20):
                
            #     best = 0
            #     best_ind = None
            #     current = 0
            #     for i in remaining_indices:
            #         current = np.dot(
            #                 (np.sum(abs(W)[V2 + [i]], axis=0) > 0), importance
            #                 )   
            #         if current >= best:
            #             best = current
            #             best_ind = i
            #     V2.append(best_ind)
            #     remaining_indices -= {best_ind}
            # print("V2", V2)

    
            
            
            # xy, _, _ = np.intersect1d(V, V2, return_indices=True)
           
            # abc = len(xy)
            # d = abc - 12
 
            # print('common b/w F and R', xy)
            #End: Code for submodular pick
        
    
            cor_co = []
            for i in range(len(rep_salF)):
                #print(len(rep_salF), len(rep_salR))
                F = rep_salF[i]
                R = rep_salR[i]
                R = np.flip(R,1)
                F = np.reshape(F, components*tp)
                R = np.reshape(R, components*tp)
                #print(F, R)
                #cosine = np.dot(F,R)/(norm(F)*norm(R))
                #cos_sim.append(cosine)
                with np.errstate(divide="ignore", invalid="ignore"): 
                    corcof = np.corrcoef(F, R)
  
                corcof[np.isnan(corcof)] = 0
                corcof = corcof[0,1]
                #print('i: ',i, corcof )
                cor_co.append(corcof)


            cor_co = np.array(cor_co)

            core = np.sort(cor_co)
            #print(core)
            print("mean_correlation",np.mean(core))

            #print("core", core)

            core_Ind = np.argsort(cor_co)
    
            #Flipping the reversed saliency maps for better comparison
            rep_salR = np.flip(rep_salR, 2)

            
            range1 = core_Ind[0:12]
            range2 = core_Ind[27:39]
            range3 = core_Ind[71:83]
    
            rep_salF1 = rep_salF[range1]  #0:14 #core_Ind[xy[d::]]
            rep_salF2 = rep_salF[range2]
            rep_salF3 = rep_salF[range3]

            ####

            rep_salR1 = rep_salR[range1]
            rep_salR2 = rep_salR[range2]
            rep_salR3 = rep_salR[range3]

            core_F1 = cor_co[range1]
            core_F2 = cor_co[range2]
            core_F3 = cor_co[range3]

            kgf = 0
            kgr = 0
            ind9 = 0
            dd = 0
            ind90 = 0
            f1,f2,f3,f4,r1,r2,r3,r4 = 0,0,0,0,0,0,0,0
            fig = plt.figure(figsize=(50, 50))
            fig.suptitle('Synthetic Data Saliency Maps', fontsize = 70)
            columns = 5 #keeping three rows empty
            rows = 1
            outer = gridspec.GridSpec(columns, rows, wspace=0.2, hspace=0.2)
            #fig.text(0.04, 0.5, 'SubModular Pick                   High Correlation                   Medium Correlation                   Low Correlation', va='center', rotation='vertical', fontsize = 50)

            #print(len(rep_salF1), len(rep_salF2),len(rep_salF3))
            #n = 12
            for i in range(columns):
                #if i == n or i == n+1 or i == n+2 or i == n+15 or i == n+16 or i == n+17 or i == n+30 or i == n+31 or i == n+32 :# or i == 36:
                #if i == 69:
                    #pass
                    
                #else:    

                    inner = gridspec.GridSpecFromSubplotSpec(2, 1,
                            subplot_spec=outer[i], wspace=0.1, hspace=0.1)
                    for j in range(2):
                        #print(i,j)
                        ax = plt.Subplot(fig, inner[j])
                        if j == 0:
            
                            # if label_FF[ind9] == Pred_FF[ind9]:
                            #     ax.spines[['top', 'right', 'left', 'bottom']].set_edgecolor('green')
                            #     ax.spines[['top', 'right','left', 'bottom']].set_linewidth(3)
                            # else:
                            #     ax.spines[['top', 'right', 'left', 'bottom']].set_edgecolor('red')
                            #     ax.spines[['top', 'right','left', 'bottom']].set_linewidth(3)
                            # ind9 = ind9+1

                            ax.set_ylabel('F   ', fontsize=70, rotation = 0)
                            if i < 12:
                                ax.imshow(rep_salF3[f3], cmap = 'Reds', interpolation='nearest', aspect = 'auto')
                                if dd == 0:
                                    ax.set_title('Crrelation Coefficient: ' + str(round(core_F3[f3],3)), fontsize = 50)
                                    dd = dd + 1
                                else:
                                    ax.set_title(str(round(core_F3[f3],3)), fontsize = 50)
                                f3= f3+1
                        #     if i >= 15 and i < 27:
                        #         ax.imshow(rep_salF2[f2], cmap = 'Reds', interpolation='nearest', aspect = 'auto')
                        #         ax.set_title(core_F2[f3])
                        #         f2= f2+1
                        #     if i >= 30 and i < 42:
                        #         ax.imshow(rep_salF3[f3], cmap = 'Reds', interpolation='nearest', aspect = 'auto')
                        #         ax.set_title(core_F3[f3])
                        #         f3= f3+1
                        #     if i >= 45 and i < 57:
                        #         ax.imshow(rep_salFSModular[f4], cmap = 'Reds', interpolation='nearest', aspect = 'auto')
                        #         ax.set_title(cor_SModular[f4])
                        #         f4= f4+1
                        elif j == 1:
                            # if label_RR[ind90] == Pred_RR[ind90]:
                            #     ax.spines[['top', 'right', 'left', 'bottom']].set_edgecolor('green')
                            #     ax.spines[['top', 'right','left', 'bottom']].set_linewidth(3)
                            # else:
                            #     ax.spines[['top', 'right', 'left', 'bottom']].set_edgecolor('red')
                            #     ax.spines[['top', 'right','left', 'bottom']].set_linewidth(3)
                            # ind90 = ind90+1
                            ax.set_ylabel('R   ', fontsize=70, rotation = 0)
                            if i < 12:
                                ax.imshow(rep_salR3[r3], cmap = 'Reds', interpolation='nearest', aspect = 'auto')
                                r3 = r3+1
                            # if i >= 15 and i < 27:
                            #     ax.imshow(rep_salR2[r2], cmap = 'Reds', interpolation='nearest', aspect = 'auto')
                            #     r2= r2+1
                            # if i >= 30 and i < 42:
                            #     ax.imshow(rep_salR3[r3], cmap = 'Reds', interpolation='nearest', aspect = 'auto')
                            #     r3= r3+1
                            # if i >= 45 and i < 57:
                            #     ax.imshow(rep_salRSModular[r4], cmap = 'Reds', interpolation='nearest', aspect = 'auto')
                            #     r4= r4+1


                        kgf = kgf+1
                            
                        #t = ax.text(0.5,0.5, 'outer=%d, inner=%d' % (i, j))
                        #t.set_ha('center')
                        ax.set_xticks([])
                        ax.set_yticks([])
                        fig.add_subplot(ax)

            print('saved here: ', Ad[ii]+"/Rep_mapstest.png")
            plt.savefig(Ad[ii]+"/Rep_mapstest.svg")
            # plt.close()

            ####
            continue

            
            t1 = label_F[range1]
            t2 = label_F[range2]
            t3 = label_F[range3]
            t4 = label_F[V[0:12]]
            label_FF = np.concatenate((t1, t2, t3, t4))

            t1 = Pred_F[range1]
            t2 = Pred_F[range2]
            t3 = Pred_F[range3]
            t4 = Pred_F[V[0:12]]
            Pred_FF = np.concatenate((t1, t2, t3, t4))

            t1 = label_R[range1]
            t2 = label_R[range2]
            t3 = label_R[range3]
            t4 = label_R[V[0:12]]
            label_RR = np.concatenate((t1, t2, t3, t4))

            t1 = Pred_R[range1]
            t2 = Pred_R[range2]
            t3 = Pred_R[range3]
            t4 = Pred_R[V[0:12]]
            Pred_RR = np.concatenate((t1, t2, t3, t4))
            #print(len(Pred_FF))
        


            rep_salFSModular = rep_salF[V[0:12]]
            rep_salR1 = rep_salR[range1]
            rep_salR2 = rep_salR[range2]
            rep_salR3 = rep_salR[range3]
            rep_salRSModular = rep_salR[V[0:12]]

            core_F1 = cor_co[range1]
            core_F2 = cor_co[range2]
            core_F3 = cor_co[range3]
            cor_SModular = cor_co[V[0:12]]

            sum_FR_L = np.add(Loss_F, Loss_R)
            
            sum_FR_LN = preprocessing.normalize([sum_FR_L])
            sum_FR_LN = np.squeeze(sum_FR_LN)
 
            cor_coN =   preprocessing.normalize([cor_co])
            cor_coN = np.squeeze(cor_coN)
        
            
        
        

            ##########################################################
            #print(Loss_F[core_Ind])
        
            # fig, ax = plt.subplots(nrows=2, ncols=1)
            # ax[0].plot(Loss_F[core_Ind], label = 'Loss_F')
            # ax[0].plot(Loss_R[core_Ind], label = 'Loss_R')
            
            # ax[0].legend()

        
            # ax[1].plot(sum_FR_LN[core_Ind], label = 'FR_comb')
            # ax[1].plot(cor_coN[core_Ind], label = 'Corr')
            # ax[1].legend()
            # ax[1].set_xlabel('subjects')

            forplot = cor_co[core_Ind]

            plt.plot(cor_co[core_Ind], label = 'Corr')
            #plt.legend()
            #plt.set_xlabel('subjects')
            plt.xlabel('Subjects')
            plt.ylabel('correlation')
    
            plt.savefig(Ad[ii]+"/Plots.png")
            plt.close()
            ##########################################################


            # # Save the data with 'H_' prefix in the file names
            # with open(os.path.join('/data/users4/ziqbal5/abc/MILC/a/', 'H_rep_salF1.pickle'), "wb") as outfile:
            #     pickle.dump(rep_salF1, outfile)
            # with open(os.path.join('/data/users4/ziqbal5/abc/MILC/a/', 'H_rep_salF2.pickle'), "wb") as outfile:
            #     pickle.dump(rep_salF2, outfile)
            # with open(os.path.join('/data/users4/ziqbal5/abc/MILC/a/', 'H_rep_salF3.pickle'), "wb") as outfile:
            #     pickle.dump(rep_salF3, outfile)
            # with open(os.path.join('/data/users4/ziqbal5/abc/MILC/a/', 'H_rep_salR1.pickle'), "wb") as outfile:
            #     pickle.dump(rep_salR1, outfile)
            # with open(os.path.join('/data/users4/ziqbal5/abc/MILC/a/', 'H_rep_salR2.pickle'), "wb") as outfile:
            #     pickle.dump(rep_salR2, outfile)
            # with open(os.path.join('/data/users4/ziqbal5/abc/MILC/a/', 'H_rep_salR3.pickle'), "wb") as outfile:
            #     pickle.dump(rep_salR3, outfile)
            # with open(os.path.join('/data/users4/ziqbal5/abc/MILC/a/', 'H_core_F1.pickle'), "wb") as outfile:
            #     pickle.dump(core_F1, outfile)
            # with open(os.path.join('/data/users4/ziqbal5/abc/MILC/a/', 'H_core_F2.pickle'), "wb") as outfile:
            #     pickle.dump(core_F2, outfile)
            # with open(os.path.join('/data/users4/ziqbal5/abc/MILC/a/', 'H_core_F3.pickle'), "wb") as outfile:
            #     pickle.dump(core_F3, outfile)
            # with open(os.path.join('/data/users4/ziqbal5/abc/MILC/a/', 'H_forplot.pickle'), "wb") as outfile:
            #     pickle.dump(forplot, outfile)
            # with open(os.path.join('/data/users4/ziqbal5/abc/MILC/a/', 'H_cor_SModular.pickle'), "wb") as outfile:
            #     pickle.dump(cor_SModular, outfile)
            # with open(os.path.join('/data/users4/ziqbal5/abc/MILC/a/', 'H_label_FF.pickle'), "wb") as outfile:
            #     pickle.dump(label_FF, outfile)
            # with open(os.path.join('/data/users4/ziqbal5/abc/MILC/a/', 'H_label_RR.pickle'), "wb") as outfile:
            #     pickle.dump(label_RR, outfile)

            # with open(os.path.join('/data/users4/ziqbal5/abc/MILC/a/', 'H_Pred_FF.pickle'), "wb") as outfile:
            #     pickle.dump(Pred_FF, outfile)
            # with open(os.path.join('/data/users4/ziqbal5/abc/MILC/a/', 'H_Pred_RR.pickle'), "wb") as outfile:
            #     pickle.dump(Pred_RR, outfile)

            # with open(os.path.join('/data/users4/ziqbal5/abc/MILC/a/', 'H_rep_salFSModular.pickle'), "wb") as outfile:
            #     pickle.dump(rep_salFSModular, outfile)
            # with open(os.path.join('/data/users4/ziqbal5/abc/MILC/a/', 'H_rep_salRSModular.pickle'), "wb") as outfile:
            #     pickle.dump(rep_salRSModular, outfile)

            # with open(os.path.join('/data/users4/ziqbal5/abc/MILC/a/', 'H_rep_salFSModular.pickle'), "wb") as outfile:
            #     pickle.dump(rep_salFSModular, outfile)
            # with open(os.path.join('/data/users4/ziqbal5/abc/MILC/a/', 'H_rep_salRSModular.pickle'), "wb") as outfile:
            #     pickle.dump(rep_salRSModular, outfile) 

            # with open(os.path.join('/data/users4/ziqbal5/abc/MILC/a/', 'H_cor_SModular.pickle'), "wb") as outfile:
            #     pickle.dump(cor_SModular, outfile)
           
            stopit
           ##########################################################
            kgf = 0
            kgr = 0
            ind9 = 0
            ind90 = 0
            f1,f2,f3,f4,r1,r2,r3,r4 = 0,0,0,0,0,0,0,0
            fig = plt.figure(figsize=(50, 50))
            fig.suptitle('HCP Saliency Maps', fontsize = 50)
            columns = 16+3 #keeping three rows empty
            rows = 3
            outer = gridspec.GridSpec(columns, rows, wspace=0.2, hspace=0.2)
            fig.text(0.04, 0.5, 'SubModular Pick                   High Correlation                   Medium Correlation                   Low Correlation', va='center', rotation='vertical', fontsize = 50)

            #print(len(rep_salF1), len(rep_salF2),len(rep_salF3))
            n = 12
            for i in range(columns*rows):
                if i == n or i == n+1 or i == n+2 or i == n+15 or i == n+16 or i == n+17 or i == n+30 or i == n+31 or i == n+32 :# or i == 36:
                #if i == 69:
                    pass
                    
                else:    

                    inner = gridspec.GridSpecFromSubplotSpec(2, 1,
                            subplot_spec=outer[i], wspace=0.1, hspace=0.1)
                    for j in range(2):
                        #print(i,j)
                        ax = plt.Subplot(fig, inner[j])
                        if j == 0:
            
                            if label_FF[ind9] == Pred_FF[ind9]:
                                ax.spines[['top', 'right', 'left', 'bottom']].set_edgecolor('green')
                                ax.spines[['top', 'right','left', 'bottom']].set_linewidth(3)
                            else:
                                ax.spines[['top', 'right', 'left', 'bottom']].set_edgecolor('red')
                                ax.spines[['top', 'right','left', 'bottom']].set_linewidth(3)
                            ind9 = ind9+1

                            ax.set_ylabel('F   ', fontsize=30, rotation = 0)
                            if i < 12:
                                ax.imshow(rep_salF1[f1], cmap = 'Reds', interpolation='nearest', aspect = 'auto')
                                ax.set_title(core_F1[f1])
                                f1= f1+1
                            if i >= 15 and i < 27:
                                ax.imshow(rep_salF2[f2], cmap = 'Reds', interpolation='nearest', aspect = 'auto')
                                ax.set_title(core_F2[f3])
                                f2= f2+1
                            if i >= 30 and i < 42:
                                ax.imshow(rep_salF3[f3], cmap = 'Reds', interpolation='nearest', aspect = 'auto')
                                ax.set_title(core_F3[f3])
                                f3= f3+1
                            if i >= 45 and i < 57:
                                ax.imshow(rep_salFSModular[f4], cmap = 'Reds', interpolation='nearest', aspect = 'auto')
                                ax.set_title(cor_SModular[f4])
                                f4= f4+1
                        elif j == 1:
                            if label_RR[ind90] == Pred_RR[ind90]:
                                ax.spines[['top', 'right', 'left', 'bottom']].set_edgecolor('green')
                                ax.spines[['top', 'right','left', 'bottom']].set_linewidth(3)
                            else:
                                ax.spines[['top', 'right', 'left', 'bottom']].set_edgecolor('red')
                                ax.spines[['top', 'right','left', 'bottom']].set_linewidth(3)
                            ind90 = ind90+1
                            ax.set_ylabel('R   ', fontsize=30, rotation = 0)
                            if i < 12:
                                ax.imshow(rep_salR1[r1], cmap = 'Reds', interpolation='nearest', aspect = 'auto')
                                r1= r1+1
                            if i >= 15 and i < 27:
                                ax.imshow(rep_salR2[r2], cmap = 'Reds', interpolation='nearest', aspect = 'auto')
                                r2= r2+1
                            if i >= 30 and i < 42:
                                ax.imshow(rep_salR3[r3], cmap = 'Reds', interpolation='nearest', aspect = 'auto')
                                r3= r3+1
                            if i >= 45 and i < 57:
                                ax.imshow(rep_salRSModular[r4], cmap = 'Reds', interpolation='nearest', aspect = 'auto')
                                r4= r4+1


                        kgf = kgf+1
                            
                        #t = ax.text(0.5,0.5, 'outer=%d, inner=%d' % (i, j))
                        #t.set_ha('center')
                        ax.set_xticks([])
                        ax.set_yticks([])
                        fig.add_subplot(ax)

            plt.savefig(Ad[ii]+"/Rep_mapstest.png")
            plt.close()


hdf = 0
for i in range(n_pages): # Number of pages to generate.
    page = i
    print("Page: ", page)
    visualization(hdf, page)
    ns = int((onepager-2)/2)
    
    hdf = hdf+ns  #5 data saliency pairs ( 10 rows)


   



