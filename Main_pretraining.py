import numpy as np
import torch
import scipy.io as sio
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable
from pathlib import Path
import pickle
import mat73
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold
from captum.attr import IntegratedGradients, GradientShap, NoiseTunnel, Occlusion
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from datetime import datetime
from sklearn.metrics import accuracy_score
import datetime
import pandas as pd
import sys
import os
import time
import gc
from pathlib import Path
import random
import math
from tqdm import tqdm
from wholeMILC import NatureOneCNN, Flatten
from lstm_attn import subjLSTM
from All_Architecture import combinedModel
from matplotlib import cm, colors, pyplot as plt
from scipy.signal import chirp
from utils import get_argparser
from Get_Data import Datasets, EarlyStopping, LSTM, LSTM2



early_stop = False
parser = get_argparser()
args = parser.parse_args()
print("JOBID: ", args.jobid)



device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
seed_values = [0,1,2,3,4,5,6,7,8,9]
fold_values = [0, 1,2,3,4,5,6,7,8,9]
seed = seed_values[args.seeds]
fold = fold_values[args.fold_v]
print("[Seed Fold]: ", seed, fold)


#to keep the dataset generation same
# random.seed(22)
# np.random.seed(22)
#################################
#this set the model initialization values
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
#################################

Trials = 1
eppochs = args.epp
Gain = [args.gainn]
dd = 00 #this is evein he.
Data = args.daata
enc = [args.encoder]     # 'cnn', 'lstmM'
attr_alg = args.attr_alg
if args.encoder == 'rnm':
    onewindow = True
else:
    onewindow = False
#aa = args.windowspecs

subjects, tp, components = args.samples, args.tp, 53  #
sample_y, window_shift = args.wsize, args.ws
samples_per_subject = args.nw #no of windows

path_dd = '/data/users4/ziqbal5/abc/MILC/Data/'
path_b = str(args.jobid) + '_'+str(args.encoder)+'_'+ args.daata + '_ws_'+str(args.ws)+ '_wsize_'+str(args.wsize)+'_nw_'+str(args.nw) + '_seed_' + str(seed) +  '_gain_'+str(args.gainn) + '_tp_' + str(args.tp) + '_samp_'+ str(args.samples) + '_ep_'+ str(args.epp)+'_'+args.attr_alg+ '_fold_' + str(fold)+ '_ptr_'+ str(args.l_ptr)
#path_b = str(args.jobid) + '-F' + str(fold)+str(args.encoder) + args.daata + 'wsh'+str(args.ws)+ 'wsi'+str(args.wsize)+'nw'+str(args.nw) + 'S' + str(seed) +  'g'+str(args.gainn) + 'tp' + str(args.tp) + 'samp'+ str(args.samples) + 'ep'+ str(args.epp)+ 'ptr'+ str(args.l_ptr)+'-'+args.attr_alg
path_a = path_dd + path_b
print("Results will be saved here: ", path_a)

Path(path_a).mkdir(parents=True, exist_ok=True)
start_time = time.time()
print("Data: ", Data)


def get_data_loader(X, Y, batch_size):
    dataLoader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Y), batch_size=batch_size, shuffle = True)

    return dataLoader


def train_model(model, loader_train, loader_Validation, loader_test, epochs, learning_rate):
    loss = torch.nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # model.cuda()
    model.to(device)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

#Earlystopping: 1/2
    pathtointer =  os.path.join(path_a, 'checkpoint.pt')
    early_stopping = EarlyStopping(patience=10, delta = 0.0001,  path = pathtointer, verbose=True)

    train_losses, valid_losses, avg_train_losses, avg_valid_losses = [],[], [],[]
    train_accuracys, valid_accuracys, avg_train_accuracys, avg_valid_accuracys = [],[], [],[]
    train_rocs, valid_rocs, avg_train_rocs, avg_valid_rocs = [],[], [],[]
    for epoch in range(epochs):
                

        #for training
        # running_loss = 0.0
        # running_accuracy = 0.0
        model.train()
        for i, data in enumerate(loader_train):
            #print('Batch: ',i+1)
            x, y = data
            optimizer.zero_grad()
            outputs= model(x)
            l = loss(outputs, y)
            _, preds = torch.max(outputs.data, 1)
            l.backward()
            optimizer.step()
            train_losses.append(l.item())
            accuracy = accuracy_score(y.cpu(), preds.cpu(), normalize=True)
            train_accuracys.append(accuracy)
            #print('t_acc: ', accuracy)
            sig = F.softmax(outputs, dim=1).to(device)
            y_scores = sig.detach()[:, 1]
            #print('yscores',y_scores)
            try:
                roc = roc_auc_score(y.to('cpu'), y_scores.to('cpu'))
            except ValueError:
                pass
            train_rocs.append(roc)
            
        #for validation
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(loader_Validation):
                x, y = data
                outputs= model(x)
                l = loss(outputs, y)
                _, preds = torch.max(outputs.data, 1)
                valid_losses.append(l.item())
                accuracy = accuracy_score(y.cpu(), preds.cpu(), normalize=True)
                valid_accuracys.append(accuracy)
                sig = F.softmax(outputs, dim=1).to(device)
                y_scores = sig.detach()[:, 1]
                try:
                    roc = roc_auc_score(y.to('cpu'), y_scores.to('cpu'))
                except ValueError:
                    pass
                valid_rocs.append(roc)
        
        train_loss,valid_loss = np.average(train_losses),np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        train_accuracy,valid_accuracy = np.average(train_accuracys),np.average(valid_accuracys)
        avg_train_accuracys.append(train_accuracy)
        avg_valid_accuracys.append(valid_accuracy)
    

        train_roc,valid_roc = np.average(train_rocs),np.average(valid_rocs)
        avg_train_rocs.append(train_roc)
        avg_valid_rocs.append(valid_roc)

        print("epoch: " + str(epoch) + ", train_loss: " + str(train_loss) + ", val_loss: " + str(valid_loss) +", train_auc: " + str(train_roc) + ", val_auc: " + str(valid_roc) +", train_acc: " + str(train_accuracy) +" , val_acc: " + str(valid_accuracy))
        train_losses, valid_losses =[], []
        train_accuracys, valid_accuracys =[], []
        train_rocs, valid_rocs =[], []
        
        #Earlystopping: 2/2
        early_stopping(valid_loss, model)
        if early_stop:
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
    model.load_state_dict(torch.load(pathtointer))        
    #Start: test set results
    x_test, y_test = next(iter(loader_test))
    outputs = model(x_test)
    _, preds = torch.max(outputs.data, 1)
    
    
    print("Test (Predicted--) :", preds)
    print("Test (GroundTruth) :", y_test)
    
    accuracy_test = accuracy_score(y_test.cpu(), preds.cpu(), normalize=True)
    sig = F.softmax(outputs, dim=1).to(device)
    y_scores = sig.detach()[:, 1]
    roc_test = roc_auc_score(y_test.to('cpu'), y_scores.to('cpu'))
    #End: test set results
        
    

    #Plot training and validation loss curves
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(avg_train_losses)+1),avg_train_losses, label='Training Loss')
    plt.plot(range(1,len(avg_valid_losses)+1),avg_valid_losses,label='Validation Loss')
    # find position of lowest validation loss
    minposs = avg_valid_losses.index(min(avg_valid_losses))+1 
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    #plt.ylim(0.5, 1) # consistent scale
    plt.xlim(0, len(avg_train_losses)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig('loss_plot.png', bbox_inches='tight')
    plt.close()
    #print("length", len(train_loss_acc), len(val_loss_acc), val_loss_acc)
    return optimizer, accuracy_test, roc_test, model


def generate_data(subjects, tp, components):
    
    
    #HCP Data
    if Data == 'HCP':
        print('Dataset Used: ', Data)
        path = Path('/data/users4/ziqbal5/MILC/Data/HCP.pickle')
        if path.is_file() == True:
            with open(path, "rb") as infile:
                data = pickle.load(infile)
    if Data == 'UKB':
        print('Dataset Used: ', Data)
        path = Path('/data/users4/ziqbal5/DataSets/UKB/UKB830.pickle')
        if path.is_file() == True:
            with open(path, "rb") as infile:
                data = pickle.load(infile)
            print(data.shape, type(data))
        else:
            AllData = []
            filename2 = '/data/users4/ziqbal5/DataSets/UKB/correct_indices_GSP.csv'
        
            cor_ind = pd.read_csv(filename2, header=None)
            cor_ind = cor_ind.to_numpy()
            i = 0
            root = '/data/qneuromark/Results/ICA/UKBioBank/'
            for name in os.listdir(root):
                
                filename = os.path.join('/data/qneuromark/Results/ICA/UKBioBank/', name, 'UKB_ica_c1-1.mat')
                #print(filename)
                data = mat73.loadmat(filename)
                data = data['tc']
                data = data.transpose()
                
                arr = []    
                for j in range(0,len(cor_ind)):
                    arr.append(data[int(cor_ind[j])-1])
                arr = np.array(arr)
                print('subject: ', i+1, 'shape: ', arr.shape)
                if arr.shape[1] != 490:
                    print("Index ",i, "doesn't have full time steps: ", arr.shape[1])
                    continue
                AllData.append(arr)
                i = i+1
                if i == 830:
                    break
            
            print('All subjects loaded successfully...')
            data = np.stack(AllData, axis=0)
            with open(path, "wb") as outfile:
                pickle.dump(data, outfile)
            print(data.shape)
        #labels fankari
        X1 = torch.Tensor(data)
        L1 = np.zeros([len(data),])
        L1 = torch.tensor(L1)

        #Reverse Direction
        ccc = np.empty([subjects, components, tp])
        for i in range(subjects):
            for j in range(components):
                ccc[i][j] = np.flip(data[i][j])
        print("Slices reversal is done")
        
        X2 = torch.Tensor(ccc)
        L2 = np.ones([len(ccc),])
        L2 = torch.tensor(L2)
        data = torch.cat((X1, X2), 0)
        labels = torch.cat((L1, L2),0)
 
    
            

    elif Data == 'HCP2':
        #####
        # path = Path('/data/qneuromark/Results/SFNC/HCP/sFNC_HCP_REST1_LR.mat')
        # data = mat73.loadmat(path)
        # data = data['sFNC']
        # #####

        print('Dataset Used: ', Data)
        path = Path('/data/users4/ziqbal5/DataSets/HCP1200realease/HCP830.pickle')
        if path.is_file() == True:
            with open(path, "rb") as infile:
                data = pickle.load(infile)
            print(data.shape, type(data))

            # c = data[400]
            # print(c.shape)
            # #c = c.transpose()
            # #c = c[0]
            # print(c.shape)
            # print(c.mean(), np.std(c))

       
        
            # fig, axx = plt.subplots(nrows=10, ncols=10, figsize=(100,100))
            # def fnc_plot(M, title, clim=(-1,1)):
            #     axx[i,j].imshow(M, interpolation=None, clim=clim,
            #             cmap=plt.cm.seismic)
                
            #     ax = plt.gca()
            #     plt.title(title)
            
            #     #plt.colorbar()
            #     # Minor ticks
            #     groups = [-0.5, 4.5, 6.5, 15.5, 24.5, 41.5, 48.5, 52.5]
            #     axx[i,j].set_xticks(groups, minor=True)
            #     axx[i,j].set_yticks(groups, minor=True)
            #     axx[i,j].grid(which='minor', color='k', linestyle='-', linewidth=0.5)
            #     # Major ticks
            #     group_centers = [2, 5.5, 11, 20, 33, 45, 50.5]
            #     axx[i,j].set_xticks(group_centers)
            #     axx[i,j].set_yticks(group_centers)

            #     # Labels for major ticks
            #     labels = ['SC', 'AU', 'SM', 'VI', 'CC', 'DM', 'CB']
            #     axx[i,j].set_xticklabels(labels)
            #     axx[i,j].set_yticklabels(labels, rotation=90, fontdict={'verticalalignment': 'center'})
            #     axx[i,j].tick_params(axis='both', which='major', length=0)

            


            
            # k0 = 0
            # for i in range(10):
            #     for j in range(10):
            #         #print(data[k0].shape)
            #         FBIRN05 = data[k0]
            #         k0 = k0+1
            #         FBIRN05 = np.corrcoef(FBIRN05)

            #         np.fill_diagonal(FBIRN05, 0)
            #         fnc_plot(FBIRN05, 'FBIRN', clim=(-0.6, 0.6))    
            # plt.savefig('h2abc.png')
            # exit()         

        else:
        
            AllData = []
            filename2 = '/data/users4/ziqbal5/DataSets/HCP1200realease/correct_indices_HCP.csv'
            cor_ind = pd.read_csv(filename2, header=None)
            cor_ind = cor_ind.to_numpy()

            
            for i in tqdm(range(1, 834)):
                
                filename = '/data/qneuromark/Results/ICA/HCP/REST1_LR/HCP1_ica_c'+str(i)+'-1.mat'
                data = sio.loadmat(filename)
                data = data['tc']   #1200x100
                data = data.transpose()  #100x1200
                arr = []    
                for j in range(0,len(cor_ind)):
                    arr.append(data[int(cor_ind[j])-1])
                arr = np.array(arr)
                if arr.shape[1] < 1200:
                    print("Index ",i, "doesn't have full time steps: ", arr.shape[1])
                    continue
                AllData.append(arr)

            print('All subjects loaded successfully...')
            data = np.stack(AllData, axis=0)
            #print(data.shape)
            with open(path, "wb") as outfile:
                pickle.dump(data, outfile)

        #labels
        X1 = torch.Tensor(data)
        L1 = np.zeros([len(data),])
        L1 = torch.tensor(L1)

        #Reverse Direction
        ccc = np.empty([subjects, components, tp])
        for i in range(subjects):
            for j in range(components):
                ccc[i][j] = np.flip(data[i][j])
        print("Slices reversal is done")
        
        X2 = torch.Tensor(ccc)
        L2 = np.ones([len(ccc),])
        L2 = torch.tensor(L2)
        data = torch.cat((X1, X2), 0)
        labels = torch.cat((L1, L2),0)

    
    elif Data == 'FBIRN':
        print("DatasetUsed: ", Data)
        path = Path('/data/users4/ziqbal5/abc/MILC/FBIRN.pickle')
        if path.is_file() == True:
            with open(path, "rb") as infile:
                data = pickle.load(infile)
            
            print(data.shape, type(data))

            # Take zscore along time points start.
            # for i in range(len(data)):
            #     temp = data[i]
            #     temp = np.reshape(temp, (1,7420))
            #     mean = np.mean(temp)
            #     std_dev = np.std(temp)
            #     z_scores = (temp - mean) / std_dev
            #     z_scores = np.reshape(z_scores, (53, -1))
            #     data[i] = z_scores
            #Take zscore along time points.
            
            # c = data[2]
            # print(c.shape)
            # #c = c.transpose()
            # #c = c[0]
            # print(c.shape)
            # print(c.mean(), np.std(c))

           
            #using the indices corresponding to brain maps
            ds = '/data/users4/ziqbal5/MILC/IndicesAndLabels/transform_to_correct_GSP.csv'
            cor_ind = pd.read_csv(ds, header=None)
            cor_ind = cor_ind[2]
            cor_ind = cor_ind.to_numpy()
            cor_ind -= 1
            for i in range(len(data)):
                temp = data[i]
                data[i] = temp[cor_ind]
            

    elif Data == 'bardata':
        print("DatasetUsed: ", Data)
        samples, t_steps, features, p_steps, seed = subjects, tp, components, 90, 6
        # if seed != None:
        #     np.random.seed(seed)
        X = np.zeros([samples, t_steps, features])
        L = np.zeros([samples])
        start_positions = np.zeros([samples])
        masks = np.zeros([samples,p_steps,features])
        for i in range(samples):
            mask = np.zeros([p_steps, features])
            #0,17 ; 27,47
            start = np.random.randint(100,(t_steps/2)-100)
            #print(start)
            start_positions[i] = start
            x = np.random.normal(0, 1, [1, t_steps, features])
            label = np.random.randint(0, 2)
            lift = np.random.normal(1, 1,[p_steps,features])
            X[i,:,:] = x
            if label:
                mask[:,0:int(features/2)] = 1
            else:
                mask[:,int(features/2):] = 1
            lift = lift*mask
            X[i,start:start+p_steps, :] += lift
            masks[i,:,:] = lift
            L[i] = int(label)
        #exit()
        data = X.transpose(0,2,1)


        with open(os.path.join(path_a, 'prev_labels.pickle'), "wb") as outfile:
            pickle.dump(L, outfile)
        with open(os.path.join(path_a, 's_pos.pickle'), "wb") as outfile:
            pickle.dump(start_positions, outfile)

    elif Data == 'bardataold':
        print("DatasetUsed: ", Data)
        samples, t_steps, features, p_steps, seed = subjects, tp, components, 10, 6
        
        #if seed != None:
            #np.random.seed(seed)
        X = np.zeros([samples, t_steps, features])
        L = np.zeros([samples])
        start_positions = np.zeros([samples])
        masks = np.zeros([samples,p_steps,features])
        for i in range(samples):
            mask = np.zeros([p_steps, features])
            #0,17 ; 27,47
            start = np.random.randint(0,t_steps-p_steps)
            start_positions[i] = start
            x = np.random.normal(0, 1, [1, t_steps, features])
            label = np.random.randint(0, 2)
            lift = np.random.normal(1, 1,[p_steps,features])
            X[i,:,:] = x
            if label:
                mask[:,0:int(features/2)] = 1
            else:
                mask[:,int(features/2):] = 1
            lift = lift*mask
            X[i,start:start+p_steps, :] += lift
            masks[i,:,:] = lift
            L[i] = int(label)
        
        X = X.transpose(0,2,1)
        X = torch.Tensor(X)
        L = torch.tensor(L)
        L = torch.squeeze(L)
        split = int(samples*.80)
        tr_data = X[:split]
        tr_labels = L[:split]

        test_data = X[split:]
        test_labels = L[split:]
    

        with open(os.path.join(path_a, 's_pos.pickle'), "wb") as outfile:
            pickle.dump(start_positions, outfile)

    elif Data == 'chirpLog':
        print('Dataset Used: ', Data)
        data = np.empty([subjects, components,tp])
        for j in range(subjects):
            fs = 7200
            T = 4
            t = np.arange(0, int(T*fs)) / fs
            f0 = random.randrange(1000,2000)
            f1 = random.randrange(100,300)
            for i in range(components):
                w = chirp(t, f0=f0, f1=f1, t1=T, method='logarithmic')
                wm = w[np.r_[50:450,15000:15400, 28400:28800]]        #print(phase.shape)
                data[j][i] = wm
    elif Data == 'chirpLog2':
        print('Dataset Used: ', Data)
        data = np.empty([subjects, components,tp])
        for j in range(subjects):
            for i in range(components):
                fs = 7200
                T = 4
                t = np.arange(0, int(T*fs)) / fs
                f0 = random.randrange(1000,2000)
                f1 = random.randrange(100,300)
                w = chirp(t, f0=f0, f1=f1, t1=T, method='logarithmic')
                wm = w[np.r_[50:100,15000:15050, 28750:28800]]        #print(phase.shape)
                data[j][i] = wm
    elif Data == 'chirp1':
        print('Dataset Used: ', Data)
        data = np.empty([subjects, components,tp])
        for j in range(subjects):
            for i in range(components):

                # Parameters
                duration = 2.0  # seconds
                start_freq =  random.uniform(5,10)  
                end_freq =  random.uniform(20,25)   
                sampling_rate = int((tp+40)/duration)  # 28

                # Time array
                t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
                t = t[40:]
                # Phase array
                phase = 2 * np.pi * (start_freq * t + 0.5 * (end_freq - start_freq) * t ** 2 / duration)
                #print(phase.shape)
                # Signal (sine wave with linearly increasing frequency)
                data[j][i] = np.sin(phase)
        #labels
        X1 = torch.Tensor(data)
        L1 = np.zeros([len(data),])
        L1 = torch.tensor(L1)

        #Reverse Direction
        ccc = np.empty([subjects, components, tp])
        for i in range(subjects):
            for j in range(components):
                ccc[i][j] = np.flip(data[i][j])
        print("Slices reversal is done")
        
        X2 = torch.Tensor(ccc)
        L2 = np.ones([len(ccc),])
        L2 = torch.tensor(L2)
        data = torch.cat((X1, X2), 0)
        labels = torch.cat((L1, L2),0)
    ####################################################
    elif Data == 'chirp1M':
        print('Dataset Used: ', Data)
        data = np.empty([subjects, components,tp])
        for j in range(subjects):
            duration = 5.0  # seconds
            start_freq =  random.uniform(5,10)  
            end_freq =  random.uniform(20,25)   
            sampling_rate = int((tp+40)/duration)  # 28
            for i in range(components):

                # Parameters


                # Time array
                t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
                t = t[40:]
                # Phase array
                phase = 2 * np.pi * (start_freq * t + 0.5 * (end_freq - start_freq) * t ** 2 / duration)
                #print(phase.shape)
                # Signal (sine wave with linearly increasing frequency)
                data[j][i] = np.sin(phase)

    
    elif Data == 'chirp2': #inphase within a subject
        print('Dataset Used: ', Data)
        data = np.empty([subjects, components,tp])
        for j in range(subjects):

            # Parameters
            duration = 5.0  # seconds
            start_freq =  random.uniform(5,15)  
            end_freq =  random.uniform(40,50)   
            sampling_rate = int(((tp/2)+40)/duration)
            for i in range(components):



                t1 = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
                t1 = t1[40::]
                t2 = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
                t2 = t2[40::]

                t = np.concatenate((t1,t2), axis=0)
                # Phase array
                phase = 2 * np.pi * (start_freq * t + 0.5 * (end_freq - start_freq) * t ** 2 / duration)
                #print(phase.shape)
                # Signal (sine wave with linearly increasing frequency)
                data[j][i] = np.sin(phase)

        #labels
        X1 = torch.Tensor(data)
        L1 = np.zeros([len(data),])
        L1 = torch.tensor(L1)

        #Reverse Direction
        ccc = np.empty([subjects, components, tp])
        for i in range(subjects):
            for j in range(components):
                ccc[i][j] = np.flip(data[i][j])
        print("Slices reversal is done")
        
        X2 = torch.Tensor(ccc)
        L2 = np.ones([len(ccc),])
        L2 = torch.tensor(L2)
        data = torch.cat((X1, X2), 0)
        labels = torch.cat((L1, L2),0)


    

    ####################################################
    elif Data == 'chirp3': #inphase
        print('Dataset Used: ', Data)
        data = np.empty([subjects, components,tp])
        for j in range(subjects):
            for i in range(components):

                duration = 5.0  # seconds
                start_freq =  random.uniform(5,15)  
                end_freq =  random.uniform(40,50)   
                sampling_rate = int(((tp/2)+40)/duration)

                t1 = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
                t1 = t1[40::]
                phase1 = 2 * np.pi * (start_freq * t1 + 0.5 * (end_freq - start_freq) * t1 ** 2 / duration)
                
                duration = 5.0  # seconds
                start_freq =  random.uniform(5,15)  
                end_freq =  random.uniform(40,50)   
                sampling_rate = int(((tp/2)+40)/duration)
                
                t2 = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
                t2 = t2[40::]
                
                phase2 = 2 * np.pi * (start_freq * t2 + 0.5 * (end_freq - start_freq) * t2 ** 2 / duration)
                
                phase = np.concatenate((phase1,phase2), axis=0)
                # Phase array
                #phase = 2 * np.pi * (start_freq * t + 0.5 * (end_freq - start_freq) * t ** 2 / duration)
                #print(phase.shape)
                # Signal (sine wave with linearly increasing frequency)
                data[j][i] = np.sin(phase)
               
    ####################################################
    elif Data == 'chirp4': #inphase
        print('Dataset Used: ', Data)
        data = np.empty([subjects, components,tp])
        for j in range(subjects):

            # Parameters

            for i in range(components):

                duration = 5.0  # seconds
                start_freq =  random.uniform(5,15)  
                end_freq =  random.uniform(40,50)   
                sampling_rate = int(((tp/2)+40)/duration)

                t1 = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
                t1 = t1[40::]
                
                
                t2 = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
                t2 = t2[40::]

                t = np.concatenate((t1,t2), axis=0)
                # Phase array
                phase = 2 * np.pi * (start_freq * t + 0.5 * (end_freq - start_freq) * t ** 2 / duration)
                #print(phase.shape)
                # Signal (sine wave with linearly increasing frequency)
                data[j][i] = np.sin(phase)
    ####################################################
    elif Data == 'chirp5': #inphase
        print('Dataset Used: ', Data)
        data = np.empty([subjects, components,tp])
        for j in range(subjects):
            T1 = random.uniform(20, 25)
            T2 = random.uniform(15, 20)
            T3 = random.uniform(10, 15)
            T4 = random.uniform(5, 10)
            for i in range(components):


                amp1 = random.uniform(10,20)
                x1 = np.linspace(0, 260, 260)
                y1 = [amp1 * math.sin(2*math.pi*i/T1) for i in x1] # first part of the function


                amp2 = random.uniform(20,30)
                x2 = np.linspace(260, 520, 260)
                y2 = [amp2 * math.cos(2*math.pi*i/T2) for i in x2] # 2nd part of the function


                amp3 = random.uniform(30,40)
                x3 = np.linspace(520, 780, 260)
                y3 = [amp3 * math.sin(2*math.pi*i/T3) for i in x3] # 3rd part of the function


                amp4 = random.uniform(40,50)
                x4 = np.linspace(780, 1040, 260)
                y4 = [amp4 * math.cos(2*math.pi*i/T4) for i in x4] # 4th part of the function
                bb = y1
                bb.extend(y2)
                bb.extend(y3)
                bb.extend(y4)
                data[j][i] = bb
    # ####################################################                
    elif Data == 'chirp6': #inphase
        print('Dataset Used: ', Data)
        data = np.empty([subjects, components,tp])
        for j in range(subjects):

            for i in range(components):
                
                T1 = random.uniform(20, 25)
                T2 = random.uniform(15, 20)
                T3 = random.uniform(10, 15)
                T4 = random.uniform(5, 10)

                amp1 = random.uniform(10,20)
                x1 = np.linspace(0, 260, 260)
                y1 = [amp1 * math.sin(2*math.pi*i/T1) for i in x1] # first part of the function


                amp2 = random.uniform(20,30)
                x2 = np.linspace(260, 520, 260)
                y2 = [amp2 * math.cos(2*math.pi*i/T2) for i in x2] # 2nd part of the function


                amp3 = random.uniform(30,40)
                x3 = np.linspace(520, 780, 260)
                y3 = [amp3 * math.sin(2*math.pi*i/T3) for i in x3] # 3rd part of the function


                amp4 = random.uniform(40,50)
                x4 = np.linspace(780, 1040, 260)
                y4 = [amp4 * math.cos(2*math.pi*i/T4) for i in x4] # 4th part of the function
                bb = y1
                bb.extend(y2)
                bb.extend(y3)
                bb.extend(y4)
                data[j][i] = bb

    Adata = torch.Tensor(data)
    data = np.zeros((len(Adata), samples_per_subject, components, sample_y))
 

    for i in range(len(Adata)):
        for j in range(samples_per_subject):
            data[i, j, :, :] = Adata[i, :, (j * window_shift):(j * window_shift) + sample_y]

   #update data and lables here:
    #print(data.shape, len(labels))
    HC = np.where(labels == 0)[0]
    print(len(HC))
    L = range(len(HC))
    amount = 83
    random.seed(1)
    rangeofinterp = [random.choice(L) for _ in range(amount)]
  

    Pat = np.where(labels == 1)[0]
    # temp_d1, temp_l1 = data[HC[120:203]], labels[HC[120:203]]
    # temp_d2, temp_l2 = data[Pat[120:203]], labels[Pat[120:203]]

    temp_d1, temp_l1 = data[HC[rangeofinterp]], labels[HC[rangeofinterp]]
    temp_d2, temp_l2 = data[Pat[rangeofinterp]], labels[Pat[rangeofinterp]]

    tdata_interp = np.concatenate((temp_d1, temp_d2))
    tlabels_interp = np.concatenate((temp_l1, temp_l2))

    #This is just for iterpretability (Choosing first 40 from each class to maintain correspondence)
    with open(os.path.join(path_a, 'test_data.pickle'), "wb") as outfile:
        pickle.dump(tdata_interp, outfile)
    with open(os.path.join(path_a, 'test_labels.pickle'), "wb") as outfile:
        pickle.dump(tlabels_interp, outfile)



 
    skf = StratifiedKFold(n_splits=10, random_state = 34, shuffle = True)
    #skf = StratifiedKFold(n_splits=5, shuffle = False) #This made difference plus the decoder.
    skf.get_n_splits(data, labels)
    Folds_train_ind = []
    Folds_test_ind = []
    for i, (train_index, test_index) in enumerate(skf.split(data, labels)):
        Folds_train_ind.append(train_index)
        Folds_test_ind.append(test_index)
    
    Folds_train_ind[fold], Folds_test_ind[fold]

    tr_data, tr_labels = data[Folds_train_ind[fold]], labels[Folds_train_ind[fold]]
    test_data, test_labels = data[Folds_test_ind[fold]], labels[Folds_test_ind[fold]] 
    return tr_data, tr_labels, test_data, test_labels

train_data, train_labels, test_data, test_labels = generate_data(subjects, tp, components)
from sklearn.model_selection import train_test_split
tr_data, val_data, tr_labels, val_labels = train_test_split(train_data,train_labels , 
                                   random_state=0,  
                                   test_size=0.20,  
                                   shuffle=True, stratify=train_labels)



for en in range(len(enc)):


    encoderr = enc[en]
    print("Encoder: ", encoderr)
        
    test_data = torch.from_numpy(test_data)
    dataLoaderTest = get_data_loader(test_data.float().to(device), test_labels.long().to(device), test_data.float().shape[0])

    accMat = []
    aucMat = []

    start_time = time.time()



    print('Gain Values Chosen:', Gain)

    dir = args.exp   # NPT or UFPT

    for i in range(len(Gain)):
        
        for restart in range(Trials):
            
            g = Gain[i]
            print("Gain: ",g)

            tr_data = torch.from_numpy(tr_data)
            val_data = torch.from_numpy(val_data)
            dataLoaderTrain =      get_data_loader(tr_data.float().to(device), tr_labels.long().to(device), args.batch_size)
            #dataLoaderTrainCheck = get_data_loader(tr_data.float().to(device), tr_labels.long().to(device),  32)
            dataLoaderValidation = get_data_loader(val_data.float().to(device), val_labels.long().to(device),  args.batch_size)

            ################################
            encoder = NatureOneCNN(53, args)
            lstm_model = subjLSTM(
                                        device,
                                        args.feature_size,
                                        args.lstm_size,
                                        num_layers=args.lstm_layers,
                                        freeze_embeddings=True,
                                        gain=g,
                                    ) 




            
            if encoderr == 'cnn':
                
                model = combinedModel(
                encoder,
                lstm_model,
                gain=g,
                PT=args.pre_training,
                exp=args.exp,
                device=device,
                oldpath=args.oldpath,
                complete_arc=args.complete_arc,
            )
            elif encoderr == 'rnn' or encoderr == 'rnm':
                #LSTM used by Mahfuz
                model = LSTM2(components, 256, 200, 121, 2, g, onewindow).float()   #X.shape[2] = 53
            starttime = datetime.datetime.now()
            optimizer, accuracy_test, auc_test,model = train_model(model, dataLoaderTrain, dataLoaderValidation, dataLoaderTest, eppochs, args.lr)#3e-4  #.0005
            endtime = datetime.datetime.now()
            elapsed_time = endtime-starttime
            torch.save(model.state_dict(),  os.path.join(path_a, 'PretrainedModel.pt'))
            savepretrainedmodelpaths = os.path.join(path_a, 'PretrainedModel.pt', '\n')
            with open('pathptrmodels.csv', 'a', newline='') as fd:
                fd.write(savepretrainedmodelpaths)  


    

            if torch.cuda.is_available():
                
                #print(f'Allocated: {torch.cuda.memory_allocated()}')
                del model
                gc.collect()
                torch.cuda.empty_cache()
                #print(f'Allocated: {torch.cuda.memory_allocated()}')
                
            
    accDataFrame = pd.DataFrame({'Accuracy':[accuracy_test], 'AUC':[auc_test]})
    #print('df', accDataFrame)
    accfname = os.path.join(path_a, 'ACC_AUC.csv')
    accDataFrame.to_csv(accfname)


print('Total Time for training (seconds) :', elapsed_time.seconds) # could be .microseconds, .days
print("Accuracy and AUC on Test Data :", accuracy_test, auc_test)


################################################################################################
################################################################################################
def Load_Data():
    #aaa = '/data/users4/ziqbal5/abc/MILC/backup/5960674_rnm_HCP2_ws_1200_wsize_1200_nw_1_seed_1_gain_0.9_tp_1200_samp_830_ep_100_GS_fold_0_ptr_F_Sel/'
    with open(os.path.join(path_a, 'test_data.pickle'), "rb") as infile:
        X = pickle.load(infile)
    with open(os.path.join(path_a, 'test_labels.pickle'), "rb") as infile:
        L = pickle.load(infile)
        
    return X, L


def Initiate_Model(path_models, gain, encoderr):
    #Initiate Model
    sample_x = components
    current_gain = gain
    encoder = NatureOneCNN(sample_x, args)
    lstm_model = subjLSTM(
                        device,
                        args.feature_size,
                        args.lstm_size,
                        num_layers=args.lstm_layers,
                        freeze_embeddings=True,
                        gain=current_gain,
                    )
    if encoderr == 'cnn':    
        model = combinedModel(
            encoder,
            lstm_model,
            gain=current_gain,
            PT=args.pre_training,
            exp=args.exp,
            device=device,
            oldpath=args.oldpath,
            complete_arc=args.complete_arc,
        )
    
    elif encoderr == 'rnn' or encoderr == 'rnm':
            #LSTM used by Mahfuz
            model = LSTM2(components, 256, 200, 121, 2, g, onewindow).float()

    path_m = path_models


    model.load_state_dict(torch.load(path_m))
    #model.load_state_dict(torch.load(path_m)['model_state_dict'])
    print("Model trained on synthetic data loaded from: ", path_m)

    return model

def Predicted_Labels(model, TestDataFD, LabelsFD):
    
    TestDataFD = torch.from_numpy(TestDataFD).float()
    LabelsFD = torch.from_numpy(LabelsFD)
    datasetFD = TensorDataset(TestDataFD, LabelsFD)
    loaderSalFD = DataLoader(datasetFD, batch_size=len(LabelsFD), shuffle=False)


    model.eval()
    model.to(device)
    
    for param in model.parameters():
        param.requires_grad = False
    
    for i, data in enumerate(loaderSalFD):
        x, y = data
        x = x.to(device)
        y = y.to(device)

        outputs = model(x)
        _, predsFD = torch.max(outputs.data, 1)
    # with open(os.path.join(path_a, 'pred_labels_S.pickle'), "wb") as outfile:
    #     pickle.dump(predsFD, outfile)

    # return predsFD
    #print("aaaaaaaa",enc[0])
    if enc[0] != 'cnn':
        loaderSalFD = DataLoader(datasetFD, batch_size=1, shuffle=False)

        model.eval()
        model.to(device)



        for param in model.parameters():
            param.requires_grad = False
        
        L, C, Pred = [], [], []
        loss = torch.nn.CrossEntropyLoss()
        for i, data in enumerate(loaderSalFD):
            
            x, y = data
            y = y.type(torch.LongTensor)
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            # _, predsFD = torch.max(outputs.data, 1)
        
            outputs = torch.squeeze(outputs)
            outputs = torch.unsqueeze(outputs,0)
            
            sig = F.softmax(outputs, dim=1).to(device)
        
            conf, classes = torch.max(sig, 1)
            l = loss(outputs, y)
            #print(l)
            #print('conf', conf)
            #print('l', l)
            
            L.append(l.item())
            C.append(conf.item())
            Pred.append(classes.item())
        LCP = np.zeros((3, len(L)))
        LCP[0], LCP[1], LCP[2] = L, C, Pred
        with open(os.path.join(path_a, 'LCP.pickle'), "wb") as outfile:
            pickle.dump(LCP, outfile)
        #print(LCP[0])
    return predsFD
  

    





def Feature_Attributions(model, TestDataFD, LabelsFD, predsFD):

    TestDataFD = torch.from_numpy(TestDataFD).float()
    datasetFD = TensorDataset(TestDataFD, predsFD)
    loaderSalFD = DataLoader(datasetFD, batch_size=1, shuffle=False)

    model.eval()
    model.to(device)



    for param in model.parameters():
        param.requires_grad = False
        
    salienciesFD = []
    
    for i, data in enumerate(loaderSalFD):
        if i % 10 == 0:
            print(f'Processing subject:{i}')
        x, y = data 
        
        #x=torch.squeeze(x, dim=0)
        
        x = x.to(device)
        y = y.to(device)
            
        #print("aa",x.shape, predsFD.shape)
        #predsFD= torch.unsqueeze(predsFD, dim=0)
        #print(predsFD)
        bl = torch.zeros(x.shape).to(device)
                
        
        x = x.to(device)
        y = y.to(device)
        
        model.train()
        x.requires_grad_()

        if attr_alg == "IG":
            # Integrated Gradeints
            sal = IntegratedGradients(model, multiply_by_inputs=False)####
            sal = IntegratedGradients(model)####
            attribution = sal.attribute(x,bl,target=y)

        elif attr_alg == "GS":
            #GradientShap
            sal = GradientShap(model)
            attribution = sal.attribute(x,bl, stdevs = 0.09,target=y)
        elif attr_alg =="GN":
            #NoiseTunnel
            ig = IntegratedGradients(model, multiply_by_inputs=False)
            sal = NoiseTunnel(ig)
            attribution = sal.attribute(x,nt_samples=10, nt_type ="smoothgrad_sq", target=y)




        salienciesFD.append(np.squeeze(attribution.cpu().detach().numpy()))

        
        
    all_salienciesFD = np.stack(salienciesFD, axis=0)
    print("sal before dumping", all_salienciesFD.shape)
    
    with open(os.path.join(path_a, 'all_saliencies_S.pickle'), "wb") as outfile:
        pickle.dump(all_salienciesFD, outfile)

for dd in range(1):
    for en in range(len(enc)):
        encoderr = enc[en]
        print("Encoder: ", encoderr)

        
    
        TestDataFD, LabelsFD = Load_Data()

        print(TestDataFD.shape, type(TestDataFD), LabelsFD.shape)
        print("TestData Ground truth labels: ", LabelsFD)
        

        for j in range(Trials):
            for g in range(len(Gain)):
            
                path_models = os.path.join(path_a, 'PretrainedModel.pt')
                gain = Gain[g]
                model = Initiate_Model(path_models, gain, encoderr) 
                predsFD = Predicted_Labels(model, TestDataFD, LabelsFD)
                print("TestData predicted labels", predsFD)
                ac = accuracy_score(LabelsFD, predsFD.cpu())
                print("accuracyforInterp: ", ac)
                Feature_Attributions(model, TestDataFD,LabelsFD, predsFD)
