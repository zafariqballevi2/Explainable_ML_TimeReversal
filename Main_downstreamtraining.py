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
import h5py
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold
from captum.attr import IntegratedGradients, GradientShap, NoiseTunnel, Occlusion
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import datetime
import pandas as pd
import sys
from sklearn.metrics import accuracy_score
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
from sklearn.metrics import accuracy_score
from Get_Data import Datasets, EarlyStopping, LSTM, LSTM2




early_stop = True
parser = get_argparser()
args = parser.parse_args()
print("JOBID: ", args.jobid)
print("Batch Size and Learning rate: ", args.batch_size, args.lr)
device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
seed_values = [0,1,2,3,4,5,6,7,8,9]
fold_values = [0,1,2,3,4,5,6,7,8,9]
seed = seed_values[args.seeds]
fold = fold_values[args.fold_v]
print("[Seed Fold]: ", seed, fold)
print("conv size: ", args.convsize)

#to keep the dataset generation same
# random.seed(22)
# np.random.seed(22)
#################################
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

Trials = 1
eppochs = args.epp
Gain = [args.gainn]
dd = 00 #this is evein he.
Data = args.daata
enc = [args.encoder]     # 'cnn', 'lstmM'
attr_alg = args.attr_alg

#aa = args.windowspecs
l_ptr = args.l_ptr
if args.encoder == 'rnm':
    onewindow = True
else:
    onewindow = False
subjects, tp, components = args.samples, args.tp, 53  #
sample_y, window_shift = args.wsize, args.ws
samples_per_subject = args.nw #no of windows

path_dd = '/data/users4/ziqbal5/abc/MILC/Data/'
#path_b = str(args.jobid) + '-F' + str(fold)+str(args.encoder) + args.daata + 'wsh'+str(args.ws)+ 'wsi'+str(args.wsize)+'nw'+str(args.nw) + 'S' + str(seed) +  'g'+str(args.gainn) + 'tp' + str(args.tp) + 'samp'+ str(args.samples) + 'ep'+ str(args.epp)+ 'ptr'+ str(args.l_ptr)+'-'+args.attr_alg
path_b = str(args.jobid) + '_'+str(args.encoder)+'_'+ args.daata + '_ws_'+str(args.ws)+ '_wsize_'+str(args.wsize)+'_nw_'+str(args.nw) + '_seed_' + str(seed) +  '_gain_'+str(args.gainn) + '_tp_' + str(args.tp) + '_samp_'+ str(args.samples) + '_ep_'+ str(args.epp)+'_'+args.attr_alg+ '_fold_' + str(fold)+ '_ptr_'+ str(args.l_ptr)
path_a = path_dd + path_b
print("Results will be saved here: ", path_a)

Path(path_a).mkdir(parents=True, exist_ok=True)
#start_time = time.time()
print("Data: ", Data)


# def find_indices_of_each_class(all_labels):
#     HC_index = (all_labels == 0).nonzero()
#     SZ_index = (all_labels == 1).nonzero()

#     return HC_index, SZ_index





def train_model(model, loader_train, loader_Validation, loader_test, epochs, learning_rate):
    loss = torch.nn.CrossEntropyLoss()

    
    #optimizer = Adam(model.parameters())
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = optim.SGD(model.parameters(), momentum=0.5)

    # model.cuda()
    model.to(device)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(dataLoaderTrain), epochs=10)
    
   
   
    #Earlystopping: 1/2
    pathtointer =  os.path.join(path_a, 'checkpoint.pt')
    early_stopping = EarlyStopping(patience=20,delta = 0.0001, path = pathtointer, verbose=True)

    train_losses, valid_losses, avg_train_losses, avg_valid_losses = [],[], [],[]
    train_accuracys, valid_accuracys, avg_train_accuracys, avg_valid_accuracys = [],[], [],[]
    train_rocs, valid_rocs, avg_train_rocs, avg_valid_rocs = [],[], [],[]
    for epoch in range(epochs):
                

        #for training
        # running_loss = 0.0
        # running_accuracy = 0.0
        model.train()
        #with torch.autograd.detect_anomaly(False):
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
            #roc = roc_auc_score(y.to('cpu'), y_scores.to('cpu'))
            #train_rocs.append(roc)
            try:
                roc = roc_auc_score(y.to('cpu'), y_scores.to('cpu'))
                train_rocs.append(roc)
            except ValueError:
                pass
            
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
                    valid_rocs.append(roc)
                except ValueError:
                    pass
                
        
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



obj_datasets = Datasets(fold)
if args.daata == 'FBIRN':
    tr_data, tr_labels, val_data, val_labels,test_data, test_labels = obj_datasets.FBIRN()
if args.daata == 'BSNIP':
    tr_data, tr_labels, val_data, val_labels,test_data, test_labels = obj_datasets.BSNIP()
if args.daata == 'COBRE':
    tr_data, tr_labels, val_data, val_labels,test_data, test_labels = obj_datasets.COBRE()


print('Dataset: ', args.daata, "Tr_data, Tr_labels, val_data, val_labels, test_data, test_labels: ",tr_data.shape, tr_labels.shape, val_data.shape, val_labels.shape,test_data.shape, test_labels.shape )
#print(tr_labels)
# c1_index = torch.where(tr_labels == 0)
# c2_index = torch.where(tr_labels == 1)
# c1_index = c1_index[0][0:15]
# c2_index = c2_index[0][0:15]
# c_index = torch.cat([c1_index, c2_index])

# tr_data, tr_labels = tr_data[c_index], tr_labels[c_index]


with open(os.path.join(path_a, 'test_data.pickle'), "wb") as outfile:
    pickle.dump(test_data, outfile)
with open(os.path.join(path_a, 'test_labels.pickle'), "wb") as outfile:
    pickle.dump(test_labels, outfile)

def get_data_loader(X, Y, batch_size):
    dataLoader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Y), batch_size=batch_size, shuffle = True)

    return dataLoader

for en in range(len(enc)):


    encoderr = enc[en]
    print("Encoder: ", encoderr)

    test_data = torch.from_numpy(test_data)
    dataLoaderTest = get_data_loader(test_data.float().to(device), test_labels.long().to(device), test_data.float().shape[0])

    accMat = []
    aucMat = []

    #start_time = time.time()

    print('Gain Values Chosen:', Gain)

    dir = args.exp   # NPT or UFPT
    #model_path = os.path.join(sbpath, Directories[Dataset[data]], dir)

    for i in range(len(Gain)):
        
        for restart in range(Trials):
            #print("Trial: ", restart)
            
            # samples = subjects_per_group[i]
            
            g = Gain[i]
            print("Gain: ",g)


            tr_data = torch.from_numpy(tr_data)
            val_data = torch.from_numpy(val_data)
            dataLoaderTrain =      get_data_loader(tr_data.float().to(device), tr_labels.long().to(device), args.batch_size)
            #dataLoaderTrainCheck = get_data_loader(tr_data.float().to(device), tr_labels.long().to(device),  32)
            dataLoaderValidation = get_data_loader(val_data.float().to(device), val_labels.long().to(device),  len(val_data))
            
        
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
                
                model = LSTM(components, 256, 200, 121, 2, g, onewindow).float()
            

            if l_ptr == 'T':

         

                #81.64 accuracy with these (lstm2)
                #path_m = '/data/users4/ziqbal5/abc/MILC/Data/6928908_rnn_HCP2_ws_1200_wsize_1200_nw_1_seed_1_gain_0.9_tp_1200_samp_830_ep_100_GS_fold_1_ptr_F/PretrainedModel.pt'
                #path_m = '/data/users4/ziqbal5/abc/MILC/Data/6928913_rnn_HCP2_ws_1200_wsize_1200_nw_1_seed_1_gain_0.9_tp_1200_samp_830_ep_100_GS_fold_6_ptr_F/PretrainedModel.pt'
                #path_m = '/data/users4/ziqbal5/abc/MILC/Data/6928914_rnn_HCP2_ws_1200_wsize_1200_nw_1_seed_1_gain_0.9_tp_1200_samp_830_ep_100_GS_fold_7_ptr_F/PretrainedModel.pt'
                #path_m = '/data/users4/ziqbal5/abc/MILC/Data/6928916_rnn_HCP2_ws_1200_wsize_1200_nw_1_seed_1_gain_0.9_tp_1200_samp_830_ep_100_GS_fold_9_ptr_F/PretrainedModel.pt'
                #path_m = '/data/users4/ziqbal5/abc/MILC/Data/6928907_rnn_HCP2_ws_1200_wsize_1200_nw_1_seed_1_gain_0.9_tp_1200_samp_830_ep_100_GS_fold_0_ptr_F/PretrainedModel.pt'
                #path_m = '/data/users4/ziqbal5/abc/MILC/Data/6928915_rnn_HCP2_ws_1200_wsize_1200_nw_1_seed_1_gain_0.9_tp_1200_samp_830_ep_100_GS_fold_8_ptr_F/PretrainedModel.pt'
                #path_m = '/data/users4/ziqbal5/abc/MILC/Data/6928910_rnn_HCP2_ws_1200_wsize_1200_nw_1_seed_1_gain_0.9_tp_1200_samp_830_ep_100_GS_fold_3_ptr_F/PretrainedModel.pt'
                #path_m = '/data/users4/ziqbal5/abc/MILC/Data/6928911_rnn_HCP2_ws_1200_wsize_1200_nw_1_seed_1_gain_0.9_tp_1200_samp_830_ep_100_GS_fold_4_ptr_F/PretrainedModel.pt'
                #path_m = '/data/users4/ziqbal5/abc/MILC/Data/6928912_rnn_HCP2_ws_1200_wsize_1200_nw_1_seed_1_gain_0.9_tp_1200_samp_830_ep_100_GS_fold_5_ptr_F/PretrainedModel.pt'

                
                if args.convsize == 0:
                    path_m ='/data/users4/ziqbal5/abc/MILC/Data_expwithlstminterp/7091964_rnm_HCP2_ws_1200_wsize_1200_nw_1_seed_1_gain_0.9_tp_1200_samp_830_ep_100_IG_fold_0_ptr_F/PretrainedModel.pt'
                if args.convsize == 1:
                    path_m ='/data/users4/ziqbal5/abc/MILC/Data_expwithlstminterp/7091965_rnm_HCP2_ws_1200_wsize_1200_nw_1_seed_1_gain_0.9_tp_1200_samp_830_ep_100_IG_fold_1_ptr_F/PretrainedModel.pt'
                if args.convsize == 2:
                    path_m ='/data/users4/ziqbal5/abc/MILC/Data_expwithlstminterp/7091966_rnm_HCP2_ws_1200_wsize_1200_nw_1_seed_1_gain_0.9_tp_1200_samp_830_ep_100_IG_fold_2_ptr_F/PretrainedModel.pt'
                if args.convsize == 3:
                    path_m ='/data/users4/ziqbal5/abc/MILC/Data_expwithlstminterp/7091967_rnm_HCP2_ws_1200_wsize_1200_nw_1_seed_1_gain_0.9_tp_1200_samp_830_ep_100_IG_fold_3_ptr_F/PretrainedModel.pt'
                if args.convsize == 4:
                    path_m ='/data/users4/ziqbal5/abc/MILC/Data_expwithlstminterp/7091968_rnm_HCP2_ws_1200_wsize_1200_nw_1_seed_1_gain_0.9_tp_1200_samp_830_ep_100_IG_fold_4_ptr_F/PretrainedModel.pt'
                if args.convsize == 5:
                    path_m ='/data/users4/ziqbal5/abc/MILC/Data_expwithlstminterp/7091969_rnm_HCP2_ws_1200_wsize_1200_nw_1_seed_1_gain_0.9_tp_1200_samp_830_ep_100_IG_fold_5_ptr_F/PretrainedModel.pt'
                if args.convsize == 6:
                    path_m ='/data/users4/ziqbal5/abc/MILC/Data_expwithlstminterp/7091970_rnm_HCP2_ws_1200_wsize_1200_nw_1_seed_1_gain_0.9_tp_1200_samp_830_ep_100_IG_fold_6_ptr_F/PretrainedModel.pt'
                if args.convsize == 7:
                    path_m ='/data/users4/ziqbal5/abc/MILC/Data_expwithlstminterp/7091971_rnm_HCP2_ws_1200_wsize_1200_nw_1_seed_1_gain_0.9_tp_1200_samp_830_ep_100_IG_fold_7_ptr_F/PretrainedModel.pt'
                if args.convsize == 8:
                    path_m ='/data/users4/ziqbal5/abc/MILC/Data_expwithlstminterp/7091972_rnm_HCP2_ws_1200_wsize_1200_nw_1_seed_1_gain_0.9_tp_1200_samp_830_ep_100_IG_fold_8_ptr_F/PretrainedModel.pt'
                if args.convsize == 9:
                    path_m ='/data/users4/ziqbal5/abc/MILC/Data_expwithlstminterp/7091976_rnm_HCP2_ws_1200_wsize_1200_nw_1_seed_1_gain_0.9_tp_1200_samp_830_ep_100_IG_fold_9_ptr_F/PretrainedModel.pt'
          
                # if args.convsize == 0:
                #     path_m ='/data/users4/ziqbal5/abc/MILC/Data_old/6060294_rnm_UKB_ws_490_wsize_490_nw_1_seed_7_gain_0.9_tp_490_samp_830_ep_100_GS_fold_0_ptr_F/PretrainedModel.pt'
                # if args.convsize == 1:
                #     path_m ='/data/users4/ziqbal5/abc/MILC/Data_old/6060295_rnm_UKB_ws_490_wsize_490_nw_1_seed_7_gain_0.9_tp_490_samp_830_ep_100_GS_fold_1_ptr_F/PretrainedModel.pt'
                # if args.convsize == 2:
                #     path_m ='/data/users4/ziqbal5/abc/MILC/Data_old/6060296_rnm_UKB_ws_490_wsize_490_nw_1_seed_7_gain_0.9_tp_490_samp_830_ep_100_GS_fold_2_ptr_F/PretrainedModel.pt'
                # if args.convsize == 3:
                #     path_m ='/data/users4/ziqbal5/abc/MILC/Data_old/6060297_rnm_UKB_ws_490_wsize_490_nw_1_seed_7_gain_0.9_tp_490_samp_830_ep_100_GS_fold_3_ptr_F/PretrainedModel.pt'
                # if args.convsize == 4:
                #     path_m ='/data/users4/ziqbal5/abc/MILC/Data_old/6060298_rnm_UKB_ws_490_wsize_490_nw_1_seed_7_gain_0.9_tp_490_samp_830_ep_100_GS_fold_4_ptr_F/PretrainedModel.pt'
                # if args.convsize == 5:
                #     path_m ='/data/users4/ziqbal5/abc/MILC/Data_old/6060299_rnm_UKB_ws_490_wsize_490_nw_1_seed_7_gain_0.9_tp_490_samp_830_ep_100_GS_fold_5_ptr_F/PretrainedModel.pt'
                # if args.convsize == 6:
                #     path_m ='/data/users4/ziqbal5/abc/MILC/Data_old/6060300_rnm_UKB_ws_490_wsize_490_nw_1_seed_7_gain_0.9_tp_490_samp_830_ep_100_GS_fold_6_ptr_F/PretrainedModel.pt'
                # if args.convsize == 7:
                #     path_m ='/data/users4/ziqbal5/abc/MILC/Data_old/6060301_rnm_UKB_ws_490_wsize_490_nw_1_seed_7_gain_0.9_tp_490_samp_830_ep_100_GS_fold_7_ptr_F/PretrainedModel.pt'
                # if args.convsize == 8:
                #     path_m ='/data/users4/ziqbal5/abc/MILC/Data_old/6060302_rnm_UKB_ws_490_wsize_490_nw_1_seed_7_gain_0.9_tp_490_samp_830_ep_100_GS_fold_8_ptr_F/PretrainedModel.pt'
                # if args.convsize == 9:
                #     path_m ='/data/users4/ziqbal5/abc/MILC/Data_old/6060293_rnm_UKB_ws_490_wsize_490_nw_1_seed_7_gain_0.9_tp_490_samp_830_ep_100_GS_fold_9_ptr_F_sel/PretrainedModel.pt'
          

                print("pretrained model loaded from: ", path_m)
                model.load_state_dict(torch.load(path_m, map_location=torch.device('cpu')))    
                
            starttime = datetime.datetime.now()
            optimizer, accuracy_test, auc_test,model = train_model(model, dataLoaderTrain, dataLoaderValidation, dataLoaderTest, eppochs, args.lr)#3e-4  #.0005
            
            endtime = datetime.datetime.now()
            elapsed_time = endtime-starttime

            torch.save(model.state_dict(),  os.path.join(path_a, 'PretrainedModel.pt'))  
 
            if torch.cuda.is_available():
                
                #print(f'Allocated: {torch.cuda.memory_allocated()}')
                del model
                gc.collect()
                torch.cuda.empty_cache()
                #print(f'Allocated: {torch.cuda.memory_allocated()}')
                
            

    #accDataFrame = pd.DataFrame(accMat, aucMat)
    accDataFrame = pd.DataFrame({'Accuracy':[accuracy_test], 'AUC':[auc_test]})
    #print('df', accDataFrame)
    accfname = os.path.join(path_a, 'ACC_AUC.csv')
    accDataFrame.to_csv(accfname)

 

print('Total Time for training (seconds) :', elapsed_time.seconds) # could be .microseconds, .days
print("Accuracy and AUC on Test Data :", accuracy_test, auc_test)

# ################################################################################################
# ################################################################################################

def Load_Data():
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
            model = LSTM(components, 256, 200, 121, 2, g, onewindow).float()

    path_m = path_models


    model.load_state_dict(torch.load(path_m))
    #model.load_state_dict(torch.load(path_m)['model_state_dict'])
    print("Model trained on synthetic data loaded from: ", path_m)

    return model

def Predicted_Labels(model, TestDataFD, LabelsFD):
        #Load Model pretrained weights
    TestDataFD = torch.from_numpy(TestDataFD).float()
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
        print(LCP[0])
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
                ac = accuracy_score(LabelsFD.cpu(), predsFD.cpu())
                print("accuracyforInterp: ", ac)
                Feature_Attributions(model, TestDataFD,LabelsFD, predsFD)
