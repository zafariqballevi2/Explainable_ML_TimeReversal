import numpy as np
import torch
from pathlib import Path
import pickle
import mat73
import h5py
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.signal import chirp
from utils import get_argparser
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import cm, colors, pyplot as plt

parser = get_argparser()
args = parser.parse_args()


class Datasets(object):

    def __init__(self,fold):
        self.fold = fold

    def FBIRN(self):
        path = Path('/data/users4/ziqbal5/DataSets/FBIRN/FBIRN_Pavel.h5')  #Pavel
        #path = Path('/data/users4/ziqbal5/DataSets/FBIRN/FBIRN.pickle')    #Old ones
        if path.is_file() == True:
 
            hf = h5py.File(path, "r")
            data = hf.get("FBIRN_dataset")
            data = np.array(data)
            print(data.shape)
            data = data.reshape(data.shape[0], 100, -1)

            # with open(path, "rb") as infile:
            #     data = pickle.load(infile)
            # print(data.shape)

        else:
            AllData = []
            for i in tqdm(range(1, args.samples+1)):
                filename = '/data/users2/zfu/Matlab/GSU/Neuromark/Results/ICA/FBIRN/FBIRN_ica_br'+str(i)+'.mat'
                data = mat73.loadmat(filename)
                data = data['compSet']['tc']
                data = data.T
                AllData.append(data[:, 0:args.tp])
            print('All subjects loaded successfully...')
            data = np.stack(AllData, axis=0)
            with open(path, "wb") as outfile:
                pickle.dump(data, outfile)

            

        ds = '/data/users4/ziqbal5/DataSets/FBIRN/transform_to_correct_GSP.csv'
        cor_ind = pd.read_csv(ds, header=None)
        cor_ind = cor_ind[1] # if you use 1 here then those are the indices coming from Dr. Fu.
        cor_ind = cor_ind.to_numpy()
        print(cor_ind)
    
        data2 = np.zeros((args.samples, 53, args.tp))
        for i in range(len(data)):
            temp = data[i]
            data2[i] = temp[cor_ind]

        #import labels
        filename = '/data/users4/ziqbal5/DataSets/FBIRN/sub_info_FBIRN.mat'
        lab = mat73.loadmat(filename)
        ab = torch.empty((len(lab['analysis_SCORE']), 1))
        for i in range(len(lab['analysis_SCORE'])):
            a = lab['analysis_SCORE'][i][2]
            ab[i][0] = a
        labels = torch.squeeze(ab)

        labels[labels==1] = 0
        labels[labels==2] = 1
        return self.split_folds(data2, labels)

    def COBRE(self):
        hf = h5py.File('/data/users4/ziqbal5/abc/MILC/data/COBRE_AllData.h5', 'r')
        data = hf.get('COBRE_dataset')
        data = np.array(data)
        data = data.reshape(len(data), 100, 140)
        
        ds = '/data/users4/ziqbal5/abc/MILC/data/bsnip/correct_indices_GSP.csv'
        cor_ind = pd.read_csv(ds, header=None)
        
        indices = pd.read_csv(ds, header=None)
        idx = indices[0].values - 1
        data = data[:, idx, :]

        # fig = plt.figure(figsize=(20, 2))
        # plt.plot(data[0].T)
        # plt.xticks([])
        # plt.yticks([])
        # plt.savefig('abc.png',transparent=True)
        # kckc
        filename = '/data/users4/ziqbal5/abc/MILC/data/labels_COBRE.csv'
        df = pd.read_csv(filename, header=None)
        all_labels = df.values
        all_labels = torch.from_numpy(all_labels).int()
        all_labels = all_labels.view(len(data))
        labels = all_labels - 1
       
        print("HC: ", len(np.where(labels == 0)[0]), "SZ: ",  len(np.argwhere(labels == 1)[0]))
        labels = torch.Tensor(labels)
        return self.split_folds(data, labels)
        
    def BSNIP(self):
        
        with np.load('/data/users4/ziqbal5/DataSets/bsnip/BSNIP_data.npz') as npzfile:
            data = npzfile['features']
            labels = npzfile['labels']

  
        #print(labels)
        #taking HC and SZ only. dropping the rest fo the classes
        ind = np.argwhere((labels== 0) | (labels == 1))

        data = data[ind]
        data = np.squeeze(data)
        labels = labels[ind]
  
        
        ds = '/data/users4/ziqbal5/DataSets/bsnip/transform_to_correct_GSP.csv'
        cor_ind = pd.read_csv(ds, header=None)
        
        indices = pd.read_csv(ds, header=None)
        idx = indices[0].values - 1
        data = data[:, idx, :]
        print(data.shape)
        print("HC: ", len(np.argwhere(labels == 0)), "SZ: ",  len(np.argwhere(labels == 1)))
        labels = torch.Tensor(labels)
        labels = torch.squeeze(labels)
        return self.split_folds(data, labels)
    
    def split_folds(self, data,labels):
        Adata = torch.Tensor(data)
      
        data = np.zeros((len(Adata), args.nw, 53, args.wsize))
        
        for i in range(len(Adata)):
            for j in range(args.nw):
                data[i, j, :, :] = Adata[i, :, (j * args.ws):(j * args.ws) + args.wsize]
        
       
        skf = StratifiedKFold(n_splits=10, random_state = 34, shuffle = True) #default 33
        skf.get_n_splits(data, labels)
        Folds_train_ind = []
        Folds_test_ind = []
        for i, (train_index, test_index) in enumerate(skf.split(data, labels)):
            Folds_train_ind.append(train_index)
            Folds_test_ind.append(test_index)
        
        #Folds_train_ind[self.fold], Folds_test_ind[self.fold]

        train_data, train_labels = data[Folds_train_ind[self.fold]], labels[Folds_train_ind[self.fold]]
        test_data, test_labels = data[Folds_test_ind[self.fold]], labels[Folds_test_ind[self.fold]] 
        tr_data, val_data, tr_labels, val_labels = train_test_split(train_data,train_labels , 
                                    random_state=100,  
                                    test_size=0.20,  
                                    shuffle=True, stratify=train_labels)
        
        return tr_data, tr_labels, val_data, val_labels,test_data, test_labels
        

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class LSTM(torch.nn.Module):
    #model = LSTM(X.shape[2], 256, 200, 121, 2, g)
    def __init__(self, enc_input_size, input_size, hidden_nodes, sequence_size, output_size, gain, onewindow):
        super(LSTM, self).__init__()
        self.sequence_size = sequence_size
        self.hidden = hidden_nodes
        self.onewindow = onewindow
        self.enc_out = input_size
        self.lstm = nn.LSTM(input_size, hidden_nodes, batch_first=True) #not using this for rnm. Fro rnn, using this and the encoder lstm layer as well.
        
        # input size for the top lstm is the hidden size for the lower
        
        self.encoder = nn.LSTM(enc_input_size, self.enc_out, batch_first = True)  #using only this for rnm

        # previously, I used 64
        self.attnenc = nn.Sequential(
             nn.Linear(self.enc_out, 64),
             nn.Linear(64, 1)
        )
     
        self.attn = nn.Sequential(
            nn.Linear(2*self.hidden, 128),
            nn.Linear(128, 1)
        )
        
        # Previously it was 64, now used 200
        
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden, 200),
            nn.Linear(200, output_size)
        )
        self.decoder2 = nn.Sequential(
   
            nn.Linear(256, 200),
            nn.Linear(200, output_size), 
            
            # #nn.Dropout(.10),
            # nn.Linear(256, 200),
            # #nn.LeakyReLU(0.1),
            # nn.Linear(200, 100),
            # nn.Linear(100, 50),            
            # nn.Linear(50, output_size),
            # nn.Sigmoid()

        )
        # self.dec1 = nn.Sequential(
        # nn.Linear(256,1),
        # nn.ReLU(),
        # )

        # self.dec2 = nn.Sequential(
        # nn.Linear(140,60),
        # nn.ReLU(),
        # nn.Dropout(0.2),
        # nn.Linear(60, output_size),
        # nn.Sigmoid() 

        # )
        
        
        self.gain = gain
        
        self.init_weight()

        
    def init_weight(self):
        
        # For UFPT experiments, initialize only the decoder
        print('Initializing fresh components')
        for name, param in self.encoder.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)
        for name, param in self.attnenc.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)
        for name, param in self.decoder.named_parameters():
            #print('Initializing Decoder:', name)
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)
        for name, param in self.decoder2.named_parameters():
            #print('Initializing Decoder:', name)
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)
        for name, param in self.attn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)

    def forward(self, x):
        #print("input", x.shape)       
        b_size = x.size(0)
        s_size = x.size(1)
        #print('xx', x.shape)
        x = x.view(-1, x.shape[2], args.wsize)
        #print("input2", x.shape)
        x = x.permute(0, 2, 1)
        
        #print('inputtoencoder: ', x.shape)
        
        out, hidden = self.encoder(x)
        #print('outputofencoder: ', out.shape)
        
        
        out = self.get_attention_enc(out)
        #print('enc_attn_output: ', out.shape)
        out = out.view(b_size, s_size, -1)
   

        # ######S: for one window ######        
        if self.onewindow == True:
            out = out.squeeze()
            lstm_out = self.decoder2(out)
            print(lstm_out.shape)
            
        # ######E: for one window ######
        else:
            lstm_out, hidden = self.lstm(out)
            #print('ext_lstm_output: ', lstm_out.shape)
            #lstm_out = out
            lstm_out = self.get_attention(lstm_out)
            #print("ext_attention_output: ",lstm_out.shape)
            lstm_out = lstm_out.view(b_size, -1)
            
            smax = torch.nn.Softmax(dim=1)

            lstm_out_smax = smax(lstm_out)
        #print("lstm_out", lstm_out.shape)
        return lstm_out #lstm_out_smax    #lstm_out_smax
        

    def get_attention(self, outputs):
        
        B= outputs[:,-1, :]
        #print('outputs: ', outputs.shape)
        B = B.unsqueeze(1).expand_as(outputs)
        outputs2 = torch.cat((outputs, B), dim=2)
        # For attention calculation
        b_size = outputs2.size(0)
        # out = outputs.view([-1, self.hidden])
        out = outputs2.reshape(-1, 2*self.hidden)
        #print("out", out.shape)
        weights = self.attn(out)
        #print("weigh", weights.shape)
        
        weights = weights.view(b_size, -1, 1)
        weights = weights.squeeze(2)
        #print('weights', weights.shape)

        # SoftMax normalization on weights
        normalized_weights = F.softmax(weights, dim=1)
        # Batch-wise multiplication of weights and lstm outputs
        #print("bmm calcualted between : ", normalized_weights.shape, outputs.shape)
        attn_applied = torch.bmm(normalized_weights.unsqueeze(1), outputs)
        #print("attapp", attn_applied.shape)
        attn_applied = attn_applied.squeeze()
        logits = self.decoder(attn_applied)


        return logits
    
    def get_attention_enc(self, outputs):
        
        b_size = outputs.size(0)

        ##### Start: Implementation of average hidden states and last hidden state
        # out = outputs

       
        # Mean_hidden_states = torch.mean(out,1,True)
        # for i in range(len(outputs)):
        #     out[i] = Mean_hidden_states[i]
        
        # # last_hidden_state = outputs[:, -1, :]
        # # last_hidden_state = last_hidden_state.unsqueeze(1)
        # # for i in range(len(outputs)):
        # #     out[i] = last_hidden_state[i]
        #out = out.reshape(-1, self.enc_out)
        ##### End: Implementation of average hidden states and last hidden state
        
        
        out = outputs.reshape(-1, self.enc_out)
        #print('1:',out.shape)
        
        weights = self.attnenc(out)
        #print('weights', weights.shape)
        weights = weights.view(b_size, -1, 1)
        weights = weights.squeeze(2)
        normalized_weights = F.softmax(weights, dim=1)
        #print('norm_weights', normalized_weights.shape)
        attn_applied = torch.bmm(normalized_weights.unsqueeze(1), outputs)
        #print("attn_appliedd", attn_applied.shape)
        attn_applied = attn_applied.squeeze()
        return attn_applied
    

class LSTM2(torch.nn.Module):
    #model = LSTM(X.shape[2], 256, 200, 121, 2, g)
    def __init__(self, enc_input_size, input_size, hidden_nodes, sequence_size, output_size, gain, onewindow):
        super(LSTM2, self).__init__()
        self.sequence_size = sequence_size
        self.hidden = input_size
        self.onewindow = onewindow
        self.enc_out = input_size
        
    
        self.encoder = nn.LSTM(enc_input_size, self.enc_out, batch_first = True)
        #self.encoder = nn.GRU(enc_input_size, self.enc_out, batch_first = True)
     
        self.attn = nn.Sequential(
            nn.Linear(2*self.hidden, 128),
            nn.Linear(128, 1)
        )

        self.attention = nn.MultiheadAttention(embed_dim=self.hidden, 
                                               num_heads=8,
                                               batch_first=True, 
                                               dropout=0.1)
        
        # Previously it was 64, now used 200
        
        #this is the one with 81.64 accuracy in fbirn
        # self.decoder = nn.Sequential(
            
        #     nn.Linear(self.hidden, 200),
        #     nn.Dropout(.20),
        #     nn.Linear(200, output_size),
        #     nn.Sigmoid()
        # )
        
        self.decoder = nn.Sequential(
            
            nn.Linear(self.hidden, 200),
            nn.Dropout(.20),
            nn.Linear(200, output_size),
            nn.Sigmoid()
        )
 

        # self.decoder = nn.Sequential(

        #     #nn.Dropout(.10),
        #     nn.Linear(self.hidden, 200),
        #     #nn.LeakyReLU(0.1),
        #     nn.Linear(200, 100),
        #     nn.Linear(100, 50),            
        #     nn.Linear(50, output_size),
        #     nn.Sigmoid()
        # )


        # self.mlp_emb = nn.Sequential(nn.Linear(args.tp, args.tp),
        #                              nn.LayerNorm(args.tp),
        #                              nn.ELU(),
        #                              nn.Linear(args.tp, 1))      
        
        self.gain = gain
        
        self.init_weight()

        
    def init_weight(self):
        
        # For UFPT experiments, initialize only the decoder
        print('Initializing fresh components')
        for name, param in self.encoder.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)
        for name, param in self.decoder.named_parameters():
            #print('Initializing Decoder:', name)
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)
        for name, param in self.attn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)

    def forward(self, x):
              
        b_size = x.size(0)
        s_size = x.size(1)
        x = x.view(-1, x.shape[2], args.wsize)
        x = x.permute(0, 2, 1)
        out, hidden = self.encoder(x)
       
        
        ######MultiheadAttention#############
        # hidden_seq = []
        # hidden_seq += [out]
        # hidden_cat = torch.cat(hidden_seq, 1)
        # attn_output, attn_output_weights = self.attention(out, hidden_cat, hidden_cat)  # Q, K, V
        # attn_output = attn_output + out                
        # attn_output = attn_output.permute(0, 2, 1) 
        # out = self.mlp_emb(attn_output)
        # out = torch.squeeze(out)
        # lstm_out = self.decoder(out)
        ######MultiheadAttention#############

        lstm_out = self.get_attention(out)
        

        lstm_out = lstm_out.view(b_size, -1)
        
    
        return lstm_out
        

    def get_attention(self, outputs):
        ##########
                #select the last hidden state for attention
        B= outputs[:,-1, :]
                #select average of all hidden states for attention
        #B = torch.mean(outputs, 1, True).squeeze()
        ##########

        B = B.unsqueeze(1).expand_as(outputs)
        
        outputs2 = torch.cat((outputs, B), dim=2)
        
        # For attention calculation
        b_size = outputs2.size(0)
        # out = outputs.view([-1, self.hidden])
        out = outputs2.reshape(-1, outputs2.shape[2])
        #print("out", out.shape)
        weights = self.attn(out)
        #print("weights1: ", weights.shape)
        
        weights = weights.view(b_size, -1, 1)
        weights = weights.squeeze(2)
        #print('weights2: ', weights.shape)

        # SoftMax normalization on weights
        normalized_weights = F.softmax(weights, dim=1)
        # Batch-wise multiplication of weights and lstm outputs
        #print("bmm calcualted between : ", normalized_weights.shape, outputs.shape)
        attn_applied = torch.bmm(normalized_weights.unsqueeze(1), outputs)
        #print("attention_output: ", attn_applied.shape)
        attn_applied = attn_applied.squeeze()
        logits = self.decoder(attn_applied)
        #print("decoder output: ", logits.shape)


        return logits
    

