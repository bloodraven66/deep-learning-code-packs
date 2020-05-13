import os
import sys
import time
import torch
import pickle
import librosa
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
from itertools import groupby
from operator import itemgetter
from python_speech_features import mfcc

class EncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout = 0):
        super(EncoderRNN, self).__init__()

        self.lstm = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.lstm(x)
        return self.relu(x[0])
class DecoderRNN(nn.Module):

    def __init__(self, hidden_size, num_layers, input_size):
        super(DecoderRNN, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, encoded_input):
        decoded_output, hidden = self.lstm(encoded_input)
        decoded_output = self.sigmoid(decoded_output)
        return decoded_output

class LSTMAutoEncoder(nn.Module):

    def __init__(self, input_size=99, hidden_size=512, num_layers=2, cuda=True):
        super(LSTMAutoEncoder, self).__init__()
        self.encoder = EncoderRNN(input_size, hidden_size, num_layers)
        self.linear = nn.Linear(512, 64)
        self.decoder = DecoderRNN(hidden_size=99, input_size=64, num_layers=num_layers)

        if cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()

    def forward(self, input):
        input = self.encoder(input)
        decoded_output = self.decoder(self.linear(input))
        return decoded_output

class nextModel(nn.Module):

    def __init__(self):
        super(nextModel, self).__init__()
        self.encoder = model.encoder
        self.linear = nn.Linear(512, 2)

    def forward(self, x):
        x = self.encoder(x)
        r_out2 = self.linear(x[:, -1, :])
        return F.log_softmax(r_out2, dim=1)

class Process():

    def __init__(self):

        self.sr = 22050
        self.nfilt = 26
        self.nfeat = 13
        self.nfft = 2205
        self.window_size = 1
        self.sliding=True
        self.slide_window_size = 0.5

    def load_data(self, path):
        print('Loading voice file..')
        if path.endswith('.wav'):
                    y, sr = librosa.load(path)
                    data_ = self.chunks(signal=y,
                                        rate=sr,
                                        window_size=self.window_size,
                                        sliding=self.sliding,
                                        slide_window_size=self.slide_window_size)
                    return data_
        else:
            print('invalid input')
            return None


    def chunks(self, signal, rate, window_size, sliding=False, slide_window_size=0):
        data = []
        chunk_size = rate*window_size
        if sliding == False:
            n_chunks = int(len(signal)/(chunk_size)) + 1
            for i in range(n_chunks):
                    chunk = signal[int(i*chunk_size):int((i+1)*chunk_size)]
                    features = mfcc(chunk, rate, numcep = self.nfeat, nfilt = self.nfilt, nfft = self.nfft)
                    data.append(features)
            return data

        elif sliding == True:
            slide_size = rate*slide_window_size
            n_chunks = int((len(signal)-chunk_size)/slide_size)
            for i in range(n_chunks):
                chunk = signal[int(i*slide_size) : int(chunk_size+(i*slide_size))]
                features = mfcc(chunk, rate, numcep = self.nfeat, nfilt = self.nfilt, nfft = self.nfft)
                data.append(features)
            return data

def output(model, train_loader):
    values = []
    continous = []
    upper_bounds = []
    result = {}
    label = ['normal', 'abnormal']
    for data_ in train_loader:
        data_ = data_[0].double()
        data_ = data_.view(data_.shape[0], data_.shape[2], data_.shape[1])
        data_ = data_.cuda()
        out = model(data_)
        pred = out.data.max(1, keepdim=True)[1]
        values.extend(pred.detach().cpu().numpy())
    idx, count = np.unique(values, return_counts=True)
    res_list = [i for i, value in enumerate(values) if value == np.array(1)]
    for k, g in groupby(enumerate(res_list), lambda ix : ix[0] - ix[1]):
        continous.append(list(map(itemgetter(1), g)))
    for val in continous:
        print(val[0]*0.5, ':',val[-1]*0.5+1)



        # print(val*0.5, ':',val*0.5+1)
        # upper_bounds.append(val*0.5+1)
    for i in range(len(idx)):
        result[label[idx[i]]] = count[i]
        print(label[idx[i]],':',count[i])

    return upper_bounds, result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', dest='path', default='normal1.wav')
    args = parser.parse_args()

    path = '../data/test_data/' + args.path
    print(args.path)

    model = LSTMAutoEncoder(cuda=False).double()
    newModel = nextModel()
    newModel.load_state_dict(torch.load("../models/stutter_noPreTraining_mar16_moreData_holdSome.pth"))
    newModel.to('cuda:0')
    newModel.double()
    process = Process()
    data = process.load_data(path)
    if data is not None:
        speech_dataset = utils.TensorDataset(torch.from_numpy(np.array(data)))
        train_loader = torch.utils.data.DataLoader(speech_dataset, batch_size=2)

        upper_bounds, result = output(newModel, train_loader)
