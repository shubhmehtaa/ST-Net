from torch import nn
import torch
import numpy as np


class SpatialCNN(nn.Module):
    def __init__(self):
        super(SpatialCNN, self).__init__()
        self.spatial_feature = nn.Sequential(
            nn.Conv3d(  
                in_channels=1, out_channels=8, kernel_size=(3,3,1), stride=1, padding=(1,1,0)),     #7
            nn.ReLU(),
            nn.Conv3d(  
                in_channels=8, out_channels=16, kernel_size=(3,3,1), stride=1),     
            nn.ReLU(),            
            nn.Conv3d(
                in_channels=16, out_channels=32, kernel_size=(3,3,1), stride=1),    
            nn.ReLU(),
            nn.Conv3d(
                in_channels=32, out_channels=64, kernel_size=(3,3,1), stride=1),    
            nn.ReLU(),
        )

    def forward(self, dce_spatial, data_ref):
        dce_spatial = dce_spatial.type(torch.cuda.FloatTensor)  
        data_ref = data_ref.type(torch.cuda.FloatTensor)  
        return torch.cat([np.squeeze(self.spatial_feature(dce_spatial)), data_ref], dim=1)


class CNNdFeature(nn.Module):
    def __init__(self):
        super(CNNdFeature, self).__init__()
        self.feature_extra = nn.Sequential(
            nn.Conv1d(
                in_channels=66, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.local = nn.Sequential(
            nn.Conv1d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.Conv1d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.Conv1d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.Conv1d(
                in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
        )
        self.wide = nn.Sequential(
            nn.Conv1d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),

            nn.Conv1d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3, dilation=3), 
            nn.ReLU(),

            nn.Conv1d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=8, dilation=8),  
            nn.ReLU(),

            nn.Conv1d(
                in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=25, dilation=25), 
            nn.ReLU(),
        )


    def forward(self, in_data):
        in_data = in_data.type(torch.cuda.FloatTensor)  
        cnn_feature = self.feature_extra(in_data)       
        local_feature = self.local(cnn_feature)         
        wide_feature = self.wide(cnn_feature)           
        return torch.cat([local_feature, wide_feature], dim=1)

class Prediction(nn.Module):
    def __init__(self):
        super(Prediction, self).__init__()
        self.merge = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
        )
        self.pre = nn.Sequential(
            nn.Dropout(0.5),        
            nn.Linear(in_features=64*84, out_features=256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),        
            nn.Linear(in_features=256, out_features=1),
        )

    def forward(self, feature):
        merge_feature = self.merge(feature)    
        out = self.pre(merge_feature.view(merge_feature.size(0), 64*84))  
        return out


class CNN_ST(nn.Module):
    def __init__(self):
        super(CNN_ST, self).__init__()
        self.spatial = SpatialCNN()
        self.temporal = CNNdFeature()
        self.predict = Prediction()
    def forward(self, *args):
        return self.predict(self.temporal(self.spatial(*args)))