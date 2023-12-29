import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout=0., act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj

class CL_Model(nn.Module):
    def __init__(self, input_feat_dim_rna, gae_hidden1_dim, gae_hidden2_dim, gcn_encoder_dropout, gcn_decoder_dropout, cl_emb_dim, image_encoder):
        super(CL_Model, self).__init__()

        self.input_feat_dim_rna = input_feat_dim_rna
        self.cl_emb_dim = cl_emb_dim                         
        self.gae_hidden1_dim = gae_hidden1_dim               
        self.gae_hidden2_dim = gae_hidden2_dim               
        self.gcn_encoder_dropout = gcn_encoder_dropout
        self.gcn_decoder_dropout = gcn_decoder_dropout
        self.image_encoder_features = image_encoder.fc.in_features
        
        #image encoder        
        self.image_encoder_f = []
        #load resnet structure
        for name, module in image_encoder.named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.image_encoder_f.append(module)
        self.image_encoder_f = nn.Sequential(*self.image_encoder_f)
        self.fc_image = nn.Linear(self.image_encoder_features, self.cl_emb_dim)

        #rna encoder
        self.gc1_rna = GraphConvolution(self.input_feat_dim_rna, self.gae_hidden1_dim, dropout=self.gcn_encoder_dropout, act=F.tanh)
        self.gc2_rna = GraphConvolution(self.gae_hidden1_dim, self.gae_hidden2_dim, dropout=self.gcn_encoder_dropout, act=F.tanh)
        self.fc_rna = nn.Linear(self.gae_hidden2_dim, self.cl_emb_dim)

        #rna decoder
        self.decoder = InnerProductDecoder(dropout=self.gcn_decoder_dropout, act=lambda x: x)

    def forward(self, x, adj, x_image):
    
        #GCN part
        hidden1_rna = self.gc1_rna(x, adj)
        hidden2_rna = self.gc2_rna(hidden1_rna, adj)
        rna_emb = self.fc_rna(hidden2_rna)

        #Resnet part
        x_image_emb = self.image_encoder_f(x_image)
        image_encoder_emb = torch.flatten(x_image_emb, start_dim=1)
        image_emb = self.fc_image(image_encoder_emb) 

        return rna_emb, hidden2_rna, None, image_emb, image_encoder_emb, None

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # TODO
        self.dropout = dropout
        # self.dropout = Parameter(torch.FloatTensor(dropout))
        self.act = act
        # self.weight = Parameter(torch.DoubleTensor(in_features, out_features))
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'