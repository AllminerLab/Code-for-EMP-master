import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn

class MLP(nn.Module):
    def __init__(self, num_layers, in_size, hidden_size, out_size):
        """
            num_layers: number of layers in MLP, if num_layers=1, then MLP is a Linear Model.
            in_size: dimensionality of input features
            hidden_size: dimensionality of hidden units at ALL layers
            out_size: number of classes for prediction
        """
        super(MLP, self).__init__()
        
        self.if_linear = True # default is True, which return a linear model.
        self.num_layers = num_layers
        
        if self.num_layers == 1:
            # a linear model
            self.Linear = nn.Linear(in_size, out_size)
        else:
            # MLP
            self.if_linear = False
            self.Linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
            
            self.Linears.append(nn.Linear(in_size, hidden_size))
            for layer in range(self.num_layers-2):
                self.Linears.append(nn.Linear(hidden_size, hidden_size))
            self.Linears.append(nn.Linear(hidden_size, out_size))
            
            # Batch Norms
            for layer in range(self.num_layers-1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_size)))
    
    def forward(self, inputs):
        if self.if_linear:
            return self.Linear(inputs)
        else:
            h = inputs
            for layer in range(self.num_layers-1):
                h = F.relu(self.batch_norms[layer](self.Linears[layer](h)))
            return self.Linears[-1](h)

class HGINLayer(nn.Module):
    def __init__(self, paths, in_size, out_size, num_mlp_layers, target_node_type):
        super(HGINLayer, self).__init__()
        self.paths = list(tuple(path) for path in paths)
        self.go_path = list( path[0] for path in self.paths)
        self.return_path = list( path[1] for path in self.paths)
        self.num_mlp_layers = num_mlp_layers
        self.target_node_type = target_node_type
        # construct parameters.

        # create a project layer for target-node-type
        self.project = nn.Linear(in_size, out_size)
        # create MLP for each meta-path's Go-Relation.
        self.MLP = nn.ModuleDict({
            name: MLP(num_mlp_layers, in_size, out_size, out_size) for name in self.go_path
        })
        # one-more MLP for Return-Message-Passing.
        self.MLP['Return'] = MLP(num_mlp_layers, out_size, out_size, out_size)
        # eps
        eps_dict = {name: nn.Parameter(torch.zeros(1)) for name in self.go_path}
        eps_dict['Return'] = nn.Parameter(torch.zeros(1))
        self.eps = nn.ParameterDict(eps_dict)
        
    def forward(self, G, feat_dict):
        """
            G: is a Heterogenous Graph in DGL.
            feat_dict: is a dictionary of node features for each type.
        """
        with G.local_scope():
            # set features for all nodes. 
            for ntype in G.ntypes:
                G.nodes[ntype].data['h'] = feat_dict[ntype]

            # GNN for GOPATH.
            for path in self.go_path:
                srctype, etype, dsttype = G.to_canonical_etype(path)
                G.send_and_recv(G[etype].edges(), fn.copy_src('h', 'm'),
                               fn.sum('m', 'a'), etype=etype)
                G.apply_nodes(lambda nodes:{'h': self.MLP[etype]((1+self.eps[etype])*nodes.data['h'] 
                                                                 + nodes.data['a'])}, ntype=dsttype)

            # GNN for RETUANPATH
            funcs = {}
            for path in self.return_path:
                srctype, etype, dsttype = G.to_canonical_etype(path)
                funcs[etype] = (G[etype].edges(), fn.copy_src('h', 'm'), fn.sum('m', 'a'))
            G.multi_send_and_recv(funcs, "sum")
            G.apply_nodes(lambda nodes: {'h': self.project(nodes.data['h'])}, ntype=self.target_node_type)
            G.apply_nodes(lambda nodes: {'h': self.MLP['Return']((1+self.eps['Return'])*nodes.data['h'] 
                                                                 + nodes.data['a'])}, ntype=self.target_node_type)
            return {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}
    
class HGIN(nn.Module):
    def __init__(self, G, features, target_node_type, paths, in_size, hidden_size, out_size, num_layers, num_mlp_layers, dropout, device):
        """
            features: input features, which is a dictionary of node features for each type
        """
        super(HGIN, self).__init__()
        self.device = device
        self.dropout = dropout
        # Use trainable node embeddings as featureless inputs.
        embed_dict = {}
        self.target_node_type = target_node_type
        for ntype in G.ntypes:
            if ntype == target_node_type:
                embed_dict[ntype] = features
            else:
                #embed_dict[ntype] = nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), in_size)).to(self.device)
                embed_dict[ntype] = torch.Tensor(G.number_of_nodes(ntype), in_size).to(self.device)
                nn.init.xavier_uniform_(embed_dict[ntype]) # initialization ramdomly
                
        self.embed = embed_dict
        
        # Create layers.
        self.layers = nn.ModuleList()
        self.layers.append(HGINLayer(paths, in_size, hidden_size, num_mlp_layers, target_node_type))
        for i in range(1,num_layers):
            self.layers.append(HGINLayer(paths, hidden_size, hidden_size, num_mlp_layers, target_node_type))
            
        self.predict = nn.Linear(hidden_size, out_size)
    
    def forward(self, G):
        h_dict = {ntype: F.dropout(embed, self.dropout, training=self.training) for ntype, embed in self.embed.items()}
        for gnn in self.layers:
            h_dict = gnn(G, h_dict)
            h_dict = {ntype: F.leaky_relu(F.dropout(embed, self.dropout, training=self.training)) for ntype, embed in h_dict.items()}
            #h_dict = {ntype: F.leaky_relu(embed) for ntype, embed in h_dict.items()}
        embeds = h_dict[self.target_node_type]

        return embeds, self.predict(h_dict[self.target_node_type])