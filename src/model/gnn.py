import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TransformerConv, GATConv


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=-1):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t, edge_attr):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x, edge_attr


class GraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=-1):
        super(GraphTransformer, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(TransformerConv(in_channels=in_channels, out_channels=hidden_channels//num_heads, heads=num_heads, edge_dim=in_channels, dropout=dropout))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.LayerNorm(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(in_channels=hidden_channels, out_channels=hidden_channels//num_heads, heads=num_heads, edge_dim=in_channels, dropout=dropout,))
            self.bns.append(torch.nn.LayerNorm(hidden_channels))
        self.convs.append(TransformerConv(in_channels=hidden_channels, out_channels=out_channels//num_heads, heads=num_heads, edge_dim=in_channels, dropout=dropout,))
        self.dropout = dropout
        
        self.input_norm = torch.nn.LayerNorm(in_channels)
        self.edge_norm = torch.nn.LayerNorm(in_channels)
        self.output_norm = torch.nn.LayerNorm(out_channels)
        
        self.residual_norms = torch.nn.ModuleList([
            torch.nn.LayerNorm(hidden_channels) for _ in range(num_layers - 1)
        ])
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=0.01)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.input_norm.reset_parameters()
        self.edge_norm.reset_parameters()
        self.output_norm.reset_parameters()
        for norm in self.residual_norms:
            norm.reset_parameters()

    def forward(self, x, adj_t, edge_attr):
        x = self.input_norm(x)
        edge_attr = self.edge_norm(edge_attr)
        
        residual = x
        
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index=adj_t, edge_attr=edge_attr)
            
            if i > 0:
                x = x + self.residual_norms[i-1](residual)
                residual = x
            
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index=adj_t, edge_attr=edge_attr)
        x = self.output_norm(x)
        
        x = F.normalize(x, p=2, dim=-1)
        edge_attr = F.normalize(edge_attr, p=2, dim=-1)
        
        return x, edge_attr

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=4):
        super(GAT, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, concat=False))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels, heads=num_heads, concat=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GATConv(hidden_channels, out_channels, heads=num_heads, concat=False))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index=edge_index, edge_attr=edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x,edge_index=edge_index, edge_attr=edge_attr)
        return x, edge_attr


load_gnn_model = {
    'gcn': GCN,
    'gat': GAT,
    'gt': GraphTransformer,
}
