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
        self.dropout = dropout
        
        # Input projection with normalization and scaling
        self.input_proj = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout)
        )
        
        # Edge projection with normalization and scaling
        self.edge_proj = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout)
        )
        
        # Main transformer layers with correct dimensions
        for _ in range(num_layers):
            self.convs.append(TransformerConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,  # Changed from hidden_channels//num_heads
                heads=num_heads,
                edge_dim=hidden_channels,
                dropout=dropout,
                bias=False,  # Disable bias for stability
                concat=True  # Changed to True to maintain dimensions
            ))
        
        # Output projection with normalization and scaling
        self.output_proj = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * num_heads, hidden_channels),  # Account for concatenated heads
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, out_channels),
            torch.nn.LayerNorm(out_channels),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout)
        )
        
        # Initialize weights with smaller values
        self.apply(self._init_weights)
        
        # Add gradient clipping hooks
        for p in self.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -0.01, 0.01))
        
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            # Use smaller initialization for stability
            torch.nn.init.xavier_uniform_(module.weight, gain=0.001)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(0.1)  # Smaller scale for LayerNorm

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for module in self.input_proj:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        for module in self.edge_proj:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        for module in self.output_proj:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def forward(self, x, adj_t, edge_attr):
        # Input normalization and projection
        x = F.normalize(x, p=2, dim=-1)  # Normalize input features
        edge_attr = F.normalize(edge_attr, p=2, dim=-1)  # Normalize edge features
        
        x = self.input_proj(x)
        edge_attr = self.edge_proj(edge_attr)
        
        # Store initial values for residual connections
        residual = x
        
        # Apply transformer layers with residual connections and scaling
        for i, conv in enumerate(self.convs):
            # Apply transformer layer
            x_new = conv(x, edge_index=adj_t, edge_attr=edge_attr)
            
            # Scale down the output
            x_new = x_new * 0.1
            
            # Add residual connection with scaling
            if i > 0:  # Skip first layer for residual
                x = x_new + residual * 0.1
                residual = x
            else:
                x = x_new
            
            # Apply activation and dropout
            x = F.gelu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Normalize after each layer
            x = F.normalize(x, p=2, dim=-1)
        
        # Output projection
        x = self.output_proj(x)
        
        # Final normalization
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
