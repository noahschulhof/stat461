import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, RGCNConv
import networkx as nx
import matplotlib.pyplot as plt
import torch_geometric as pyg
from parameters import GATAEParameters, RSTAEParameters, STAEParameters, GATSTAEParameters, GraphAEParameters, TransformerAEParameters, MLPAEParameters
from datautils import generate_edges

class GraphEncoder(nn.Module):
    def __init__(self, num_features, hidden_dim, latent_dim, num_layers, dropout_percentage=0.1):
        super().__init__()
        self.num_layers = num_layers

        # Define a ModuleList to store the dynamic number of RGCNConv layers
        self.conv_layers = nn.ModuleList([GCNConv(num_features, hidden_dim)])
        for _ in range(self.num_layers - 1):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))

        self.fc = nn.Linear(hidden_dim, latent_dim)

        self.dropout_percentage = dropout_percentage
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Loop through the RGCNConv layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x, edge_index, edge_attr)
            x = F.relu(x)

        x = F.dropout(x, p=self.dropout_percentage, training=self.training)
        x = pyg.nn.global_mean_pool(x, data.batch)
        z = self.fc(x)
        return z
    

class LatentTemporalAggregator(nn.Module):
    def __init__(self, latent_size, hidden_size, num_layers):
        super(LatentTemporalAggregator, self).__init__()
        self.lstm = nn.LSTM(input_size=latent_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, latent_size)

    def forward(self, z):
        # Basically a LSTM with batch size 1
        z = z.unsqueeze(0)
        z, (_, _) = self.lstm(z)
        return self.head(z)
    

# class GraphDecoder(nn.Module):
#     def __init__(self, num_features, hidden_dim, latent_dim, num_nodes=196, replicate_latent=False):
#         super(GraphDecoder, self).__init__()
#         self.fc = nn.Linear(latent_dim, num_nodes * latent_dim)
#         self.conv1 = GCNConv(latent_dim, hidden_dim)
#         # self.conv2 = GCNConv(hidden_dim, num_features)
#         self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
#         self.conv3 = GCNConv(hidden_dim, hidden_dim)
#         self.conv4 = GCNConv(hidden_dim, num_features)
        
#         # self.head = nn.Linear(num_features)

#         self.hidden_dim = hidden_dim
#         self.num_features = num_features
#         self.latent_dim = latent_dim
#         self.replicate_latent = replicate_latent
#         self.num_nodes = num_nodes

#     def forward(self, z, edge_index):
#         if self.replicate_latent:
#             # 'Trick' to make the decoding work
#             # Replicate the latent vector for every node
#             # Suggested by ChatGPT
#             z = z.unsqueeze(0).expand(self.num_nodes, -1)
#         else:
#             # Nonlinear projection to increased size and give fixed latent space to each node
#             # Makes the speed reconstruction much more expressive than above
#             z = z.unsqueeze(0)
#             z = self.fc(z)
#             z = F.relu(z)
#             z = z.view(self.num_nodes, self.latent_dim)

#         x = self.conv1(z, edge_index)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index)
#         x = F.relu(x)
#         x = self.conv3(x, edge_index)
#         x = F.relu(x)
#         x = self.conv4(x, edge_index)
#         return x
    
class GraphDecoder(nn.Module):
    def __init__(self, num_features, hidden_dim, latent_dim, num_nodes=196, replicate_latent=False, num_gcn_layers=5):
        super(GraphDecoder, self).__init__()

        self.fc = nn.Linear(latent_dim, num_nodes * latent_dim)
        self.gcn_layers = nn.ModuleList([GCNConv(latent_dim, hidden_dim)])
        for _ in range(num_gcn_layers - 1):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
        self.gcn_layers.append(GCNConv(hidden_dim, num_features))

        self.hidden_dim = hidden_dim
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.replicate_latent = replicate_latent
        self.num_nodes = num_nodes

    def forward(self, z, edge_index):
        if self.replicate_latent:
            x = z.unsqueeze(0).expand(self.num_nodes, -1)
        else:
            x = z.unsqueeze(0)
            x = self.fc(x)
            x= F.relu(x)
            x = x.view(self.num_nodes, self.latent_dim)

        for i, layer in enumerate(self.gcn_layers):
            x = layer(x, edge_index)
            if i < len(self.gcn_layers) - 1:  # Apply ReLU for all layers except the last one
                x = F.relu(x)

        return x

    

class SpatioTemporalAutoencoder(nn.Module):
    def __init__(self, params: STAEParameters):
        super(SpatioTemporalAutoencoder, self).__init__()
        self.enc = GraphEncoder(num_features=params.num_features, hidden_dim=params.gcn_hidden_dim, latent_dim=params.latent_dim, dropout_percentage=params.dropout, num_layers=2)
        self.temp_agg = LatentTemporalAggregator(latent_size=params.latent_dim, hidden_size=params.lstm_hidden_dim, num_layers=params.lstm_num_layers)
        self.dec = GraphDecoder(num_features=params.num_features, hidden_dim=params.gcn_hidden_dim, latent_dim=params.latent_dim, num_gcn_layers=2)

    def forward(self, temporal_graphs):
        edge_index = temporal_graphs[-1].edge_index
        # For each graph in the time window, apply the GraphEncoder
        latent_vectors = [self.enc(graph) for graph in temporal_graphs]

        # This gives a matrix of latent features
        latent_mat = torch.cat(latent_vectors) 

        # Run this through LatentTemporalAggregator
        aggregated = self.temp_agg(latent_mat).squeeze()[-1,:]

        # Feed the temporal aggregation through the graph decoder to construct a graph of the same structure
        graph_hat = self.dec(aggregated, edge_index)
        return graph_hat
    
    

class GATDecoder(nn.Module):
    def __init__(self, num_features, hidden_dim, latent_dim, num_nodes=196, replicate_latent=False, heads=1):
        super(GATDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, num_nodes * latent_dim)
        self.conv1 = GATConv(latent_dim, hidden_dim, heads=heads) # same as above but using GAT instead of GCN
        self.conv2 = GATConv(hidden_dim * heads, num_features, heads=heads)

        self.hidden_dim = hidden_dim
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.replicate_latent = replicate_latent
        self.num_nodes = num_nodes
    
    def forward(self, z, edge_index):
        if self.replicate_latent:
            # 'Trick' to make the decoding work
            # Replicate the latent vector for every node
            # Suggested by ChatGPT
            z = z.unsqueeze(0).expand(self.num_nodes, -1)
        else:
            z = z.unsqueeze(0)
            z = self.fc(z)
            z = F.relu(z)
            z = z.view(self.num_nodes, self.latent_dim)

        x = self.conv1(z, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
    

class GATSpatioTemporalAutoencoder(nn.Module):
    def __init__(self, params: GATSTAEParameters):
        super(GATSpatioTemporalAutoencoder, self).__init__()
        self.enc = GATEncoder(num_features=params.num_features, hidden_dim=params.gcn_hidden_dim, latent_dim=params.latent_dim, dropout_percentage=params.dropout, heads=params.gat_heads)
        self.temp_agg = LatentTemporalAggregator(latent_size=params.latent_dim, hidden_size=params.lstm_hidden_dim, num_layers=params.lstm_num_layers)
        self.dec = GATDecoder(num_features=params.num_features, hidden_dim=params.gcn_hidden_dim, latent_dim=params.latent_dim, heads=params.gat_heads)

    def forward(self, temporal_graphs):
        edge_index = temporal_graphs[-1].edge_index
        # For each graph in the time window, apply the GATEncoder
        latent_vectors = [self.enc(graph) for graph in temporal_graphs]

        # This gives a matrix of latent features
        latent_mat = torch.cat(latent_vectors) 

        # Run this through LatentTemporalAggregator
        aggregated = self.temp_agg(latent_mat).squeeze()[-1, :]

        # Feed the temporal aggregation through the GATDecoder to construct a graph of the same structure
        graph_hat = self.dec(aggregated, edge_index)
        return graph_hat
    
class GATEncoder(nn.Module):
    def __init__(self, num_features, hidden_dim, latent_dim, num_layers, dropout_percentage=0.1, num_heads=1):
        super().__init__()
        self.num_layers = num_layers

        self.conv_layers = nn.ModuleList([GATConv(num_features, hidden_dim, num_heads=num_heads)])
        for _ in range(self.num_layers - 1):
            self.conv_layers.append(GATConv(hidden_dim, hidden_dim, num_relations=num_heads))

        self.fc = nn.Linear(hidden_dim, latent_dim)

        self.dropout_percentage = dropout_percentage
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Loop through the RGCNConv layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x, edge_index)
            x = F.relu(x)

        x = F.dropout(x, p=self.dropout_percentage, training=self.training)
        x = pyg.nn.global_mean_pool(x, data.batch)
        z = self.fc(x)
        return z
    
class GATDecoder(nn.Module):
    def __init__(self, num_features, hidden_dim, latent_dim, num_nodes=196, replicate_latent=False, num_layers=3, num_heads=1):
        super().__init__()

        self.fc = nn.Linear(latent_dim, num_nodes * latent_dim)
        self.gat_layers = nn.ModuleList([GATConv(latent_dim, hidden_dim, num_heads=num_heads)])
        for _ in range(num_layers - 1):
            self.gat_layers.append(GATConv(hidden_dim, hidden_dim, num_heads=num_heads))
        self.gat_layers.append(GATConv(hidden_dim, num_features, num_heads=num_heads))

        self.hidden_dim = hidden_dim
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.replicate_latent = replicate_latent
        self.num_nodes = num_nodes

    def forward(self, z, edge_index):
        if self.replicate_latent:
            x = z.unsqueeze(0).expand(self.num_nodes, -1)
        else:
            x = z.unsqueeze(0)
            x = self.fc(x)
            x= F.relu(x)
            x = x.view(self.num_nodes, self.latent_dim)

        for i, layer in enumerate(self.gat_layers):
            x = layer(x, edge_index)
            if i < len(self.gat_layers) - 1:  # Apply ReLU for all layers except the last one
                x = F.relu(x)

        return x
    
class RelationalGraphEncoder(nn.Module):
    def __init__(self, num_features, hidden_dim, latent_dim, num_layers, dropout_percentage=0.1):
        super().__init__()
        self.num_layers = num_layers

        # Define a ModuleList to store the dynamic number of RGCNConv layers
        self.conv_layers = nn.ModuleList([RGCNConv(num_features, hidden_dim, num_relations=6)])
        for _ in range(self.num_layers - 1):
            self.conv_layers.append(RGCNConv(hidden_dim, hidden_dim, num_relations=6))

        self.fc = nn.Linear(hidden_dim, latent_dim)

        self.dropout_percentage = dropout_percentage
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Loop through the RGCNConv layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x, edge_index, edge_attr)
            x = F.relu(x)

        x = F.dropout(x, p=self.dropout_percentage, training=self.training)
        x = pyg.nn.global_mean_pool(x, data.batch)
        z = self.fc(x)
        return z
    
class RelationalSTAE(nn.Module):
    def __init__(self, parameters: RSTAEParameters):
        super().__init__()
        self.reconstructed_index = generate_edges(list(range(49))) # should probably pass the milemarker programitaclly
        self.enc = RelationalGraphEncoder(num_features=parameters.num_features, hidden_dim=parameters.gcn_hidden_dim, latent_dim=parameters.latent_dim, dropout_percentage=parameters.dropout, num_layers=parameters.num_gcn)
        self.dec = GraphDecoder(num_features=parameters.num_features, hidden_dim=parameters.gcn_hidden_dim, latent_dim=parameters.latent_dim, num_nodes=196, num_gcn_layers=parameters.num_gcn)

    def forward(self, x):
        z = self.enc(x)
        xhat = self.dec(z, self.reconstructed_index)
        return xhat
    
class GraphAE(nn.Module):
    def __init__(self, parameters: GraphAEParameters):
        super().__init__()
        self.reconstructed_index = generate_edges(list(range(49))) # should probably pass the milemarker programitaclly
        self.enc = GraphEncoder(num_features=parameters.num_features, hidden_dim=parameters.gcn_hidden_dim, latent_dim=parameters.latent_dim, dropout_percentage=parameters.dropout, num_layers=parameters.num_gcn)
        self.dec = GraphDecoder(num_features=parameters.num_features, hidden_dim=parameters.gcn_hidden_dim, latent_dim=parameters.latent_dim, num_nodes=196, num_gcn_layers=parameters.num_gcn)

    def forward(self, x):
        z = self.enc(x)
        xhat = self.dec(z, self.reconstructed_index)
        return xhat
    
    
class TransformerAutoencoder(nn.Module):
    def __init__(self, parameters: TransformerAEParameters):
        super(TransformerAutoencoder, self).__init__()
        self.input_dim = parameters.num_features
        self.sequence_length = parameters.sequence_length 

        # The positional embeddings are learned.
        
        # Encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=parameters.num_features,
                nhead=parameters.num_heads,
                dim_feedforward=parameters.hidden_dim,
                batch_first=True
            ),
            num_layers=parameters.num_layers
        )

        # Projection to Latent Dimension
        # self.projection = nn.Linear(self.input_dim*self.sequence_length, parameters.latent_dim)
        self.projection = nn.Linear(self.input_dim, parameters.latent_dim)
        

        # Decoder
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=parameters.latent_dim,  # Change input dimension for decoder
                nhead=parameters.num_heads,
                dim_feedforward=parameters.hidden_dim,
                batch_first=True
            ),
            num_layers=parameters.num_layers
        )

        # Fully connected layer for output
        self.fc_out = nn.Linear(parameters.latent_dim, parameters.num_features)

    def forward(self, x):
        # Encoding
        encoded = self.encoder(x)
        # encoded = encoded.view(-1, self.input_dim*self.sequence_length)
        projected = self.projection(encoded)
        # projected = projected.unsqueeze(1)
        # print(projected.shape)

        # Decoding
        decoded = self.decoder(projected)

        # Fully connected layer
        output = self.fc_out(decoded)[:,-1,:]

        return output
    
class GATAE(nn.Module):
    def __init__(self, parameters: GATAEParameters):
        super().__init__()
        self.reconstructed_index = generate_edges(list(range(49)))
        self.enc = GATEncoder(num_features=parameters.num_features, hidden_dim=parameters.gcn_hidden_dim, latent_dim=parameters.latent_dim, dropout_percentage=parameters.dropout, num_layers=parameters.num_layers, num_heads=parameters.num_heads)
        self.dec = GATDecoder(num_features=parameters.num_features, hidden_dim=parameters.gcn_hidden_dim, latent_dim=parameters.latent_dim, num_nodes=196, num_layers=parameters.num_layers, num_heads=parameters.num_heads)

    def forward(self, x):
        z = self.enc(x)
        xhat = self.dec(z, self.reconstructed_index)
        return xhat
    
    
class MLPAutoencoder(nn.Module):
    def __init__(self, parameters: MLPAEParameters):
        super(MLPAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(parameters.num_features, parameters.hidden_dim),
            nn.ReLU(),
            nn.Linear(parameters.hidden_dim, parameters.latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(parameters.latent_dim, parameters.hidden_dim),
            nn.ReLU(),
            nn.Linear(parameters.hidden_dim, parameters.num_features)
        )

    def forward(self, x):
        # Encoding
        encoded = self.encoder(x)
        
        # Decoding
        decoded = self.decoder(encoded)

        return decoded
    



# =========================
# New models (append below)
# =========================

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Simple causal TCN blocks for latent aggregation ---

class _CausalConv1d(nn.Module):
    """Causal 1D conv implemented by left-padding then regular Conv1d (no right lookahead)."""
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x):
        # x: [B, C, T]
        x = F.pad(x, (self.pad, 0))  # left pad only (causal)
        return self.conv(x)


class TemporalConvBlock(nn.Module):
    """Residual TCN block: (Conv -> ReLU -> Dropout -> Conv) + residual."""
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        c_in, c_out = channels
        self.conv1 = _CausalConv1d(c_in, c_out, kernel_size, dilation)
        self.conv2 = _CausalConv1d(c_out, c_out, kernel_size, dilation)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.norm = nn.LayerNorm(c_out)

    def forward(self, x):
        # x: [B, C, T]
        residual = self.downsample(x)
        out = self.conv1(x)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.dropout(out)
        out = out + residual
    
        out = out.transpose(1, 2)
        out = self.norm(out)
        out = out.transpose(1, 2)
        out = F.relu(out)
        return out


class TemporalConvNet(nn.Module):
    """Stack of TemporalConvBlock with exponentially increasing dilations."""
    def __init__(self, in_ch, hidden_ch=64, num_blocks=4, kernel_size=3, dropout=0.1):
        super().__init__()
        chans = [in_ch] + [hidden_ch] * num_blocks
        dilations = [2 ** i for i in range(num_blocks)]
        blocks = []
        for i in range(num_blocks):
            blocks.append(
                TemporalConvBlock(
                    channels=(chans[i], chans[i + 1]),
                    kernel_size=kernel_size,
                    dilation=dilations[i],
                    dropout=dropout,
                )
            )
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        # x: [B, C, T]
        return self.net(x)


# -------------------------------------------------------------------
# 1) SpatioTemporal TCN Autoencoder (GCN encoder + TCN + GCN decoder)
#     – drop-in replacement for the LSTM-based STAE
# -------------------------------------------------------------------

class SpatioTemporalTCNAutoencoder(nn.Module):
    """
    Same input pattern as your SpatioTemporalAutoencoder:
      forward(temporal_graphs: List[pyg.data.Data]) -> reconstruction of last frame
    Uses GraphEncoder -> TCN over latent sequence -> GraphDecoder.
    """
    def __init__(self, params: STAEParameters, tcn_hidden=128, tcn_blocks=3, tcn_dropout=0.05637794327123946):
        super().__init__()
        self.enc = GraphEncoder(
            num_features=params.num_features,
            hidden_dim=params.gcn_hidden_dim,
            latent_dim=params.latent_dim,
            num_layers=2,
            dropout_percentage=params.dropout,
        )

        self.tcn = TemporalConvNet(
            in_ch=params.latent_dim,
            hidden_ch=tcn_hidden,
            num_blocks=tcn_blocks,
            kernel_size=3,
            dropout=tcn_dropout,
        )
        self.tcn_proj = nn.Conv1d(tcn_hidden, params.latent_dim, kernel_size=1)

        self.dec = GraphDecoder(
            num_features=params.num_features,
            hidden_dim=params.gcn_hidden_dim,
            latent_dim=params.latent_dim,
            num_gcn_layers=2,
        )

    def forward(self, temporal_graphs):
        latents = [self.enc(g) for g in temporal_graphs]               
        processed = []
        for z in latents:
            # z may be [1, D] (batch dim 1); squeeze it to [D]
            if z.dim() == 2 and z.size(0) == 1:
                z = z.squeeze(0)
            processed.append(z)

        Z = torch.stack(processed, dim=0)
        Z = Z.transpose(0, 1).unsqueeze(0)  # [1, D, T]


        H = self.tcn(Z)                      # [1, H, T]
        H = self.tcn_proj(H)                 # [1, D, T]
        z_last = H[:, :, -1].squeeze(0)      # [D]
        xhat = self.dec(z_last, temporal_graphs[-1].edge_index)
        return xhat


