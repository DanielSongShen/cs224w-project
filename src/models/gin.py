"""Graph Neural Network models for heterogeneous graph classification"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GINConv,
    GATv2Conv,
    HeteroConv,
    Linear,
    global_mean_pool,
    global_add_pool,
    GlobalAttention,
)
from torch_geometric.data import HeteroData


class NodeEncoder(nn.Module):
    """
    Encodes node features (level, category, thought_index) into continuous embeddings.
    Uses learned embeddings for category and level, but sinusoidal PE for thought index.
    
    For use with 'tree' format: [level, cate, thought_idx]
    """
    def __init__(self, hidden_channels: int, max_level: int = 1000, max_thoughts: int = 5000):
        super().__init__()
        self.hidden_channels = hidden_channels
        
        # Learned embeddings for categorical features
        self.cate_emb = nn.Embedding(5, hidden_channels)
        self.level_emb = nn.Embedding(max_level, hidden_channels)
        
        # Fixed Sinusoidal Positional Encoding for thought_index
        # Pre-compute PE matrix
        pe = torch.zeros(max_thoughts, hidden_channels)
        position = torch.arange(0, max_thoughts, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_channels, 2).float() * (-torch.log(torch.tensor(10000.0)) / hidden_channels))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a learnable parameter)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [N, 3] containing (level, cate, thought_index)
        x = x.long()
        
        # Clamp values
        level = x[:, 0].clamp(0, self.level_emb.num_embeddings - 1)
        cate = x[:, 1].clamp(0, self.cate_emb.num_embeddings - 1)
        thought = x[:, 2].clamp(0, self.pe.size(0) - 1)
        
        # Sum embeddings
        # Note: cate_emb and level_emb are learned, pe is fixed
        return self.cate_emb(cate) + self.level_emb(level) + self.pe[thought]


class GraphNodeEncoder(nn.Module):
    """
    Encodes node features (node_id, out_degree, in_degree) into continuous embeddings.
    Uses sinusoidal PE for node_id and learned embeddings for degrees.
    
    For use with 'graph' format: [node_id, out_degree, in_degree]
    """
    def __init__(self, hidden_channels: int, max_nodes: int = 5000, max_degree: int = 200):
        super().__init__()
        self.hidden_channels = hidden_channels
        
        # Learned embeddings for degree features
        self.out_degree_emb = nn.Embedding(max_degree, hidden_channels)
        self.in_degree_emb = nn.Embedding(max_degree, hidden_channels)
        
        # Fixed Sinusoidal Positional Encoding for node_id
        # Pre-compute PE matrix
        pe = torch.zeros(max_nodes, hidden_channels)
        position = torch.arange(0, max_nodes, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_channels, 2).float() * (-torch.log(torch.tensor(10000.0)) / hidden_channels))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a learnable parameter)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [N, 3] containing (node_id, out_degree, in_degree)
        x = x.long()
        
        # Clamp values
        node_id = x[:, 0].clamp(0, self.pe.size(0) - 1)
        out_deg = x[:, 1].clamp(0, self.out_degree_emb.num_embeddings - 1)
        in_deg = x[:, 2].clamp(0, self.in_degree_emb.num_embeddings - 1)
        
        # Sum embeddings
        # Note: degree embeddings are learned, node_id PE is fixed
        return self.pe[node_id] + self.out_degree_emb(out_deg) + self.in_degree_emb(in_deg)


class HeteroGINConv(nn.Module):
    """
    Heterogeneous GIN convolution layer.
    
    Applies separate GIN convolutions for each edge type, then aggregates
    the results for each node type.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_types: list,
        aggr: str = "sum",
    ):
        super().__init__()
        
        # Create a GIN MLP for each edge type
        convs = {}
        for edge_type in edge_types:
            mlp = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            )
            convs[edge_type] = GINConv(mlp)
        
        self.conv = HeteroConv(convs, aggr=aggr)
    
    def forward(self, x_dict, edge_index_dict):
        return self.conv(x_dict, edge_index_dict)


class HeteroGATConv(nn.Module):
    """
    Heterogeneous GAT convolution layer.
    
    Applies separate GAT convolutions for each edge type.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_types: list,
        heads: int = 1,
        aggr: str = "sum",
    ):
        super().__init__()
        
        convs = {}
        for edge_type in edge_types:
            convs[edge_type] = GATv2Conv(
                in_channels,
                out_channels,
                heads=heads,
                concat=False,
                add_self_loops=False,
            )
        
        self.conv = HeteroConv(convs, aggr=aggr)
    
    def forward(self, x_dict, edge_index_dict):
        return self.conv(x_dict, edge_index_dict)


class HeteroGraphClassifier(nn.Module):
    """
    Heterogeneous GNN for graph-level binary classification.
    
    This model handles HeteroData graphs with multiple edge types.
    It applies heterogeneous message passing, then pools node representations
    to produce graph-level predictions.
    
    Args:
        in_channels: Number of input features per node
        hidden_channels: Hidden layer dimension
        out_channels: Number of output classes (2 for binary classification)
        edge_types: List of edge type tuples, e.g., [('thought', 'continuous_logic', 'thought'), ...]
        num_layers: Number of message passing layers
        conv_type: Type of convolution - "gin" or "gat"
        dropout: Dropout rate
        pool: Pooling method - "mean" or "add"
        encoder_type: Type of node encoder - "tree" for NodeEncoder, "graph" for GraphNodeEncoder
        
    Example:
        >>> # Get edge types from a sample graph
        >>> sample = dataset[0]
        >>> edge_types = sample.edge_types
        >>> 
        >>> model = HeteroGraphClassifier(
        ...     in_channels=3,
        ...     hidden_channels=64,
        ...     out_channels=2,
        ...     edge_types=edge_types,
        ... )
        >>> 
        >>> # Forward pass
        >>> out = model(batch)  # batch is a HeteroData batch
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        edge_types: list,
        num_layers: int = 3,
        conv_type: str = "gin",
        dropout: float = 0.0,
        pool: str = "mean",
        use_node_encoder: bool = True,
        encoder_type: str = "tree",
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.pool = pool
        self.use_node_encoder = use_node_encoder
        self.encoder_type = encoder_type
        
        # Input projection or encoding
        if use_node_encoder and in_channels == 3:
            if encoder_type == "graph":
                self.node_encoder = GraphNodeEncoder(hidden_channels)
            else:
                self.node_encoder = NodeEncoder(hidden_channels)
            self.input_lin = None
        else:
            self.node_encoder = None
            self.input_lin = Linear(in_channels, hidden_channels)
        
        # Message passing layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            if conv_type == "gin":
                conv = HeteroGINConv(
                    hidden_channels,
                    hidden_channels,
                    edge_types,
                )
            elif conv_type == "gat":
                conv = HeteroGATConv(
                    hidden_channels,
                    hidden_channels,
                    edge_types,
                    heads=4,
                )
            else:
                raise ValueError(f"Unknown conv_type: {conv_type}")
            
            self.convs.append(conv)
            self.norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )
        
    def forward(self, data: HeteroData) -> torch.Tensor:
        """
        Forward pass for graph classification.
        
        Args:
            data: HeteroData object (can be batched)
            
        Returns:
            Logits of shape [batch_size, out_channels]
        """
        # Get node features and edge indices
        x_dict = {key: data[key].x for key in data.node_types}
        edge_index_dict = {
            edge_type: data[edge_type].edge_index
            for edge_type in data.edge_types
        }
        
        # Input projection
        if self.node_encoder is not None:
            x_dict = {key: self.node_encoder(x) for key, x in x_dict.items()}
        else:
            x_dict = {key: self.input_lin(x) for key, x in x_dict.items()}
        
        # Message passing
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_dict = conv(x_dict, edge_index_dict)
            
            # Apply normalization and activation
            x_dict = {
                key: F.relu(norm(x)) 
                for key, x in x_dict.items()
            }
            
            # Apply dropout (except last layer)
            if i < self.num_layers - 1 and self.dropout > 0:
                x_dict = {
                    key: F.dropout(x, p=self.dropout, training=self.training)
                    for key, x in x_dict.items()
                }
        
        # Global pooling - pool over all node types
        # For graphs with single node type "thought"
        pooled = []
        for node_type in x_dict:
            x = x_dict[node_type]
            
            # Get batch assignment for this node type
            if hasattr(data[node_type], "batch"):
                batch = data[node_type].batch
            else:
                # Single graph case
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
            if self.pool == "mean":
                pooled.append(global_mean_pool(x, batch))
            else:
                pooled.append(global_add_pool(x, batch))
        
        # Combine pooled representations from all node types
        x = torch.stack(pooled, dim=0).sum(dim=0)
        
        # Classification
        out = self.classifier(x)
        
        return out


class SimpleHeteroGNN(nn.Module):
    """
    Simplified heterogeneous GNN that converts to homogeneous for processing.
    
    This is useful when you want to use standard GNN layers but have
    heterogeneous input data. It projects all edge types to a single
    representation.
    
    Args:
        in_channels: Number of input features per node
        hidden_channels: Hidden layer dimension
        out_channels: Number of output classes
        num_layers: Number of GIN layers
        dropout: Dropout rate
        encoder_type: Type of node encoder - "tree" for NodeEncoder, "graph" for GraphNodeEncoder
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.0,
        use_node_encoder: bool = True,
        encoder_type: str = "tree",
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_node_encoder = use_node_encoder
        self.encoder_type = encoder_type
        
        # Encoder
        if use_node_encoder and in_channels == 3:
            if encoder_type == "graph":
                self.node_encoder = GraphNodeEncoder(hidden_channels)
            else:
                self.node_encoder = NodeEncoder(hidden_channels)
            mlp_in_channels = hidden_channels
        else:
            self.node_encoder = None
            mlp_in_channels = in_channels
        
        # Build GIN layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # First layer
        mlp = nn.Sequential(
            nn.Linear(mlp_in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.convs.append(GINConv(mlp))
        self.norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.convs.append(GINConv(mlp))
            self.norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Attention pooling: learns which nodes are important
        # gate_nn produces attention scores, allowing model to focus on "mistake" nodes
        gate_nn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )
        self.pool = GlobalAttention(gate_nn)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )
    
    def forward(self, data: HeteroData) -> torch.Tensor:
        """
        Forward pass - converts hetero to homo internally.
        
        Args:
            data: HeteroData object
            
        Returns:
            Logits of shape [batch_size, out_channels]
        """
        # Get node features (assuming single node type "thought")
        x = data["thought"].x
        
        # Apply encoder if enabled
        if self.node_encoder is not None:
            x = self.node_encoder(x)
        
        # Combine all edge types into single edge_index
        # Only use forward edges (not reverse) to avoid duplicates
        edge_indices = []
        for edge_type in data.edge_types:
            # Skip reverse edges to avoid duplicates
            if edge_type[1].startswith("rev_"):
                continue
            edge_indices.append(data[edge_type].edge_index)
        
        if edge_indices:
            edge_index = torch.cat(edge_indices, dim=1)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=x.device)
        
        # Get batch assignment
        if hasattr(data["thought"], "batch"):
            batch = data["thought"].batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Message passing
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            
            if i < self.num_layers - 1 and self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Attention pooling - learns to focus on important nodes
        x = self.pool(x, batch)
        
        # Classification
        out = self.classifier(x)
        
        return out


def create_model(
    model_type: str,
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    edge_types: list = None,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create GNN models.
    
    Args:
        model_type: One of "hetero_gin", "hetero_gat", "simple_gin"
        in_channels: Input feature dimension
        hidden_channels: Hidden dimension
        out_channels: Number of output classes
        edge_types: List of edge types (required for hetero models)
        **kwargs: Additional arguments passed to model constructor
            - num_layers: Number of message passing layers
            - dropout: Dropout rate
            - pool: Pooling method (hetero models only)
            - use_node_encoder: Whether to use learned node encoder
            - encoder_type: "tree" for NodeEncoder, "graph" for GraphNodeEncoder
        
    Returns:
        Initialized model
    """
    # Common kwargs accepted by all models
    common_kwargs = {
        k: v for k, v in kwargs.items() 
        if k in ("num_layers", "dropout", "use_node_encoder", "encoder_type")
    }
    
    # Kwargs only for HeteroGraphClassifier
    hetero_kwargs = {
        k: v for k, v in kwargs.items()
        if k in ("num_layers", "dropout", "pool", "use_node_encoder", "encoder_type")
    }
    
    if model_type == "hetero_gin":
        if edge_types is None:
            raise ValueError("edge_types required for hetero_gin model")
        return HeteroGraphClassifier(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            edge_types=edge_types,
            conv_type="gin",
            **hetero_kwargs,
        )
    elif model_type == "hetero_gat":
        if edge_types is None:
            raise ValueError("edge_types required for hetero_gat model")
        return HeteroGraphClassifier(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            edge_types=edge_types,
            conv_type="gat",
            **hetero_kwargs,
        )
    elif model_type == "simple_gin":
        return SimpleHeteroGNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            **common_kwargs,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
