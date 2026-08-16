

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
import itertools


class MultiOrderNeighborhood(nn.Module):
    """Multi-order neighborhood system"""
    
    def __init__(self, max_order: int = 3):
        """
        Initialize multi-order neighborhood
        
        Args:
            max_order: Maximum neighborhood order
        """
        super().__init__()
        self.max_order = max_order
    
    def get_neighborhood(self, 
                        node_idx: int,
                        adjacency_index: torch.Tensor,
                        order: int) -> List[int]:
        """
        Get neighborhood nodes of specified order
        
        Args:
            node_idx: Center node index
            adjacency_index: Adjacency index matrix [num_nodes, num_directions]
            order: Neighborhood order
            
        Returns:
            List of neighborhood node indices
        """
        if order == 1:
            neighbors = adjacency_index[node_idx]
            valid_neighbors = neighbors[neighbors != -1].tolist()
            return valid_neighbors
        
        first_order = self.get_neighborhood(node_idx, adjacency_index, 1)
        higher_order = set()
        
        for neighbor in first_order:
            if neighbor != node_idx:
                neighbor_higher = self.get_neighborhood(neighbor, adjacency_index, order-1)
                higher_order.update(neighbor_higher)
        
        higher_order.discard(node_idx)
        
        return list(higher_order)
    
    def get_multi_order_neighborhoods(self,
                                     node_idx: int,
                                     adjacency_index: torch.Tensor) -> Dict[int, List[int]]:
        neighborhoods = {}
        
        for order in range(1, self.max_order + 1):
            neighborhoods[order] = self.get_neighborhood(node_idx, adjacency_index, order)
        
        return neighborhoods
    
    def build_neighborhood_matrix(self,
                                 adjacency_index: torch.Tensor,
                                 order: int) -> torch.Tensor:
        num_nodes = adjacency_index.size(0)
        neighborhood_matrix = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
        
        for i in range(num_nodes):
            neighbors = self.get_neighborhood(i, adjacency_index, order)
            for j in neighbors:
                neighborhood_matrix[i, j] = 1.0
        
        return neighborhood_matrix


class MultiOrderContextAggregator(nn.Module):
    """Multi-order context aggregator"""
    
    def __init__(self,
                 feature_dim: int,
                 max_order: int = 3,
                 use_attention: bool = True,
                 num_directions: int = 4):
        """
        Initialize multi-order context aggregator
        
        Args:
            feature_dim: Feature dimension
            max_order: Maximum neighborhood order
            use_attention: Whether to use attention mechanism
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.max_order = max_order
        self.use_attention = use_attention
        self.num_directions = num_directions
        
        self.neighborhood_system = MultiOrderNeighborhood(max_order)
        
        if use_attention:
            self.query_proj = nn.Linear(feature_dim, feature_dim)
            self.key_proj = nn.Linear(feature_dim, feature_dim)
            self.value_proj = nn.Linear(feature_dim, feature_dim)
    
    def forward(self,
                features: torch.Tensor,
                adjacency_index: torch.Tensor) -> torch.Tensor:
        """
        Aggregate multi-order context features
        
        Args:
            features: Input features [batch_size, num_nodes, feature_dim]
            adjacency_index: Adjacency index matrix [num_nodes, num_directions]
            
        Returns:
            Aggregated features [batch_size, num_nodes, feature_dim]
        """
        batch_size, num_nodes, feature_dim = features.shape
        
        aggregated_features = torch.zeros_like(features)
        
        for node_idx in range(num_nodes):
            neighborhoods = self.neighborhood_system.get_multi_order_neighborhoods(
                node_idx, adjacency_index
            )
            
            for order in range(1, self.max_order + 1):
                neighbors = neighborhoods[order]
                
                if len(neighbors) > 0:
                    neighbor_features = features[:, neighbors, :]
                    
                    center_features = features[:, node_idx:node_idx+1, :]
                    
                    if self.use_attention:
                        order_features = self.attention_aggregation(
                            center_features, neighbor_features
                        )
                    else:
                        order_features = neighbor_features.mean(dim=1, keepdim=True)
                    
                    aggregated_features[:, node_idx:node_idx+1, :] += order_features
        
        aggregated_features /= self.max_order
        
        return aggregated_features
    
    def attention_aggregation(self,
                             center_features: torch.Tensor,
                             neighbor_features: torch.Tensor) -> torch.Tensor:
        """
        Attention aggregation
        
        Args:
            center_features: Center features [batch_size, 1, feature_dim]
            neighbor_features: Neighbor features [batch_size, num_neighbors, feature_dim]
            
        Returns:
            Aggregated features [batch_size, 1, feature_dim]
        """
        Q = self.query_proj(center_features)
        K = self.key_proj(neighbor_features)
        V = self.value_proj(neighbor_features)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.feature_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        aggregated = torch.matmul(attention_weights, V)
        
        return aggregated
    
    def get_order_specific_features(self,
                                  features: torch.Tensor,
                                  adjacency_index: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Get order-specific features
        
        Args:
            features: Input features [batch_size, num_nodes, feature_dim]
            adjacency_index: Adjacency index matrix
            
        Returns:
            Dictionary: order -> features [batch_size, num_nodes, feature_dim]
        """
        batch_size, num_nodes, feature_dim = features.shape
        order_features = {}
        
        for order in range(1, self.max_order + 1):
            order_feat = torch.zeros_like(features)
            
            for node_idx in range(num_nodes):
                neighbors = self.neighborhood_system.get_neighborhood(
                    node_idx, adjacency_index, order
                )
                
                if len(neighbors) > 0:
                    neighbor_feat = features[:, neighbors, :]
                    
                    if self.use_attention:
                        center_feat = features[:, node_idx:node_idx+1, :]
                        aggregated = self.attention_aggregation(center_feat, neighbor_feat)
                    else:
                        aggregated = neighbor_feat.mean(dim=1, keepdim=True)
                    
                    order_feat[:, node_idx:node_idx+1, :] = aggregated
            
            order_features[order] = order_feat
        
        return order_features


class MultiOrderContextLayer(nn.Module):
    """Multi-order context layer (integration module)"""
    
    def __init__(self,
                 feature_dim: int,
                 max_order: int = 3,
                 use_attention: bool = True,
                 dropout: float = 0.1):
        """
        Initialize multi-order context layer
        
        Args:
            feature_dim: Feature dimension
            max_order: Maximum order
            use_attention: Whether to use attention
            dropout: Dropout rate
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.max_order = max_order
        
        self.aggregator = MultiOrderContextAggregator(
            feature_dim=feature_dim,
            max_order=max_order,
            use_attention=use_attention
        )
        
        self.transform = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim)
        )
    
    def forward(self,
                features: torch.Tensor,
                adjacency_index: torch.Tensor) -> torch.Tensor:
        context_features = self.aggregator(features, adjacency_index)
        
        combined = torch.cat([features, context_features], dim=-1)
        
        output = self.transform(combined)
        
        return output
