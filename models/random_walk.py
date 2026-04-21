"""
Random walk attention module

Implements context information propagation based on random walk.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Any
import math


class TransitionProbabilityCalculator(nn.Module):
    """Transition probability calculator"""
    
    def __init__(self, feature_dim: int, temperature: float = 1.0):
        """
        Initialize transition probability calculator
        
        Args:
            feature_dim: Feature dimension
            temperature: Temperature parameter for softmax
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.temperature = temperature
        
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
    
    def forward(self,
                center_features: torch.Tensor,
                neighbor_features: torch.Tensor) -> torch.Tensor:
        """
        Calculate transition probability
        
        Args:
            center_features: Center features [batch_size, feature_dim] or [batch_size, 1, feature_dim]
            neighbor_features: Neighbor features [batch_size, num_neighbors, feature_dim]
            
        Returns:
            Transition probability [batch_size, num_neighbors]
        """
        if center_features.dim() == 2:
            center_features = center_features.unsqueeze(1)
        
        Q = self.query_proj(center_features)
        K = self.key_proj(neighbor_features)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.feature_dim)
        attention_scores = attention_scores / self.temperature
        
        transition_probs = F.softmax(attention_scores, dim=-1)
        
        return transition_probs.squeeze(1)


class AttentionBasedRandomWalk(nn.Module):
    """Attention-based random walk"""
    
    def __init__(self, 
                 feature_dim: int,
                 use_higher_order: bool = True,
                 threshold: Optional[float] = None):
        """
        Initialize random walk module
        
        Args:
            feature_dim: Feature dimension
            use_higher_order: Whether to use higher-order context
            threshold: Random walk threshold
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.use_higher_order = use_higher_order
        self.threshold = threshold
        
        self.transition_calculator = TransitionProbabilityCalculator(feature_dim)
        
        self.value_proj = nn.Linear(feature_dim, feature_dim)
    
    def forward(self,
                center_features: torch.Tensor,
                first_order_features: torch.Tensor,
                second_order_features: torch.Tensor) -> torch.Tensor:
        """
        Perform random walk
        
        Args:
            center_features: Center features [batch_size, feature_dim]
            first_order_features: First-order neighbor features [batch_size, num_first, feature_dim]
            second_order_features: Second-order neighbor features [batch_size, num_second, feature_dim]
            
        Returns:
            Random walk aggregated features [batch_size, feature_dim]
        """
        batch_size = center_features.size(0)
        
        if self.use_higher_order:
            scores = self.transition_calculator(
                center_features.unsqueeze(1),
                second_order_features
            )
            
            if self.threshold is not None:
                mask = scores > self.threshold
                scores = scores * mask.float()
                scores = scores / (scores.sum(dim=1, keepdim=True) + 1e-8)
            
            second_order_values = self.value_proj(second_order_features)
            weighted_features = torch.bmm(scores.unsqueeze(1), second_order_values)
            
            aggregated = weighted_features.squeeze(1)
            
        else:
            scores = self.transition_calculator(
                center_features.unsqueeze(1),
                first_order_features
            )
            
            first_order_values = self.value_proj(first_order_features)
            aggregated = torch.bmm(scores.unsqueeze(1), first_order_values).squeeze(1)
        
        return aggregated
    
    def multi_step_random_walk(self,
                              features: torch.Tensor,
                              adjacency_index: torch.Tensor,
                              steps: int = 2) -> torch.Tensor:
        """
        Multi-step random walk
        
        Args:
            features: Node features [batch_size, num_nodes, feature_dim]
            adjacency_index: Adjacency index matrix [num_nodes, num_directions]
            steps: Number of walk steps
            
        Returns:
            Multi-step walked features [batch_size, num_nodes, feature_dim]
        """
        batch_size, num_nodes, feature_dim = features.shape
        device = features.device
        
        walk_states = features.clone()
        
        for step in range(steps):
            new_states = torch.zeros_like(walk_states)
            
            for node_idx in range(num_nodes):
                neighbors = adjacency_index[node_idx]
                valid_mask = neighbors != -1
                valid_neighbors = neighbors[valid_mask]
                
                if len(valid_neighbors) > 0:
                    neighbor_features = features[:, valid_neighbors, :]
                    
                    center_feat = walk_states[:, node_idx, :]
                    probs = self.transition_calculator(
                        center_feat.unsqueeze(1),
                        neighbor_features
                    )
                    
                    neighbor_states = walk_states[:, valid_neighbors, :]
                    weighted = torch.bmm(probs.unsqueeze(1), neighbor_states)
                    
                    new_states[:, node_idx, :] = weighted.squeeze(1)
                else:
                    new_states[:, node_idx, :] = walk_states[:, node_idx, :]
            
            walk_states = new_states
        
        return walk_states


class RandomWalkContextAggregator(nn.Module):
    """Random walk context aggregator"""
    
    def __init__(self,
                 feature_dim: int,
                 use_higher_order: bool = True,
                 threshold: float = 0.71,
                 dropout: float = 0.1):
        """
        Initialize random walk aggregator
        
        Args:
            feature_dim: Feature dimension
            use_higher_order: Whether to use higher-order context
            threshold: Random walk threshold
            dropout: Dropout rate
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.use_higher_order = use_higher_order
        self.threshold = threshold
        
        self.random_walk = AttentionBasedRandomWalk(
            feature_dim=feature_dim,
            use_higher_order=use_higher_order,
            threshold=threshold
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim)
        )
    
    def forward(self,
                features: torch.Tensor,
                adjacency_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            features: Input features [batch_size, num_nodes, feature_dim]
            adjacency_index: Adjacency index matrix [num_nodes, num_directions]
            
        Returns:
            Aggregated features [batch_size, num_nodes, feature_dim]
        """
        batch_size, num_nodes, feature_dim = features.shape
        device = features.device
        
        aggregated_features = torch.zeros_like(features)
        
        for node_idx in range(num_nodes):
            first_order = self._get_neighbors(node_idx, adjacency_index, order=1)
            second_order = self._get_neighbors(node_idx, adjacency_index, order=2)
            
            if len(first_order) > 0:
                center_feat = features[:, node_idx, :]
                first_feat = features[:, first_order, :]
                
                if self.use_higher_order and len(second_order) > 0:
                    second_feat = features[:, second_order, :]
                    rw_feat = self.random_walk(center_feat, first_feat, second_feat)
                else:
                    rw_feat = self.random_walk(center_feat, first_feat, first_feat)
                
                combined = torch.cat([
                    center_feat.unsqueeze(1),
                    rw_feat.unsqueeze(1)
                ], dim=-1)
                
                fused = self.fusion(combined)
                aggregated_features[:, node_idx:node_idx+1, :] = fused
        
        return aggregated_features
    
    def _get_neighbors(self, 
                      node_idx: int, 
                      adjacency_index: torch.Tensor, 
                      order: int) -> List[int]:
        """
        Get neighbors of specified order
        
        Args:
            node_idx: Node index
            adjacency_index: Adjacency index matrix
            order: Order
            
        Returns:
            List of neighbor indices
        """
        if order == 1:
            neighbors = adjacency_index[node_idx]
            valid_neighbors = neighbors[neighbors != -1].tolist()
            return valid_neighbors
        
        first_order = self._get_neighbors(node_idx, adjacency_index, 1)
        second_order = set()
        
        for neighbor in first_order:
            if neighbor != node_idx:
                neighbor_first = self._get_neighbors(neighbor, adjacency_index, 1)
                second_order.update(neighbor_first)
        
        second_order.discard(node_idx)
        for n in first_order:
            second_order.discard(n)
        
        return list(second_order)