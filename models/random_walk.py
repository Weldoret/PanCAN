"""
Random walk attention module

Implements context information propagation based on random walk.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Any, Union
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
                neighbor_features: torch.Tensor,
                prior: Optional[torch.Tensor] = None) -> torch.Tensor:
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

        if prior is not None:
            if prior.dim() == 1:
                prior = prior.view(1, 1, -1)
            elif prior.dim() == 2:
                prior = prior.unsqueeze(1)
            else:
                raise ValueError("prior must have shape [neighbors] or [batch, neighbors]")
            attention_scores = attention_scores + prior.to(
                device=attention_scores.device,
                dtype=attention_scores.dtype,
            ).abs().clamp_min(1e-8).log()
        
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
                masked_scores = scores * mask.float()
                normalizer = masked_scores.sum(dim=1, keepdim=True)

                # A threshold can remove every transition for a node. Keep the
                # most likely transition in that case so the walk remains a
                # valid probability distribution instead of returning zeros.
                fallback = F.one_hot(
                    scores.argmax(dim=1), num_classes=scores.size(1)
                ).to(dtype=scores.dtype)
                normalized_scores = masked_scores / normalizer.clamp_min(1e-8)
                scores = torch.where(
                    (normalizer <= 1e-8), fallback, normalized_scores
                )
            
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


class RandomWalkAttention(nn.Module):
    """Random-walk context aggregation from PanCAN's Eqs. (8)--(16).

    Each directional adjacency matrix is treated as its own neighborhood type
    ``c``. For every order ``k``, this module uses separate query, matching, and
    value projections, computes transition probabilities over only
    ``N_c^(k)(x)``, and concatenates the resulting contexts before reducing the
    representation with a pointwise projection. ``num_heads`` is retained for
    compatibility with the repository's configuration; the paper applies
    multi-head attention in the cross-scale module, not in RWCA.
    """

    def __init__(self,
                 feature_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 use_threshold: bool = True,
                 threshold: float = 0.71,
                 max_order: int = 2,
                 num_directions: int = 4,
                 gamma: float = 1.0):
        super().__init__()

        if max_order < 1:
            raise ValueError("max_order must be positive")
        if num_directions < 1:
            raise ValueError("num_directions must be positive")
        if gamma < 0:
            raise ValueError("gamma must be non-negative")

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.max_order = max_order
        self.num_directions = num_directions
        self.use_threshold = use_threshold
        self.threshold = threshold
        self.sqrt_gamma = math.sqrt(gamma)

        # These modules are the paper's W_q^k, W_m^k, and W_v^k. The existing
        # TransitionProbabilityCalculator supplies the first two projections
        # and the scaled dot-product calculation.
        self.transition_calculators = nn.ModuleList(
            TransitionProbabilityCalculator(feature_dim)
            for _ in range(max_order)
        )
        self.value_projections = nn.ModuleList(
            nn.Linear(feature_dim, feature_dim) for _ in range(max_order)
        )

        full_feature_dim = feature_dim * (1 + num_directions * max_order)
        self.dimension_reduction = nn.Conv1d(
            in_channels=full_feature_dim,
            out_channels=feature_dim,
            kernel_size=1,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                features: torch.Tensor,
                adjacency: Union[torch.Tensor, List[torch.Tensor], Any]) -> torch.Tensor:
        """Return the reduced multi-order context representation ``[B, N, F]``.

        The same cell map supplies queries, matches, and values, exactly as
        Eqs. (6)--(8). Order-specific features are concatenated per direction,
        transformed by the corresponding learned P_c, and finally concatenated
        with the intrinsic map as prescribed by Eqs. (16)--(17).
        """
        if features.dim() != 3:
            raise ValueError("features must be a 3D tensor")
        if features.size(-1) != self.feature_dim:
            raise ValueError(
                f"expected feature dimension {self.feature_dim}, got {features.size(-1)}"
            )

        directional_matrices = self._as_directional_matrices(
            adjacency, features.device
        )
        directional_neighbors = [
            self._matrix_to_neighbors(matrix) for matrix in directional_matrices
        ]
        if len(directional_neighbors) != self.num_directions:
            raise ValueError(
                f"expected {self.num_directions} directional neighborhoods, "
                f"got {len(directional_neighbors)}"
            )
        if len(directional_neighbors[0]) != features.size(1):
            raise ValueError("adjacency and feature node counts do not match")

        context_parts = [features]
        for direction_index, direction_neighbors in enumerate(directional_neighbors):
            order_contexts = []
            for order in range(1, self.max_order + 1):
                order_contexts.append(
                    self._aggregate_order(
                        features,
                        direction_neighbors,
                        order,
                    )
                )
            directional_context = torch.cat(order_contexts, dim=-1)
            matrix = directional_matrices[direction_index].to(
                device=features.device, dtype=features.dtype
            )
            context_parts.append(
                self.sqrt_gamma * torch.matmul(matrix, directional_context)
            )

        full_context = torch.cat(context_parts, dim=-1)
        reduced = self.dimension_reduction(full_context.transpose(1, 2))
        reduced = reduced.transpose(1, 2)
        return self.dropout(reduced)

    def _aggregate_order(self,
                         features: torch.Tensor,
                         first_order_neighbors: List[List[int]],
                         order: int) -> torch.Tensor:
        """Evaluate Eqs. (6)--(8) for one neighborhood type and order."""
        batch_size, num_nodes, _ = features.shape
        aggregated = features.new_zeros(
            batch_size, num_nodes, self.feature_dim
        )
        calculator = self.transition_calculators[order - 1]
        value_projection = self.value_projections[order - 1]

        for node_idx in range(num_nodes):
            neighbors = self._get_order_neighbors(
                first_order_neighbors, node_idx, order
            )
            if not neighbors:
                continue

            center = features[:, node_idx, :]
            neighbor_features = features[:, neighbors, :]
            probabilities = calculator(center, neighbor_features)

            if self.use_threshold:
                # The paper retains only high-probability cells in Eq. (8); it
                # does not define a second normalization after selection.
                probabilities = probabilities * (
                    probabilities > self.threshold
                ).to(probabilities.dtype)

            values = value_projection(neighbor_features)
            aggregated[:, node_idx, :] = torch.bmm(
                probabilities.unsqueeze(1), values
            ).squeeze(1)

        return aggregated

    @staticmethod
    def _get_order_neighbors(first_order_neighbors: List[List[int]],
                             node_idx: int,
                             order: int) -> List[int]:
        """Build the recursive k-th order neighborhood for one direction."""
        if order == 1:
            return sorted(set(first_order_neighbors[node_idx]))

        previous = RandomWalkAttention._get_order_neighbors(
            first_order_neighbors, node_idx, order - 1
        )
        neighbors = set()
        for neighbor in previous:
            if neighbor != node_idx:
                neighbors.update(RandomWalkAttention._get_order_neighbors(
                    first_order_neighbors, neighbor, order - 1
                ))
        neighbors.discard(node_idx)
        return sorted(neighbors)

    @staticmethod
    def _as_directional_neighbors(
            adjacency: Union[torch.Tensor, List[torch.Tensor], Any],
            device: torch.device) -> List[List[List[int]]]:
        """Convert adjacency input into one neighbor list per context type."""
        matrices = RandomWalkAttention._as_directional_matrices(adjacency, device)
        return [RandomWalkAttention._matrix_to_neighbors(matrix)
                for matrix in matrices]

    @staticmethod
    def _as_directional_matrices(
            adjacency: Union[torch.Tensor, List[torch.Tensor], Any],
            device: torch.device) -> List[torch.Tensor]:
        """Convert adjacency input into weighted matrices per context type."""
        if hasattr(adjacency, "adjacency_matrices"):
            adjacency = adjacency.adjacency_matrices

        if isinstance(adjacency, torch.Tensor):
            if adjacency.dim() != 2:
                raise ValueError("adjacency tensor must be 2D")
            if adjacency.dtype in (torch.int8, torch.int16, torch.int32,
                                   torch.int64, torch.uint8):
                # An index matrix has one column per directional context.
                matrices = []
                num_nodes = adjacency.size(0)
                rows = torch.arange(num_nodes, device=device)
                for direction in range(adjacency.size(1)):
                    matrix = torch.zeros(
                        num_nodes, num_nodes, device=device, dtype=torch.float32
                    )
                    neighbors = adjacency[:, direction].to(device=device)
                    valid = neighbors >= 0
                    matrix[rows[valid], neighbors[valid].long()] = 1.0
                    matrices.append(matrix)
                return matrices
            else:
                adjacency = [adjacency]

        if not isinstance(adjacency, (list, tuple)) or not adjacency:
            raise ValueError("adjacency must contain at least one matrix")

        directional_matrices = []
        for matrix in adjacency:
            matrix = matrix.to(device=device)
            if matrix.dim() == 1:
                num_nodes = matrix.size(0)
                converted = torch.zeros(
                    num_nodes, num_nodes, device=device, dtype=torch.float32
                )
                rows = torch.arange(num_nodes, device=device)
                valid = matrix >= 0
                converted[rows[valid], matrix[valid].long()] = 1.0
                matrix = converted
            elif matrix.dim() == 2:
                if matrix.size(0) != matrix.size(1):
                    raise ValueError(
                        "dense adjacency matrices must be square"
                    )
            else:
                raise ValueError("each adjacency entry must be a vector or matrix")
            directional_matrices.append(matrix.to(dtype=torch.float32))

        num_nodes = directional_matrices[0].size(0)
        if any(matrix.size(0) != num_nodes for matrix in directional_matrices):
            raise ValueError("all directional neighborhoods must have equal node counts")
        return directional_matrices

    @staticmethod
    def _matrix_to_neighbors(matrix: torch.Tensor) -> List[List[int]]:
        support = matrix.abs() > 1e-8
        return [
            torch.nonzero(support[node_idx], as_tuple=False).flatten().tolist()
            for node_idx in range(matrix.size(0))
        ]
