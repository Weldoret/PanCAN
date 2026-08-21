
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
import math


class DirectionalNeighborhood:
    """Directional neighborhood definition"""
    
    DIRECTIONS = {
        'up': (-1, 0),
        'down': (1, 0),
        'left': (0, -1),
        'right': (0, 1)
    }
    
    def __init__(self, num_directions=4):
        """
        Initialize directional neighborhood
        
        Args:
            num_directions: Number of directions, default 4 (up, down, left, right)
        """
        self.num_directions = num_directions
        self.directions = self._get_directions()
    
    def _get_directions(self) -> List[Tuple[int, int]]:
        """Get direction list"""
        if self.num_directions == 4:
            return [
                self.DIRECTIONS['up'],
                self.DIRECTIONS['down'],
                self.DIRECTIONS['left'],
                self.DIRECTIONS['right']
            ]
        elif self.num_directions == 8:
            return [
                self.DIRECTIONS['up'],
                self.DIRECTIONS['down'],
                self.DIRECTIONS['left'],
                self.DIRECTIONS['right'],
                (-1, -1),  # top-left
                (-1, 1),   # top-right
                (1, -1),   # bottom-left
                (1, 1)     # bottom-right
            ]
        else:
            raise ValueError(f"Unsupported number of directions: {self.num_directions}")
    
    def get_direction_index(self, direction: str) -> int:
        """Get direction index"""
        direction_map = {
            'up': 0,
            'down': 1,
            'left': 2,
            'right': 3
        }
        return direction_map.get(direction, -1)


class AdjacencyMatrixGenerator:
    """Adjacency matrix generator"""
    
    def __init__(self, rows: int, cols: int, directions: Optional[List[Tuple[int, int]]] = None):
        """
        Initialize adjacency matrix generator
        
        Args:
            rows: Number of grid rows
            cols: Number of grid columns
            directions: List of directions
        """
        if rows < 1 or cols < 1:
            raise ValueError("grid dimensions must be positive")
        self.rows = rows
        self.cols = cols
        self.num_nodes = rows * cols
        
        if directions is None:
            directional = DirectionalNeighborhood()
            self.directions = directional.directions
            self.num_directions = directional.num_directions
        else:
            if not directions:
                raise ValueError("directions must not be empty")
            self.directions = list(directions)
            self.num_directions = len(directions)
    
    def generate_adjacency_matrices(self) -> List[torch.Tensor]:
        """
        Generate adjacency matrix list (one matrix per direction)
        
        Returns:
            List of adjacency matrices, each with shape [num_nodes, num_nodes]
        """
        adjacency_matrices = []
        
        for direction in self.directions:
            adj_matrix = self._generate_directional_adjacency(direction)
            adjacency_matrices.append(adj_matrix)
        
        return adjacency_matrices
    
    def _generate_directional_adjacency(self, direction: Tuple[int, int]) -> torch.Tensor:
        dr, dc = direction
        adj_matrix = torch.zeros(self.num_nodes, self.num_nodes, dtype=torch.float32)
        
        for i in range(self.rows):
            for j in range(self.cols):
                node_idx = i * self.cols + j
                
                ni, nj = i + dr, j + dc
                
                if 0 <= ni < self.rows and 0 <= nj < self.cols:
                    neighbor_idx = ni * self.cols + nj
                    adj_matrix[node_idx, neighbor_idx] = 1.0
        
        return adj_matrix
    
    def generate_adjacency_index_tensor(self) -> torch.Tensor:
        index_matrix = torch.full((self.num_nodes, self.num_directions), -1, dtype=torch.long)
        
        for i in range(self.rows):
            for j in range(self.cols):
                node_idx = i * self.cols + j
                
                for d, (dr, dc) in enumerate(self.directions):
                    ni, nj = i + dr, j + dc
                    
                    if 0 <= ni < self.rows and 0 <= nj < self.cols:
                        neighbor_idx = ni * self.cols + nj
                        index_matrix[node_idx, d] = neighbor_idx
        
        return index_matrix
    
    def generate_weight_matrix(self, normalization: str = 'none') -> torch.Tensor:
        index_matrix = self.generate_adjacency_index_tensor()
        weight_matrix = torch.zeros(self.num_nodes, self.num_directions, dtype=torch.float32)
        
        for node_idx in range(self.num_nodes):
            valid_indices = index_matrix[node_idx] != -1
            num_valid = valid_indices.sum().item()
            
            if num_valid > 0:
                if normalization == 'average':
                    weight = 1.0 / num_valid
                elif normalization == 'degree':
                    weight = 1.0 / math.sqrt(num_valid)
                elif normalization == 'none':
                    weight = 1.0
                else:
                    raise ValueError(f"Unknown normalization: {normalization}")
                
                weight_matrix[node_idx, valid_indices] = weight
        
        return weight_matrix


def generate_adjacency_index_matrix(rows: int, cols: int, directions: Optional[List[Tuple[int, int]]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = AdjacencyMatrixGenerator(rows, cols, directions)
    index_matrix = generator.generate_adjacency_index_tensor()
    weights_flag = generator.generate_weight_matrix()
    
    return index_matrix, weights_flag


def generate_weight_matrix(rows: int, cols: int, normalization: str = 'none') -> torch.Tensor:
    generator = AdjacencyMatrixGenerator(rows, cols)
    return generator.generate_weight_matrix(normalization)


class NeighborhoodSystem(nn.Module):
    """Neighborhood system module"""
    
    def __init__(self, rows: int, cols: int, directions: Optional[List[Tuple[int, int]]] = None):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.num_nodes = rows * cols
        
        generator = AdjacencyMatrixGenerator(rows, cols, directions)
        self._adjacency_buffer_names = []
        for direction_idx, matrix in enumerate(generator.generate_adjacency_matrices()):
            name = f"adjacency_matrix_{direction_idx}"
            self.register_buffer(name, matrix)
            self._adjacency_buffer_names.append(name)
        self.register_buffer(
            'adjacency_index', generator.generate_adjacency_index_tensor()
        )
        self.register_buffer(
            'adjacency_weight', generator.generate_weight_matrix()
        )
        
        self.directions = generator.directions
        self.num_directions = generator.num_directions

    @property
    def adjacency_matrices(self) -> List[torch.Tensor]:
        """Return directional adjacency matrices on the module's current device."""
        return [getattr(self, name) for name in self._adjacency_buffer_names]
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply neighborhood system
        
        Args:
            features: Input features [batch_size, num_nodes, feature_dim]
            
        Returns:
            Neighborhood aggregated features
        """
        if features.dim() != 3 or features.size(1) != self.num_nodes:
            raise ValueError(
                f"features must have shape [batch, {self.num_nodes}, feature_dim]"
            )
        _, num_nodes, _ = features.shape
        
        aggregated_features = torch.zeros_like(features)
        
        for node_idx in range(num_nodes):
            neighbor_indices = self.adjacency_index[node_idx]
            weights = self.adjacency_weight[node_idx]
            
            valid_mask = neighbor_indices != -1
            valid_indices = neighbor_indices[valid_mask]
            valid_weights = weights[valid_mask]
            
            if len(valid_indices) > 0:
                neighbor_features = features[:, valid_indices, :]
                
                weighted_neighbors = neighbor_features * valid_weights.view(1, -1, 1)
                
                aggregated = weighted_neighbors.sum(dim=1)
                
                aggregated_features[:, node_idx, :] = aggregated
        
        return aggregated_features
    
    def get_neighbors(self, node_idx: int, include_self: bool = False) -> List[int]:
        """
        Get neighbors of specified node
        
        Args:
            node_idx: Node index
            include_self: Whether to include self
            
        Returns:
            List of neighbor indices
        """
        self._validate_node(node_idx)
        neighbor_indices = self.adjacency_index[node_idx]
        
        valid_mask = neighbor_indices != -1
        neighbors = neighbor_indices[valid_mask].tolist()
        
        if include_self:
            neighbors.append(node_idx)
        
        return neighbors
    
    def get_neighborhood_matrix(self, direction_idx: int) -> torch.Tensor:
        """
        Get adjacency matrix for specified direction
        
        Args:
            direction_idx: Direction index
            
        Returns:
            Adjacency matrix
        """
        if 0 <= direction_idx < len(self.adjacency_matrices):
            return self.adjacency_matrices[direction_idx]
        else:
            raise ValueError(f"Direction index out of range: {direction_idx}")
    
    def visualize_neighborhood(self, node_idx: int, order: int = 1) -> np.ndarray:
        grid = np.zeros((self.rows, self.cols))
        
        center_row = node_idx // self.cols
        center_col = node_idx % self.cols
        grid[center_row, center_col] = 2  # Center node marked as 2
        
        if order == 1:
            neighbors = self.get_neighbors(node_idx)
        else:
            neighbors = self.get_higher_order_neighbors(node_idx, order)
        
        for neighbor_idx in neighbors:
            row = neighbor_idx // self.cols
            col = neighbor_idx % self.cols
            grid[row, col] = 1  # Neighbors marked as 1
        
        return grid
    
    def get_directional_neighbors(
            self, node_idx: int, direction_idx: int, order: int = 1) -> List[int]:
        """Return the exact N_c^(order)(x) defined by paper Eq. (5)."""
        self._validate_node(node_idx)
        if not 0 <= direction_idx < self.num_directions:
            raise ValueError("direction index out of range")
        if order < 1:
            raise ValueError("order must be positive")

        if order == 1:
            neighbor = int(self.adjacency_index[node_idx, direction_idx])
            return [] if neighbor < 0 else [neighbor]

        previous = self.get_directional_neighbors(
            node_idx, direction_idx, order - 1
        )
        neighbors = set()
        for neighbor in previous:
            if neighbor != node_idx:
                neighbors.update(self.get_directional_neighbors(
                    neighbor, direction_idx, order - 1
                ))
        neighbors.discard(node_idx)
        return sorted(neighbors)

    def get_higher_order_neighbors(self, node_idx: int, order: int) -> List[int]:
        """
        Get higher-order neighbors
        
        Args:
            node_idx: Node index
            order: Order
            
        Returns:
            List of higher-order neighbor indices
        """
        if order < 1:
            raise ValueError("order must be positive")
        return sorted({
            neighbor
            for direction_idx in range(self.num_directions)
            for neighbor in self.get_directional_neighbors(
                node_idx, direction_idx, order
            )
        })

    def _validate_node(self, node_idx: int) -> None:
        if not 0 <= node_idx < self.num_nodes:
            raise ValueError("node index out of range")
