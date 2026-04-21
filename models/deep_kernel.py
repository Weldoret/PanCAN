"""
Deep Kernel Mapping Module

Implements deep kernel mapping network, extending context-aware kernel mapping to deep network structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
import math


class KernelRecursiveMapping(nn.Module):
    """Kernel recursive mapping layer"""
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 num_directions: int = 4,
                 gamma: float = 0.5,
                 use_residual: bool = True):
        """
        Initialize kernel recursive mapping layer
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            num_directions: Number of directions
            gamma: Context influence factor
            use_residual: Whether to use residual connection
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_directions = num_directions
        self.gamma = gamma
        self.use_residual = use_residual
        
        self.conv_reduce = nn.Conv2d(
            in_channels=input_dim * (num_directions + 1),
            out_channels=output_dim,
            kernel_size=1,
            bias=False
        )
        
        self.bn = nn.BatchNorm2d(output_dim)
        
        self.activation = nn.ReLU(inplace=True)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        nn.init.kaiming_normal_(self.conv_reduce.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, 
                phi_t: torch.Tensor, 
                directional_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass, implementing equation (17)
        
        Args:
            phi_t: t-th layer features [batch_size, num_nodes, input_dim]
            directional_features: List of directional features, each with shape [batch_size, num_nodes, input_dim]
            
        Returns:
            (t+1)-th layer features [batch_size, num_nodes, output_dim]
        """
        batch_size, num_nodes, input_dim = phi_t.shape
        
        features_to_concat = [phi_t]
        
        for i, directional_feature in enumerate(directional_features):
            weighted_feature = math.sqrt(self.gamma) * directional_feature
            features_to_concat.append(weighted_feature)
        
        concatenated = torch.cat(features_to_concat, dim=-1)
        
        concatenated = concatenated.permute(0, 2, 1).unsqueeze(-1)
        
        reduced = self.conv_reduce(concatenated)
        
        reduced = self.bn(reduced)
        reduced = self.activation(reduced)
        
        reduced = reduced.squeeze(-1).permute(0, 2, 1)
        
        if self.use_residual and self.input_dim == self.output_dim:
            reduced = reduced + phi_t
        
        return reduced


class ContextAwareKernelLayer(nn.Module):
    """Context-aware kernel layer"""
    
    def __init__(self,
                 feature_dim: int,
                 num_directions: int = 4,
                 gamma: float = 0.5,
                 use_dimension_reduction: bool = True,
                 reduction_ratio: float = 0.5):
        """
        Initialize context-aware kernel layer
        
        Args:
            feature_dim: Feature dimension
            num_directions: Number of directions
            gamma: Context influence factor
            use_dimension_reduction: Whether to use dimension reduction
            reduction_ratio: Dimension reduction ratio
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_directions = num_directions
        self.gamma = gamma
        self.use_dimension_reduction = use_dimension_reduction
        
        if use_dimension_reduction:
            self.reduced_dim = max(1, int(feature_dim * reduction_ratio))
        else:
            self.reduced_dim = feature_dim
        
        if use_dimension_reduction:
            self.dimension_reduction = nn.Sequential(
                nn.Conv2d(feature_dim, self.reduced_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.reduced_dim),
                nn.ReLU(inplace=True)
            )
        
        self.recursive_mapping = KernelRecursiveMapping(
            input_dim=self.reduced_dim,
            output_dim=feature_dim,
            num_directions=num_directions,
            gamma=gamma,
            use_residual=True
        )
    
    def forward(self,
                phi_t: torch.Tensor,
                adjacency_matrices: List[torch.Tensor],
                directional_features: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            phi_t: t-th layer features [batch_size, num_nodes, feature_dim]
            adjacency_matrices: List of adjacency matrices
            directional_features: List of directional features (optional)
            
        Returns:
            (t+1)-th layer features
        """
        batch_size, num_nodes, feature_dim = phi_t.shape
        
        if directional_features is None:
            directional_features = self._compute_directional_features(phi_t, adjacency_matrices)
        
        if self.use_dimension_reduction:
            phi_t_conv = phi_t.permute(0, 2, 1).unsqueeze(-1)
            phi_t_reduced = self.dimension_reduction(phi_t_conv)
            phi_t_reduced = phi_t_reduced.squeeze(-1).permute(0, 2, 1)
            
            reduced_directional_features = []
            for feature in directional_features:
                feature_conv = feature.permute(0, 2, 1).unsqueeze(-1)
                feature_reduced = self.dimension_reduction(feature_conv)
                feature_reduced = feature_reduced.squeeze(-1).permute(0, 2, 1)
                reduced_directional_features.append(feature_reduced)
            
            phi_t = phi_t_reduced
            directional_features = reduced_directional_features
        
        phi_t_plus_1 = self.recursive_mapping(phi_t, directional_features)
        
        return phi_t_plus_1
    
    def _compute_directional_features(self,
                                     features: torch.Tensor,
                                     adjacency_matrices: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Compute directional features
        
        Args:
            features: Input features [batch_size, num_nodes, feature_dim]
            adjacency_matrices: List of adjacency matrices
            
        Returns:
            List of directional features
        """
        batch_size, num_nodes, feature_dim = features.shape
        directional_features = []
        
        for P_c in adjacency_matrices:
            P_c_expanded = P_c.unsqueeze(0).expand(batch_size, -1, -1)
            
            directional_feature = torch.bmm(P_c_expanded, features)
            directional_features.append(directional_feature)
        
        return directional_features


class ExplicitKernelMap(nn.Module):
    """Explicit kernel mapping"""
    
    def __init__(self,
                 feature_dim: int,
                 num_layers: int = 3,
                 num_directions: int = 4,
                 gamma: float = 0.5):
        """
        Initialize explicit kernel mapping
        
        Args:
            feature_dim: Feature dimension
            num_layers: Number of layers
            num_directions: Number of directions
            gamma: Context influence factor
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.gamma = gamma
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = ContextAwareKernelLayer(
                feature_dim=feature_dim,
                num_directions=num_directions,
                gamma=gamma,
                use_dimension_reduction=(i > 0),
                reduction_ratio=0.5
            )
            self.layers.append(layer)
    
    def forward(self,
                phi_0: torch.Tensor,
                adjacency_matrices: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass, implementing recursive expansion of equation (4)
        
        Args:
            phi_0: Initial feature mapping [batch_size, num_nodes, feature_dim]
            adjacency_matrices: List of adjacency matrices
            
        Returns:
            Deep kernel mapping features [batch_size, num_nodes, feature_dim]
        """
        phi_t = phi_0
        
        for layer in self.layers:
            phi_t = layer(phi_t, adjacency_matrices)
        
        return phi_t
    
    def compute_kernel_value(self,
                            x_i: torch.Tensor,
                            x_j: torch.Tensor,
                            adjacency_matrices: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute kernel value between two units, implementing equation (18)
        
        Args:
            x_i: Features of unit i [feature_dim]
            x_j: Features of unit j [feature_dim]
            adjacency_matrices: List of adjacency matrices
            
        Returns:
            Kernel value
        """
        x_i = x_i.unsqueeze(0).unsqueeze(0)
        x_j = x_j.unsqueeze(0).unsqueeze(0)
        
        phi_i = self.forward(x_i, adjacency_matrices)
        phi_j = self.forward(x_j, adjacency_matrices)
        
        kernel_value = torch.sum(phi_i * phi_j, dim=-1)
        
        return kernel_value.squeeze()


class DynamicKernelNetwork(nn.Module):
    """Dynamic kernel network"""
    
    def __init__(self,
                 input_dim: int,
                 num_layers: int = 3,
                 num_directions: int = 4,
                 gamma: float = 0.5,
                 adaptive_depth: bool = True,
                 max_iterations: int = 10):
        """
        Initialize dynamic kernel network
        
        Args:
            input_dim: Input dimension
            num_layers: Initial number of layers
            num_directions: Number of directions
            gamma: Context influence factor
            adaptive_depth: Whether to use adaptive depth
            max_iterations: Maximum iterations (for adaptive depth)
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.gamma = gamma
        self.adaptive_depth = adaptive_depth
        self.max_iterations = max_iterations
        
        self.kernel_map = ExplicitKernelMap(
            feature_dim=input_dim,
            num_layers=num_layers,
            num_directions=num_directions,
            gamma=gamma
        )
        
        if adaptive_depth:
            self.convergence_threshold = nn.Parameter(torch.tensor(1e-4))
    
    def forward(self,
                phi_0: torch.Tensor,
                adjacency_matrices: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass
        
        Args:
            phi_0: Initial features [batch_size, num_nodes, feature_dim]
            adjacency_matrices: List of adjacency matrices
            
        Returns:
            phi_K: Final features
            info: Computation information
        """
        batch_size, num_nodes, feature_dim = phi_0.shape
        
        if self.adaptive_depth:
            return self._forward_adaptive(phi_0, adjacency_matrices)
        else:
            phi_K = self.kernel_map(phi_0, adjacency_matrices)
            info = {'num_layers_used': self.num_layers, 'converged': False}
            return phi_K, info
    
    def _forward_adaptive(self,
                         phi_0: torch.Tensor,
                         adjacency_matrices: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Adaptive depth forward pass
        
        Args:
            phi_0: Initial features
            adjacency_matrices: List of adjacency matrices
            
        Returns:
            phi_K: Final features
            info: Computation information
        """
        phi_t = phi_0
        prev_phi_t = phi_0
        
        for iteration in range(self.max_iterations):
            phi_t_plus_1 = self._apply_single_layer(phi_t, adjacency_matrices, iteration)
            
            delta = torch.norm(phi_t_plus_1 - prev_phi_t, p='fro').item() / torch.norm(prev_phi_t, p='fro').item()
            
            prev_phi_t = phi_t
            phi_t = phi_t_plus_1
            
            if delta < self.convergence_threshold.item():
                break
        
        info = {
            'num_layers_used': iteration + 1,
            'converged': iteration < self.max_iterations - 1,
            'final_delta': delta
        }
        
        return phi_t, info
    
    def _apply_single_layer(self,
                           phi_t: torch.Tensor,
                           adjacency_matrices: List[torch.Tensor],
                           layer_index: int) -> torch.Tensor:
        """
        Apply single layer mapping
        
        Args:
            phi_t: Current features
            adjacency_matrices: List of adjacency matrices
            layer_index: Layer index
            
        Returns:
            Next layer features
        """
        batch_size, num_nodes, feature_dim = phi_t.shape
        
        directional_features = []
        for P_c in adjacency_matrices:
            P_c_expanded = P_c.unsqueeze(0).expand(batch_size, -1, -1)
            directional_feature = torch.bmm(P_c_expanded, phi_t)
            directional_features.append(directional_feature)
        
        features_to_concat = [phi_t]
        for feature in directional_features:
            features_to_concat.append(math.sqrt(self.gamma) * feature)
        
        concatenated = torch.cat(features_to_concat, dim=-1)
        
        concatenated = concatenated.permute(0, 2, 1).unsqueeze(-1)
        
        conv_layer = nn.Conv2d(
            in_channels=concatenated.shape[1],
            out_channels=feature_dim,
            kernel_size=1,
            bias=False
        ).to(phi_t.device)
        
        nn.init.kaiming_normal_(conv_layer.weight, mode='fan_out', nonlinearity='relu')
        
        output = conv_layer(concatenated)
        output = output.squeeze(-1).permute(0, 2, 1)
        
        output = output + phi_t
        
        return output


class DeepKernelMappingNetwork(nn.Module):
    """Deep kernel mapping network"""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 num_directions: int = 4,
                 gamma: float = 0.5,
                 use_explicit_map: bool = True):
        """
        Initialize deep kernel mapping network
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            num_directions: Number of directions
            gamma: Context influence factor
            use_explicit_map: Whether to use explicit mapping
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)
        self.num_directions = num_directions
        self.gamma = gamma
        self.use_explicit_map = use_explicit_map
        
        self.layers = nn.ModuleList()
        
        self.layers.append(
            ContextAwareKernelLayer(
                feature_dim=input_dim,
                num_directions=num_directions,
                gamma=gamma,
                use_dimension_reduction=False,
                reduction_ratio=0.5
            )
        )
        
        prev_dim = input_dim
        for hidden_dim in hidden_dims[:-1]:
            layer = ContextAwareKernelLayer(
                feature_dim=prev_dim,
                num_directions=num_directions,
                gamma=gamma,
                use_dimension_reduction=True,
                reduction_ratio=0.5
            )
            self.layers.append(layer)
            prev_dim = hidden_dim
        
        if len(hidden_dims) > 0:
            self.layers.append(
                ContextAwareKernelLayer(
                    feature_dim=prev_dim,
                    num_directions=num_directions,
                    gamma=gamma,
                    use_dimension_reduction=False,
                    reduction_ratio=0.5
                )
            )
        
        if use_explicit_map:
            self.explicit_map = ExplicitKernelMap(
                feature_dim=hidden_dims[-1] if hidden_dims else input_dim,
                num_layers=2,
                num_directions=num_directions,
                gamma=gamma
            )
    
    def forward(self,
                phi_0: torch.Tensor,
                adjacency_matrices: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            phi_0: Initial features [batch_size, num_nodes, input_dim]
            adjacency_matrices: List of adjacency matrices
            
        Returns:
            Deep mapped features
        """
        phi_t = phi_0
        
        for layer in self.layers:
            phi_t = layer(phi_t, adjacency_matrices)
        
        if self.use_explicit_map:
            phi_t = self.explicit_map(phi_t, adjacency_matrices)
        
        return phi_t
    
    def compute_image_kernel(self,
                           image1_cells: torch.Tensor,
                           image2_cells: torch.Tensor,
                           adjacency_matrices: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute kernel function value between two images, implementing equation (20)
        
        Args:
            image1_cells: Image 1 cell features [num_cells1, feature_dim]
            image2_cells: Image 2 cell features [num_cells2, feature_dim]
            adjacency_matrices: List of adjacency matrices
            
        Returns:
            Image kernel value
        """
        batch_size1, num_cells1, feature_dim = image1_cells.shape
        batch_size2, num_cells2, _ = image2_cells.shape
        
        phi_image1 = self.forward(image1_cells, adjacency_matrices)
        phi_image2 = self.forward(image2_cells, adjacency_matrices)
        
        image1_representation = torch.sum(phi_image1, dim=1)
        image2_representation = torch.sum(phi_image2, dim=1)
        
        kernel_values = torch.sum(image1_representation * image2_representation, dim=-1)
        
        return kernel_values
    
    def get_kernel_mapping(self, cells: torch.Tensor, adjacency_matrices: List[torch.Tensor]) -> torch.Tensor:
        """
        Get kernel mapping representation, implementing equation (21)
        
        Args:
            cells: Cell features [batch_size, num_cells, feature_dim]
            adjacency_matrices: List of adjacency matrices
            
        Returns:
            Kernel mapping representation [batch_size, output_dim]
        """
        phi_cells = self.forward(cells, adjacency_matrices)
        
        phi_image = torch.sum(phi_cells, dim=1)
        
        return phi_image