
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
import numpy as np


class FidelityCriterion(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, K: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        loss = -torch.trace(K @ S.t())
        return loss


class ContextCriterion(nn.Module):
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, K: torch.Tensor, P: List[torch.Tensor]) -> torch.Tensor:
        loss = 0.0
        for P_c in P:
            term = torch.trace(K @ P_c @ K.t() @ P_c.t())
            loss -= self.alpha * term
        
        return loss


class KernelRegularizer(nn.Module):
    def __init__(self, beta: float = 1.0, norm_type: str = 'frobenius'):
        super().__init__()
        self.beta = beta
        self.norm_type = norm_type
    
    def forward(self, K: torch.Tensor) -> torch.Tensor:
        if self.norm_type == 'frobenius':
            norm = torch.norm(K, p='fro')
        elif self.norm_type == 'l2':
            norm = torch.norm(K, p=2)
        else:
            raise ValueError(f"Unsupported norm type: {self.norm_type}")
        
        loss = 0.5 * self.beta * (norm ** 2)
        return loss


class ContextAwareKernel(nn.Module):
    def __init__(self, 
                 alpha: float = 0.5,
                 beta: float = 1.0,
                 num_directions: int = 4,
                 max_iterations: int = 10,
                 convergence_threshold: float = 1e-6):
        super().__init__()
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if beta <= 0:
            raise ValueError("beta must be positive")
        if num_directions < 1:
            raise ValueError("num_directions must be positive")
        self.alpha = alpha
        self.beta = beta
        self.gamma = alpha / beta
        self.num_directions = num_directions
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        
        self.fidelity_criterion = FidelityCriterion()
        self.context_criterion = ContextCriterion(alpha)
        self.regularizer = KernelRegularizer(beta)
    
    def forward(self, 
                S: torch.Tensor, 
                P: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        self._validate_inputs(S, P)
        
        K = S.clone()
        
        for iteration in range(self.max_iterations):
            K_prev = K.clone()
            
            context_term = torch.zeros_like(K)
            for P_c in P:
                context_term += P_c @ K @ P_c.t()
            
            K = S + self.gamma * context_term
            
            delta = torch.norm(K - K_prev, p='fro').item()
            if delta < self.convergence_threshold:
                break
        
        total_loss = self.compute_total_loss(K, S, P)
        
        info = {
            'iterations': iteration + 1,
            'converged': delta < self.convergence_threshold,
            'final_delta': delta,
            'total_loss': total_loss.item()
        }
        
        return K, info
    
    def compute_total_loss(self, 
                          K: torch.Tensor, 
                          S: torch.Tensor, 
                          P: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute total loss
        
        Args:
            K: Kernel matrix
            S: Similarity matrix
            P: List of adjacency matrices
            
        Returns:
            Total loss value
        """
        fidelity_loss = self.fidelity_criterion(K, S)
        context_loss = self.context_criterion(K, P)
        regularizer_loss = self.regularizer(K)
        
        total_loss = fidelity_loss + context_loss + regularizer_loss
        return total_loss
    
    def _validate_inputs(self, S: torch.Tensor, P: List[torch.Tensor]):
        """Validate inputs"""
        assert len(S.shape) == 2, "S must be a 2D matrix"
        assert len(P) == self.num_directions, f"Expected {self.num_directions} adjacency matrices"
        
        for i, P_c in enumerate(P):
            assert P_c.shape == S.shape, f"P[{i}] shape {P_c.shape} doesn't match S shape {S.shape}"


class KernelMappingLayer(nn.Module):
    """Kernel mapping layer"""
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 kernel_type: str = 'gaussian',
                 learnable: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_type = kernel_type
        self.learnable = learnable
        
        if kernel_type == 'linear':
            self.W = nn.Parameter(torch.randn(input_dim, output_dim), requires_grad=learnable)
        elif kernel_type == 'polynomial':
            self.W = nn.Parameter(torch.randn(input_dim, output_dim), requires_grad=learnable)
            self.c = nn.Parameter(torch.tensor(1.0), requires_grad=learnable)
            self.d = nn.Parameter(torch.tensor(2.0), requires_grad=learnable)
        elif kernel_type == 'gaussian':
            self.sigma = nn.Parameter(torch.tensor(1.0), requires_grad=learnable)
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_type == 'linear':
            return x @ self.W
        
        elif self.kernel_type == 'polynomial':
            linear_term = x @ self.W
            return (self.c + linear_term) ** self.d
        
        elif self.kernel_type == 'gaussian':
            return x
        
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")
    
    def compute_kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.kernel_type == 'linear':
            phi1 = self.forward(x1)
            phi2 = self.forward(x2)
            return phi1 @ phi2.t()
        
        elif self.kernel_type == 'polynomial':
            phi1 = self.forward(x1)
            phi2 = self.forward(x2)
            return (self.c + phi1 @ phi2.t()) ** self.d
        
        elif self.kernel_type == 'gaussian':
            n1 = x1.size(0)
            n2 = x2.size(0)
            
            x1_norm = (x1 ** 2).sum(dim=1).view(n1, 1)
            x2_norm = (x2 ** 2).sum(dim=1).view(1, n2)
            dist_sq = x1_norm + x2_norm - 2.0 * x1 @ x2.t()
            
            return torch.exp(-dist_sq / (2.0 * self.sigma ** 2))
        
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")


class ContextAwareKernelMap(nn.Module):
    """Explicit context-aware kernel map from the paper's Eqs. (3)--(6).

    Each mapping layer concatenates the initial cell map with directional
    aggregates of the current map, scaled by ``sqrt(alpha / beta)``, then
    applies a pointwise projection to keep the representation usable by later
    modules.
    """
    
    def __init__(self,
                 feature_dim: int,
                 kernel_dim: int,
                 alpha: float = 0.5,
                 beta: float = 1.0,
                 num_directions: int = 4,
                 kernel_type: str = 'gaussian',
                 num_layers: int = 1,
                 num_nodes: Optional[int] = None,
                 learnable_neighborhoods: bool = True):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")

        self.feature_dim = feature_dim
        self.kernel_dim = kernel_dim
        self.num_directions = num_directions
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        
        self.kernel_mapping = KernelMappingLayer(
            input_dim=feature_dim,
            output_dim=kernel_dim,
            kernel_type=kernel_type,
            learnable=True
        )
        # The Gaussian mapping intentionally keeps the original feature space
        # for similarity computation, but the rest of the network consumes the
        # configured kernel dimension. Project that downstream representation
        # explicitly instead of reshaping a tensor with the wrong size.
        self.feature_projection = (
            nn.Linear(feature_dim, kernel_dim)
            if feature_dim != kernel_dim else nn.Identity()
        )
        
        self.context_kernel = ContextAwareKernel(
            alpha=alpha,
            beta=beta,
            num_directions=num_directions
        )
        self.mapping_layers = nn.ModuleList([
            nn.Conv1d(
                kernel_dim * (num_directions + 1),
                kernel_dim,
                kernel_size=1,
            )
            for _ in range(num_layers)
        ])

        if learnable_neighborhoods and num_nodes is not None:
            initial_logit = torch.log(torch.expm1(torch.tensor(1.0))).item()
            self.neighborhood_logits = nn.Parameter(
                torch.full((num_layers, num_directions, num_nodes), initial_logit)
            )
        else:
            self.register_parameter('neighborhood_logits', None)
    
    def forward(self,
                features: torch.Tensor,
                adjacency_matrices: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_nodes, feature_dim = features.shape

        if feature_dim != self.feature_dim:
            raise ValueError(
                f"expected feature dimension {self.feature_dim}, got {feature_dim}"
            )
        if len(adjacency_matrices) != self.num_directions:
            raise ValueError(
                f"expected {self.num_directions} adjacency matrices, "
                f"got {len(adjacency_matrices)}"
            )

        # Phi^[0] is the approximate feature map initialized from the visual
        # representation, as in Eq. (3)'s definition of Phi^[0].
        kernel_features = self.kernel_mapping(features.reshape(-1, feature_dim))
        if kernel_features.size(-1) != self.kernel_dim:
            kernel_features = self.feature_projection(kernel_features)
        kernel_features = kernel_features.view(batch_size, num_nodes, self.kernel_dim)
        initial_features = kernel_features

        matrices = []
        for matrix in adjacency_matrices:
            matrix = matrix.to(
                device=kernel_features.device,
                dtype=kernel_features.dtype,
            )
            if matrix.shape != (num_nodes, num_nodes):
                raise ValueError(
                    "adjacency matrices must match the number of feature nodes"
                )
            matrices.append(matrix)

        sqrt_gamma = self.context_kernel.gamma ** 0.5
        for layer_index, layer in enumerate(self.mapping_layers):
            layer_matrices = self._weighted_adjacencies(matrices, layer_index)
            directional_features = [
                torch.bmm(
                    matrix.unsqueeze(0).expand(batch_size, -1, -1),
                    kernel_features,
                )
                for matrix in layer_matrices
            ]
            unfolded = torch.cat(
                [initial_features]
                + [sqrt_gamma * feature for feature in directional_features],
                dim=-1,
            )
            kernel_features = layer(unfolded.transpose(1, 2)).transpose(1, 2)

        # The Gram matrix is the inner product of the unfolded cell maps
        # (Eq. (4)); retaining the batch dimension keeps every image's map
        # independent while allowing the network to consume Phi^[T].
        kernel_matrix = torch.bmm(
            kernel_features,
            kernel_features.transpose(1, 2),
        )
        return kernel_matrix, kernel_features

    def get_adjacency_matrices(
            self, adjacency_matrices: List[torch.Tensor]) -> List[torch.Tensor]:
        """Return the final learned directional neighborhoods for downstream stages."""
        if self.neighborhood_logits is None:
            return adjacency_matrices
        matrices = [matrix.to(self.neighborhood_logits.device)
                    for matrix in adjacency_matrices]
        return self._weighted_adjacencies(matrices, self.num_layers - 1)

    def _weighted_adjacencies(
            self,
            adjacency_matrices: List[torch.Tensor],
            layer_index: int,
    ) -> List[torch.Tensor]:
        if self.neighborhood_logits is None:
            return adjacency_matrices
        if len(adjacency_matrices) != self.num_directions:
            raise ValueError("adjacency count does not match neighborhood parameters")
        if any(matrix.size(0) != self.num_nodes for matrix in adjacency_matrices):
            raise ValueError("adjacency size does not match neighborhood parameters")

        weights = F.softplus(self.neighborhood_logits[layer_index])
        return [matrix * weights[direction].unsqueeze(-1)
                for direction, matrix in enumerate(adjacency_matrices)]
    
    def _compute_similarity_matrix(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, feature_dim = features.shape
        
        sample_features = features[0]
        
        S = self.kernel_mapping.compute_kernel(sample_features, sample_features)
        
        return S
    
    def compute_image_kernel(self,
                           image1_features: torch.Tensor,
                           image2_features: torch.Tensor,
                           adjacency_matrices: List[torch.Tensor]) -> torch.Tensor:
        _, mapped1 = self.forward(image1_features.unsqueeze(0), adjacency_matrices)
        _, mapped2 = self.forward(image2_features.unsqueeze(0), adjacency_matrices)

        image1_representation = mapped1.sum(dim=1)
        image2_representation = mapped2.sum(dim=1)
        return torch.sum(image1_representation * image2_representation)
