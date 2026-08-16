

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict, Any
import math


class CenteredSelfAttention(nn.Module):
    """Centered self-attention module"""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, 
                center_feature: torch.Tensor, 
                connected_features: torch.Tensor) -> torch.Tensor:
        if center_feature.dim() == 1:
            center_feature = center_feature.unsqueeze(0)
        
        Q = self.query(center_feature)
        K = self.key(connected_features)
        V = self.value(connected_features)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))
        attention_probs = self.softmax(attention_scores)
        
        max_score_index = torch.argmax(attention_probs, dim=-1, keepdim=True)
        
        if V.dim() == 2:
            V = V.unsqueeze(0)
        
        most_similar_feature = torch.gather(
            V, 
            1, 
            max_score_index.unsqueeze(-1).expand(-1, -1, V.size(-1))
        )
        
        max_attention_prob = attention_probs.gather(1, max_score_index)
        
        weighted_most_similar_feature = most_similar_feature * max_attention_prob.unsqueeze(-1)
        
        return weighted_most_similar_feature.squeeze(1)


class AnchorBoxGenerator(nn.Module):
    """Anchor box generator"""
    
    def __init__(self, 
                 grid_size: Tuple[int, int],
                 anchor_sizes: List[Tuple[int, int]],
                 strides: Optional[Tuple[int, int]] = None):
        super().__init__()
        self.grid_rows, self.grid_cols = grid_size
        self.anchor_sizes = anchor_sizes
        self.strides = strides if strides else (1, 1)
    
    def generate_anchor_boxes(self) -> List[Tuple[int, int, int, int]]:
        """
        Generate anchor boxes
        
        Returns:
            List of anchor boxes, each as (top, left, bottom, right)
        """
        anchor_boxes = []
        stride_h, stride_w = self.strides
        
        for i in range(0, self.grid_rows, stride_h):
            for j in range(0, self.grid_cols, stride_w):
                for (h, w) in self.anchor_sizes:
                    top = i
                    left = j
                    bottom = min(i + h, self.grid_rows)
                    right = min(j + w, self.grid_cols)
                    
                    if bottom > top and right > left:
                        anchor_boxes.append((top, left, bottom, right))
        
        return anchor_boxes
    
    def get_cells_in_anchor(self, 
                           anchor_box: Tuple[int, int, int, int]) -> List[int]:
        """
        Get cell indices within anchor box
        
        Args:
            anchor_box: Anchor box (top, left, bottom, right)
            
        Returns:
            List of cell indices
        """
        top, left, bottom, right = anchor_box
        cell_indices = []
        
        for i in range(top, bottom):
            for j in range(left, right):
                cell_idx = i * self.grid_cols + j
                cell_indices.append(cell_idx)
        
        return cell_indices


class SubregionSelector(nn.Module):
    """Subregion selector"""
    
    def __init__(self, 
                 feature_dim: int,
                 top_k: int = 3,
                 use_nms: bool = True):
        """
        Initialize subregion selector
        
        Args:
            feature_dim: Feature dimension
            top_k: Number of top-k anchors to select
            use_nms: Whether to use non-maximum suppression
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.top_k = top_k
        self.use_nms = use_nms
        
    def forward(self, 
                features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select saliency high anchors
        
        Args:
            features: Features [batch_size, num_cells, feature_dim]
            
        Returns:
            anchors: Anchor indices [batch_size, top_k]
            saliency: Saliency scores [batch_size, num_cells]
        """
        batch_size, num_cells, feature_dim = features.shape
        
        saliency = torch.norm(features, p=2, dim=-1)
        
        if self.use_nms:
            anchors = self.non_maximum_suppression(saliency)
        else:
            _, anchors = torch.topk(saliency, k=self.top_k, dim=-1)
        
        return anchors, saliency
    
    def non_maximum_suppression(self, 
                               saliency: torch.Tensor,
                               suppression_radius: int = 2) -> torch.Tensor:
        """
        Non-maximum suppression
        
        Args:
            saliency: Saliency scores [batch_size, num_cells]
            suppression_radius: Suppression radius
            
        Returns:
            Anchor indices [batch_size, top_k]
        """
        batch_size, num_cells = saliency.shape
        device = saliency.device
        
        anchors_list = []
        
        for b in range(batch_size):
            batch_saliency = saliency[b]
            selected_indices = []
            
            remaining_saliency = batch_saliency.clone()
            
            for _ in range(self.top_k):
                if remaining_saliency.max() <= 0:
                    break
                
                max_idx = torch.argmax(remaining_saliency).item()
                selected_indices.append(max_idx)
                
                max_idx = min(max_idx, num_cells - 1)
                
                for i in range(max(0, max_idx - suppression_radius), min(num_cells, max_idx + suppression_radius + 1)):
                    if i != max_idx:
                        remaining_saliency[i] = 0
            
            while len(selected_indices) < self.top_k:
                selected_indices.append(0)
            
            anchors_list.append(torch.tensor(selected_indices, device=device))
        
        return torch.stack(anchors_list, dim=0)


class MultiHeadScaleAttention(nn.Module):
    """Multi-head scale attention"""
    
    def __init__(self, 
                 feature_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor) -> torch.Tensor:
        batch_size, num_queries, _ = query.shape
        _, num_keys, _ = key.shape
        
        Q = self.q_proj(query).view(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, num_keys, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, num_keys, self.num_heads, self.head_dim).transpose(1, 2)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context = torch.matmul(attention_probs, V)
        
        context = context.transpose(1, 2).contiguous().view(batch_size, num_queries, self.feature_dim)
        
        output = self.out_proj(context)
        
        return output


class MultiScaleFeatureAggregator(nn.Module):
    """Multi-scale feature aggregator"""
    
    def __init__(self,
                 feature_dim: int,
                 scales: List[Tuple[int, int]],
                 anchor_sizes: List[Tuple[int, int]],
                 top_k: int = 3,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 attention_heads: Optional[int] = None,
                 stride: Optional[int] = None):
        super().__init__()
        if attention_heads is not None:
            num_heads = attention_heads

        self.feature_dim = feature_dim
        self.scales = scales
        self.anchor_sizes = anchor_sizes
        self.top_k = top_k
        self.stride = stride
        
        self.selector = SubregionSelector(feature_dim, top_k=top_k)
        
        self.multi_head_attention = MultiHeadScaleAttention(
            feature_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.scale_aggregators = nn.ModuleList()
        base_grid = scales[0]
        for scale in scales:
            stride = (base_grid[0] // scale[0], base_grid[1] // scale[1])
            anchor_count = len(
                AnchorBoxGenerator(
                    grid_size=base_grid,
                    anchor_sizes=anchor_sizes,
                    strides=stride,
                ).generate_anchor_boxes()
            )
            aggregator = nn.Sequential(
                nn.Linear(feature_dim * anchor_count, feature_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.scale_aggregators.append(aggregator)
    
    def forward(self, 
                features: torch.Tensor,
                grid_size: Tuple[int, int]) -> torch.Tensor:
        batch_size, num_cells, feature_dim = features.shape
        grid_rows, grid_cols = grid_size
        
        assert num_cells == grid_rows * grid_cols, "Number of features doesn't match grid size"
        
        scale_features = []
        
        for scale_idx, (scale_rows, scale_cols) in enumerate(self.scales):
            anchor_generator = AnchorBoxGenerator(
                grid_size=(grid_rows, grid_cols),
                anchor_sizes=self.anchor_sizes,
                strides=(grid_rows // scale_rows, grid_cols // scale_cols)
            )
            
            anchor_boxes = anchor_generator.generate_anchor_boxes()
            
            anchor_features_list = []
            
            for anchor_box in anchor_boxes:
                cell_indices = anchor_generator.get_cells_in_anchor(anchor_box)
                
                if len(cell_indices) > 0:
                    anchor_feat = features[:, cell_indices, :]
                    
                    anchors, _ = self.selector(anchor_feat)
                    
                    aggregated_anchor_feat = self.aggregate_anchor_features(anchor_feat, anchors)
                    anchor_features_list.append(aggregated_anchor_feat)
            
            if anchor_features_list:
                all_anchor_features = torch.stack(anchor_features_list, dim=1)
                
                scale_feat = self.scale_aggregators[scale_idx](
                    all_anchor_features.view(batch_size, -1)
                )
                
                scale_features.append(scale_feat.unsqueeze(1))
        
        if scale_features:
            multi_scale_features = torch.cat(scale_features, dim=1)
            
            query = multi_scale_features.mean(dim=1, keepdim=True)
            
            fused_features = self.multi_head_attention(
                query=query,
                key=multi_scale_features,
                value=multi_scale_features
            )
            
            return fused_features.squeeze(1)
        else:
            return features.mean(dim=1)
    
    def aggregate_anchor_features(self,
                                 anchor_features: torch.Tensor,
                                 anchors: torch.Tensor) -> torch.Tensor:
        batch_size, num_cells, feature_dim = anchor_features.shape
        top_k = anchors.size(1)
        
        anchor_feat_list = []
        
        for b in range(batch_size):
            batch_anchors = anchors[b]
            batch_features = anchor_features[b]
            
            batch_anchors = torch.clamp(batch_anchors, 0, num_cells - 1)
            
            anchor_feat = batch_features[batch_anchors]
            anchor_feat_list.append(anchor_feat)
        
        all_anchor_feat = torch.stack(anchor_feat_list, dim=0)
        
        query = all_anchor_feat[:, 0:1, :]
        key = all_anchor_feat
        value = all_anchor_feat
        
        aggregated = self.multi_head_attention(query, key, value)
        
        return aggregated.squeeze(1)
