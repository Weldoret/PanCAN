"""
Network architecture module

Implements the complete multi-scale context-aware deep kernel mapping network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any, Sequence
import torchvision.models as models

# Import other modules
from .neighborhood import (
    DirectionalNeighborhood,
    NeighborhoodSystem,
    generate_adjacency_index_matrix,
)
from .context_kernel import ContextAwareKernelMap
from .multi_order import MultiOrderContextAggregator
from .random_walk import RandomWalkAttention
from .multi_scale import MultiScaleFeatureAggregator, CenteredSelfAttention
from .deep_kernel import DeepKernelMappingNetwork, ExplicitKernelMap


class FeatureExtractor(nn.Module):
    """Feature extractor"""
    
    def __init__(self, backbone_name: str = "resnet101", pretrained: bool = True):
        """
        Initialize feature extractor
        
        Args:
            backbone_name: Backbone network name
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        self.backbone_name = backbone_name
        self.pretrained = pretrained
        
        self.backbone = self._create_backbone(backbone_name, pretrained)
        
        self.feature_dim = self._get_feature_dim(backbone_name)
        
        self._freeze_layers()
    
    def _create_backbone(self, backbone_name: str, pretrained: bool) -> nn.Module:
        """Create backbone network"""
        if backbone_name == "resnet101":
            backbone = models.resnet101(pretrained=pretrained)
            modules = list(backbone.children())[:-2]
            return nn.Sequential(*modules)
        
        elif backbone_name == "resnet50":
            backbone = models.resnet50(pretrained=pretrained)
            modules = list(backbone.children())[:-2]
            return nn.Sequential(*modules)
        
        elif backbone_name == "resnet34":
            backbone = models.resnet34(pretrained=pretrained)
            modules = list(backbone.children())[:-2]
            return nn.Sequential(*modules)
        
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
    
    def _get_feature_dim(self, backbone_name: str) -> int:
        """Get feature dimension"""
        if backbone_name.startswith("resnet"):
            if backbone_name.endswith("101"):
                return 2048
            elif backbone_name.endswith("50"):
                return 2048
            elif backbone_name.endswith("34"):
                return 512
            else:
                return 2048
        else:
            return 2048
    
    def _freeze_layers(self, freeze_ratio: float = 0.5):
        """Freeze partial layers"""
        if self.pretrained:
            num_layers = len(list(self.backbone.parameters()))
            num_freeze = int(num_layers * freeze_ratio)
            
            for i, param in enumerate(self.backbone.parameters()):
                if i < num_freeze:
                    param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input image [batch_size, 3, H, W]
            
        Returns:
            Feature map [batch_size, feature_dim, H', W']
        """
        return self.backbone(x)


class RotatedGridPositionFeatures(nn.Module):
    """Inject 2D rotary positional features into grid-cell descriptors.

    The paper specifies rotated positional features but does not define their
    frequency schedule. This uses the standard fixed RoPE base while keeping
    the backbone feature dimension unchanged.
    """

    def __init__(self, rows: int, cols: int, feature_dim: int, base: float = 10000.0):
        super().__init__()
        if rows < 1 or cols < 1:
            raise ValueError("grid dimensions must be positive")
        if feature_dim < 4 or feature_dim % 4:
            raise ValueError("feature_dim must be divisible by four for 2D rotation")

        pair_dim = feature_dim // 4
        inverse_frequency = 1.0 / (
            base ** (torch.arange(pair_dim, dtype=torch.float32) / pair_dim)
        )
        row_index = torch.arange(rows, dtype=torch.float32).repeat_interleave(cols)
        col_index = torch.arange(cols, dtype=torch.float32).repeat(rows)
        row_angles = row_index[:, None] * inverse_frequency[None, :]
        col_angles = col_index[:, None] * inverse_frequency[None, :]

        self.rows = rows
        self.cols = cols
        self.feature_dim = feature_dim
        self.register_buffer("row_cos", row_angles.cos())
        self.register_buffer("row_sin", row_angles.sin())
        self.register_buffer("col_cos", col_angles.cos())
        self.register_buffer("col_sin", col_angles.sin())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 3 or features.size(1) != self.rows * self.cols:
            raise ValueError(
                f"expected [batch, {self.rows * self.cols}, features] grid input"
            )
        if features.size(-1) != self.feature_dim:
            raise ValueError(
                f"expected feature dimension {self.feature_dim}, got {features.size(-1)}"
            )

        pair_dim = self.feature_dim // 4
        row_features = features[..., :2 * pair_dim].reshape(
            features.size(0), features.size(1), pair_dim, 2
        )
        col_features = features[..., 2 * pair_dim:].reshape(
            features.size(0), features.size(1), pair_dim, 2
        )

        row_features = self._rotate(
            row_features,
            self.row_cos.to(device=features.device, dtype=features.dtype),
            self.row_sin.to(device=features.device, dtype=features.dtype),
        )
        col_features = self._rotate(
            col_features,
            self.col_cos.to(device=features.device, dtype=features.dtype),
            self.col_sin.to(device=features.device, dtype=features.dtype),
        )
        return torch.cat((row_features, col_features), dim=-1)

    @staticmethod
    def _rotate(features, cos, sin):
        x, y = features.unbind(dim=-1)
        return torch.stack((x * cos - y * sin, x * sin + y * cos), dim=-1).flatten(-2)


class BackboneFactory:
    """Backbone network factory"""
    
    @staticmethod
    def create_backbone(backbone_name: str, pretrained: bool = True) -> nn.Module:
        """
        Create backbone network
        
        Args:
            backbone_name: Backbone network name
            pretrained: Whether to use pretrained weights
            
        Returns:
            Backbone network
        """
        return FeatureExtractor(backbone_name, pretrained)
    
    @staticmethod
    def get_available_backbones() -> List[str]:
        """Get available backbone network list"""
        return ["resnet34", "resnet50", "resnet101"]
    
    @staticmethod
    def get_feature_dim(backbone_name: str) -> int:
        """Get feature dimension"""
        dim_map = {
            "resnet34": 512,
            "resnet50": 2048,
            "resnet101": 2048
        }
        return dim_map.get(backbone_name, 2048)


class MultiLabelClassifier(nn.Module):
    """Multi-label classifier"""
    
    def __init__(self,
                 input_dim: int,
                 num_classes: int,
                 use_grouped_fc: bool = True,
                 num_groups: int = 5,
                 dropout_rate: float = 0.5,
                 class_groups: Optional[Sequence[Sequence[int]]] = None,
                 group_weights: Optional[Sequence[float]] = None):
        """
        Initialize multi-label classifier
        
        Args:
            input_dim: Input dimension
            num_classes: Number of classes
            use_grouped_fc: Whether to use grouped fully connected layers
            num_groups: Number of groups
            dropout_rate: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.use_grouped_fc = use_grouped_fc
        self.dropout_rate = dropout_rate

        self.dropout = nn.Dropout(dropout_rate)

        if use_grouped_fc:
            if class_groups is None:
                class_groups = [list(range(num_classes))]
            self.class_groups = tuple(tuple(int(index) for index in group) for group in class_groups)
            self._validate_groups(self.class_groups, num_classes)
            self.num_groups = len(self.class_groups)
            if group_weights is None:
                group_weights = [1.0] * self.num_groups
            if len(group_weights) != self.num_groups:
                raise ValueError("group_weights must match class_groups")
            self.register_buffer(
                "group_weights",
                torch.as_tensor(group_weights, dtype=torch.float32),
            )
            self.group_classifiers = nn.ModuleList()
            self._group_index_names = []
            for group_index, group in enumerate(self.class_groups):
                self.register_buffer(
                    f"group_indices_{group_index}",
                    torch.tensor(group, dtype=torch.long),
                )
                self._group_index_names.append(f"group_indices_{group_index}")
                self.group_classifiers.append(nn.Linear(input_dim, len(group)))
        else:
            self.class_groups = ((tuple(range(num_classes))),)
            self.num_groups = 1
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(512, num_classes)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Classification logits [batch_size, num_classes]
        """
        x = self.dropout(x)
        
        if self.use_grouped_fc:
            logits = x.new_zeros(x.size(0), self.num_classes)
            for classifier, index_name in zip(self.group_classifiers, self._group_index_names):
                logits = logits.index_copy(1, getattr(self, index_name), classifier(x))
        else:
            logits = self.classifier(x)

        return logits

    @staticmethod
    def _validate_groups(class_groups, num_classes):
        if any(not group for group in class_groups):
            raise ValueError("class_groups must not contain empty groups")
        flattened = sorted(index for group in class_groups for index in group)
        if flattened != list(range(num_classes)):
            raise ValueError("class_groups must partition every class exactly once")

    def grouped_l2(self) -> torch.Tensor:
        if not self.use_grouped_fc:
            return next(self.parameters()).new_zeros(())
        return sum(
            classifier.weight.pow(2).sum()
            for classifier in self.group_classifiers
        )


class ScaleContextBlock(nn.Module):
    """Apply multi-order context to one coarser cell grid."""

    def __init__(self, rows, cols, feature_dim, config, directions, max_order):
        super().__init__()
        self.neighborhood = NeighborhoodSystem(rows, cols, directions=directions)
        self.context_kernel = ContextAwareKernelMap(
            feature_dim=feature_dim,
            kernel_dim=feature_dim,
            alpha=config.alpha,
            beta=config.beta,
            num_directions=config.num_directions,
            num_layers=config.context_layers,
            num_nodes=rows * cols,
        )
        self.random_walk = RandomWalkAttention(
            feature_dim=feature_dim,
            num_heads=config.attention_heads,
            dropout=config.attention_dropout,
            threshold=config.random_walk_threshold,
            max_order=max_order,
            num_directions=config.num_directions,
        )
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, features):
        adjacency = self.neighborhood.adjacency_matrices
        _, mapped = self.context_kernel(features, adjacency)
        learned_adjacency = self.context_kernel.get_adjacency_matrices(adjacency)
        walked = self.random_walk(mapped, mapped, learned_adjacency)
        return self.fusion(torch.cat((mapped, walked), dim=-1))


class MultiScaleContextAwareNetwork(nn.Module):
    """Multi-scale context-aware network"""
    
    def __init__(self,
                 backbone: nn.Module,
                 num_classes: int,
                 config,
                 device: Optional[torch.device] = None,
                 class_groups: Optional[Sequence[Sequence[int]]] = None,
                 group_weights: Optional[Sequence[float]] = None):
        """
        Initialize multi-scale context-aware network
        
        Args:
            backbone: Backbone network
            num_classes: Number of classes
            config: Configuration object
            device: Device
        """
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.config = config
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if hasattr(backbone, 'feature_dim'):
            self.feature_dim = backbone.feature_dim
        else:
            self.feature_dim = config.network.backbone_feature_dim

        self.rotated_position = RotatedGridPositionFeatures(
            rows=config.network.grid_rows,
            cols=config.network.grid_cols,
            feature_dim=self.feature_dim,
        ).to(self.device)
        
        directions = DirectionalNeighborhood(config.network.num_directions).directions
        self.neighborhood_system = NeighborhoodSystem(
            rows=config.network.grid_rows,
            cols=config.network.grid_cols,
            directions=directions,
        ).to(self.device)
        
        self.context_kernel = ContextAwareKernelMap(
            feature_dim=self.feature_dim,
            kernel_dim=self.feature_dim * 2,
            alpha=config.network.alpha,
            beta=config.network.beta,
            num_directions=config.network.num_directions,
            kernel_type='gaussian',
            num_layers=config.network.context_layers,
            num_nodes=config.network.num_blocks,
        ).to(self.device)
        
        self.multi_order_aggregator = MultiOrderContextAggregator(
            max_order=config.network.max_order,
            feature_dim=self.feature_dim * 2,
            num_directions=config.network.num_directions
        ).to(self.device)

        self.random_walk = RandomWalkAttention(
            feature_dim=self.feature_dim * 2,
            num_heads=config.network.attention_heads,
            dropout=config.network.attention_dropout,
            use_threshold=True,
            threshold=config.network.random_walk_threshold,
            max_order=config.network.max_order,
            num_directions=config.network.num_directions,
        ).to(self.device)
        
        self.multi_scale_aggregator = MultiScaleFeatureAggregator(
            scales=config.network.scales,
            anchor_sizes=config.network.anchor_sizes,
            feature_dim=self.feature_dim * 2,
            attention_heads=config.network.attention_heads,
            stride=config.network.sliding_window_stride
        ).to(self.device)
        
        self.deep_kernel_network = DeepKernelMappingNetwork(
            input_dim=self.feature_dim * 2,
            hidden_dims=config.network.kernel_feature_dims,
            num_directions=config.network.num_directions,
            gamma=config.network.alpha / config.network.beta,
            use_explicit_map=True
        ).to(self.device)

        self.coarse_scale_context = nn.ModuleList([
            ScaleContextBlock(
                rows=rows,
                cols=cols,
                feature_dim=self.feature_dim * 2,
                config=config.network,
                directions=directions,
                max_order=config.network.coarse_max_order,
            )
            for rows, cols in config.network.scales[1:]
        ]).to(self.device)
        
        self.multi_scale_fusion = nn.Sequential(
            nn.Linear(config.network.final_feature_dim, config.network.final_feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(config.network.classifier_dropout),
            nn.Linear(config.network.final_feature_dim // 2, config.network.final_feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(config.network.classifier_dropout)
        ).to(self.device)
        
        self.classifier = MultiLabelClassifier(
            input_dim=config.network.final_feature_dim // 4,
            num_classes=num_classes,
            use_grouped_fc=config.network.use_grouped_fc,
            num_groups=config.network.num_groups,
            dropout_rate=config.network.classifier_dropout,
            class_groups=class_groups,
            group_weights=group_weights,
        ).to(self.device)
        
        self.batch_norm = nn.BatchNorm1d(self.feature_dim).to(self.device)
        
        self._initialize_weights()

    @property
    def adjacency_matrices(self) -> List[torch.Tensor]:
        return self.neighborhood_system.adjacency_matrices
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def extract_grid_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Extract grid features from feature map
        
        Args:
            features: Feature map [batch_size, feature_dim, H, W]
            
        Returns:
            Grid features [batch_size, num_blocks, feature_dim]
        """
        batch_size, feature_dim, H, W = features.shape
        grid_rows, grid_cols = self.config.network.grid_rows, self.config.network.grid_cols
        
        block_h = H // grid_rows
        block_w = W // grid_cols
        
        if H % grid_rows != 0 or W % grid_cols != 0:
            target_h = block_h * grid_rows
            target_w = block_w * grid_cols
            features = F.interpolate(features, size=(target_h, target_w), mode='bilinear', align_corners=False)
            H, W = target_h, target_w
        
        grid_features = []
        for i in range(grid_rows):
            for j in range(grid_cols):
                block = features[:, :, i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
                
                block_pooled = F.adaptive_avg_pool2d(block, (1, 1))
                block_pooled = block_pooled.squeeze(-1).squeeze(-1)
                
                grid_features.append(block_pooled)
        
        grid_features = torch.stack(grid_features, dim=1)
        
        return grid_features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input image [batch_size, 3, H, W]
            
        Returns:
            Classification logits [batch_size, num_classes]
        """
        batch_size = x.size(0)
        
        features = self.backbone(x)
        
        grid_features = self.extract_grid_features(features)
        grid_features = self.rotated_position(grid_features)

        grid_features = grid_features.permute(0, 2, 1)
        grid_features = self.batch_norm(grid_features)
        grid_features = grid_features.permute(0, 2, 1)
        
        kernel_matrix, kernel_features = self.context_kernel(grid_features, self.adjacency_matrices)
        learned_adjacency_matrices = self.context_kernel.get_adjacency_matrices(
            self.adjacency_matrices
        )
        
        multi_order_features = self.multi_order_aggregator(
            kernel_features,
            self.neighborhood_system.adjacency_index
        )
        
        random_walk_features = self.random_walk(
            kernel_features,
            multi_order_features,
            learned_adjacency_matrices
        )
        
        # Keep the lattice intact while applying the learned kernel mapping;
        # pooling first would reduce the graph to a single identity node.
        deep_kernel_features = self.deep_kernel_network(
            random_walk_features,
            learned_adjacency_matrices
        )

        fused_features = self.multi_scale_aggregator(
            deep_kernel_features,
            (self.config.network.grid_rows, self.config.network.grid_cols),
            coarse_context_processors=self.coarse_scale_context,
        )
        
        fused_features = self.multi_scale_fusion(fused_features)
        
        logits = self.classifier(fused_features)
        
        return logits


class CustomNetwork(nn.Module):
    """Custom network (compatible with original code)"""
    
    def __init__(self, num_classes, connect_idx, weights_flag, device, num_blocks):
        """
        Initialize custom network
        
        Args:
            num_classes: Number of classes
            connect_idx: Adjacency index matrix
            weights_flag: Weight matrix
            device: Device
            num_blocks: Number of image blocks
        """
        super(CustomNetwork, self).__init__()
        
        self.feature_aggregation = FeatureAggregation(connect_idx, weights_flag, device)
        self.feature_aggregation_1 = FeatureAggregation_1(connect_idx, weights_flag, device)
        self.bn = nn.BatchNorm1d(num_features=2048).to(device)
        self.weighted_sum_layer = WeightedSumLayer(num_blocks=num_blocks, feature_dim=6144)
        self.classifier = nn.Linear(6144, num_classes)
    
    def forward(self, x):
        """Forward pass"""
        x = x.reshape(x.size(0), x.size(1) * x.size(2), x.size(3))
        reshaped_features = x.permute(0, 2, 1).contiguous()
        reshaped_features = self.bn(reshaped_features)
        reshaped_features = reshaped_features.permute(0, 2, 1).contiguous()
        
        min_values, _ = torch.min(reshaped_features, dim=2, keepdim=True)
        max_values, _ = torch.max(reshaped_features, dim=2, keepdim=True)
        diff = max_values - min_values
        normalized_features = (reshaped_features - min_values) / diff
        
        aggregated_features, x_start = self.feature_aggregation(normalized_features)
        aggregated_features_1 = self.feature_aggregation_1(aggregated_features, x_start)
        
        sum_output = self.weighted_sum_layer(aggregated_features_1)
        
        output = self.classifier(sum_output)
        
        return output


# Helper classes for CustomNetwork (maintain compatibility with original code)

class CenteredSelfAttention(nn.Module):
    def __init__(self, feature_dim):
        super(CenteredSelfAttention, self).__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, center_feature, connected_features):
        Q = self.query(center_feature)
        K = self.key(connected_features)
        V = self.value(connected_features)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))
        attention_probs = self.softmax(attention_scores)
        max_score_index = torch.argmax(attention_probs, dim=-1, keepdim=True)
        most_similar_feature = torch.gather(V, 0, max_score_index.expand(-1, V.size(-1)))
        max_attention_prob = attention_probs.gather(1, max_score_index)
        weighted_most_similar_feature = most_similar_feature * max_attention_prob
        return weighted_most_similar_feature.squeeze(1)


class FeatureAggregation(nn.Module):
    def __init__(self, connect_idx, weights_flag, device):
        super(FeatureAggregation, self).__init__()
        self.connect_idx = connect_idx
        self.l0l1_w = nn.Parameter(weights_flag.clone().float(), requires_grad=True).to(device)
        self.l1l2_w = nn.Parameter(torch.sqrt(torch.ones_like(weights_flag, dtype=torch.float32)),
                           requires_grad=True).to(device)
        self.conv_reduce_dim = nn.Conv2d(in_channels=8192, out_channels=2048, kernel_size=1).to(device)
        self.centered_attention = CenteredSelfAttention(2048).to(device)
    
    def forward(self, x):
        num_blocks = x.size(1)
        block_aggregated_features = torch.zeros(x.size(0),
                                                x.size(1),
                                                4,
                                                x.size(2),
                                                device=x.device)
        for i in range(0, x.size(0)):
            block_features = x[i]
            for j in range(block_features.size(0)):
                connected_feature_idx = self.connect_idx[j]
                valid_indices = connected_feature_idx != -1
                valid_feature_idx = connected_feature_idx[valid_indices]
                valid_feature_idx = valid_feature_idx.long()
                connected_features_1_step = block_features[valid_feature_idx]
                
                for k, idx in enumerate(valid_feature_idx):
                    connected_to_idx = self.connect_idx[idx]
                    valid_connected_indices = connected_to_idx != -1
                    valid_connected_to_idx = connected_to_idx[valid_connected_indices].long()
                    connected_features_2_step = block_features[valid_connected_to_idx]
                    
                    most_similar_feature = self.centered_attention(block_features[j].unsqueeze(0),
                                                                   connected_features_2_step)
                    connected_features_1_step[k] = connected_features_1_step[k] + most_similar_feature.squeeze(0)
                
                weighted_features = connected_features_1_step * self.l0l1_w[j, valid_indices].unsqueeze(1)
                weighted_features_result = weighted_features * self.l1l2_w[j, valid_indices].unsqueeze(1)
                block_aggregated_features[i, j, valid_indices] = weighted_features_result
        
        block_aggregated_features = block_aggregated_features.reshape(
            block_aggregated_features.size(0),
            block_aggregated_features.size(1),
            block_aggregated_features.size(2) * block_aggregated_features.size(3)
        )
        block_aggregated_features_result = self.conv_reduce_dim(
            block_aggregated_features.unsqueeze(-1).permute(0, 2, 1, 3)
        ).squeeze(-1).permute(0, 2, 1)
        
        concatenated_features_result = torch.cat((x, block_aggregated_features_result), dim=-1)
        return concatenated_features_result, x


class FeatureAggregation_1(nn.Module):
    def __init__(self, connect_idx, weights_flag, device):
        super(FeatureAggregation_1, self).__init__()
        self.connect_idx = connect_idx
        self.l0l1_w = nn.Parameter(weights_flag.clone().float(), requires_grad=True).to(device)
        self.l1l2_w = nn.Parameter(torch.sqrt(torch.ones_like(weights_flag, dtype=torch.float32)),
                                   requires_grad=True).to(device)
        self.conv_reduce_dim_1 = nn.Conv2d(in_channels=16384, out_channels=4096, kernel_size=1)
        self.centered_attention = CenteredSelfAttention(4096).to(device)
    
    def forward(self, x, x_start):
        num_blocks = x.size(1)
        block_aggregated_features_1 = torch.zeros(x.size(0),
                                                x.size(1),
                                                4,
                                                x.size(2),
                                                device=x.device)
        for i in range(0, x.size(0)):
            block_features = x[i]
            for j in range(block_features.size(0)):
                connected_feature_idx = self.connect_idx[j]
                valid_indices = connected_feature_idx != -1
                valid_feature_idx = connected_feature_idx[valid_indices]
                valid_feature_idx = valid_feature_idx.long()
                connected_features_1_step = block_features[valid_feature_idx]
                
                for k, idx in enumerate(valid_feature_idx):
                    connected_to_idx = self.connect_idx[idx]
                    valid_connected_indices = connected_to_idx != -1
                    valid_connected_to_idx = connected_to_idx[valid_connected_indices].long()
                    connected_features_2_step = block_features[valid_connected_to_idx]
                    most_similar_feature = self.centered_attention(block_features[j].unsqueeze(0),
                                                                   connected_features_2_step)
                    connected_features_1_step[k] = connected_features_1_step[k] + most_similar_feature.squeeze(0)
                
                weighted_features = connected_features_1_step * self.l0l1_w[j, valid_indices].unsqueeze(1)
                weighted_features_result = weighted_features * self.l1l2_w[j, valid_indices].unsqueeze(1)
                block_aggregated_features_1[i, j, valid_indices] = weighted_features_result
        
        block_aggregated_features_1 = block_aggregated_features_1.reshape(
            block_aggregated_features_1.size(0),
            block_aggregated_features_1.size(1),
            block_aggregated_features_1.size(2) * block_aggregated_features_1.size(3)
        )
        block_aggregated_features_result = self.conv_reduce_dim_1(
            block_aggregated_features_1.unsqueeze(-1).permute(0, 2, 1, 3)
        ).squeeze(-1).permute(0, 2, 1)
        
        concatenated_features_result = torch.cat((x_start, block_aggregated_features_result), dim=-1)
        return concatenated_features_result


class WeightedSumLayer(nn.Module):
    def __init__(self, num_blocks, feature_dim):
        super(WeightedSumLayer, self).__init__()
        self.weights = nn.Parameter(torch.full((num_blocks,), 1.0 / num_blocks))
        self.feature_dim = feature_dim
    
    def forward(self, x):
        weights = self.weights.unsqueeze(0).unsqueeze(-1)
        weighted_sum = torch.sum(x * weights, dim=1)
        norm = torch.norm(weighted_sum, dim=1, keepdim=True)
        normalized_output = weighted_sum / norm
        return normalized_output
