# PanCAN: Panoptic Context Aggregation Network for Multi-label Image Classification

Mingyuan Jiu, Hailong Zhu, Wenchuan Wei

School of Computer and Artificial Intelligence, Zhengzhou University


## Overview

Here is the English translation, aligned with the academic terminology from your paper:In multi-label image classification, current approaches often focus on basic geometric relationships or localized features, neglecting cross-scale contextual interactions between objects. 
PanCAN hierarchically integrates multi-order geometric contexts through cross-scale feature aggregation in a high-dimensional Hilbert space.The core innovation of this model lies in combining random walks with an attention mechanism to learn multi-order neighborhood relationships at each scale. Modules from different scales are cascaded, utilizing attention mechanisms to dynamically fuse neighborhood features, thereby significantly enhancing the understanding of complex scenes.
Multi-order Contexts: The core innovation of the paper, which extracts fine-grained local details and coarse-grained global contextual information through random walk and attention mechanisms.
Cross-scale Aggregation: Progressively constructs features from micro-cells to macro-cells, fully capturing visual and structural information across different scales.
End-to-end Learning: Supports hierarchical label propagation, dynamically learning adjacency matrices and attention weights.
