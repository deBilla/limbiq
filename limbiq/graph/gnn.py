"""
Graph Neural Network for Active Graph Propagation — Phase 2
=============================================================
A lightweight GNN (pure PyTorch, no PyG dependency) that learns
propagation dynamics from limbiq's graph structure.

Three learned functions:
1. Node classifier — predict: noise / valuable / priority
2. Edge predictor — predict: should two memories merge?
3. Activation propagator — learned message-passing for activation dynamics

Architecture: Graph Attention Network (GAT) with typed edges.
Total params: ~2-5M depending on embedding dimension.
Trains in <1 min on CPU for graphs up to ~1000 nodes.
"""

import os
import json
import time
import math
import struct
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ─── Node Feature Extraction ─────────────────────────────────────

@dataclass
class NodeFeatures:
    """Raw features for a single memory node."""
    memory_id: str
    embedding: np.ndarray       # Semantic embedding (384-dim)
    confidence: float
    access_count: int
    session_count: int
    is_priority: bool
    is_suppressed: bool
    tier: str                   # short, mid, long, priority
    content_length: int
    has_web_prefix: bool        # [Web] tagged
    has_user_prefix: bool       # "User said:" tagged


def extract_node_features(store_db, embedding_dim: int = 384) -> list[NodeFeatures]:
    """Extract features for all memory nodes from the database."""
    from limbiq.store.memory_store import _deserialize_embedding

    cursor = store_db.execute(
        "SELECT id, content, tier, confidence, access_count, session_count, "
        "is_priority, is_suppressed, embedding FROM memories"
    )

    features = []
    for row in cursor.fetchall():
        emb = _deserialize_embedding(row[8])
        if emb is None:
            continue

        emb_array = np.array(emb[:embedding_dim])
        if len(emb_array) < embedding_dim:
            emb_array = np.pad(emb_array, (0, embedding_dim - len(emb_array)))

        content = row[1]
        features.append(NodeFeatures(
            memory_id=row[0],
            embedding=emb_array,
            confidence=row[3],
            access_count=row[4],
            session_count=row[5],
            is_priority=bool(row[6]),
            is_suppressed=bool(row[7]),
            tier=row[2],
            content_length=len(content),
            has_web_prefix=content.startswith("[Web]"),
            has_user_prefix=content.startswith("User said:"),
        ))

    return features


def features_to_tensors(features: list[NodeFeatures], embedding_dim: int = 384):
    """Convert node features to PyTorch tensors for the GNN."""
    n = len(features)
    if n == 0:
        return None

    # Semantic embeddings [n, embedding_dim]
    embeddings = np.stack([f.embedding for f in features])

    # Scalar features [n, 8]
    tier_map = {"short": 0, "mid": 1, "long": 2, "priority": 3}
    scalars = np.zeros((n, 8), dtype=np.float32)
    for i, f in enumerate(features):
        scalars[i, 0] = f.confidence
        scalars[i, 1] = min(f.access_count / 30.0, 1.0)  # Normalized access
        scalars[i, 2] = 1.0 / (1.0 + f.session_count * 0.1)  # Recency
        scalars[i, 3] = float(f.is_priority)
        scalars[i, 4] = float(f.is_suppressed)
        scalars[i, 5] = tier_map.get(f.tier, 0) / 3.0  # Normalized tier
        scalars[i, 6] = float(f.has_web_prefix)
        scalars[i, 7] = float(f.has_user_prefix)

    return {
        "embeddings": torch.tensor(embeddings, dtype=torch.float32),
        "scalars": torch.tensor(scalars, dtype=torch.float32),
        "ids": [f.memory_id for f in features],
    }


# ─── Adjacency Construction ──────────────────────────────────────

def build_adjacency(embeddings: torch.Tensor, threshold: float = 0.4,
                    k_nearest: int = 10) -> torch.Tensor:
    """
    Build sparse adjacency matrix from embedding similarity.
    Uses k-nearest neighbors + threshold for efficiency.
    Returns [n, n] adjacency weights (0 where no edge).
    """
    n = embeddings.shape[0]

    # Normalize embeddings
    norms = embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normalized = embeddings / norms

    # Cosine similarity matrix
    sim = normalized @ normalized.T

    # Zero out self-loops
    sim.fill_diagonal_(0)

    # Apply threshold
    sim[sim < threshold] = 0

    # Keep only top-k neighbors per node (sparse)
    if k_nearest < n:
        topk_vals, topk_idx = sim.topk(min(k_nearest, n - 1), dim=1)
        sparse_sim = torch.zeros_like(sim)
        sparse_sim.scatter_(1, topk_idx, topk_vals)
        # Symmetrize
        sim = (sparse_sim + sparse_sim.T) / 2

    return sim


# ─── GNN Architecture ────────────────────────────────────────────

class GraphAttentionLayer(nn.Module):
    """Single Graph Attention layer (GAT)."""

    def __init__(self, in_features: int, out_features: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.out_per_head = out_features // heads

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a_src = nn.Parameter(torch.randn(heads, self.out_per_head))
        self.a_dst = nn.Parameter(torch.randn(heads, self.out_per_head))
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x: [n, in_features]
        adj: [n, n] adjacency weights
        Returns: [n, out_features]
        """
        n = x.shape[0]
        h = self.W(x)  # [n, out_features]
        h = h.view(n, self.heads, self.out_per_head)  # [n, heads, out_per_head]

        # Attention scores
        attn_src = (h * self.a_src).sum(dim=-1)  # [n, heads]
        attn_dst = (h * self.a_dst).sum(dim=-1)  # [n, heads]

        # Pairwise attention: [n, n, heads]
        attn = attn_src.unsqueeze(1) + attn_dst.unsqueeze(0)
        attn = self.leaky_relu(attn)

        # Mask by adjacency
        mask = (adj == 0).unsqueeze(-1).expand_as(attn)
        attn = attn.masked_fill(mask, float('-inf'))

        # Softmax over neighbors
        attn = F.softmax(attn, dim=1)
        attn = torch.nan_to_num(attn, 0.0)  # Handle isolated nodes
        attn = self.dropout(attn)

        # Weighted aggregation
        # attn: [n, n, heads], h: [n, heads, out_per_head]
        out = torch.einsum('ijh,jhd->ihd', attn, h)  # [n, heads, out_per_head]
        out = out.reshape(n, -1)  # [n, out_features]

        return out


class LimbiqGNN(nn.Module):
    """
    Graph Neural Network for limbiq's active graph propagation.

    Three output heads:
    1. Node quality classifier (noise / normal / priority)
    2. Activation level predictor (continuous 0-1)
    3. Node embedding (for merge detection via cosine similarity)
    """

    def __init__(self, embedding_dim: int = 384, scalar_dim: int = 8,
                 hidden_dim: int = 128, num_layers: int = 3, heads: int = 4):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Input projection: combine embedding + scalar features
        self.input_proj = nn.Sequential(
            nn.Linear(embedding_dim + scalar_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # GAT layers
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim, heads=heads)
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Output heads
        # 1. Node quality: 3 classes (noise=0, normal=1, priority=2)
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

        # 2. Activation level: scalar 0-1
        self.activation_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # 3. Merge embedding: for detecting duplicates
        self.merge_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
        )

        self._count_params()

    def _count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"LimbiqGNN: {total:,} params ({trainable:,} trainable)")
        self.total_params = total

    def forward(self, embeddings: torch.Tensor, scalars: torch.Tensor,
                adj: torch.Tensor) -> dict:
        """
        Forward pass.

        Args:
            embeddings: [n, embedding_dim] semantic embeddings
            scalars: [n, scalar_dim] node scalar features
            adj: [n, n] adjacency weights

        Returns dict with:
            quality_logits: [n, 3] — noise/normal/priority
            activation: [n, 1] — predicted activation level
            merge_emb: [n, 64] — embedding for merge detection
            hidden: [n, hidden_dim] — final hidden states
        """
        # Combine inputs
        x = torch.cat([embeddings, scalars], dim=-1)
        x = self.input_proj(x)

        # GAT layers with residual connections
        for gat, norm in zip(self.gat_layers, self.layer_norms):
            residual = x
            x = gat(x, adj)
            x = norm(x + residual)  # Residual + LayerNorm
            x = F.gelu(x)

        # Output heads
        quality_logits = self.quality_head(x)
        activation = self.activation_head(x)
        merge_emb = self.merge_head(x)

        return {
            "quality_logits": quality_logits,
            "activation": activation.squeeze(-1),
            "merge_emb": merge_emb,
            "hidden": x,
        }


# ─── Training ────────────────────────────────────────────────────

@dataclass
class TrainingLabels:
    """Labels generated from Phase 1 propagation outputs."""
    quality: torch.Tensor       # [n] — 0=noise, 1=normal, 2=priority
    activation: torch.Tensor    # [n] — target activation 0-1
    merge_pairs: list           # [(i, j)] — pairs that should merge
    non_merge_pairs: list       # [(i, j)] — pairs that should NOT merge


def generate_training_labels(store_db, features: list[NodeFeatures],
                             phase1_activations: list = None) -> TrainingLabels:
    """
    Generate training labels from Phase 1 hand-written rules.
    The GNN learns to replicate (and eventually exceed) Phase 1's decisions.
    """
    import re

    n = len(features)
    quality = torch.ones(n, dtype=torch.long)  # Default: normal
    activation = torch.zeros(n, dtype=torch.float32)

    # Noise patterns from Phase 1
    noise_patterns = [
        r"user'?s?\s+greeting\s+was", r"user\s+initiated\s+the\s+conversation",
        r"user'?s?\s+question\s+was\s+a\s+simple", r"I'm a helpful AI assistant",
        r"I don't recognize users", r"current date is", r"current day of the week",
        r"falls in week \d+", r"There are \d+ days in the month",
        r"conversation started from scratch", r"information found does not list",
    ]
    meta_patterns = [
        r"i'm a helpful ai", r"i don't recognize", r"i'm a .* assistant",
        r"conversation started", r"no stored information",
    ]

    for i, f in enumerate(features):
        # Get content
        row = store_db.execute(
            "SELECT content FROM memories WHERE id = ?", (f.memory_id,)
        ).fetchone()
        content = row[0] if row else ""
        content_lower = content.lower()

        # Quality labels
        is_noise = False
        for pattern in noise_patterns + meta_patterns:
            if re.search(pattern, content_lower, re.IGNORECASE):
                is_noise = True
                break

        # Short echoed questions are noise
        if content_lower.startswith("who is my") and len(content) < 30:
            is_noise = True

        if is_noise:
            quality[i] = 0  # noise
        elif f.is_priority and not is_noise:
            quality[i] = 2  # genuine priority
        else:
            quality[i] = 1  # normal

        # Activation labels (from Phase 1 or heuristic)
        if phase1_activations and i < len(phase1_activations):
            activation[i] = phase1_activations[i]
        else:
            # Heuristic: combine confidence, access, priority, recency
            recency = 1.0 / (1.0 + f.session_count * 0.1)
            popularity = min(1.0, f.access_count / 20.0)
            act = f.confidence * 0.3 + recency * 0.2 + popularity * 0.2
            if f.is_priority and not is_noise:
                act += 0.3
            if is_noise:
                act = 0.05
            if f.is_suppressed:
                act = 0.02
            activation[i] = min(1.0, act)

    # Merge pairs: find near-identical content
    merge_pairs = []
    non_merge_pairs = []

    embs = np.stack([f.embedding for f in features])
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    norm_embs = embs / norms
    sim = norm_embs @ norm_embs.T

    for i in range(n):
        for j in range(i + 1, min(i + 50, n)):  # Limit comparisons
            if sim[i][j] > 0.92:
                merge_pairs.append((i, j))
            elif 0.3 < sim[i][j] < 0.7:
                non_merge_pairs.append((i, j))

    # Balance non-merge pairs
    if len(non_merge_pairs) > len(merge_pairs) * 3:
        np.random.shuffle(non_merge_pairs)
        non_merge_pairs = non_merge_pairs[:len(merge_pairs) * 3]

    return TrainingLabels(
        quality=quality,
        activation=activation,
        merge_pairs=merge_pairs,
        non_merge_pairs=non_merge_pairs,
    )


def train_gnn(model: LimbiqGNN, tensors: dict, adj: torch.Tensor,
              labels: TrainingLabels, epochs: int = 200,
              lr: float = 0.001) -> dict:
    """
    Train the GNN on Phase 1 labels.
    Returns training metrics.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Loss weights
    quality_weight = 2.0
    activation_weight = 1.0
    merge_weight = 1.5

    # Class weights for imbalanced quality labels
    class_counts = torch.bincount(labels.quality, minlength=3).float().clamp(min=1)
    class_weights = (1.0 / class_counts)
    class_weights = class_weights / class_weights.sum() * 3

    history = {"loss": [], "quality_acc": [], "activation_mae": []}
    best_loss = float('inf')
    best_state = None

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        out = model(tensors["embeddings"], tensors["scalars"], adj)

        # Quality classification loss
        loss_quality = F.cross_entropy(
            out["quality_logits"], labels.quality, weight=class_weights
        ) * quality_weight

        # Activation regression loss
        loss_activation = F.mse_loss(
            out["activation"], labels.activation
        ) * activation_weight

        # Merge contrastive loss
        loss_merge = torch.tensor(0.0)
        if labels.merge_pairs:
            merge_emb = F.normalize(out["merge_emb"], dim=-1)
            # Positive pairs should be close
            for i, j in labels.merge_pairs[:50]:  # Limit for speed
                pos_sim = F.cosine_similarity(
                    merge_emb[i:i+1], merge_emb[j:j+1]
                )
                loss_merge = loss_merge + (1 - pos_sim).mean()

            # Negative pairs should be far
            for i, j in labels.non_merge_pairs[:50]:
                neg_sim = F.cosine_similarity(
                    merge_emb[i:i+1], merge_emb[j:j+1]
                )
                loss_merge = loss_merge + F.relu(neg_sim - 0.3).mean()

            n_pairs = min(50, len(labels.merge_pairs)) + min(50, len(labels.non_merge_pairs))
            if n_pairs > 0:
                loss_merge = loss_merge / n_pairs * merge_weight

        total_loss = loss_quality + loss_activation + loss_merge

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Metrics
        with torch.no_grad():
            preds = out["quality_logits"].argmax(dim=-1)
            acc = (preds == labels.quality).float().mean().item()
            mae = (out["activation"] - labels.activation).abs().mean().item()

        history["loss"].append(total_loss.item())
        history["quality_acc"].append(acc)
        history["activation_mae"].append(mae)

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 50 == 0:
            logger.info(
                f"Epoch {epoch+1}/{epochs}: loss={total_loss.item():.4f}, "
                f"quality_acc={acc:.3f}, activation_mae={mae:.4f}"
            )
            print(
                f"  Epoch {epoch+1}/{epochs}: loss={total_loss.item():.4f}, "
                f"quality_acc={acc:.3f}, activation_mae={mae:.4f}"
            )

    # Restore best
    if best_state:
        model.load_state_dict(best_state)

    return history


# ─── GNN-Powered Propagation ─────────────────────────────────────

class GNNPropagation:
    """
    Phase 2: GNN-based propagation replacing hand-written rules.

    Uses the trained LimbiqGNN for:
    - Node quality classification (noise detection + priority validation)
    - Activation computation (learned message-passing)
    - Merge detection (learned similarity in GNN embedding space)
    """

    def __init__(self, store, graph, embedding_engine=None,
                 user_name: str = "Dimuthu", model_dir: str = "data/gnn"):
        self.store = store
        self.graph = graph
        self.embeddings = embedding_engine
        self.user_name = user_name
        self.model_dir = model_dir
        self.model = None
        self.embedding_dim = 384

    def train_and_save(self, phase1_activations: list = None,
                       epochs: int = 200) -> dict:
        """
        Train GNN from current graph state and Phase 1 labels.
        Saves model to model_dir.
        """
        print("  Extracting node features...")
        features = extract_node_features(self.store.db, self.embedding_dim)
        if len(features) < 5:
            print("  Too few nodes for training")
            return {"status": "insufficient_data"}

        print(f"  {len(features)} nodes extracted")

        tensors = features_to_tensors(features, self.embedding_dim)
        adj = build_adjacency(tensors["embeddings"], threshold=0.4, k_nearest=10)

        print("  Generating training labels from Phase 1...")
        labels = generate_training_labels(
            self.store.db, features, phase1_activations
        )

        noise_count = (labels.quality == 0).sum().item()
        normal_count = (labels.quality == 1).sum().item()
        priority_count = (labels.quality == 2).sum().item()
        print(f"  Labels: noise={noise_count}, normal={normal_count}, priority={priority_count}")
        print(f"  Merge pairs: {len(labels.merge_pairs)}, non-merge: {len(labels.non_merge_pairs)}")

        # Initialize model
        self.model = LimbiqGNN(
            embedding_dim=self.embedding_dim,
            scalar_dim=8,
            hidden_dim=128,
            num_layers=3,
            heads=4,
        )
        print(f"  Model: {self.model.total_params:,} parameters")

        # Train
        print(f"  Training for {epochs} epochs...")
        start = time.time()
        history = train_gnn(self.model, tensors, adj, labels, epochs=epochs)
        duration = time.time() - start

        # Save
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, "limbiq_gnn.pt")
        torch.save({
            "model_state": self.model.state_dict(),
            "config": {
                "embedding_dim": self.embedding_dim,
                "scalar_dim": 8,
                "hidden_dim": 128,
                "num_layers": 3,
                "heads": 4,
            },
            "training": {
                "epochs": epochs,
                "final_loss": history["loss"][-1],
                "final_quality_acc": history["quality_acc"][-1],
                "final_activation_mae": history["activation_mae"][-1],
                "duration_seconds": duration,
                "num_nodes": len(features),
            },
        }, model_path)
        print(f"  Model saved to {model_path}")
        print(f"  Training completed in {duration:.1f}s")
        print(f"  Final: loss={history['loss'][-1]:.4f}, "
              f"quality_acc={history['quality_acc'][-1]:.3f}, "
              f"mae={history['activation_mae'][-1]:.4f}")

        return {
            "status": "trained",
            "params": self.model.total_params,
            "duration": duration,
            "final_loss": history["loss"][-1],
            "final_accuracy": history["quality_acc"][-1],
            "final_mae": history["activation_mae"][-1],
        }

    def load_model(self) -> bool:
        """Load a previously trained model."""
        model_path = os.path.join(self.model_dir, "limbiq_gnn.pt")
        if not os.path.exists(model_path):
            return False

        checkpoint = torch.load(model_path, weights_only=False)
        config = checkpoint["config"]
        self.model = LimbiqGNN(**config)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.embedding_dim = config["embedding_dim"]
        return True

    @torch.no_grad()
    def propagate(self) -> dict:
        """
        Run GNN-based propagation. Returns results dict.
        Falls back to Phase 1 if no trained model.
        """
        if self.model is None:
            if not self.load_model():
                print("  No GNN model found — falling back to Phase 1")
                from limbiq.graph.propagation import ActiveGraphPropagation
                p1 = ActiveGraphPropagation(
                    self.store, self.graph, self.embeddings, self.user_name
                )
                return {"phase": 1, "result": p1.propagate()}

        start = time.time()
        self.model.eval()

        # Extract features
        features = extract_node_features(self.store.db, self.embedding_dim)
        if len(features) < 2:
            return {"phase": 2, "status": "insufficient_nodes"}

        tensors = features_to_tensors(features, self.embedding_dim)
        adj = build_adjacency(tensors["embeddings"], threshold=0.4, k_nearest=10)

        # Run GNN
        out = self.model(tensors["embeddings"], tensors["scalars"], adj)

        # Apply quality predictions
        quality_preds = out["quality_logits"].argmax(dim=-1)
        activations = out["activation"]
        merge_embs = F.normalize(out["merge_emb"], dim=-1)

        # 1. Suppress predicted noise
        noise_suppressed = 0
        for i, (pred, feat) in enumerate(zip(quality_preds, features)):
            if pred.item() == 0 and not feat.is_suppressed:  # Noise
                self.store.suppress(feat.memory_id, "gnn_noise")
                noise_suppressed += 1

        # 2. Deflate wrong priorities
        priority_deflated = 0
        for i, (pred, feat) in enumerate(zip(quality_preds, features)):
            if pred.item() != 2 and feat.is_priority and not feat.is_suppressed:
                # GNN says not priority, but it's marked priority
                # Only deflate if GNN is confident
                probs = F.softmax(out["quality_logits"][i], dim=-1)
                if probs[2].item() < 0.3:  # Low priority probability
                    self.store.db.execute(
                        "UPDATE memories SET is_priority = 0, confidence = 0.6 WHERE id = ?",
                        (feat.memory_id,)
                    )
                    self.store.db.commit()
                    priority_deflated += 1

        # 3. Merge duplicates using GNN merge embeddings
        duplicates_merged = 0
        already_merged = set()
        merge_sim = merge_embs @ merge_embs.T
        merge_sim.fill_diagonal_(0)

        for i in range(len(features)):
            if i in already_merged or features[i].is_suppressed:
                continue
            for j in range(i + 1, len(features)):
                if j in already_merged or features[j].is_suppressed:
                    continue
                if merge_sim[i][j].item() > 0.85:
                    # Merge: keep the one with higher activation
                    if activations[i] >= activations[j]:
                        self.store.suppress(features[j].memory_id, "gnn_merge")
                        already_merged.add(j)
                    else:
                        self.store.suppress(features[i].memory_id, "gnn_merge")
                        already_merged.add(i)
                    duplicates_merged += 1

        # 4. Run entity repair (still use Phase 1 for this — it's rule-based)
        from limbiq.graph.propagation import ActiveGraphPropagation
        p1 = ActiveGraphPropagation(
            self.store, self.graph, self.embeddings, self.user_name
        )
        repair_stats = p1.repair_graph()

        # Run inference
        from limbiq.graph.inference import InferenceEngine
        inference = InferenceEngine(self.graph)
        inferred = inference.run_full_inference()

        duration = (time.time() - start) * 1000

        return {
            "phase": 2,
            "noise_suppressed": noise_suppressed,
            "priority_deflated": priority_deflated,
            "duplicates_merged": duplicates_merged,
            "entities_created": repair_stats["entities_created"],
            "relations_created": repair_stats["relations_created"] + inferred,
            "duration_ms": duration,
        }

    @torch.no_grad()
    def compute_activations(self, query_embedding=None) -> list:
        """
        Compute GNN-based activations for all nodes.
        Returns list of (memory_id, activation_value) sorted by activation.
        """
        if self.model is None:
            if not self.load_model():
                return []

        self.model.eval()
        features = extract_node_features(self.store.db, self.embedding_dim)
        if not features:
            return []

        tensors = features_to_tensors(features, self.embedding_dim)
        adj = build_adjacency(tensors["embeddings"], threshold=0.4, k_nearest=10)

        out = self.model(tensors["embeddings"], tensors["scalars"], adj)
        activations = out["activation"]

        # Query relevance boost (same as Phase 1 but additive with GNN output)
        if query_embedding is not None:
            q = torch.tensor(query_embedding[:self.embedding_dim], dtype=torch.float32)
            if len(q) < self.embedding_dim:
                q = F.pad(q, (0, self.embedding_dim - len(q)))
            q = q / q.norm().clamp(min=1e-8)

            emb_norm = tensors["embeddings"] / tensors["embeddings"].norm(
                dim=1, keepdim=True
            ).clamp(min=1e-8)
            relevance = (emb_norm @ q).clamp(min=0)

            # Blend GNN activation with query relevance
            activations = activations * 0.6 + relevance * 0.4

        results = [
            (features[i].memory_id, activations[i].item())
            for i in range(len(features))
            if not features[i].is_suppressed
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results
