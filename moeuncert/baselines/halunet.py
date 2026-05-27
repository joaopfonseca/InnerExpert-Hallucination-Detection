"""
HaluNet baseline (Tong et al., 2025, arXiv 2512.24562).

A lightweight trainable neural framework that integrates multi-granular
token-level uncertainties by combining semantic embeddings with probabilistic
confidence and distributional uncertainty. Its multi-branch architecture
adaptively fuses what the model knows with the uncertainty expressed in its
outputs, enabling efficient one-pass hallucination detection.

Architecture (from paper Section 3.4):
- 3 branches: log-likelihoods, entropies, hidden-state embeddings
- Scalar branches (log-likelihoods, entropies): mean pooling + 2-layer MLP
- Embedding branch (hidden states): 2-layer 1D Conv + ReLU + adaptive avg pooling
- Fusion: attention-based (default) or concatenation + MLP
- Output: single logit → sigmoid

Training:
- Binary cross-entropy loss
- Supervised by LLM-as-a-Judge labels (or any binary hallucination labels)
- Sequence length L = 50 (zero-padded)

Paper: HaluNet: Multi-Granular Uncertainty Modeling for Efficient
       Hallucination Detection in LLM Question Answering (arXiv 2512.24562)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import BaseBaseline


class HaluNetModel(nn.Module):
    """HaluNet architecture as described in the paper (Section 3.4).

    Args:
        embedding_dim: Dimension of hidden-state embeddings (d in paper).
            Default 768 for base models.
        hidden_dim: Dimension of branch latent vectors (dh in paper).
            Default 128.
        max_seq_len: Maximum sequence length (L in paper). Default 50.
        dropout: Dropout probability. Default 0.5 (from paper Fig. 2b).
        fusion: Fusion mechanism. 'attention' (Eq. 4) or 'mlp' (Eq. 5).
    """

    def __init__(self, embedding_dim=768, hidden_dim=128, max_seq_len=50,
                 dropout=0.5, fusion='attention'):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        self.fusion = fusion

        # Branch 1: Log-likelihoods (scalar feature)
        # Mean pooling (implicit) + 2-layer MLP
        self.ll_branch = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Branch 2: Entropies (scalar feature)
        # Mean pooling (implicit) + 2-layer MLP
        self.entropy_branch = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Branch 3: Hidden-state embeddings (sequence feature)
        # 2-layer 1D Conv (kernel=3, padding=1) + ReLU + adaptive avg pooling
        self.embedding_branch = nn.Sequential(
            nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # outputs (B, hidden_dim, 1)
        )

        # Fusion layer
        num_branches = 3
        if fusion == 'attention':
            # Eq. 4: attention-based fusion
            self.Wa = nn.Linear(hidden_dim, hidden_dim)
            self.w = nn.Linear(hidden_dim, 1, bias=False)
        elif fusion == 'mlp':
            # Eq. 5: concatenation + MLP
            self.fusion_mlp = nn.Sequential(
                nn.Linear(num_branches * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            raise ValueError(f"Unknown fusion: {fusion}. Use 'attention' or 'mlp'.")

        # Output projection (Eq. 6)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, log_likelihoods, entropies, embeddings, mask=None):
        """
        Args:
            log_likelihoods: (B, L) — token-level log probabilities.
            entropies: (B, L) — token-level entropies.
            embeddings: (B, L, D) — token-level hidden-state embeddings.
            mask: (B, L) — optional padding mask (True = valid token).

        Returns:
            logits: (B,) — hallucination logits (pre-sigmoid).
        """
        B, L, D = embeddings.shape

        # Pad sequences to max_seq_len
        if L < self.max_seq_len:
            pad_len = self.max_seq_len - L
            log_likelihoods = F.pad(log_likelihoods, (0, pad_len), value=0.0)
            entropies = F.pad(entropies, (0, pad_len), value=0.0)
            embeddings = F.pad(embeddings, (0, 0, 0, pad_len), value=0.0)
            if mask is not None:
                mask = F.pad(mask, (0, pad_len), value=False)
        elif L > self.max_seq_len:
            log_likelihoods = log_likelihoods[:, :self.max_seq_len]
            entropies = entropies[:, :self.max_seq_len]
            embeddings = embeddings[:, :self.max_seq_len]
            if mask is not None:
                mask = mask[:, :self.max_seq_len]

        # Compute mean for scalar features (Eq. 1)
        if mask is not None:
            # Masked mean
            ll_mean = (log_likelihoods * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp(min=1)
            entropy_mean = (entropies * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            ll_mean = log_likelihoods.mean(dim=1, keepdim=True)
            entropy_mean = entropies.mean(dim=1, keepdim=True)

        # Branch outputs
        h_ll = self.ll_branch(ll_mean)              # (B, hidden_dim)
        h_entropy = self.entropy_branch(entropy_mean)  # (B, hidden_dim)

        # Embedding branch: (B, L, D) → (B, D, L) for Conv1d
        emb = embeddings.transpose(1, 2)           # (B, D, L)
        h_emb = self.embedding_branch(emb)          # (B, hidden_dim, 1)
        h_emb = h_emb.squeeze(-1)                    # (B, hidden_dim)

        # Stack branch outputs (Eq. 3)
        H = torch.stack([h_ll, h_entropy, h_emb], dim=1)  # (B, 3, hidden_dim)

        # Fusion
        if self.fusion == 'attention':
            # Eq. 4: attention-based fusion
            # tanh(Wa * h) for each branch
            attn_input = torch.tanh(self.Wa(H))        # (B, 3, hidden_dim)
            attn_scores = self.w(attn_input).squeeze(-1)  # (B, 3)
            alpha = F.softmax(attn_scores, dim=1)       # (B, 3)
            h_fused = (alpha.unsqueeze(-1) * H).sum(dim=1)  # (B, hidden_dim)
        elif self.fusion == 'mlp':
            # Eq. 5: concatenation + MLP
            h_flat = H.view(B, -1)                     # (B, 3 * hidden_dim)
            h_fused = self.fusion_mlp(h_flat)            # (B, hidden_dim)

        # Output projection (Eq. 6)
        logits = self.output(h_fused).squeeze(-1)      # (B,)
        return logits


class HaluNet(BaseBaseline):
    """
    HaluNet baseline for hallucination detection.

    A trainable multi-branch neural network that fuses three types of
    token-level uncertainty signals:
    1. Log-likelihoods (probabilistic confidence)
    2. Entropies (distributional uncertainty)
    3. Hidden-state embeddings (semantic trajectory)

    This is the only trainable baseline in our lineup, serving as an
    upper-bound comparison. Our method is also trainable (lightweight
    classifier on MoE signals), so the comparison is fair.

    Paper: HaluNet: Multi-Granular Uncertainty Modeling for Efficient
           Hallucination Detection in LLM Question Answering
           (arXiv 2512.24562)

    Args:
        embedding_dim: Dimension of hidden-state embeddings. Default 768.
        hidden_dim: Branch latent dimension. Default 128.
        max_seq_len: Maximum sequence length (zero-padded). Default 50.
        dropout: Dropout probability. Default 0.5.
        fusion: 'attention' or 'mlp'. Default 'attention'.
        device: torch device. Default 'cuda' if available, else 'cpu'.
        lr: Learning rate. Default 1e-3.
        epochs: Training epochs. Default 20.
        batch_size: Training batch size. Default 32.
    """

    def __init__(self, embedding_dim=768, hidden_dim=128, max_seq_len=50,
                 dropout=0.5, fusion='attention', device=None,
                 lr=1e-3, epochs=20, batch_size=32):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.model = HaluNetModel(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len,
            dropout=dropout,
            fusion=fusion,
        ).to(device)

        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold = 0.5  # Default; can be tuned via fit

    def _prepare_batch(self, log_likelihoods_list, entropies_list,
                       embeddings_list, labels=None):
        """Prepare a batch of data for the model.

        Args:
            log_likelihoods_list: List of (L,) arrays — per-token log probs.
            entropies_list: List of (L,) arrays — per-token entropies.
            embeddings_list: List of (L, D) arrays — per-token embeddings.
            labels: Optional list of binary labels.

        Returns:
            Dict of tensors ready for model forward pass.
        """
        B = len(log_likelihoods_list)

        # Find max sequence length in this batch
        max_len = max(len(ll) for ll in log_likelihoods_list)

        # Pad all sequences to max_len
        ll_padded = []
        ent_padded = []
        emb_padded = []
        mask = []

        for i in range(B):
            L = len(log_likelihoods_list[i])
            ll_padded.append(np.pad(log_likelihoods_list[i], (0, max_len - L),
                                    mode='constant', constant_values=0.0))
            ent_padded.append(np.pad(entropies_list[i], (0, max_len - L),
                                     mode='constant', constant_values=0.0))
            emb_padded.append(np.pad(embeddings_list[i], ((0, max_len - L), (0, 0)),
                                     mode='constant', constant_values=0.0))
            mask.append(np.concatenate([np.ones(L), np.zeros(max_len - L)]))

        ll_tensor = torch.tensor(np.stack(ll_padded), dtype=torch.float32, device=self.device)
        ent_tensor = torch.tensor(np.stack(ent_padded), dtype=torch.float32, device=self.device)
        emb_tensor = torch.tensor(np.stack(emb_padded), dtype=torch.float32, device=self.device)
        mask_tensor = torch.tensor(np.stack(mask), dtype=torch.bool, device=self.device)

        result = {
            'log_likelihoods': ll_tensor,
            'entropies': ent_tensor,
            'embeddings': emb_tensor,
            'mask': mask_tensor,
        }

        if labels is not None:
            result['labels'] = torch.tensor(labels, dtype=torch.float32, device=self.device)

        return result

    def fit(self, log_likelihoods_list, entropies_list, embeddings_list, labels,
            val_split=0.1, verbose=True):
        """
        Train HaluNet on labeled data.

        Args:
            log_likelihoods_list: List of (L,) arrays — per-token log probs.
            entropies_list: List of (L,) arrays — per-token entropies.
            embeddings_list: List of (L, D) arrays — per-token embeddings.
            labels: Binary hallucination labels (0=factual, 1=hallucinated).
            val_split: Fraction of data for validation (threshold tuning).
            verbose: Print training progress.
        """
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy().astype(float)
        else:
            labels = np.array(labels, dtype=float)

        N = len(labels)
        indices = np.random.permutation(N)
        split_idx = int(N * (1 - val_split))
        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        self.model.train()
        for epoch in range(self.epochs):
            # Training
            np.random.shuffle(train_idx)
            train_loss = 0.0

            for start in range(0, len(train_idx), self.batch_size):
                batch_idx = train_idx[start:start + self.batch_size]
                batch = self._prepare_batch(
                    [log_likelihoods_list[i] for i in batch_idx],
                    [entropies_list[i] for i in batch_idx],
                    [embeddings_list[i] for i in batch_idx],
                    [labels[i] for i in batch_idx],
                )

                optimizer.zero_grad()
                logits = self.model(
                    batch['log_likelihoods'],
                    batch['entropies'],
                    batch['embeddings'],
                    batch['mask'],
                )
                loss = criterion(logits, batch['labels'])
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(batch_idx)

            train_loss /= len(train_idx)

            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {train_loss:.4f}")

        # Validation: tune threshold
        if len(val_idx) > 0:
            self.model.eval()
            with torch.no_grad():
                val_batch = self._prepare_batch(
                    [log_likelihoods_list[i] for i in val_idx],
                    [entropies_list[i] for i in val_idx],
                    [embeddings_list[i] for i in val_idx],
                )
                val_logits = self.model(
                    val_batch['log_likelihoods'],
                    val_batch['entropies'],
                    val_batch['embeddings'],
                    val_batch['mask'],
                )
                val_probs = torch.sigmoid(val_logits).cpu().numpy()
                val_labels = labels[val_idx]

                # Find best threshold
                best_threshold = 0.5
                best_accuracy = 0.0
                for threshold in np.linspace(val_probs.min(), val_probs.max(), 100):
                    preds = (val_probs >= threshold).astype(int)
                    accuracy = (preds == val_labels).mean()
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_threshold = threshold

                self.threshold = best_threshold
                if verbose:
                    print(f"Validation accuracy: {best_accuracy:.4f}, Threshold: {self.threshold:.4f}")

    def predict_proba(self, log_likelihoods, entropies, embeddings):
        """
        Predict hallucination probability for a single answer.

        Args:
            log_likelihoods: (L,) array — per-token log probabilities.
            entropies: (L,) array — per-token entropies.
            embeddings: (L, D) array — per-token hidden-state embeddings.

        Returns:
            float — hallucination probability in [0, 1].
        """
        self.model.eval()
        with torch.no_grad():
            batch = self._prepare_batch(
                [log_likelihoods],
                [entropies],
                [embeddings],
            )
            logits = self.model(
                batch['log_likelihoods'],
                batch['entropies'],
                batch['embeddings'],
                batch['mask'],
            )
            prob = torch.sigmoid(logits).item()
        return prob

    def predict(self, log_likelihoods, entropies, embeddings):
        """
        Predict binary hallucination label for a single answer.

        Args:
            log_likelihoods: (L,) array — per-token log probabilities.
            entropies: (L,) array — per-token entropies.
            embeddings: (L, D) array — per-token hidden-state embeddings.

        Returns:
            int — binary label (0=factual, 1=hallucinated).
        """
        prob = self.predict_proba(log_likelihoods, entropies, embeddings)
        return int(prob >= self.threshold)

    def save(self, path):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'threshold': self.threshold,
            'config': {
                'embedding_dim': self.model.embedding_branch[0].in_channels,
                'hidden_dim': self.model.hidden_dim,
                'max_seq_len': self.model.max_seq_len,
                'fusion': self.model.fusion,
            }
        }, path)

    def load(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        # Reconstruct model with the exact architecture used during training.
        # The checkpoint always stores config (since PR fix in save()).
        config = checkpoint.get('config')
        if config is not None:
            self.model = HaluNetModel(**config).to(self.device)
        # If config is missing (very old checkpoints), fall back to current self.model.
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.threshold = checkpoint['threshold']