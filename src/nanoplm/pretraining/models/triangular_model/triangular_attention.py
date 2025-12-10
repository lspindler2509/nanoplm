"""
Triangular Attention implementation for protein structure-aware modeling.
Based on AlphaFold/Boltzmann-2 concepts using NVIDIA cuEquivariance optimizations.

Requires: pip install cuequivariance-torch

This implementation exclusively uses NVIDIA's cuEquivariance library for 
high-performance triangular attention operations.
"""

import sys
import math
from nanoplm.pretraining.models.triangular_model.attention import AttentionPairBias
from nanoplm.pretraining.models.triangular_model.transition import Transition
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from scipy.stats import truncnorm
from nanoplm.pretraining.models.triangular_model.positional_encoding import RelativePositionEncoding
from einops.layers.torch import Rearrange
import nanoplm.pretraining.models.triangular_model.initialize as init

class TriangularSelfAttention(nn.Module):
    """
    Triangular self-attention module implementing the geometric constraints
    from AlphaFold/Boltzmann-2 for protein structure modeling.
    """
    
    def __init__(
        self,
        pair_dim: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        dropout: float = 0.1,
        orientation: str = "per_row"  # "per_row" or "per_column"
    ):
        super().__init__()
                
        self.pair_dim = pair_dim
        self.num_heads = num_heads
        self.head_dim = head_dim or pair_dim // num_heads
        self.dropout = dropout
        self.orientation = orientation
        
        # Ensure dimensions are compatible
        assert self.head_dim * num_heads == pair_dim, f"pair_dim ({pair_dim}) must be divisible by num_heads ({num_heads})"
        
        # Validate cuEquivariance performance constraints
        self._validate_cuequivariance_constraints()
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(pair_dim, pair_dim, bias=False)
        self.k_proj = nn.Linear(pair_dim, pair_dim, bias=False)
        self.v_proj = nn.Linear(pair_dim, pair_dim, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(pair_dim, pair_dim)
        
        # Gating mechanism (from Boltzmann-2)
        self.gate_proj = nn.Linear(pair_dim, pair_dim)
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(pair_dim)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling for gradient stability"""
        # Initialize Q, K, V projections with smaller gain for stability
        for module in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(module.weight, gain=0.25)
        
        # Initialize output projection with even smaller gain
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)
        nn.init.zeros_(self.out_proj.bias)
        
        # Initialize gate to favor residual (conservative start)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, -1.5)  # sigmoid(-1.5) ≈ 0.18, favors original input

    def forward(
        self, 
        pair_repr: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        ta_layer_num: Optional[int] = None
    ) -> torch.Tensor:
        """
        Forward pass of triangular attention.
        
        Args:
            pair_repr: Pair representation tensor of shape (B, N, N, C)
            attention_mask: Optional attention attention_mask of shape (B, N, N)
            
        Returns:
            Updated pair representation of shape (B, N, N, C)
        """
        B, N, N_check, C = pair_repr.shape
        assert N == N_check, f"Expected square pair matrix, got {N} x {N_check}"
        assert C == self.pair_dim, f"Expected pair_dim {self.pair_dim}, got {C}"
        
        # Apply layer norm
        normed_input = self.norm(pair_repr).to(torch.bfloat16)
        
        # Always use NVIDIA cuEquivariance (availability checked in __init__)
        output = self._cuequivariance_triangular_attention(normed_input, attention_mask)
        
        # Apply gating (residual connection with learned gate)
        gate = torch.sigmoid(self.gate_proj(pair_repr))
        if torch.rand(1).item() < 0.0001:
            print(f"[TriAttn] gate stats at layer {ta_layer_num}: min={gate.min().item():.4f}, max={gate.max().item():.4f}")
            print(f"[TriAttn] mean gate = {gate.mean().item():.4f}")
            delta = (output - normed_input).abs().mean().item()
            print(f"[TriAttn] delta input→output = {delta:.4f}")
            print("\n")

        output = gate * output + (1 - gate) * normed_input
        
        return output
    
    def _cuequivariance_triangular_attention(
        self, 
        pair_repr: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Use NVIDIA cuEquivariance optimized triangular attention kernel"""
        B, N, _, C = pair_repr.shape
        
        # Compute Q, K, V projections
        q = self.q_proj(pair_repr)  # (B, N, N, C)
        k = self.k_proj(pair_repr)  # (B, N, N, C)
        v = self.v_proj(pair_repr)  # (B, N, N, C)
        
        # Reshape for NVIDIA cuEquivariance format: (B, N, H, Q, D) where Q=K=N
        # From (B, N, N, C) -> (B, N, N, H, D) -> (B, N, H, N, D)
        # Note: In triangular attention, each row/column attends to all positions, so Q=K=N
        q = q.view(B, N, N, self.num_heads, self.head_dim)  # (B, N, N, H, D)
        k = k.view(B, N, N, self.num_heads, self.head_dim)  # (B, N, N, H, D)  
        v = v.view(B, N, N, self.num_heads, self.head_dim)  # (B, N, N, H, D)
        
        # Transpose to get cuEquivariance format: (B, N, H, Q, D) where Q=N (second N becomes Q)
        q = q.permute(0, 3, 1, 2, 4)  # (B, N, N, H, D) -> (B, N, H, N, D) (wanted B H N N D)
        k = k.permute(0, 3, 1, 2, 4)  # (B, N, N, H, D) -> (B, N, H, N, D)
        v = v.permute(0, 3, 1, 2, 4)  # (B, N, N, H, D) -> (B, N, H, N, D)

        # Create bias tensor (required parameter): (B, 1, H, Q, K) = (B, 1, H, N, N) (wanted B H N N)
        bias = torch.zeros(B, self.num_heads, N, N, device=q.device, dtype=torch.bfloat16)
        
        # Convert mask format for cuEquivariance: (B, N, 1, 1, K) where K=N
        attn_mask = None
        if attention_mask is not None:
            
            # Convert mask to boolean first if needed
            if attention_mask.dtype != torch.bool:
                # Assume 1 = valid, 0 = invalid for non-boolean masks
                attention_mask = attention_mask.bool()

            # Handle different mask dimensions and convert to cuEquivariance format (B, N, 1, 1, K)
            if attention_mask.dim() == 2:  # (B, N) sequence mask
                # Expand to cuEquivariance format: (B, N, 1, 1, N)
                # Each query position N can attend to all valid key positions
                B_mask, N_mask = attention_mask.shape
                # Create query mask: each query position gets the full sequence mask
                # (B, N) -> (B, N, 1, 1, N)
                # attn_mask = attention_mask.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # (B, N, 1, 1, 1)
                # attn_mask = attn_mask.expand(B_mask, N_mask, 1, 1, N_mask)  # (B, N, 1, 1, N)
                # # Apply sequence validity: both query and key must be valid
                # key_valid = attention_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)  # (B, 1, 1, 1, N)
                # attn_mask = attn_mask & key_valid  # (B, N, 1, 1, N)
                attn_mask = attention_mask.unsqueeze(2) & attention_mask.unsqueeze(1)  # (B, N, N)
            else:
                print(f"⚠️  Unsupported mask dimension: {attention_mask.dim()}, using no mask", file=sys.stderr)
                attn_mask = None
        
        # API: triangle_attention(q, k, v, bias, mask=None, scale=None, return_aux=False)
        try:
            #from cuequivariance_torch import triangle_attention
            #from cuequivariance_ops_torch import init_triton_cache
            #init_triton_cache()
            from trifast import triangle_attention
        except ImportError as e:
            raise ImportError(
                f"Failed to import cuEquivariance triangle_attention. "
                f"Make sure you're running on a GPU node with CUDA drivers loaded. "
                f"Error: {e}"
            )
        
        scale = 1.0 / math.sqrt(self.head_dim)
        
        # Debug: Check all tensor properties before cuEquivariance call
        
        if self.orientation == "per_row":
            # Starting update: triangular attention where we update (i,j) based on (i,k) and (k,j)
            try:
                attn_output = triangle_attention(q, k, v, bias, attn_mask)
                # attn_output = triangle_attention(
                #     q=q, k=k, v=v,
                #     bias=bias,
                #     mask=attn_mask,
                #     scale=scale,
                #     return_aux=False
                # )
            except Exception as e:
                print(f"💥 TRIANGLE_ATTENTION ERROR (per_row): {type(e).__name__}: {e}", file=sys.stderr)
                raise e
        else:
            # Ending update: column-wise triangular attention  
            # For column-wise updates, we need to transpose the sequence dimensions
            # This changes how we attend: instead of row i attending to all columns j,
            # we want column j attending to all rows i
            q_t = q.transpose(2, 3)  # (B, N, H, N, D) -> (B, N, H, N, D) - swap query/key sequence dims
            k_t = k.transpose(2, 3)  # (B, N, H, N, D) -> (B, N, H, N, D)
            v_t = v.transpose(2, 3)  # (B, N, H, N, D) -> (B, N, H, N, D) BHNND
            bias_t = bias.transpose(2, 3)  # (B, 1, H, N, N) -> (B, 1, H, N, N) - swap bias dims BHNN
            
            attn_mask_t = None
            if attn_mask is not None:
                attn_mask_t = attn_mask.transpose(1, 2)  # (B, N, 1, 1, N) -> (B, N, 1, 1, N) - swap mask dims
            
            try:
                # attn_output = triangle_attention(
                #     q=q_t, k=k_t, v=v_t,
                #     bias=bias_t,
                #     mask=attn_mask_t,
                #     scale=scale,
                #     return_aux=False
                # )
                attn_output = triangle_attention(q_t, k_t, v_t, bias_t, attn_mask_t)
                
                # Transpose back for ending update
                attn_output = attn_output.transpose(2, 3)  # Transpose back from column-wise
            except Exception as e:
                print(f"💥 TRIANGLE_ATTENTION ERROR (per_column): {type(e).__name__}: {e}", file=sys.stderr)
                raise e
        
        # Reshape back to (B, N, N, C) from (B, N, H, Q, D) where Q=N
        # (B, N, H, N, D) -> (B, N, N, H, D) -> (B, N, N, C)
        attn_output = attn_output.permute(0, 2, 3, 1, 4)  # (B, N, H, N, D) BHNND -> (B, N, N, H, D)
        attn_output = attn_output.contiguous().view(B, N, N, C)  # (B, N, N, H*D)
        
        # Output projection
        output = self.out_proj(attn_output)
        return output
    
    def _validate_cuequivariance_constraints(self):
        """
        Validate NVIDIA cuEquivariance performance constraints for optimal CUDA kernel usage.
        
        Performance Constraints from NVIDIA Docs:
        - tf32/fp32: hidden_dim <= 32 & divisible by 4
        - bf16/fp16: hidden_dim <= 128 & divisible by 8
        
        Falls back to PyTorch implementation if constraints not met (no error).
        """
        import warnings
        
        # Check both precision constraints (hidden_dim = pair_dim in our case)
        hidden_dim = self.pair_dim
        is_optimal_tf32 = hidden_dim <= 32 and hidden_dim % 4 == 0
        is_optimal_bf16 = hidden_dim <= 128 and hidden_dim % 8 == 0
        
        if not (is_optimal_tf32 or is_optimal_bf16):
            warnings.warn(
                f"Triangular Attention Performance Warning: "
                f"hidden_dim={hidden_dim} does not meet cuEquivariance optimal constraints. "
                f"tf32/fp32: hidden_dim <= 32 & divisible by 4. "
                f"bf16/fp16: hidden_dim <= 128 & divisible by 8. "
                f"Will fallback to PyTorch implementation.",
                UserWarning
            )
        else:

            print(f"✅ cuEquivariance constraints met for 'tf32/fp32': {is_optimal_tf32} (hidden_dim={hidden_dim})'")
            print(f"✅ cuEquivariance constraints met for 'bf16/fp16': {is_optimal_bf16} (hidden_dim={hidden_dim})")

class PairwiseTriangularBlock(nn.Module):
    """
    Pairwise triangular attention block combining row- and column-wise updates.
    Includes learnable mixing between normalized and raw pair states and
    conservative initialization for gradient stability.
    """

    def __init__(
        self,
        residue_dim: int,
        pair_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_multiplication: bool = True,
        use_positional_encoding: bool = False,
        num_attention_heads: Optional[int] = None  # For residue attention
    ):
        super().__init__()
        
        self.residue_dim = residue_dim
        self.pair_dim = pair_dim
        self.num_heads = num_heads
        self.head_dim = pair_dim // num_heads
        self.dropout = dropout
        self.use_multiplication = use_multiplication
        self.use_positional_encoding = use_positional_encoding
        
        # Number of attention heads for residue attention (default: residue_dim // 64)
        if num_attention_heads is None:
            num_attention_heads = max(1, residue_dim // 64)
        self.num_attention_heads = num_attention_heads
        
        # Ensure residue_dim is divisible by num_attention_heads
        if residue_dim % num_attention_heads != 0:
            raise ValueError(
                f"residue_dim ({residue_dim}) must be divisible by num_attention_heads ({num_attention_heads})"
            )
        self.residue_head_dim = residue_dim // num_attention_heads
        self.pre_norm_residue = nn.LayerNorm(residue_dim)
        
        if not self.use_multiplication:

            # --- Triangular attention modules ---
            self.tri_attn_row = TriangularSelfAttention(
                pair_dim=pair_dim,
                num_heads=num_heads,
                head_dim=self.head_dim,
                dropout=dropout,
                orientation="per_row"
            )
            self.tri_attn_col = TriangularSelfAttention(
                pair_dim=pair_dim,
                num_heads=num_heads,
                head_dim=self.head_dim,
                dropout=dropout,
                orientation="per_column"
            )
        else:
            self.norm_in = nn.LayerNorm(pair_dim, eps=1e-5)
            self.p_in = nn.Linear(pair_dim, 2 * pair_dim, bias=False)
            self.g_in = nn.Linear(pair_dim, 2 * pair_dim, bias=False)

            self.norm_out = nn.LayerNorm(pair_dim)
            self.p_out = nn.Linear(pair_dim, pair_dim, bias=False)
            self.g_out = nn.Linear(pair_dim, pair_dim, bias=False)

            init.bias_init_one_(self.norm_in.weight)
            init.bias_init_zero_(self.norm_in.bias)

            init.lecun_normal_init_(self.p_in.weight)
            init.gating_init_(self.g_in.weight)

            init.bias_init_one_(self.norm_out.weight)
            init.bias_init_zero_(self.norm_out.bias)

            init.final_init_(self.p_out.weight)
            init.gating_init_(self.g_out.weight)
            
        if self.use_positional_encoding:
            self.relpos = RelativePositionEncoding(
                r_max=32,
                s_max=2,
                dim_out=pair_dim
            )

        # --- Residue → Pair projections (AlphaFold-style) ---
        self.residue_to_pair_left = nn.Linear(residue_dim, pair_dim)
        self.residue_to_pair_right = nn.Linear(residue_dim, pair_dim)

        self.attention = AttentionPairBias(residue_dim, pair_dim, num_heads)
        
        self.transition_z = Transition(pair_dim, pair_dim * 4)
        self.transition_s = Transition(residue_dim, residue_dim * 4)
        self.s_post_norm = nn.LayerNorm(residue_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights conservatively to avoid gradient explosions."""
        nn.init.xavier_uniform_(self.residue_to_pair_left.weight)
        nn.init.zeros_(self.residue_to_pair_left.bias)
        nn.init.xavier_uniform_(self.residue_to_pair_right.weight)
        nn.init.zeros_(self.residue_to_pair_right.bias)

        print("✅ PairwiseTriangularBlock initialized")

    def forward(
        self,
        residue_repr: torch.Tensor,
        pair_repr: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        ta_layer_num: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass combining residue and pair representations.
        
        Args:
            residue_repr: (B, N, residue_dim)
            pair_repr: Optional (B, N, N, pair_dim)
            attention_mask: Optional (B, N, N)
        """
        # Initialize pair representation if missing
        if pair_repr is None:
            pair_repr = self._create_pair_representation(residue_repr, self.use_positional_encoding)
        
        if not self.use_multiplication:

            # --- Triangular attention updates ---
            pair_repr = pair_repr + self.tri_attn_row(pair_repr, attention_mask=attention_mask, ta_layer_num=ta_layer_num) # already includes norm & gating
            pair_repr = pair_repr + self.tri_attn_col(pair_repr, attention_mask=attention_mask, ta_layer_num=ta_layer_num+1) # already includes norm & gating
        
        else:
            attn_mask = None
            if attention_mask is not None:
                if attention_mask.dtype != torch.bool:
                    attention_mask = attention_mask.bool()
                if attention_mask.dim() == 2:
                    attn_mask = attention_mask.unsqueeze(2) & attention_mask.unsqueeze(1)  # (B, N, N)
                elif attention_mask.dim() == 4:
                    # [B, 1, N, N] → [B, N, N]
                    attn_mask = attention_mask[:, 0]

            from cuequivariance_torch import triangle_multiplicative_update
            pair_repr = pair_repr + triangle_multiplicative_update(
                x=pair_repr,
                direction="outgoing",  # or "incoming"
                mask=attn_mask,
                norm_in_weight=self.norm_in.weight,
                norm_in_bias=self.norm_in.bias,
                p_in_weight=self.p_in.weight,
                g_in_weight=self.g_in.weight,
                norm_out_weight=self.norm_out.weight,
                norm_out_bias=self.norm_out.bias,
                p_out_weight=self.p_out.weight,
                g_out_weight=self.g_out.weight,
                eps=1e-5,
            )
            pair_repr = pair_repr + triangle_multiplicative_update(
                x=pair_repr,
                direction="incoming",  # or "outgoing"
                mask=attn_mask,
                norm_in_weight=self.norm_in.weight,
                norm_in_bias=self.norm_in.bias,
                p_in_weight=self.p_in.weight,
                g_in_weight=self.g_in.weight,
                norm_out_weight=self.norm_out.weight,
                norm_out_bias=self.norm_out.bias,
                p_out_weight=self.p_out.weight,
                g_out_weight=self.g_out.weight,
                eps=1e-5,
            )

        # Transition layer on pair representations (like Boltz - applied before attention bias)
        pair_repr = (pair_repr + self.transition_z(pair_repr)).to(torch.bfloat16)
        
        # Compute sequence stack
        residue_repr_normed = self.pre_norm_residue(residue_repr)
        residue_repr = residue_repr + self.attention(
            s=residue_repr_normed, z=pair_repr, mask=attention_mask, k_in=residue_repr_normed
        )
        residue_repr = residue_repr + self.transition_s(residue_repr)
        residue_repr = self.s_post_norm(residue_repr).to(torch.bfloat16)
        
        return residue_repr, pair_repr


    def _create_pair_representation(self, residue_repr: torch.Tensor, use_positional_encoding: bool) -> torch.Tensor:
        """
        AlphaFold-style pair representation:
        pair[i,j] = linear_left(res_i) + linear_right(res_j)

        Args:
            residue_repr: Residue representations (B, N, residue_dim)

        Returns:
            Pair representation (B, N, N, pair_dim)
        """
        B, N, residue_dim = residue_repr.shape
        if use_positional_encoding:
            device = residue_repr.device
            pair_repr = self.relpos(
                batch_size=B,
                seq_len=N,
                device=device,
                complex_chain_break_indices=None
            )
            pair_repr = pair_repr.to(torch.bfloat16)
            return pair_repr

        left_proj = self.residue_to_pair_left(residue_repr)      # (B, N, pair_dim)
        right_proj = self.residue_to_pair_right(residue_repr)    # (B, N, pair_dim)

        # Broadcast + addition (AlphaFold)
        pair = left_proj.unsqueeze(2) + right_proj.unsqueeze(1)  # (B, N, N, pair_dim)

        return pair


def create_triangular_attention_layer(
    residue_dim: int,
    pair_dim: Optional[int] = None,
    num_heads: Optional[int] = None,
    dropout: float = 0.1,
    num_attention_heads: Optional[int] = None
) -> PairwiseTriangularBlock:
    """
    Factory function to create a triangular attention layer.
    
    Args:
        residue_dim: Dimension of residue representations
        pair_dim: Dimension of pair representations (default: residue_dim)
        num_heads: Number of attention heads for triangular attention (default: 8)
        dropout: Dropout rate
        num_attention_heads: Number of attention heads for residue attention (default: residue_dim // 64)
        
    Returns:
        PairwiseTriangularBlock instance
    """
    # Set defaults if not specified
    if pair_dim is None:
        pair_dim = residue_dim
    if num_heads is None:
        num_heads = 8
    
    print(f"🔧 Creating Triangular Attention Layer: residue_dim={residue_dim}, pair_dim={pair_dim}, num_heads={num_heads}, dropout={dropout}, num_attention_heads={num_attention_heads}")
        
    return PairwiseTriangularBlock(
        residue_dim=residue_dim,
        pair_dim=pair_dim,
        num_heads=num_heads,
        dropout=dropout,
        num_attention_heads=num_attention_heads
    )