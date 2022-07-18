from torch.overrides import (
    has_torch_function, has_torch_function_unary, has_torch_function_variadic,
    handle_torch_function)
from torch import _VF
from typing import (
    Tuple, Optional, List
)

import math

import torch
import torch.nn.functional as F
Tensor = torch.Tensor

"""
Code is taken from torch functional.py and adapted to allow to pass a custom attention function. All credits belong to the original authors.
"""


def _scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    key_pos=None,
    query_pos=None,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    **kwargs
) -> Tuple[Tensor, Tensor]:
    r"""
    Computes scaled dot product attention on query, key and value tensors, using
    an optional attention mask if passed, and applying dropout if a probability
    greater than 0.0 is specified.
    Returns a tensor pair containing attended values and attention weights.

    Args:
        q, k, v: query, key and value tensors. See Shape section for shape details.
        attn_mask: optional tensor containing mask values to be added to calculated
            attention. May be 2D or 3D; see Shape section for details.
        dropout_p: dropout probability. If greater than 0.0, dropout is applied.

    Shape:
        - q: :math:`(B, Nt, E)` where B is batch size, Nt is the target sequence length,
            and E is embedding dimension.
        - key: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - value: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - attn_mask: either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
            shape :math:`(Nt, Ns)`.

        - Output: attention values have shape :math:`(B, Nt, E)`; attention weights
            have shape :math:`(B, Nt, Ns)`
    """

    SENSORS, B, Nt, E = q.shape

    # normalize before stacking to not rescale the direction vector
    q = q / math.sqrt(E)

    # stack query / key with pos features if present
    if key_pos is not None:
        k = torch.cat([k, key_pos], dim=-1)
    if query_pos is not None:
        q = torch.cat([q, query_pos], dim=-1)

    patches_per_img = k.shape[2]

    attn_full = torch.empty((SENSORS, B, Nt, patches_per_img), device=q.device)
    for sensor_idx in range(SENSORS):
        # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
        attn = torch.bmm(q[sensor_idx], k[sensor_idx].transpose(-2, -1))

        attn = F.softmax(attn, dim=-1)
        if dropout_p > 0.0:
            attn = F.dropout(attn, p=dropout_p)

        attn_full[sensor_idx] = attn

    # sensors x B x Nt x patches -> B x Nt x sensors x patches
    attn_full = attn_full.permute(1, 2, 0, 3)
    attn_full = attn_full.reshape(B, Nt, -1)

    # sensors x b x patches x dims -> b x sensors x patches x dims
    v = v.permute(1, 0, 2, 3)
    v = v.reshape(B, -1, E)

    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn_full, v)
    return output, attn_full


def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    key_pos: Optional[Tensor] = None,
    query_pos: Optional[Tensor] = None,
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    attn_func=_scaled_dot_product_attention,
    **kwargs,
) -> Tuple[Tensor, Optional[Tensor]]:

    # TODO refactor: change interface to only supported options e.g. without masks
    tens_ops = (query, key, value, in_proj_weight, in_proj_bias,
                bias_k, bias_v, out_proj_weight, out_proj_bias)
    if F.has_torch_function(tens_ops):
        return handle_torch_function(
            multi_head_attention_forward,
            tens_ops,
            query,
            key,
            value,
            embed_dim_to_check,
            num_heads,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            add_zero_attn,
            dropout_p,
            out_proj_weight,
            out_proj_bias,
            key_pos=key_pos,
            query_pos=query_pos,
            training=training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            static_k=static_k,
            static_v=static_v,
            attn_func=attn_func,
            **kwargs
        )

    # set up shape vars
    sensors, tgt_len, bsz, embed_dim = query.shape
    _, src_len, _, _ = key.shape
    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * \
        num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert key.shape[:2] == value.shape[:2], \
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #

    # for projection we need batch first
    # q: sensors x queries x bs x dims
    # k: sensors x keys x bs x dims
    # v: sensors  x vals x  bs x dims
    # -> bs x sensors x p  x dims
    query = query.permute(2, 0, 1, 3)
    key = key.permute(2, 0, 1, 3)
    value = value.permute(2, 0, 1, 3)

    if not use_separate_proj_weight:
        query, key, value = _in_projection_packed(
            query, key, value, in_proj_weight, in_proj_bias)
    else:
        raise NotImplementedError

    # add bias along batch dimension (currently first)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        # TODO critical: if used with bias test if this works
        key = torch.cat([key, bias_k.repeat(bsz, sensors, 1, 1)])
        value = torch.cat([value, bias_v.repeat(bsz, sensors, 1, 1)])

    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention
    #
    # TODO refactor to faster operations
    # permute first
    # bs x sensors x p x full_dim -> sensors x p x bs x dim
    query = query.permute(1, 2, 0, 3)
    key = key.permute(1, 2, 0, 3)
    value = value.permute(1, 2, 0, 3)

    # now split in heads
    query = query.reshape(sensors, tgt_len, bsz*num_heads, head_dim)
    key = key.reshape(sensors, src_len, bsz*num_heads, head_dim)
    value = value.reshape(sensors, src_len, bsz*num_heads, head_dim)

    # bring batch to front
    # sensors x p x bs * heads x head_dim -> sensors x bs*heads x p x head_dim
    query = query.permute(0, 2, 1, 3)
    key = key.permute(0, 2, 1, 3)
    value = value.permute(0, 2, 1, 3)

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #
    attn_output, attn_output_weights = attn_func(
        q=query, k=key, v=value, key_pos=key_pos, query_pos=query_pos, attn_mask=None, dropout_p=dropout_p, **kwargs)

    attn_output = attn_output.transpose(
        0, 1).contiguous().view(tgt_len, bsz, embed_dim)

    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        raise NotImplementedError(
            "not working with sensor specific attention yet")
    else:
        return attn_output, None


def _in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
) -> List[Tensor]:

    w_q, w_k, w_v = w.chunk(3)
    if b is None:
        b_q = b_k = b_v = None
    else:
        b_q, b_k, b_v = b.chunk(3)
    return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)
