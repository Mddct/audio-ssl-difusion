from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from difusion import DiffLlamaPrefix


class SSL_DiT(nn.Module):
    """
    SSL_DiT: A simplified Diffusion Transformer model for Text-to-Audio (TTS/Music/TTA)
    that is directly conditioned on features from a pre-trained SSL model.

    This architecture consists of:
    1. A conditioning projection layer for SSL features.
    2. A Diffusion Transformer (DiT) decoder that generates the output feature
       (e.g., VAE latent) from the SSL condition, optional text conditioning,
       and a prompt, using a flow-matching diffusion process.
    This version does not use any form of Vector Quantization.
    """

    def __init__(
            self,
            # Target feature settings
            target_dim: int = 128,
            ssl_feature_dim: int = 1024,
            # Transformer settings
            hidden_size: int = 1024,
            decoder_num_layers: int = 16,
            num_heads: int = 16,
            # Conditioning settings
            use_text_cond: bool = True,
            text_vocab_size: int = 32100,
            context_drop_p: float = 0.2,  # Dropout for prompt context
            # Diffusion settings
        sigma: float = 1e-5,
            time_scheduler: str = "linear",
            cfg: Optional[Any] = None,  # Config object for overriding
    ):
        """
        Initializes the SSL_DiT model.
        """
        super().__init__()

        # Override parameters with config object if provided
        if cfg is not None:
            target_dim = getattr(cfg, "target_dim", target_dim)
            ssl_feature_dim = getattr(cfg, "ssl_feature_dim", ssl_feature_dim)
            hidden_size = getattr(cfg, "hidden_size", hidden_size)
            decoder_num_layers = getattr(cfg, "decoder_num_layers",
                                         decoder_num_layers)
            num_heads = getattr(cfg, "num_heads", num_heads)
            use_text_cond = getattr(cfg, "use_text_cond", use_text_cond)
            text_vocab_size = getattr(cfg, "text_vocab_size", text_vocab_size)
            context_drop_p = getattr(cfg, "context_drop_p", context_drop_p)
            sigma = getattr(cfg, "sigma", sigma)
            time_scheduler = getattr(cfg, "time_scheduler", time_scheduler)

        self.target_dim = target_dim
        self.ssl_feature_dim = ssl_feature_dim
        self.hidden_size = hidden_size
        self.decoder_num_layers = decoder_num_layers
        self.num_heads = num_heads
        self.use_text_cond = use_text_cond
        self.text_vocab_size = text_vocab_size
        self.context_drop_p = context_drop_p
        self.sigma = sigma
        self.time_scheduler = time_scheduler

        # Text embedding layer
        if self.use_text_cond:
            self.text_emb = nn.Embedding(text_vocab_size, hidden_size)

        # Projection layer for SSL features to match hidden_size
        self.ssl_cond_projection = nn.Linear(self.ssl_feature_dim,
                                             self.hidden_size)

        # Decoder: The Diffusion Transformer (DiT)
        self.decoder = DiffLlamaPrefix(
            hidden_size=hidden_size,
            num_layers=decoder_num_layers,
            num_heads=num_heads,
            in_dim=self.target_dim,
            out_dim=self.target_dim,
            use_text_emb=use_text_cond,
            use_diff_step=True,
            use_cond=True,
        )

    def forward_diffusion(
        self, x: torch.Tensor, t: torch.Tensor, x_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor]:
        """
        Performs the forward diffusion process based on flow matching.
        It takes the clean data `x` (target latent) and a timestep `t` to produce a noisy sample `xt`.
        It also creates a prompt/target mask based on `x_mask`.
        """
        z = torch.randn(x.shape,
                        dtype=x.dtype,
                        device=x.device,
                        requires_grad=False)

        # Per-sample prompt length generation based on actual lengths
        actual_lengths = x_mask.sum(dim=1)
        keep_prompt_mask = (torch.rand(x.shape[0], device=x.device)
                            > self.context_drop_p)
        rand_ratios = (torch.rand(x.shape[0], device=x.device) * 0.3 + 0.1
                       )  # Range [0.1, 0.4)
        prompt_len_if_kept = (actual_lengths * rand_ratios).int()
        prompt_len = prompt_len_if_kept * keep_prompt_mask
        prompt_len = torch.clamp(prompt_len, max=actual_lengths - 1)
        prompt_len = torch.clamp(prompt_len, min=0)

        # Create a mask to distinguish prompt from target
        is_prompt = torch.zeros_like(x[:, :, 0])
        col_indices = torch.arange(is_prompt.shape[1],
                                   device=prompt_len.device).repeat(
                                       is_prompt.shape[0], 1)
        is_prompt[col_indices < prompt_len.unsqueeze(1)] = 1
        mask = torch.ones_like(x[:, :, 0])
        mask[is_prompt.bool()] = 0
        mask = mask.unsqueeze(-1)

        t = t.unsqueeze(-1).unsqueeze(-1)
        xt = ((1 - (1 - self.sigma) * t) * z + t * x) * mask + x * (1 - mask)

        return xt, z, t.squeeze(), prompt_len, mask

    def forward(
        self,
        target_latent: torch.Tensor,
        target_mask: torch.Tensor,
        ssl_features: torch.Tensor,
        ssl_mask: torch.Tensor,
        text_ids: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        The main training-time forward pass of the model.

        Args:
            target_latent (torch.Tensor): The target latent representation (e.g., from VAE or mel), shape `(B, T_target, target_dim)`.
            target_mask (torch.Tensor): Padding mask for `target_latent`.
            ssl_features (torch.Tensor): Input SSL features for conditioning, shape `(B, T_ssl, ssl_dim)`.
            ssl_mask (torch.Tensor): Padding mask for `ssl_features`.
            text_ids (Optional[torch.Tensor]): Input text token IDs for conditioning.
            text_mask (Optional[torch.Tensor]): Padding mask for `text_ids`.

        Returns:
            Dict for loss computation.
        """
        # 1. Prepare SSL-based condition
        # Project SSL features to the hidden size of the decoder
        cond_emb = self.ssl_cond_projection(ssl_features)

        # Upsample cond_emb to match the length of target_latent if necessary
        if cond_emb.shape[1] != target_latent.shape[1]:
            cond_emb = cond_emb.transpose(1, 2)
            cond_emb = F.interpolate(cond_emb,
                                     size=target_latent.shape[1],
                                     mode="linear")
            cond_emb = cond_emb.transpose(1, 2)

        # 2. Prepare text-based condition
        if self.use_text_cond:
            text_emb = self.text_emb(text_ids)
        else:
            text_emb = None

        # 3. Perform diffusion on the target latent
        t = torch.rand(target_latent.shape[0],
                       device=target_latent.device,
                       requires_grad=False)
        t = torch.clamp(t, 1e-5, 1.0)
        xt, z, new_t, prompt_len, mask = self.forward_diffusion(
            target_latent, t, target_mask)

        # 4. Decoder pass (The DiT)
        flow_pred = self.decoder(
            x=xt,
            x_mask=target_mask,
            text_embedding=text_emb,
            text_mask=text_mask,
            cond=cond_emb,
            diffusion_step=new_t,
        )

        # Predict the clean latent `x0_pred` from the noisy input `xt` and the predicted flow
        # Note: The original `t` of shape (B,) is used here, and needs unsqueezing.
        x_pred = xt + (1 - t.unsqueeze(-1).unsqueeze(-1)) * flow_pred

        final_mask = mask * target_mask.unsqueeze(-1)

        return {
            "noise": z,
            "x": target_latent,
            "flow_pred": flow_pred,
            "x_pred": x_pred,
            "final_mask": final_mask,
            "prompt_len": prompt_len,
        }

    @torch.no_grad()
    def reverse_diffusion(
        self,
        ssl_features: torch.Tensor,
        target_len: int,
        text_ids: Optional[torch.Tensor] = None,
        prompt_latent: Optional[torch.Tensor] = None,
        n_timesteps: int = 32,
        cfg: float = 1.0,
    ) -> torch.Tensor:
        """
        Performs the reverse diffusion process for inference.
        """
        # Prepare SSL condition
        cond_emb = self.ssl_cond_projection(ssl_features)

        # Upsample to target length
        if cond_emb.shape[1] != target_len:
            cond_emb = cond_emb.transpose(1, 2)
            cond_emb = F.interpolate(cond_emb, size=target_len, mode="linear")
            cond_emb = cond_emb.transpose(1, 2)

        # Prepare text condition
        if self.use_text_cond and text_ids is not None:
            text_emb = self.text_emb(text_ids)
            text_mask = torch.ones_like(text_ids)
        else:
            text_emb, text_mask = None, None

        # Handle prompt
        if prompt_latent is None:
            prompt_latent = torch.zeros(cond_emb.shape[0],
                                        0,
                                        self.target_dim,
                                        device=cond_emb.device)

        prompt_len = prompt_latent.shape[1]
        generation_len = target_len - prompt_len

        xt_mask = torch.ones(cond_emb.shape[0],
                             target_len,
                             device=cond_emb.device)

        # Initialize with random noise
        z = torch.randn(
            (cond_emb.shape[0], generation_len, self.target_dim),
            dtype=cond_emb.dtype,
            device=cond_emb.device,
        )
        xt = z
        h = 1.0 / n_timesteps

        # Iterative denoising loop
        for i in range(n_timesteps):
            xt_input = torch.cat([prompt_latent, xt], dim=1)
            t = (0 + (i + 0.5) * h) * torch.ones(
                z.shape[0], dtype=z.dtype, device=z.device)

            # Get conditional flow prediction
            flow_pred = self.decoder(
                x=xt_input,
                x_mask=xt_mask,
                text_embedding=text_emb,
                text_mask=text_mask,
                cond=cond_emb,
                diffusion_step=t,
            )
            flow_pred = flow_pred[:, prompt_len:, :]

            # Optional: Add Classifier-Free Guidance (CFG) logic here if needed
            # ...

            dxt = flow_pred * h
            xt = xt + dxt

        return xt
