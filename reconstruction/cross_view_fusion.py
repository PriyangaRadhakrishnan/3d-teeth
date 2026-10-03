from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from reconstruction.blender_depth_fusion import (
    rotation_matrix_from_euler,
    resolve_file,
    load_depth_map,
    load_mask
)


class RelativePoseEncoder(nn.Module):
    """
    Encodes the pairwise relative camera geometry between views (relative rotation,
    translation baseline, distance, and angular separation) into geometric feature
    embeddings that guide multi-view cross-attention.
    """
    def __init__(self, out_dim: int = 64):
        super().__init__()
        # Input: 3 (rel translation) + 1 (baseline dist) + 1 (angular diff) + 9 (rel rotation matrix) = 14 dims
        self.mlp = nn.Sequential(
            nn.Linear(14, 64),
            nn.SiLU(),
            nn.Linear(64, out_dim),
            nn.SiLU()
        )
        self.affinity_bias = nn.Linear(out_dim, 1)

    def forward(self, rel_geo_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            rel_geo_feats: (V, V, 14) pairwise relative geometric features
        Returns:
            geo_emb: (V, V, out_dim)
            affinity: (V, V, 1) scalar pairwise geometric affinity bias
        """
        geo_emb = self.mlp(rel_geo_feats)
        bias = self.affinity_bias(geo_emb)
        return geo_emb, bias


class GeometryWarpingModule(nn.Module):
    """
    Computes dense 3D reprojection coordinate grids between calibrated intraoral views
    using metric depth maps and camera extrinsics/intrinsics, enabling exact epipolar
    feature sampling across views.
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def compute_warp_grid(
        cam_src: Dict[str, Any],
        cam_tgt: Dict[str, Any],
        depth_tgt: np.ndarray,
        mask_tgt: np.ndarray,
        feature_size: Tuple[int, int],
        orig_size: Tuple[int, int] = (640, 640)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Backprojects target view pixels to 3D world coordinates using target camera geometry
        and metric depth, then projects them onto the source view image plane to construct
        a normalized sampling grid for F.grid_sample.

        Returns:
            grid: (1, H_feat, W_feat, 2) in normalized [-1, 1] coords for grid_sample
            valid_mask: (1, 1, H_feat, W_feat) binary float tensor indicating valid overlap
        """
        H_feat, W_feat = feature_size
        W_orig, H_orig = orig_size

        # Downsample target depth and mask to feature resolution
        depth_pil = Image.fromarray(depth_tgt)
        depth_feat = np.array(depth_pil.resize((W_feat, H_feat), Image.BILINEAR), dtype=np.float32)
        mask_pil = Image.fromarray(mask_tgt)
        mask_feat = np.array(mask_pil.resize((W_feat, H_feat), Image.NEAREST), dtype=np.float32)

        # Scale target camera intrinsics to feature resolution
        sx = W_feat / float(W_orig)
        sy = H_feat / float(H_orig)
        f_tgt = cam_tgt["focal_length_pixels"] * sx
        cx_tgt, cy_tgt = cam_tgt["principal_point"][0] * sx, cam_tgt["principal_point"][1] * sy
        pos_tgt = np.array(cam_tgt["position"], dtype=np.float32)
        R_tgt = rotation_matrix_from_euler(*cam_tgt["rotation_euler"]).astype(np.float32)

        # Source camera parameters
        f_src = cam_src["focal_length_pixels"] * sx
        cx_src, cy_src = cam_src["principal_point"][0] * sx, cam_src["principal_point"][1] * sy
        pos_src = np.array(cam_src["position"], dtype=np.float32)
        R_src = rotation_matrix_from_euler(*cam_src["rotation_euler"]).astype(np.float32)

        # Build target pixel grid
        xs = np.arange(W_feat, dtype=np.float32)
        ys = np.arange(H_feat, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)

        # Camera coordinate back-projection in target frame
        # Conventions match Blender / blender_depth_fusion.py:
        # cam_x = (x - cx) * depth / focal, cam_y = (y - cy) * depth / focal, cam_z = -depth
        cam_x = (grid_x - cx_tgt) * depth_feat / max(f_tgt, 1e-6)
        cam_y = (grid_y - cy_tgt) * depth_feat / max(f_tgt, 1e-6)
        cam_z = -depth_feat
        pt_cam_tgt = np.stack([cam_x, cam_y, cam_z], axis=0).reshape(3, -1)  # (3, H*W)

        # Transform to 3D world coordinates: P_world = R_tgt @ P_cam_tgt + pos_tgt
        pt_world = R_tgt @ pt_cam_tgt + pos_tgt[:, None]

        # Transform 3D world coordinates into source camera frame: P_cam_src = R_src.T @ (P_world - pos_src)
        pt_cam_src = R_src.T @ (pt_world - pos_src[:, None])
        depth_src = -pt_cam_src[2, :].reshape(H_feat, W_feat)

        # Project onto source image plane
        front_facing = depth_src > 0.05
        safe_depth = np.where(front_facing, depth_src, 1e-4)
        u_src = pt_cam_src[0, :].reshape(H_feat, W_feat) * f_src / safe_depth + cx_src
        v_src = pt_cam_src[1, :].reshape(H_feat, W_feat) * f_src / safe_depth + cy_src

        # Check visibility and bounds in source view
        in_bounds = front_facing & (u_src >= 0.0) & (u_src < float(W_feat)) & (v_src >= 0.0) & (v_src < float(H_feat))
        valid = in_bounds & (mask_feat > 0.15) & (depth_feat > 0.05) & (depth_feat < 4.0)

        # Normalize coordinates to [-1, 1] for torch.nn.functional.grid_sample
        norm_u = (u_src / max(W_feat - 1, 1)) * 2.0 - 1.0
        norm_v = (v_src / max(H_feat - 1, 1)) * 2.0 - 1.0

        grid = np.stack([norm_u, norm_v], axis=-1)  # (H, W, 2)
        grid_tensor = torch.from_numpy(grid).unsqueeze(0).float()
        valid_tensor = torch.from_numpy(valid.astype(np.float32)).unsqueeze(0).unsqueeze(0).float()

        return grid_tensor, valid_tensor


class MultiScaleCrossViewBlock(nn.Module):
    """
    Performs geometry-aware cross-view feature fusion at a specific scale.
    Combines:
    1. Dense epipolar/depth warped feature alignment from all other views
    2. Relative pose geometric modulation
    3. Multi-view cross-attention with visibility gating
    4. Local feature refinement and residual update
    """
    def __init__(self, in_channels: int, geo_dim: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.proj_dim = min(in_channels, 256)

        # Feature projection and compression for efficient multi-view attention
        self.feat_proj = nn.Conv2d(in_channels, self.proj_dim, kernel_size=1, bias=False)
        self.geo_proj = nn.Linear(geo_dim, self.proj_dim)

        # Cross-view attention scoring
        self.query_conv = nn.Conv2d(self.proj_dim, self.proj_dim // 2, kernel_size=1)
        self.key_conv = nn.Conv2d(self.proj_dim, self.proj_dim // 2, kernel_size=1)
        self.val_conv = nn.Conv2d(self.proj_dim, self.proj_dim, kernel_size=1)

        # Adaptive fusion gating and output projection
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(in_channels + self.proj_dim, self.proj_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, self.proj_dim),
            nn.SiLU(),
            nn.Conv2d(self.proj_dim, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.out_refine = nn.Sequential(
            nn.Conv2d(self.proj_dim, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8 if in_channels % 8 == 0 else 4, in_channels),
            nn.SiLU()
        )

    def forward(
        self,
        features: torch.Tensor,
        warp_grids: Dict[Tuple[int, int], torch.Tensor],
        valid_masks: Dict[Tuple[int, int], torch.Tensor],
        geo_embs: torch.Tensor,
        affinity_biases: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            features: (V, C, H, W) ControlNet multiscale features for this scale
            warp_grids: dict mapping (src_idx, tgt_idx) -> (1, H, W, 2)
            valid_masks: dict mapping (src_idx, tgt_idx) -> (1, 1, H, W)
            geo_embs: (V, V, geo_dim) pairwise geometric relative pose embeddings
            affinity_biases: (V, V, 1) scalar geometric affinity biases
        Returns:
            fused_features: (V, C, H, W) enhanced with cross-view geometric information
            stats: summary dictionary of cross-view alignment statistics
        """
        V, C, H, W = features.shape
        device = features.device

        # Project features to attention space
        proj_feats = self.feat_proj(features)  # (V, proj_dim, H, W)

        fused_views = []
        overlap_stats = []

        for tgt_idx in range(V):
            target_feat = proj_feats[tgt_idx:tgt_idx + 1]  # (1, proj_dim, H, W)
            orig_target = features[tgt_idx:tgt_idx + 1]    # (1, C, H, W)

            warped_feats_list = []
            attn_weights_list = []

            for src_idx in range(V):
                if src_idx == tgt_idx:
                    # Self-reference view
                    warped = target_feat
                    val_mask = torch.ones((1, 1, H, W), device=device, dtype=features.dtype)
                else:
                    grid = warp_grids[(src_idx, tgt_idx)].to(device, dtype=features.dtype)
                    val_mask = valid_masks[(src_idx, tgt_idx)].to(device, dtype=features.dtype)
                    src_feat = proj_feats[src_idx:src_idx + 1]
                    warped = F.grid_sample(src_feat, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
                    warped = warped * val_mask
                    overlap_stats.append(float(val_mask.mean().item()))

                # Geometric pose modulation: inject relative camera transform
                geo_mod = self.geo_proj(geo_embs[src_idx, tgt_idx]).view(1, self.proj_dim, 1, 1)
                modulated_src = warped + geo_mod
                warped_feats_list.append(modulated_src)

                # Cross-view correlation attention score
                q = self.query_conv(target_feat)
                k = self.key_conv(modulated_src)
                score = (q * k).sum(dim=1, keepdim=True) / math.sqrt(self.proj_dim // 2)
                bias = affinity_biases[src_idx, tgt_idx].view(1, 1, 1, 1)
                score = score + bias

                # Mask out occluded/out-of-bounds regions (give large negative score)
                masked_score = torch.where(val_mask > 0.5, score, score - 1e4)
                attn_weights_list.append(masked_score)

            # Softmax over all source views
            all_scores = torch.cat(attn_weights_list, dim=1)  # (1, V, H, W)
            attn_weights = F.softmax(all_scores, dim=1)       # (1, V, H, W)

            # Weighted sum of aligned source features
            all_warped = torch.stack(warped_feats_list, dim=1)  # (1, V, proj_dim, H, W)
            cross_view_agg = (all_warped * attn_weights.unsqueeze(2)).sum(dim=1)  # (1, proj_dim, H, W)

            # Adaptive gating between target feature and aggregated cross-view feature
            gate_in = torch.cat([orig_target, cross_view_agg], dim=1)
            gate = self.fusion_gate(gate_in)
            refined_cross = self.out_refine(cross_view_agg)

            # Residual update: preserves original view characteristics while enriching with multi-view context
            fused_target = orig_target + gate * refined_cross
            fused_views.append(fused_target)

        fused_tensor = torch.cat(fused_views, dim=0)  # (V, C, H, W)
        mean_overlap = float(np.mean(overlap_stats)) if overlap_stats else 1.0

        return fused_tensor, {"mean_view_overlap_pct": mean_overlap * 100.0}


class CrossViewFeatureFusion(nn.Module):
    """
    Stage 4B: Genuine Multi-View Cross-View Feature Fusion for Sparse-View Dental Reconstruction.

    Fuses multi-scale feature representations from the ControlNet Feature Extractor across
    the 5 intraoral views using:
    - Calibrated camera extrinsics & intrinsics from cameras.json
    - Metric camera depth maps from depth/ and depth_vis/
    - Precise 3D back-projection & epipolar coordinate remapping
    - Pairwise relative camera geometry encoding
    - Multi-scale geometry-aware cross-view attention with visibility masking
    """
    FUSION_METHOD_NAME = "Geometry-Guided Epipolar Warping & Cross-View Attention Fusion v1.0"

    def __init__(self, device: Optional[str] = None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Relative camera pose encoder
        self.geo_dim = 64
        self.pose_encoder = RelativePoseEncoder(out_dim=self.geo_dim)

        # Multi-scale fusion blocks for each hierarchical diffusion scale
        self.fusion_blocks = nn.ModuleDict({
            "scale_1_8": MultiScaleCrossViewBlock(in_channels=320, geo_dim=self.geo_dim),
            "scale_1_16": MultiScaleCrossViewBlock(in_channels=640, geo_dim=self.geo_dim),
            "scale_1_32": MultiScaleCrossViewBlock(in_channels=1280, geo_dim=self.geo_dim),
            "scale_1_64": MultiScaleCrossViewBlock(in_channels=1280, geo_dim=self.geo_dim),
        })

        self.to(self.device)
        self.eval()

    def build_relative_pose_features(
        self,
        cameras: Dict[str, Any],
        views: List[str]
    ) -> torch.Tensor:
        """
        Builds pairwise relative camera transformation features for all (src, tgt) pairs among views.
        Returns tensor of shape (V, V, 14)
        """
        V = len(views)
        pairs_matrix = np.zeros((V, V, 14), dtype=np.float32)

        for i, vi in enumerate(views):
            ci = cameras[vi]
            pos_i = np.array(ci["position"], dtype=np.float32)
            R_i = rotation_matrix_from_euler(*ci["rotation_euler"]).astype(np.float32)

            for j, vj in enumerate(views):
                cj = cameras[vj]
                pos_j = np.array(cj["position"], dtype=np.float32)
                R_j = rotation_matrix_from_euler(*cj["rotation_euler"]).astype(np.float32)

                # Relative translation baseline
                rel_t = pos_i - pos_j
                dist = np.linalg.norm(rel_t)

                # Relative rotation matrix R_rel = R_i.T @ R_j
                R_rel = R_i.T @ R_j
                cos_theta = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
                theta = float(np.arccos(cos_theta))

                # 14-dim geometric vector: [t_x, t_y, t_z, dist, theta, R_rel elements]
                feat_vec = np.zeros(14, dtype=np.float32)
                feat_vec[0:3] = rel_t
                feat_vec[3] = dist
                feat_vec[4] = theta
                feat_vec[5:14] = R_rel.flatten()

                pairs_matrix[i, j] = feat_vec

        return torch.from_numpy(pairs_matrix).to(self.device)

    @torch.no_grad()
    def fuse_multiscale_features(
        self,
        controlnet_features: Dict[str, torch.Tensor],
        cameras: Dict[str, Any],
        depth_maps: Dict[str, np.ndarray],
        masks: Dict[str, np.ndarray],
        views: List[str]
    ) -> Dict[str, Any]:
        """
        Executes cross-view feature fusion across all 4 scales using geometric warping
        and attention.

        Args:
            controlnet_features: dict with scales 'scale_1_8', 'scale_1_16', 'scale_1_32', 'scale_1_64'
            cameras: camera parameters dict
            depth_maps: dict mapping view_name -> metric depth map (np.ndarray)
            masks: dict mapping view_name -> binary mask (np.ndarray)
            views: list of 5 view names
        """
        V = len(views)
        assert V == 5, f"Expected 5 intraoral views, got {V}"

        # 1. Compute relative pose embeddings
        rel_geo_feats = self.build_relative_pose_features(cameras, views)
        geo_embs, affinity_biases = self.pose_encoder(rel_geo_feats)

        fused_scales: Dict[str, torch.Tensor] = {}
        shapes_summary: Dict[str, List[int]] = {}
        scale_stats: Dict[str, Any] = {}

        # 2. Iterate through each hierarchical scale
        for scale_name, scale_tensor in controlnet_features.items():
            if scale_name not in self.fusion_blocks:
                continue

            scale_tensor = scale_tensor.to(self.device)
            _, C, H, W = scale_tensor.shape
            feat_size = (H, W)

            # Precompute warping grids and valid overlap masks for all pairs at this resolution
            warp_grids: Dict[Tuple[int, int], torch.Tensor] = {}
            valid_masks: Dict[Tuple[int, int], torch.Tensor] = {}

            for src_idx, src_view in enumerate(views):
                for tgt_idx, tgt_view in enumerate(views):
                    if src_idx == tgt_idx:
                        continue
                    cam_src = cameras[src_view]
                    cam_tgt = cameras[tgt_view]
                    depth_tgt = depth_maps[tgt_view]
                    mask_tgt = masks[tgt_view]

                    grid, val_mask = GeometryWarpingModule.compute_warp_grid(
                        cam_src=cam_src,
                        cam_tgt=cam_tgt,
                        depth_tgt=depth_tgt,
                        mask_tgt=mask_tgt,
                        feature_size=feat_size
                    )
                    warp_grids[(src_idx, tgt_idx)] = grid
                    valid_masks[(src_idx, tgt_idx)] = val_mask

            # Fuse features for this scale
            fusion_block = self.fusion_blocks[scale_name]
            fused_tensor, stats = fusion_block(
                features=scale_tensor,
                warp_grids=warp_grids,
                valid_masks=valid_masks,
                geo_embs=geo_embs,
                affinity_biases=affinity_biases
            )

            fused_scales[scale_name] = fused_tensor.detach().cpu()
            shapes_summary[scale_name] = list(fused_tensor.shape)
            scale_stats[scale_name] = stats

        return {
            "fused_multiscale_features": fused_scales,
            "fused_feature_shapes": shapes_summary,
            "fusion_stats": scale_stats,
            "fusion_method": self.FUSION_METHOD_NAME
        }

    @classmethod
    def fuse_sample(
        cls,
        sample_dir: Union[str, Path],
        device: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        High-level execution method that operates on a real processed sample directory:
        1. Loads existing ControlNet multiscale features from features.pt
        2. Loads cameras.json and view manifests
        3. Loads calibrated depth maps and SAM masks for all 5 views
        4. Runs genuine Cross-View Feature Fusion across all scales
        5. Updates features.pt with backward compatibility
        """
        sample_path = Path(sample_dir).resolve()
        features_file = sample_path / "features.pt"
        if not features_file.exists():
            raise FileNotFoundError(f"features.pt not found at: {features_file}")

        # Load existing ControlNet features
        existing_data = torch.load(features_file, map_location="cpu", weights_only=False)

        # Identify original ControlNet multiscale features
        if "controlnet_multiscale_features" in existing_data:
            orig_features = existing_data["controlnet_multiscale_features"]
        elif "multiscale_features" in existing_data:
            orig_features = existing_data["multiscale_features"]
        else:
            raise KeyError("No multiscale features found in features.pt")

        # Load camera manifest and parameters
        metadata_file = sample_path / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"metadata.json not found at: {metadata_file}")
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        cameras_file = sample_path / metadata.get("camera_file", "cameras/cameras.json")
        cameras = json.loads(cameras_file.read_text(encoding="utf-8"))

        views = existing_data.get("views", metadata.get("views", [
            "frontal", "left_buccal", "right_buccal", "maxillary_occlusal", "mandibular_occlusal"
        ]))

        # Load depth maps and masks for all views
        depth_maps: Dict[str, np.ndarray] = {}
        masks: Dict[str, np.ndarray] = {}

        for v in views:
            d_path = resolve_file(sample_path, "depth_vis", v, [".png"])
            if not d_path or not d_path.exists():
                d_path = resolve_file(sample_path, "depth", v, [".exr", ".png"])
            if not d_path or not d_path.exists():
                raise FileNotFoundError(f"Could not resolve depth map for view '{v}' in {sample_path}")

            w, h, depth_arr = load_depth_map(d_path)
            depth_maps[v] = depth_arr

            m_path = resolve_file(sample_path, "masks", v, [".png"])
            if m_path and m_path.exists():
                mask_arr = load_mask(m_path)
            else:
                mask_arr = np.ones((h, w), dtype=np.float32)
            masks[v] = mask_arr

        # Initialize fusion module and execute
        fusion_engine = cls(device=device)
        print(f"[Cross-View Feature Fusion] Fusing ControlNet features across {len(views)} views using {cls.FUSION_METHOD_NAME}...")
        fuse_result = fusion_engine.fuse_multiscale_features(
            controlnet_features=orig_features,
            cameras=cameras,
            depth_maps=depth_maps,
            masks=masks,
            views=views
        )

        fused_multiscale = fuse_result["fused_multiscale_features"]

        # Build updated features.pt payload with complete backward compatibility
        updated_payload = {
            # Metadata
            "model_type": existing_data.get("model_type", "ControlNet Feature Extractor"),
            "checkpoint": existing_data.get("checkpoint", "lllyasviel/control_v11p_sd15_normalbae"),
            "sample_id": existing_data.get("sample_id", sample_path.name),
            "views": views,
            "fusion_method": fuse_result["fusion_method"],
            # Original ControlNet multiscale features preserved explicitly
            "controlnet_multiscale_features": orig_features,
            # Newly generated fused cross-view multiscale features
            "cross_view_fused_features": fused_multiscale,
            # Backward-compatibility alias pointing to fused features
            "multiscale_features": fused_multiscale,
            # Explicit shapes tracking
            "feature_shapes": {
                "controlnet_shapes": {k: list(t.shape) for k, t in orig_features.items()},
                "fused_shapes": fuse_result["fused_feature_shapes"]
            },
            "fusion_stats": fuse_result["fusion_stats"]
        }

        # Save to features.pt
        torch.save(updated_payload, features_file)
        print(f"[Cross-View Feature Fusion] Saved updated feature artifact to {features_file.name} ({features_file.stat().st_size} bytes)")

        return updated_payload
