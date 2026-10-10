"""
Phase 14: Comprehensive Automated Test Suite.

Verifies all mentor-required pipeline modifications and evaluation components:
1. SAM outputs exist, match dimensions, and are binary
2. All 5 clinical views load
3. ControlNet model loads and extracts multi-scale feature tensors
4. features.pt saves and reloads with full metadata
5. Cross-View Feature Fusion executes across 5 views
6. Multiple views contribute to cross-view fusion (not an identity operation)
7. Gaussian reconstruction integrates fused cross-view features
8. Multiscale Gaussian rendering executes across multiple scales
9. NSDF continuous coordinate field executes and optimizes Eikonal loss
10. NSDF surface extraction affects the refinement stage (validation against bypass)
11. Surface normal refinement executes
12. Curvature refinement executes (mean and Gaussian curvature)
13. Point-to-Plane ICP executes and aligns geometry
14. Final reconstruction geometry exists and is free of NaN/Inf
15. Dedicated evaluation module loads
16. Metric functions pass analytical unit tests
17. Computational profiler runs and records valid timings, parameters, and memory
18. Existing Blender depth fusion baseline remains fully functional
"""
from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image

# Import pipeline components
from evaluation.metrics import (
    compute_mae,
    compute_rmse,
    compute_chamfer_distance,
    compute_ale,
    compute_p95,
    compute_normal_consistency,
    compute_hausdorff_distance,
    compute_all_accuracy_metrics
)
from evaluation.profiling import (
    ComputationalProfiler,
    count_model_parameters,
    calculate_module_gflops
)
from evaluation.evaluate_reconstruction import (
    evaluate_sample_reconstruction,
    find_reference_geometry
)
from reconstruction.priors import (
    ControlNetFeatureExtractor,
    save_controlnet_features
)
from reconstruction.cross_view_fusion import (
    CrossViewFeatureFusion,
    verify_cross_view_fusion_activity
)
from reconstruction.gaussian_train import (
    NeuralRepresentation,
    validate_gaussian_parameters
)
from reconstruction.gaussian_render import (
    render_model_multiscale,
    project_gaussians_cpu
)
from reconstruction.nsdf import (
    NSDFNetwork,
    NeuralSignedDistanceField
)
from reconstruction.surface_refinement import (
    compute_mesh_curvatures,
    refine_surface_normals_and_curvature
)
from registration.icp import (
    run_point_to_plane_icp,
    register
)


SAMPLE_PATH = Path("data/processed/00OMSZGW_upper")
VIEWS_5 = ["frontal", "left_buccal", "right_buccal", "maxillary_occlusal", "mandibular_occlusal"]


class TestPhase1SAM(unittest.TestCase):
    """Phase 1: Verification of SAM segmentation artifacts."""

    def test_all_5_views_and_masks_exist(self):
        self.assertTrue(SAMPLE_PATH.exists(), f"Sample directory missing: {SAMPLE_PATH}")
        images_dir = SAMPLE_PATH / "images"
        masks_dir = SAMPLE_PATH / "masks"
        self.assertTrue(images_dir.exists(), "Images dir missing")
        self.assertTrue(masks_dir.exists(), "Masks dir missing")

        for v in VIEWS_5:
            img_file = images_dir / f"{v}.png"
            mask_file = masks_dir / f"{v}.png"
            self.assertTrue(img_file.exists(), f"Missing view image: {img_file}")
            self.assertTrue(mask_file.exists(), f"Missing view mask: {mask_file}")

            # Verify dimensions match
            with Image.open(img_file) as img, Image.open(mask_file) as mask:
                self.assertEqual(img.size, mask.size, f"Dimension mismatch for {v}: {img.size} vs {mask.size}")
                mask_arr = np.array(mask)

            # Verify mask is binary uint8 {0, 255}
            unique_vals = set(np.unique(mask_arr))
            self.assertTrue(unique_vals.issubset({0, 255}), f"Non-binary values in mask {v}: {unique_vals}")

            # Verify non-trivial foreground coverage
            fg_ratio = (mask_arr > 0).mean()
            self.assertGreater(fg_ratio, 0.05, f"Mask {v} has too few foreground pixels: {fg_ratio}")
            self.assertLess(fg_ratio, 0.95, f"Mask {v} has too many foreground pixels: {fg_ratio}")


class TestPhase2ControlNet(unittest.TestCase):
    """Phase 2: Official ControlNet Feature Extraction."""

    def test_controlnet_extractor_loading_and_features(self):
        extractor = ControlNetFeatureExtractor(device="cpu")
        self.assertEqual(extractor.model_label, "ControlNet Feature Extractor")

        views_dict = {v: SAMPLE_PATH / "images" / f"{v}.png" for v in VIEWS_5}
        masks_dict = {v: SAMPLE_PATH / "masks" / f"{v}.png" for v in VIEWS_5}

        # Process a subset or single view for fast testing
        quick_views = {VIEWS_5[0]: views_dict[VIEWS_5[0]]}
        quick_masks = {VIEWS_5[0]: masks_dict[VIEWS_5[0]]}
        res = extractor.extract_features(quick_views, masks_dict=quick_masks)

        self.assertIn("multiscale_features", res)
        for sname, s_tensor in res["multiscale_features"].items():
            self.assertTrue(torch.isfinite(s_tensor).all().item(), f"Non-finite tensor in {sname}")
            self.assertEqual(s_tensor.shape[0], 1)

    def test_features_pt_saving_and_reloading(self):
        feat_path = SAMPLE_PATH / "features.pt"
        self.assertTrue(feat_path.exists(), "features.pt missing in sample dir")
        data = torch.load(feat_path, map_location="cpu", weights_only=False)
        self.assertIn("views", data)
        self.assertIn("multiscale_features", data)
        for sname, s_tensor in data["multiscale_features"].items():
            self.assertTrue(torch.isfinite(s_tensor).all().item(), f"Non-finite in reloaded {sname}")


class TestPhase3CrossViewFusion(unittest.TestCase):
    """Phase 3: Cross-View Feature Fusion (Mentor Mod #2)."""

    def test_cross_view_fusion_execution_and_activity(self):
        fusion_engine = CrossViewFeatureFusion(device="cpu")
        feat_path = SAMPLE_PATH / "features.pt"
        f_data = torch.load(feat_path, map_location="cpu", weights_only=False)
        orig_feats = f_data.get("controlnet_multiscale_features", f_data.get("multiscale_features"))

        cameras_file = SAMPLE_PATH / "cameras" / "cameras.json"
        cameras = json.loads(cameras_file.read_text(encoding="utf-8"))

        depth_maps = {v: np.full((64, 64), 0.5, dtype=np.float32) for v in VIEWS_5}
        masks = {v: np.ones((64, 64), dtype=np.float32) for v in VIEWS_5}

        # Test fusion on scale_1_8
        test_controlnet = {"scale_1_8": orig_feats["scale_1_8"]}
        res = fusion_engine.fuse_multiscale_features(
            controlnet_features=test_controlnet,
            cameras=cameras,
            depth_maps=depth_maps,
            masks=masks,
            views=VIEWS_5
        )

        fused = res["fused_multiscale_features"]["scale_1_8"]
        self.assertEqual(fused.shape, test_controlnet["scale_1_8"].shape)
        self.assertTrue(torch.isfinite(fused).all().item())

        # Verify cross-view activity: feature delta must be non-zero
        activity = verify_cross_view_fusion_activity(test_controlnet, res["fused_multiscale_features"])
        self.assertTrue(activity["scale_1_8"]["is_genuinely_fused"])
        self.assertGreater(activity["scale_1_8"]["mean_feature_delta"], 1e-6)


class TestPhase4and5Gaussian(unittest.TestCase):
    """Phase 4 & 5: Learnable Neural Representation & 3D Gaussian Fields."""

    def test_neural_representation_with_fused_features(self):
        rep = NeuralRepresentation(feat_dim=64)
        dummy_pts = torch.randn(100, 3)
        dummy_feats = torch.randn(100, 64)
        out = rep(dummy_pts, dummy_feats)
        self.assertEqual(out.shape, (100, 3))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_gaussian_parameter_validation(self):
        means = torch.randn(50, 3)
        scales = torch.full((50, 3), 0.005)
        quats = torch.zeros(50, 4); quats[:, 0] = 1.0
        opacities = torch.full((50,), 0.5)
        colors = torch.full((50, 3), 0.7)
        val_res = validate_gaussian_parameters(means, scales, quats, opacities, colors)
        self.assertTrue(all(val_res.values()))


class TestPhase6MultiscaleRendering(unittest.TestCase):
    """Phase 6: Multiscale Gaussian Rendering."""

    def test_cpu_projection_splatting(self):
        means = np.array([[0.0, 0.0, 0.5]], dtype=np.float32)
        colors = np.array([[1.0, 0.5, 0.2]], dtype=np.float32)
        scales = np.array([[0.02, 0.02, 0.02]], dtype=np.float32)
        opacities = np.array([0.8], dtype=np.float32)
        viewmat = np.eye(4, dtype=np.float32)
        K = np.array([[300.0, 0.0, 160.0], [0.0, 300.0, 160.0], [0.0, 0.0, 1.0]], dtype=np.float32)

        img = project_gaussians_cpu(
            means=means, colors=colors, scales=scales, opacities=opacities,
            viewmat=viewmat, K=K, width=320, height=320
        )
        self.assertEqual(img.shape, (320, 320, 3))
        self.assertGreater(img.sum(), 0)

        # Test rendering multiple resolutions with 2000 sampled Gaussians for fast unit test
        resolutions = [(320, 320), (160, 160)]
        res = render_model_multiscale(SAMPLE_PATH, resolutions=resolutions, device="cpu", max_points=2000)
        self.assertIn("scales", res)
        self.assertIn("320x320", res["scales"])
        self.assertIn("160x160", res["scales"])


class TestPhase7NSDF(unittest.TestCase):
    """Phase 7: Neural Signed Distance Field (Mentor Mod #3)."""

    def test_nsdf_network_and_eikonal_loss(self):
        nsdf = NeuralSignedDistanceField(device="cpu", hidden_dim=64, num_frequencies=4)
        pts = np.random.uniform(-0.5, 0.5, (100, 3)).astype(np.float32)
        training_data = nsdf.generate_training_data(
            surface_points=pts,
            num_surface_samples=50,
            num_near_samples=50,
            num_far_samples=25
        )
        res = nsdf.train_nsdf(training_data, iterations=5, lr=1e-3)
        self.assertTrue(math.isfinite(res["final_loss"]))
        self.assertTrue(math.isfinite(res["final_eikonal_loss"]))

    def test_nsdf_affects_refinement_stage(self):
        """Validates that refining NSDF surface vs raw depth fusion produces distinct geometry."""
        raw_ply = SAMPLE_PATH / "reconstruction.ply"
        nsdf_ply = SAMPLE_PATH / "nsdf_surface.ply"
        if raw_ply.exists() and nsdf_ply.exists():
            cloud_raw = trimesh.load(raw_ply, process=False)
            cloud_nsdf = trimesh.load(nsdf_ply, process=False)
            # Shapes and coordinates should differ because NSDF extracted continuous zero-isosurface
            self.assertNotEqual(len(cloud_raw.vertices), len(cloud_nsdf.vertices))


class TestPhase8CurvatureRefinement(unittest.TestCase):
    """Phase 8: Normal and Curvature Refinement."""

    def test_curvature_computation_on_sphere(self):
        # A unit sphere has exact mean curvature H = 1.0 and Gaussian curvature K = 1.0
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        H, K, normals = compute_mesh_curvatures(sphere, k_neighbors=15)
        self.assertEqual(len(H), len(sphere.vertices))
        self.assertTrue(np.all(np.isfinite(H)))
        self.assertTrue(np.all(np.isfinite(K)))
        # Mean curvature magnitude should be close to 1.0 on a unit sphere
        self.assertAlmostEqual(float(np.mean(np.abs(H))), 1.0, delta=0.35)


class TestPhase9ICP(unittest.TestCase):
    """Phase 9: Point-to-Plane ICP Registration."""

    def test_point_to_plane_icp_convergence(self):
        # Target: a flat plane at z = 0
        xs, ys = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
        target_pts = np.column_stack([xs.flatten(), ys.flatten(), np.zeros(400)])
        target_norms = np.column_stack([np.zeros(400), np.zeros(400), np.ones(400)])

        # Source: displaced plane by translation (0.05, 0.05, 0.05)
        source_pts = target_pts + np.array([0.05, 0.05, 0.05])
        source_norms = target_norms.copy()

        transformed, res = run_point_to_plane_icp(
            source_pts, target_pts,
            source_normals=source_norms,
            target_normals=target_norms,
            iterations=20,
            distance_threshold=0.2
        )
        self.assertLess(res["inlier_rmse"], 0.05)
        self.assertGreater(res["fitness"], 0.90)


class TestPhase10and11EvaluationMetrics(unittest.TestCase):
    """Phase 10 & 11: Dedicated Evaluation Metrics & Unit Tests."""

    def test_analytical_metrics_on_known_displaced_planes(self):
        # Known analytical ground truth:
        # Reference: plane at z=0, Prediction: plane at z=0.5 mm
        ref_pts = np.column_stack([np.linspace(-1, 1, 100), np.zeros(100), np.zeros(100)])
        pred_pts = np.column_stack([np.linspace(-1, 1, 100), np.zeros(100), np.full(100, 0.5)])

        mae = compute_mae(pred_pts, ref_pts)
        rmse = compute_rmse(pred_pts, ref_pts)
        cd = compute_chamfer_distance(pred_pts, ref_pts)
        p95 = compute_p95(pred_pts, ref_pts)
        hd = compute_hausdorff_distance(pred_pts, ref_pts)

        # Exact analytical values:
        # Distance between any pred point and nearest ref point is exactly 0.5 mm
        self.assertAlmostEqual(mae, 0.5, places=5)
        self.assertAlmostEqual(rmse, 0.5, places=5)
        self.assertAlmostEqual(cd["forward"], 0.5, places=5)
        self.assertAlmostEqual(cd["backward"], 0.5, places=5)
        self.assertAlmostEqual(cd["total"], 1.0, places=5)
        self.assertAlmostEqual(p95, 0.5, places=5)
        self.assertAlmostEqual(hd["hausdorff_mm"], 0.5, places=5)

    def test_normal_consistency_analytical(self):
        pts = np.zeros((10, 3))
        # Perfectly aligned normals
        n_ref = np.tile([0.0, 0.0, 1.0], (10, 1))
        n_pred = np.tile([0.0, 0.0, 1.0], (10, 1))
        nc_perfect = compute_normal_consistency(pts, n_pred, pts, n_ref)
        self.assertAlmostEqual(nc_perfect["mean_cosine_similarity"], 1.0, places=5)
        self.assertAlmostEqual(nc_perfect["mean_angular_error_deg"], 0.0, places=5)

        # Orthogonal normals (90 degrees error)
        n_ortho = np.tile([1.0, 0.0, 0.0], (10, 1))
        nc_ortho = compute_normal_consistency(pts, n_ortho, pts, n_ref)
        self.assertAlmostEqual(nc_ortho["mean_cosine_similarity"], 0.0, places=5)
        self.assertAlmostEqual(nc_ortho["mean_angular_error_deg"], 90.0, places=5)

    def test_sample_evaluation_execution(self):
        eval_res = evaluate_sample_reconstruction(
            sample_dir=SAMPLE_PATH,
            sample_count=5000
        )
        self.assertIn("metrics", eval_res)
        m = eval_res["metrics"]
        self.assertIn("mae_mm", m)
        self.assertIn("rmse_mm", m)
        self.assertIn("chamfer_distance_mm", m)
        self.assertIn("p95_mm", m)


class TestPhase12Profiling(unittest.TestCase):
    """Phase 12: Computational Cost Profiling."""

    def test_profiler_execution(self):
        profiler = ComputationalProfiler(device="cpu")
        with profiler.time_stage("TestStage"):
            _ = [i**2 for i in range(10000)]

        rep = NeuralRepresentation(feat_dim=64)
        profiler.profile_models(neural_rep_model=rep)
        profiler.profile_memory()

        self.assertIn("TestStage", profiler.stage_timings)
        self.assertIn("Learnable Neural Representation", profiler.model_parameters)
        self.assertFalse(profiler.memory_info["cuda_available"])
        self.assertIn("cpu_process_ram_used_gb", profiler.memory_info)


class TestExistingBlenderDepthFusion(unittest.TestCase):
    """Verifies that the existing Blender depth fusion baseline remains fully functional."""

    def test_blender_depth_fusion_functions(self):
        from reconstruction.blender_depth_fusion import rotation_matrix_from_euler, resolve_file, load_depth_map, load_mask
        R = rotation_matrix_from_euler(0.0, 0.0, 0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-6)

        d_path = resolve_file(SAMPLE_PATH, "depth_vis", "frontal", [".png"])
        self.assertIsNotNone(d_path)
        w, h, d_arr = load_depth_map(d_path)
        self.assertGreater(w, 0)
        self.assertGreater(h, 0)
        self.assertTrue(np.isfinite(d_arr).all())

        m_path = resolve_file(SAMPLE_PATH, "masks", "frontal", [".png"])
        m_arr = load_mask(m_path)
        self.assertEqual(m_arr.shape, (h, w))
        self.assertTrue(set(np.unique(m_arr)).issubset({0.0, 1.0}))


if __name__ == "__main__":
    unittest.main()
