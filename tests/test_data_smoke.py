"""Smoke tests for the data pipeline (dataset, transforms, pool)."""

import os
import tempfile
import json

import pytest
import torch
from PIL import Image
from torchvision import transforms

from src.config import Config
from src.data.dataset import EuroSATDataset, discover_images, stratified_split
from src.data.transforms import (
    compute_norm_stats,
    save_norm_stats,
    load_norm_stats,
    get_train_transform,
    get_eval_transform,
)
from src.data.pool import build_unlabeled_pool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dummy_dataset(tmp_path, class_names, n_per_class=5):
    """Create a temp directory with dummy 64x64 .jpg images per class."""
    root = os.path.join(tmp_path, "data")
    for cls in class_names:
        cls_dir = os.path.join(root, cls)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(n_per_class):
            img = Image.new("RGB", (64, 64), color=(i * 30 % 256, 100, 150))
            img.save(os.path.join(cls_dir, f"{cls}_{i+1}.jpg"))
    return root


# ---------------------------------------------------------------------------
# 3.1 — dataset.py
# ---------------------------------------------------------------------------

class TestDiscoverImages:
    def test_basic_discovery(self, tmp_path):
        classes = ["A", "B", "C"]
        root = _make_dummy_dataset(tmp_path, classes, n_per_class=4)
        paths, labels = discover_images(root, classes)
        assert len(paths) == 12
        assert len(labels) == 12
        assert set(labels) == {0, 1, 2}

    def test_missing_directory_raises(self, tmp_path):
        root = _make_dummy_dataset(tmp_path, ["A"], n_per_class=2)
        with pytest.raises(FileNotFoundError):
            discover_images(root, ["A", "Missing"])

    def test_empty_directory_raises(self, tmp_path):
        root = _make_dummy_dataset(tmp_path, ["A"], n_per_class=2)
        empty_dir = os.path.join(root, "Empty")
        os.makedirs(empty_dir)
        with pytest.raises(ValueError, match="No valid .jpg images"):
            discover_images(root, ["A", "Empty"])


class TestStratifiedSplit:
    def test_no_overlap_and_sizes(self, tmp_path):
        classes = ["X", "Y"]
        root = _make_dummy_dataset(tmp_path, classes, n_per_class=20)
        paths, labels = discover_images(root, classes)

        tp, vp, tep, tl, vl, tel = stratified_split(
            paths, labels, 0.7, 0.15, 0.15, seed=42
        )
        # No overlap
        assert set(tp).isdisjoint(set(vp))
        assert set(tp).isdisjoint(set(tep))
        assert set(vp).isdisjoint(set(tep))
        # Total preserved
        assert len(tp) + len(vp) + len(tep) == len(paths)

    def test_determinism(self, tmp_path):
        classes = ["X", "Y"]
        root = _make_dummy_dataset(tmp_path, classes, n_per_class=20)
        paths, labels = discover_images(root, classes)

        r1 = stratified_split(paths, labels, 0.7, 0.15, 0.15, seed=99)
        r2 = stratified_split(paths, labels, 0.7, 0.15, 0.15, seed=99)
        assert r1 == r2


class TestEuroSATDataset:
    def test_getitem_returns_tensor_and_label(self, tmp_path):
        classes = ["A"]
        root = _make_dummy_dataset(tmp_path, classes, n_per_class=3)
        paths, labels = discover_images(root, classes)
        ds = EuroSATDataset(paths, labels, transform=transforms.ToTensor())
        tensor, label = ds[0]
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 64, 64)
        assert label == 0

    def test_len(self, tmp_path):
        classes = ["A", "B"]
        root = _make_dummy_dataset(tmp_path, classes, n_per_class=5)
        paths, labels = discover_images(root, classes)
        ds = EuroSATDataset(paths, labels)
        assert len(ds) == 10


# ---------------------------------------------------------------------------
# 3.3 — transforms.py
# ---------------------------------------------------------------------------

class TestNormStats:
    def test_compute_and_roundtrip(self, tmp_path):
        classes = ["A"]
        root = _make_dummy_dataset(tmp_path, classes, n_per_class=3)
        paths, labels = discover_images(root, classes)
        ds = EuroSATDataset(paths, labels, transform=transforms.ToTensor())

        mean, std = compute_norm_stats(ds)
        assert len(mean) == 3
        assert len(std) == 3
        assert all(0 <= m <= 1 for m in mean)
        assert all(s >= 0 for s in std)

        # Save / load round-trip
        stats_path = os.path.join(tmp_path, "stats.json")
        save_norm_stats(mean, std, stats_path)
        mean2, std2 = load_norm_stats(stats_path)
        assert mean == mean2
        assert std == std2


class TestTransforms:
    def _make_config(self, aug_overrides=None):
        aug = {
            "horizontal_flip": True,
            "vertical_flip": True,
            "random_rotation": 15,
            "color_jitter": {
                "brightness": 0.2,
                "contrast": 0.2,
                "saturation": 0.2,
                "hue": 0.1,
            },
        }
        if aug_overrides:
            aug.update(aug_overrides)
        return Config(
            dataset_root="EuroSAT/2750",
            output_dir="outputs",
            weights_path="outputs/best_model.pt",
            norm_stats_path="outputs/norm_stats.json",
            known_classes=["A"],
            ghost_classes=["B"],
            augmentation=aug,
        )

    def test_train_transform_includes_augmentations(self):
        cfg = self._make_config()
        t = get_train_transform(cfg, [0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
        type_names = [type(op).__name__ for op in t.transforms]
        assert "ToTensor" in type_names
        assert "Normalize" in type_names
        assert "RandomHorizontalFlip" in type_names
        assert "RandomVerticalFlip" in type_names
        assert "RandomRotation" in type_names
        assert "ColorJitter" in type_names

    def test_eval_transform_no_augmentation(self):
        t = get_eval_transform([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
        type_names = [type(op).__name__ for op in t.transforms]
        assert type_names == ["ToTensor", "Normalize"]

    def test_disabled_augmentations_excluded(self):
        cfg = self._make_config(
            aug_overrides={
                "horizontal_flip": False,
                "vertical_flip": False,
                "random_rotation": 0,
                "color_jitter": None,
            }
        )
        t = get_train_transform(cfg, [0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
        type_names = [type(op).__name__ for op in t.transforms]
        assert "RandomHorizontalFlip" not in type_names
        assert "RandomVerticalFlip" not in type_names
        assert "RandomRotation" not in type_names
        assert "ColorJitter" not in type_names


# ---------------------------------------------------------------------------
# 3.5 — pool.py
# ---------------------------------------------------------------------------

class TestBuildUnlabeledPool:
    def test_pool_size_and_labels(self, tmp_path):
        known_classes = ["K1", "K2"]
        ghost_classes = ["G1", "G2"]
        root = _make_dummy_dataset(
            tmp_path, known_classes + ghost_classes, n_per_class=10
        )
        known_paths, known_labels = discover_images(root, known_classes)
        # Use 5 known samples
        pool_paths, gt_labels = build_unlabeled_pool(
            known_paths, known_labels, root, ghost_classes,
            n_known_samples=5, seed=42,
        )
        assert len(pool_paths) == 5 + 20  # 5 known + 20 ghost
        assert len(gt_labels) == len(pool_paths)
        assert set(gt_labels) == {0, 1}
        assert gt_labels.count(0) == 5
        assert gt_labels.count(1) == 20

    def test_determinism(self, tmp_path):
        known_classes = ["K1"]
        ghost_classes = ["G1"]
        root = _make_dummy_dataset(
            tmp_path, known_classes + ghost_classes, n_per_class=10
        )
        known_paths, known_labels = discover_images(root, known_classes)

        r1 = build_unlabeled_pool(known_paths, known_labels, root, ghost_classes, 5, seed=7)
        r2 = build_unlabeled_pool(known_paths, known_labels, root, ghost_classes, 5, seed=7)
        assert r1 == r2
