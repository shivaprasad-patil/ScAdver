import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.preprocessing import LabelEncoder

from scadver import core
from scadver.core import (
    _build_reference_neighbor_targets,
    _encode_labels_with_reference,
    _get_domain_mixing_labels,
    adversarial_batch_correction,
)
from scadver.model import AdversarialBatchCorrector


def _make_adata(X, obs):
    return AnnData(X=np.asarray(X, dtype=np.float32), obs=pd.DataFrame(obs))


def test_encode_labels_with_reference_preserves_reference_indices_for_subset_labels():
    ref_classes = [
        "acinar",
        "activated_stellate",
        "alpha",
        "beta",
        "delta",
        "ductal",
        "endothelial",
        "epsilon",
        "gamma",
        "macrophage",
        "mast",
        "quiescent_stellate",
        "schwann",
        "t_cell",
    ]
    query_classes = [label for label in ref_classes if label != "ductal"]
    label_to_idx = {label: idx for idx, label in enumerate(ref_classes)}

    encoded, stats = _encode_labels_with_reference(query_classes, label_to_idx)

    assert stats["matched_class_ratio"] == 1.0
    assert stats["matched_cell_ratio"] == 1.0
    assert [int(idx) for idx in encoded] == [label_to_idx[label] for label in query_classes]


def test_build_reference_neighbor_targets_can_balance_across_batches():
    ref_embeddings = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [10.0, 0.0],
            [10.1, 0.0],
        ],
        dtype=np.float32,
    )
    ref_labels = np.asarray(["A", "A", "A", "A"])
    ref_batches = np.asarray(["v2", "v2", "v3", "v3"])
    query_embeddings = np.asarray([[0.05, 0.0]], dtype=np.float32)
    query_labels = np.asarray(["A"])

    unbalanced, matched_unbalanced = _build_reference_neighbor_targets(
        ref_embeddings=ref_embeddings,
        ref_labels=ref_labels,
        query_embeddings=query_embeddings,
        query_labels=query_labels,
        k=2,
    )
    balanced, matched_balanced = _build_reference_neighbor_targets(
        ref_embeddings=ref_embeddings,
        ref_labels=ref_labels,
        query_embeddings=query_embeddings,
        query_labels=query_labels,
        k=2,
        ref_batch_labels=ref_batches,
    )

    assert matched_unbalanced[0]
    assert matched_balanced[0]
    assert np.allclose(unbalanced[0], [0.05, 0.0], atol=1e-5)
    assert np.allclose(balanced[0], [5.05, 0.0], atol=1e-5)


def test_reference_only_training_disables_single_class_source_adversary():
    adata = _make_adata(
        X=[
            [0.0, 0.2, 0.1, 0.3],
            [0.1, 0.1, 0.0, 0.4],
            [1.0, 0.9, 1.1, 0.8],
            [0.9, 1.0, 0.8, 1.1],
            [0.2, 0.0, 0.1, 0.2],
            [0.3, 0.1, 0.2, 0.1],
            [1.1, 1.0, 0.9, 0.8],
            [0.8, 0.9, 1.0, 1.1],
        ],
        obs={
            "celltype": ["A", "A", "B", "B", "A", "A", "B", "B"],
            "batch": ["b0", "b1", "b0", "b1", "b0", "b1", "b0", "b1"],
            "Source": ["Reference", "Reference", "Reference", "Reference", "Query", "Query", "Query", "Query"],
        },
    )

    _, model, metrics = adversarial_batch_correction(
        adata=adata,
        bio_label="celltype",
        batch_label="batch",
        reference_data="Reference",
        query_data="Query",
        latent_dim=4,
        epochs=0,
        device="cpu",
        seed=0,
    )

    assert model.source_discriminator is None
    assert 0.0 <= metrics["source_integration"] <= 1.0


def test_detect_domain_shift_routes_neighbor_for_high_shared_coverage():
    rng = np.random.default_rng(0)
    ref_labels = ["A"] * 6 + ["B"] * 6 + ["C"] * 6
    query_labels = ["A"] * 4 + ["B"] * 4 + ["C"] * 4

    adata_ref = _make_adata(
        X=rng.normal(size=(len(ref_labels), 5)),
        obs={
            "celltype": ref_labels,
            "batch": ["ref_a", "ref_b"] * (len(ref_labels) // 2),
        },
    )
    adata_query = _make_adata(
        X=rng.normal(size=(len(query_labels), 5)),
        obs={
            "celltype": query_labels,
            "batch": ["query_a", "query_b"] * (len(query_labels) // 2),
        },
    )

    model = AdversarialBatchCorrector(
        input_dim=5,
        latent_dim=3,
        n_bio_labels=3,
        n_batches=2,
        n_sources=None,
    )
    model.bio_encoder = LabelEncoder().fit(np.asarray(["A", "B", "C"]))

    result = core.detect_domain_shift(
        model=model,
        adata_query=adata_query,
        adata_reference=adata_ref,
        bio_label="celltype",
        device="cpu",
        seed=0,
    )

    assert result["shared_cell_ratio"] == 1.0
    assert result["shared_class_ratio"] == 1.0
    assert result["shared_class_count"] == 3
    assert result["min_shared_ref_cells"] == 6
    assert result["min_shared_query_cells"] == 4
    assert result["recommended_mode"] == "neighbor"


def test_detect_domain_shift_avoids_neighbor_when_shared_class_support_is_too_low():
    rng = np.random.default_rng(2)
    ref_labels = ["A"] * 6 + ["B"] * 6 + ["C"] * 6
    query_labels = ["A"] * 4 + ["B"] * 4 + ["C"] * 1

    adata_ref = _make_adata(
        X=rng.normal(size=(len(ref_labels), 5)),
        obs={
            "celltype": ref_labels,
            "batch": ["ref_a", "ref_b"] * (len(ref_labels) // 2),
        },
    )
    adata_query = _make_adata(
        X=rng.normal(size=(len(query_labels), 5)),
        obs={
            "celltype": query_labels,
            "batch": ["query_a", "query_b"] * (len(query_labels) // 2) + ["query_a"] * (len(query_labels) % 2),
        },
    )

    model = AdversarialBatchCorrector(
        input_dim=5,
        latent_dim=3,
        n_bio_labels=3,
        n_batches=2,
        n_sources=None,
    )
    model.bio_encoder = LabelEncoder().fit(np.asarray(["A", "B", "C"]))

    result = core.detect_domain_shift(
        model=model,
        adata_query=adata_query,
        adata_reference=adata_ref,
        bio_label="celltype",
        device="cpu",
        seed=0,
    )

    assert result["shared_cell_ratio"] == 1.0
    assert result["shared_class_ratio"] == 1.0
    assert result["min_shared_ref_cells"] == 6
    assert result["min_shared_query_cells"] == 1
    assert result["recommended_mode"] == "auto"


def test_transform_query_adaptive_still_runs_probe_with_large_reference_class_counts(monkeypatch, capsys):
    rng = np.random.default_rng(1)
    n_ref_classes = 120
    ref_labels = [f"class_{idx:03d}" for idx in range(n_ref_classes) for _ in range(2)]
    query_labels = [f"class_{idx:03d}" for idx in range(5) for _ in range(3)]

    adata_ref = _make_adata(
        X=rng.normal(size=(len(ref_labels), 5)),
        obs={
            "celltype": ref_labels,
            "batch": ["ref_a", "ref_b"] * n_ref_classes,
        },
    )
    adata_query = _make_adata(
        X=rng.normal(size=(len(query_labels), 5)),
        obs={
            "celltype": query_labels,
            "batch": ["query_a", "query_b"] * (len(query_labels) // 2) + ["query_a"] * (len(query_labels) % 2),
        },
    )

    model = AdversarialBatchCorrector(
        input_dim=5,
        latent_dim=3,
        n_bio_labels=n_ref_classes,
        n_batches=2,
        n_sources=None,
    )
    model.bio_encoder = LabelEncoder().fit(np.asarray(ref_labels))

    called = {"probe": False}

    def fake_detect_domain_shift(*args, **kwargs):
        called["probe"] = True
        return {
            "needs_adapter": False,
            "adapter_dim": 0,
            "residual_magnitude": 0.05,
            "residual_std": 0.01,
            "confidence": "medium",
            "recommended_mode": "auto",
            "raw_domain_metric": "global mean offset",
            "shared_cell_ratio": None,
            "shared_class_ratio": None,
            "shared_class_count": None,
        }

    monkeypatch.setattr(core, "detect_domain_shift", fake_detect_domain_shift)

    adata_out = core.transform_query_adaptive(
        model=model,
        adata_query=adata_query,
        adata_reference=adata_ref,
        bio_label="celltype",
        device="cpu",
        seed=0,
    )

    captured = capsys.readouterr().out
    assert called["probe"] is True
    assert "analytical" not in captured.lower()
    assert adata_out.obsm["X_ScAdver"].shape == (len(query_labels), 3)


def test_transform_query_adaptive_ignores_unmatched_cells_in_neural_bio_supervision(monkeypatch, capsys):
    rng = np.random.default_rng(1)
    ref_labels = ["acinar"] * 6 + ["beta"] * 6 + ["ductal"] * 6 + ["gamma"] * 6
    query_labels = ["acinar"] * 4 + ["beta"] * 4 + ["gamma"] * 4 + ["orphan"] * 4

    adata_ref = _make_adata(
        X=rng.normal(size=(len(ref_labels), 6)),
        obs={
            "celltype": ref_labels,
            "batch": ["r0", "r1"] * (len(ref_labels) // 2),
        },
    )
    adata_query = _make_adata(
        X=rng.normal(size=(len(query_labels), 6)),
        obs={
            "celltype": query_labels,
            "batch": ["q0", "q1"] * (len(query_labels) // 2),
        },
    )

    model = AdversarialBatchCorrector(
        input_dim=6,
        latent_dim=4,
        n_bio_labels=4,
        n_batches=2,
        n_sources=None,
    )
    model.bio_encoder = LabelEncoder().fit(np.asarray(["acinar", "beta", "ductal", "gamma"]))

    monkeypatch.setattr(
        core,
        "detect_domain_shift",
        lambda *args, **kwargs: {
            "needs_adapter": True,
            "adapter_dim": 128,
            "residual_magnitude": 0.2,
            "residual_std": 0.0,
            "confidence": "high",
        },
    )

    adata_out = core.transform_query_adaptive(
        model=model,
        adata_query=adata_query,
        adata_reference=adata_ref,
        bio_label="celltype",
        adaptation_epochs=1,
        max_epochs=1,
        warmup_epochs=0,
        patience=1,
        learning_rate=0.0005,
        device="cpu",
        seed=0,
    )

    captured = capsys.readouterr().out
    assert "ignoring 4 unmatched cells" in captured
    assert adata_out.obsm["X_ScAdver"].shape == (len(query_labels), 4)


def test_transform_query_adaptive_keeps_bio_supervision_with_partial_nan_labels(monkeypatch, capsys):
    rng = np.random.default_rng(7)
    ref_labels = ["acinar"] * 6 + ["beta"] * 6 + ["gamma"] * 6
    query_labels = ["acinar"] * 2 + ["beta"] * 2 + [np.nan] * 16

    adata_ref = _make_adata(
        X=rng.normal(size=(len(ref_labels), 6)),
        obs={
            "celltype": ref_labels,
            "batch": ["r0", "r1"] * (len(ref_labels) // 2),
        },
    )
    adata_query = _make_adata(
        X=rng.normal(size=(len(query_labels), 6)),
        obs={
            "celltype": query_labels,
            "batch": ["q0", "q1"] * (len(query_labels) // 2),
        },
    )

    model = AdversarialBatchCorrector(
        input_dim=6,
        latent_dim=4,
        n_bio_labels=3,
        n_batches=2,
        n_sources=None,
    )
    model.bio_encoder = LabelEncoder().fit(np.asarray(["acinar", "beta", "gamma"]))

    monkeypatch.setattr(
        core,
        "detect_domain_shift",
        lambda *args, **kwargs: {
            "needs_adapter": True,
            "adapter_dim": 128,
            "residual_magnitude": 0.2,
            "residual_std": 0.0,
            "confidence": "high",
        },
    )

    adata_out = core.transform_query_adaptive(
        model=model,
        adata_query=adata_query,
        adata_reference=adata_ref,
        bio_label="celltype",
        adaptation_epochs=1,
        max_epochs=1,
        warmup_epochs=0,
        patience=1,
        learning_rate=0.0005,
        device="cpu",
        seed=0,
    )

    captured = capsys.readouterr().out
    assert "labeled cells matched: 100.0%" in captured
    assert "Bio supervision ENABLED" in captured
    assert "Bio supervision DISABLED" not in captured
    assert "ignoring 16 unmatched cells" in captured
    assert adata_out.obsm["X_ScAdver"].shape == (len(query_labels), 4)


def test_get_domain_mixing_labels_uses_reference_query_roles():
    adata_ref = _make_adata(
        X=np.zeros((4, 2)),
        obs={
            "Source": ["source_ref", "source_ref", "source_ref", "source_ref"],
            "batch": ["b0", "b1", "b0", "b1"],
        },
    )
    adata_query = _make_adata(
        X=np.zeros((4, 2)),
        obs={
            "Source": ["source_query", "source_query", "source_query", "source_query"],
            "batch": ["b0", "b1", "b0", "b1"],
        },
    )

    col, ref_labels, query_labels, use_ref_ceiling = _get_domain_mixing_labels(adata_ref, adata_query)

    assert col == "role"
    assert np.all(ref_labels == "reference")
    assert np.all(query_labels == "query")
    assert use_ref_ceiling is False
