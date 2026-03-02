#!/usr/bin/env python
"""
Local benchmark for the saved ScAdver simulation example.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from anndata import read_h5ad
from sklearn.neighbors import NearestNeighbors

from scadver import load_model, transform_query_adaptive
from scadver.core import _resolve_device


def _to_dense_float32(matrix):
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def _encode_reference(model, adata, device, batch_size=4096):
    model = model.to(device)
    model.eval()
    x_tensor = torch.as_tensor(_to_dense_float32(adata.X), dtype=torch.float32, device=device)
    embeddings = []

    with torch.no_grad():
        for start in range(0, x_tensor.shape[0], batch_size):
            embeddings.append(model.encoder(x_tensor[start:start + batch_size]).cpu().numpy())

    return np.vstack(embeddings)


def _knn_indices(embeddings, n_neighbors):
    n_neighbors = min(n_neighbors + 1, embeddings.shape[0])
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(embeddings)
    return nn.kneighbors(embeddings, return_distance=False)[:, 1:]


def _neighbor_mixing(indices, labels):
    labels = np.asarray(labels)
    return float(np.mean([np.mean(labels[idx] != labels[i]) for i, idx in enumerate(indices)]))


def _label_transfer_accuracy(ref_emb, query_emb, ref_labels, query_labels, n_neighbors):
    nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(ref_emb)), metric="euclidean").fit(ref_emb)
    neighbor_idx = nn.kneighbors(query_emb, return_distance=False)

    query_labels = np.asarray(query_labels)
    ref_labels = np.asarray(ref_labels)
    matches = np.array(
        [np.any(ref_labels[row] == query_labels[i]) for i, row in enumerate(neighbor_idx)],
        dtype=float,
    )
    matched_mask = np.isin(query_labels, np.unique(ref_labels))

    overall = float(matches.mean()) if len(matches) else float("nan")
    matched = float(matches[matched_mask].mean()) if matched_mask.any() else float("nan")
    orphan_fraction = float((~matched_mask).mean()) if len(matched_mask) else 0.0
    return overall, matched, orphan_fraction


def _balanced_shared_query_subset(adata_query, adata_ref, bio_label, shared_classes, cells_per_class, seed):
    query_labels = np.asarray(adata_query.obs[bio_label]).astype(str)
    ref_labels = np.asarray(adata_ref.obs[bio_label]).astype(str)

    query_counts = {}
    ref_counts = {}
    for label in np.unique(query_labels):
        query_counts[label] = int(np.sum(query_labels == label))
    for label in np.unique(ref_labels):
        ref_counts[label] = int(np.sum(ref_labels == label))

    eligible = [
        label for label in query_counts
        if label in ref_counts and query_counts[label] >= cells_per_class and ref_counts[label] >= max(cells_per_class, 8)
    ]
    eligible = sorted(
        eligible,
        key=lambda label: (min(query_counts[label], ref_counts[label]), query_counts[label], ref_counts[label]),
        reverse=True,
    )

    if not eligible:
        raise ValueError(
            "No shared classes have enough cells for balanced sampling. "
            "Lower --cells-per-class or disable --shared-classes."
        )

    selected_classes = eligible[:shared_classes]
    rng = np.random.default_rng(seed)
    selected_idx = []

    for label in selected_classes:
        label_idx = np.flatnonzero(query_labels == label)
        take = min(len(label_idx), cells_per_class)
        sampled = rng.choice(label_idx, size=take, replace=False)
        selected_idx.extend(sampled.tolist())

    return adata_query[np.sort(np.asarray(selected_idx, dtype=np.int64))].copy(), selected_classes


def run_benchmark(args):
    model_path = Path(args.model_path)
    data_path = Path(args.data_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Missing example dataset: {data_path}")

    device = _resolve_device(args.device)
    print(f"Loading model from {model_path}")
    model = load_model(str(model_path), device=device)

    print(f"Loading data from {data_path}")
    adata = read_h5ad(data_path)
    adata_ref = adata[adata.obs["Source"] == args.reference_source].copy()
    adata_query = adata[adata.obs["Source"] == args.query_source].copy()

    if args.shared_classes > 0 and args.cells_per_class > 0:
        adata_query, selected_classes = _balanced_shared_query_subset(
            adata_query=adata_query,
            adata_ref=adata_ref,
            bio_label=args.bio_label,
            shared_classes=args.shared_classes,
            cells_per_class=args.cells_per_class,
            seed=args.seed,
        )
        print(
            f"Balanced shared query subset: {len(selected_classes)} classes × up to "
            f"{args.cells_per_class} cells ({len(adata_query):,} cells total)"
        )
    elif args.query_limit and len(adata_query) > args.query_limit:
        rng = np.random.default_rng(args.seed)
        subset_idx = rng.choice(len(adata_query), args.query_limit, replace=False)
        adata_query = adata_query[np.sort(subset_idx)].copy()

    print(f"Reference cells: {len(adata_ref):,}")
    print(f"Query cells:     {len(adata_query):,}")
    print(f"Reference classes: {adata_ref.obs[args.bio_label].nunique()}")
    print(f"Query classes:     {adata_query.obs[args.bio_label].nunique()}")

    restore_refinement = None
    if args.disable_large_scale_refinement:
        import scadver.core as core_module

        restore_refinement = core_module._run_large_scale_refinement
        core_module._run_large_scale_refinement = (
            lambda **kwargs: (
                kwargs["analytical_embeddings"],
                {"ran": False, "reason": "disabled from benchmark"},
            )
        )

    try:
        start = time.perf_counter()
        adata_query_corrected = transform_query_adaptive(
            model=model,
            adata_query=adata_query,
            adata_reference=adata_ref,
            bio_label=args.bio_label,
            device=device,
            seed=args.seed,
        )
        runtime = time.perf_counter() - start
    finally:
        if restore_refinement is not None:
            import scadver.core as core_module

            core_module._run_large_scale_refinement = restore_refinement

    ref_embeddings = _encode_reference(model, adata_ref, device=device)
    query_embeddings = adata_query_corrected.obsm["X_ScAdver"]

    ref_labels = np.asarray(adata_ref.obs[args.bio_label]).astype(str)
    query_labels = np.asarray(adata_query_corrected.obs[args.bio_label]).astype(str)
    ref_batches = np.asarray(adata_ref.obs[args.batch_label]).astype(str)
    combined_batches = np.concatenate(
        [
            ref_batches,
            np.asarray(adata_query_corrected.obs[args.batch_label]).astype(str),
        ]
    )
    sources = np.concatenate(
        [
            np.full(len(adata_ref), args.reference_source),
            np.full(len(adata_query_corrected), args.query_source),
        ]
    )
    combined_embeddings = np.vstack([ref_embeddings, query_embeddings])

    lta_overall, lta_matched, orphan_fraction = _label_transfer_accuracy(
        ref_emb=ref_embeddings,
        query_emb=query_embeddings,
        ref_labels=ref_labels,
        query_labels=query_labels,
        n_neighbors=args.k_neighbors,
    )
    source_mixing = _neighbor_mixing(_knn_indices(combined_embeddings, args.k_neighbors), sources)
    ref_batch_mixing = _neighbor_mixing(_knn_indices(ref_embeddings, args.k_neighbors), ref_batches)
    batch_mixing = _neighbor_mixing(_knn_indices(combined_embeddings, args.k_neighbors), combined_batches)

    p_query = float(np.mean(sources == args.query_source))
    p_ref = 1.0 - p_query
    expected_source_mixing = 2.0 * p_ref * p_query

    print("\nBenchmark")
    print("=" * 60)
    print(f"Runtime (projection only): {runtime:.2f}s")
    print(f"Embedding shape:          {query_embeddings.shape}")
    print(f"LTA overall:              {lta_overall:.4f}")
    print(f"LTA matched only:         {lta_matched:.4f}")
    print(f"Orphan query fraction:    {orphan_fraction:.1%}")
    print(f"Source mixing:            {source_mixing:.4f}  (expected ≈ {expected_source_mixing:.4f})")
    print(f"Batch mixing:             {batch_mixing:.4f}  (ref ceiling = {ref_batch_mixing:.4f})")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run the local ScAdver simulation benchmark.")
    parser.add_argument("--model-path", default="examples/scadver_ref_sim_model.pt")
    parser.add_argument("--data-path", default="examples/simulation_2.h5ad")
    parser.add_argument("--reference-source", default="AZ")
    parser.add_argument("--query-source", default="Phenaros")
    parser.add_argument("--bio-label", default="perturbation")
    parser.add_argument("--batch-label", default="imaging_batch")
    parser.add_argument("--query-limit", type=int, default=2000)
    parser.add_argument("--shared-classes", type=int, default=0)
    parser.add_argument("--cells-per-class", type=int, default=0)
    parser.add_argument("--disable-large-scale-refinement", action="store_true")
    parser.add_argument("--k-neighbors", type=int, default=15)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
