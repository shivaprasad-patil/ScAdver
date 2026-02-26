"""
Example usage of ScAdver with the human pancreas dataset.

Demonstrates the full two-stage pipeline:
  1. adversarial_batch_correction  – train on Reference cells only
  2. transform_query_adaptive      – auto-detect domain shift and project Query
"""

import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')   # non-interactive; safe for scripts and notebooks
import matplotlib.pyplot as plt

import numpy as np
import scanpy as sc

from scadver import (
    adversarial_batch_correction,
    transform_query_adaptive,
    set_global_seed,
)


# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
DATA_H5AD = os.path.join(_HERE, "human_pancreas_norm_complexBatch.h5ad")
FIGURES   = os.path.join(_HERE, "figures")
os.makedirs(FIGURES, exist_ok=True)


def main(seed: int = 42):
    """Run the full ScAdver pipeline on the pancreas dataset.

    Parameters
    ----------
    seed : int
        Master random seed — all stochastic steps use this value,
        guaranteeing identical results across runs.
    """

    set_global_seed(seed)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("Loading pancreas dataset...")
    adata = sc.read(DATA_H5AD)
    print(f"  {adata.shape[0]} cells × {adata.shape[1]} genes")
    print(f"  Technologies : {sorted(adata.obs['tech'].unique())}")
    print(f"  Cell types   : {sorted(adata.obs['celltype'].unique())}")

    # ── 2. HVG selection ─────────────────────────────────────────────────────
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key="tech", subset=True)
    print(f"  After HVG: {adata.shape[1]} genes")

    # ── 3. Reference / Query split ────────────────────────────────────────────
    query_techs = ["smartseq2", "celseq2"]
    adata_query = adata[adata.obs["tech"].isin(query_techs)].copy()
    adata_ref   = adata[~adata.obs["tech"].isin(query_techs)].copy()

    print(f"\nData split:")
    print(f"  Reference : {adata_ref.shape[0]} cells | {dict(adata_ref.obs['tech'].value_counts())}")
    print(f"  Query     : {adata_query.shape[0]} cells | {dict(adata_query.obs['tech'].value_counts())}")

    # ── 4. Train ScAdver on Reference ─────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 1 — Train ScAdver on Reference data")
    print("="*60)

    adata_ref_corrected, model, ref_metrics = adversarial_batch_correction(
        adata=adata_ref,
        bio_label='celltype',
        batch_label='tech',
        latent_dim=256,
        epochs=500,
        bio_weight=20.0,
        batch_weight=0.5,
        learning_rate=0.001,
        device='auto',
        return_reconstructed=True,
        seed=seed,
    )

    print(f"\n✅ Reference training complete!")
    print(f"   Latent embedding          : {adata_ref_corrected.obsm['X_ScAdver'].shape}")
    print(f"   Reconstructed expression  : {adata_ref_corrected.layers['ScAdver_reconstructed'].shape}")
    print(f"\n📈 Reference metrics:")
    for key, value in ref_metrics.items():
        print(f"   {key}: {value:.4f}")

    # ── 5. Project Query (fully automatic) ────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 2 — Automatic query projection")
    print("="*60)

    adata_query_corrected = transform_query_adaptive(
        model=model,
        adata_query=adata_query,
        adata_reference=adata_ref,
        bio_label='celltype',
        adaptation_epochs=200,
        warmup_epochs=40,
        patience=30,
        learning_rate=0.0005,
        device='auto',
        return_reconstructed=True,
        seed=seed,
    )

    print(f"\n✅ Query projection complete!")
    print(f"   Latent embedding         : {adata_query_corrected.obsm['X_ScAdver'].shape}")
    print(f"   Reconstructed expression : {adata_query_corrected.layers['ScAdver_reconstructed'].shape}")

    # ── 6. Combine & UMAP ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 3 — UMAP visualisation")
    print("="*60)

    adata_ref_corrected.obs['Source']   = 'Reference'
    adata_query_corrected.obs['Source'] = 'Query'
    adata_all = sc.concat([adata_ref_corrected, adata_query_corrected])

    sc.pp.neighbors(adata_all, use_rep='X_ScAdver', n_neighbors=15)
    sc.tl.umap(adata_all)
    print("  UMAP computed")

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    sc.pl.umap(adata_all, color='Source',   ax=axes[0], show=False, title='Data Source')
    sc.pl.umap(adata_all, color='celltype', ax=axes[1], show=False, title='Cell Types')
    sc.pl.umap(adata_all, color='tech',     ax=axes[2], show=False, title='Technology')
    plt.tight_layout()
    out_path = os.path.join(FIGURES, "pancreas_result.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path}")

    # ── 7. Integration metrics ────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 4 — Integration quality metrics")
    print("="*60)

    from sklearn.metrics import silhouette_score
    emb = adata_all.obsm['X_ScAdver']

    source_sil = silhouette_score(emb, adata_all.obs['Source'])
    bio_sil    = silhouette_score(emb, adata_all.obs['celltype'])
    tech_sil   = silhouette_score(emb, adata_all.obs['tech'])

    source_mix = (1 - source_sil) / 2 + 0.5
    bio_pres   = (bio_sil  + 1)   / 2
    tech_mix   = (1 - tech_sil)   / 2 + 0.5

    print(f"\n  Source mixing     : {source_mix:.4f}  (sil={source_sil:+.4f},  1=fully mixed)")
    print(f"  Biology preserved : {bio_pres:.4f}   (sil={bio_sil:+.4f}, 1=well-structured)")
    print(f"  Tech mixing       : {tech_mix:.4f}  (sil={tech_sil:+.4f},  1=fully mixed)")

    print(f"  Source   : {'✅ Good' if source_mix > 0.75 else ('⚙️  Moderate' if source_mix > 0.60 else '⚠️  Poor')}")
    print(f"  Biology  : {'✅ Good' if bio_pres   > 0.65 else '⚠️  Degraded'}")
    print(f"  Tech mix : {'✅ Good' if tech_mix   > 0.75 else ('⚙️  Moderate' if tech_mix > 0.60 else '⚠️  Poor')}")

    # ── 8. DE analysis on reconstructed expression ───────────────────────────
    print("\n" + "="*60)
    print("STEP 5 — Differential expression on reconstructed expression")
    print("="*60)

    adata_recon = adata_all.copy()
    adata_recon.X = adata_all.layers['ScAdver_reconstructed']
    sc.tl.rank_genes_groups(adata_recon, groupby='celltype', method='wilcoxon')

    markers = sc.get.rank_genes_groups_df(adata_recon, group=None)
    print("  Top 10 marker genes across all cell types:")
    print(markers.head(10).to_string(index=False))

    print("\n🎉 Example completed successfully!")
    print("\n📝 Summary:")
    print("   - Latent embeddings  → UMAP, clustering, integration metrics")
    print("   - Reconstructed expr → differential expression, downstream analysis")
    print(f"   - UMAP saved to      → {out_path}")

    return adata_all, model, ref_metrics


if __name__ == "__main__":
    main()
