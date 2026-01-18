"""
Visual demonstration of encoder training and query projection.
This script generates diagrams showing how the encoder learns and applies transformations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

def create_training_phase_diagram():
    """Visualize the training phase with adversarial objectives."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'TRAINING PHASE: Encoder Learning from Reference Data', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Input data
    input_box = FancyBboxPatch((0.5, 9), 1.5, 1.5, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='black', facecolor='lightblue', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.25, 10.3, 'Reference\nData (X)', fontsize=10, ha='center', fontweight='bold')
    ax.text(1.25, 9.5, 'batch1, batch2\nbatch3', fontsize=8, ha='center')
    ax.text(1.25, 8.7, 'T-cells, B-cells\nMonocytes', fontsize=8, ha='center', style='italic')
    
    # Encoder
    arrow1 = FancyArrowPatch((2, 9.75), (2.8, 9.75), 
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow1)
    
    encoder_box = FancyBboxPatch((2.8, 8.5), 2, 2.5,
                                boxstyle="round,pad=0.1",
                                edgecolor='green', facecolor='lightgreen', linewidth=3)
    ax.add_patch(encoder_box)
    ax.text(3.8, 10.3, 'ENCODER', fontsize=12, ha='center', fontweight='bold', color='darkgreen')
    ax.text(3.8, 9.8, '(Training)', fontsize=9, ha='center', style='italic')
    ax.text(3.8, 9.4, '2000→2048→1024', fontsize=8, ha='center')
    ax.text(3.8, 9.0, '→512→256', fontsize=8, ha='center')
    ax.text(3.8, 8.6, 'Weights: Learning', fontsize=8, ha='center', 
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # Latent Z
    arrow2 = FancyArrowPatch((4.8, 9.75), (5.6, 9.75),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow2)
    
    latent_box = FancyBboxPatch((5.6, 9), 1.3, 1.5,
                               boxstyle="round,pad=0.1",
                               edgecolor='purple', facecolor='lavender', linewidth=2)
    ax.add_patch(latent_box)
    ax.text(6.25, 10, 'Latent Z', fontsize=11, ha='center', fontweight='bold', color='purple')
    ax.text(6.25, 9.5, '(256-dim)', fontsize=9, ha='center')
    
    # Three objectives branching from Z
    # 1. Decoder
    arrow_decoder = FancyArrowPatch((6.25, 8.9), (6.25, 7.8),
                                   arrowstyle='->', mutation_scale=15, linewidth=2, color='blue')
    ax.add_patch(arrow_decoder)
    
    decoder_box = FancyBboxPatch((5.3, 6.8), 1.9, 1,
                                boxstyle="round,pad=0.05",
                                edgecolor='blue', facecolor='lightcyan', linewidth=2)
    ax.add_patch(decoder_box)
    ax.text(6.25, 7.5, 'DECODER', fontsize=10, ha='center', fontweight='bold', color='blue')
    ax.text(6.25, 7.1, 'Reconstruction', fontsize=8, ha='center')
    ax.text(6.25, 6.4, '✅ Minimize MSE', fontsize=8, ha='center', color='green')
    
    # 2. Bio-classifier
    arrow_bio = FancyArrowPatch((6.9, 9.5), (8.2, 7.8),
                               arrowstyle='->', mutation_scale=15, linewidth=2, color='darkgreen')
    ax.add_patch(arrow_bio)
    
    bio_box = FancyBboxPatch((7.8, 6.8), 2, 1,
                            boxstyle="round,pad=0.05",
                            edgecolor='darkgreen', facecolor='lightgreen', linewidth=2)
    ax.add_patch(bio_box)
    ax.text(8.8, 7.5, 'Bio-Classifier', fontsize=10, ha='center', fontweight='bold', color='darkgreen')
    ax.text(8.8, 7.1, 'Cell Type Pred', fontsize=8, ha='center')
    ax.text(8.8, 6.4, '✅ Maximize Accuracy', fontsize=8, ha='center', color='green')
    ax.text(8.8, 6.0, 'weight: 20.0', fontsize=7, ha='center', 
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # 3. Batch-discriminator
    arrow_batch = FancyArrowPatch((5.6, 9.5), (2.8, 7.8),
                                 arrowstyle='->', mutation_scale=15, linewidth=2, color='red')
    ax.add_patch(arrow_batch)
    
    batch_box = FancyBboxPatch((1.8, 6.8), 2, 1,
                              boxstyle="round,pad=0.05",
                              edgecolor='red', facecolor='lightcoral', linewidth=2)
    ax.add_patch(batch_box)
    ax.text(2.8, 7.5, 'Batch-Discriminator', fontsize=10, ha='center', fontweight='bold', color='red')
    ax.text(2.8, 7.1, 'Batch Prediction', fontsize=8, ha='center')
    ax.text(2.8, 6.4, '❌ Minimize Accuracy', fontsize=8, ha='center', color='red')
    ax.text(2.8, 6.0, 'weight: -0.5 ⚔️', fontsize=7, ha='center',
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    # Adversarial game explanation
    game_box = FancyBboxPatch((0.5, 4.5), 9, 1.3,
                             boxstyle="round,pad=0.1",
                             edgecolor='orange', facecolor='lightyellow', linewidth=2)
    ax.add_patch(game_box)
    ax.text(5, 5.5, '⚔️ ADVERSARIAL GAME ⚔️', fontsize=11, ha='center', fontweight='bold', color='orange')
    ax.text(5, 5.1, 'Encoder learns: "Keep biology patterns (green ✅), Remove batch patterns (red ❌)"', 
           fontsize=9, ha='center')
    ax.text(5, 4.7, 'Result: Z = Biology-rich + Batch-free latent representation', 
           fontsize=9, ha='center', style='italic')
    
    # Training iterations
    iterations_box = FancyBboxPatch((0.5, 2.8), 9, 1.5,
                                   boxstyle="round,pad=0.1",
                                   edgecolor='gray', facecolor='white', linewidth=2)
    ax.add_patch(iterations_box)
    ax.text(5, 4.1, 'Training Progress (500 epochs):', fontsize=10, ha='center', fontweight='bold')
    
    ax.text(2, 3.6, 'Epoch 1-100:', fontsize=9, ha='left', fontweight='bold')
    ax.text(2, 3.3, '• Batch Disc: 90% → Learning', fontsize=8, ha='left')
    ax.text(2, 3.0, '• Bio Classif: 70% → Learning', fontsize=8, ha='left')
    
    ax.text(5, 3.6, 'Epoch 100-300:', fontsize=9, ha='left', fontweight='bold')
    ax.text(5, 3.3, '• Batch Disc: 70% → Improving', fontsize=8, ha='left')
    ax.text(5, 3.0, '• Bio Classif: 85% → Improving', fontsize=8, ha='left')
    
    ax.text(7.8, 3.6, 'Epoch 300-500:', fontsize=9, ha='left', fontweight='bold')
    ax.text(7.8, 3.3, '• Batch Disc: ~50% ✅ Success!', fontsize=8, ha='left', color='green')
    ax.text(7.8, 3.0, '• Bio Classif: ~95% ✅ Success!', fontsize=8, ha='left', color='green')
    
    # Result
    result_box = FancyBboxPatch((0.5, 0.3), 9, 2.3,
                               boxstyle="round,pad=0.1",
                               edgecolor='darkgreen', facecolor='lightgreen', linewidth=3, alpha=0.3)
    ax.add_patch(result_box)
    ax.text(5, 2.3, '🎉 TRAINING COMPLETE: Encoder Weights FIXED', 
           fontsize=12, ha='center', fontweight='bold', color='darkgreen')
    ax.text(5, 1.9, '✅ Learned Transformation: X → Z (batch-free, biology-rich)', 
           fontsize=10, ha='center')
    ax.text(5, 1.5, '✅ Biology patterns: Recognized and preserved', 
           fontsize=9, ha='center')
    ax.text(5, 1.1, '✅ Batch patterns: Recognized and removed', 
           fontsize=9, ha='center')
    ax.text(5, 0.7, '✅ Weights encode: ~6 million parameters representing this transformation', 
           fontsize=9, ha='center', style='italic')
    
    plt.tight_layout()
    return fig


def create_projection_phase_diagram():
    """Visualize the query projection phase with frozen encoder."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'PROJECTION PHASE: Query Data Through Frozen Encoder', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Query input
    query_box = FancyBboxPatch((0.5, 9), 1.5, 1.5,
                              boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor='lightcoral', linewidth=2)
    ax.add_patch(query_box)
    ax.text(1.25, 10.3, 'Query Data\n(NEW!)', fontsize=10, ha='center', fontweight='bold')
    ax.text(1.25, 9.5, 'smartseq2 batch', fontsize=8, ha='center', color='red')
    ax.text(1.25, 8.7, 'Same cell types', fontsize=8, ha='center', style='italic')
    
    # Frozen encoder
    arrow1 = FancyArrowPatch((2, 9.75), (2.8, 9.75),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow1)
    
    encoder_box = FancyBboxPatch((2.8, 8.5), 2, 2.5,
                                boxstyle="round,pad=0.1",
                                edgecolor='blue', facecolor='lightblue', linewidth=3)
    ax.add_patch(encoder_box)
    ax.text(3.8, 10.5, '🧊 FROZEN ENCODER', fontsize=12, ha='center', fontweight='bold', color='darkblue')
    ax.text(3.8, 10, '(No Training!)', fontsize=9, ha='center', style='italic', color='blue')
    ax.text(3.8, 9.5, '2000→2048→1024', fontsize=8, ha='center')
    ax.text(3.8, 9.1, '→512→256', fontsize=8, ha='center')
    ax.text(3.8, 8.7, 'Weights: FIXED ❄️', fontsize=9, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='cyan', alpha=0.7))
    
    # Latent Z query
    arrow2 = FancyArrowPatch((4.8, 9.75), (5.6, 9.75),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow2)
    
    latent_box = FancyBboxPatch((5.6, 9), 1.8, 1.5,
                               boxstyle="round,pad=0.1",
                               edgecolor='purple', facecolor='lavender', linewidth=2)
    ax.add_patch(latent_box)
    ax.text(6.5, 10, 'Z_query', fontsize=11, ha='center', fontweight='bold', color='purple')
    ax.text(6.5, 9.5, '(Batch-corrected!)', fontsize=8, ha='center', color='green')
    
    # No training indicators
    no_train_box1 = FancyBboxPatch((2.5, 7.8), 2.7, 0.5,
                                  boxstyle="round,pad=0.05",
                                  edgecolor='red', facecolor='white', linewidth=2, linestyle='--')
    ax.add_patch(no_train_box1)
    ax.text(3.85, 8.05, '❌ No backward pass', fontsize=8, ha='center', color='red')
    
    no_train_box2 = FancyBboxPatch((2.5, 7.2), 2.7, 0.5,
                                  boxstyle="round,pad=0.05",
                                  edgecolor='red', facecolor='white', linewidth=2, linestyle='--')
    ax.add_patch(no_train_box2)
    ax.text(3.85, 7.45, '❌ No optimizer step', fontsize=8, ha='center', color='red')
    
    no_train_box3 = FancyBboxPatch((2.5, 6.6), 2.7, 0.5,
                                  boxstyle="round,pad=0.05",
                                  edgecolor='red', facecolor='white', linewidth=2, linestyle='--')
    ax.add_patch(no_train_box3)
    ax.text(3.85, 6.85, '❌ No weight updates', fontsize=8, ha='center', color='red')
    
    # What happens
    process_box = FancyBboxPatch((0.5, 4.5), 9, 1.7,
                                boxstyle="round,pad=0.1",
                                edgecolor='blue', facecolor='aliceblue', linewidth=2)
    ax.add_patch(process_box)
    ax.text(5, 6, '🔄 What Happens During Projection:', fontsize=11, ha='center', fontweight='bold', color='darkblue')
    ax.text(5, 5.6, '1. Query gene expression (X_query) enters frozen encoder', fontsize=9, ha='center')
    ax.text(5, 5.2, '2. Encoder applies SAME transformation learned from reference data', fontsize=9, ha='center')
    ax.text(5, 4.8, '3. Biology patterns → Preserved (learned: "CD3+CD8+ = T-cell")', fontsize=9, ha='center', color='green')
    ax.text(5, 4.4, '4. Batch patterns → Removed (learned: "Ignore library size, protocol noise")', fontsize=9, ha='center', color='red')
    
    # Why it works
    why_box = FancyBboxPatch((0.5, 2.3), 9, 2,
                            boxstyle="round,pad=0.1",
                            edgecolor='green', facecolor='honeydew', linewidth=2)
    ax.add_patch(why_box)
    ax.text(5, 4.1, '💡 Why Batch Correction Happens Automatically:', fontsize=11, ha='center', fontweight='bold', color='darkgreen')
    
    ax.text(1, 3.7, '✅ Generalization:', fontsize=9, ha='left', fontweight='bold', color='darkgreen')
    ax.text(1.2, 3.4, 'Encoder learned PATTERNS, not memorized cells', fontsize=8, ha='left')
    ax.text(1.2, 3.1, '"CD3+CD8+ = T-cell" works for ANY batch', fontsize=8, ha='left', style='italic')
    ax.text(1.2, 2.8, 'Pattern recognition generalizes to new data', fontsize=8, ha='left', style='italic')
    
    ax.text(5.5, 3.7, '✅ Fixed Transformation:', fontsize=9, ha='left', fontweight='bold', color='darkgreen')
    ax.text(5.7, 3.4, 'Same weights apply to reference and query', fontsize=8, ha='left')
    ax.text(5.7, 3.1, 'Batch removal was "baked into" the weights', fontsize=8, ha='left', style='italic')
    ax.text(5.7, 2.8, 'Biology preservation was "baked into" weights', fontsize=8, ha='left', style='italic')
    
    ax.text(1, 2.5, '✅ Adversarial Learning:', fontsize=9, ha='left', fontweight='bold', color='darkgreen')
    ax.text(1.2, 2.2, 'Encoder learned to extract batch-invariant features', fontsize=8, ha='left')
    ax.text(1.2, 1.9, 'These features work regardless of batch origin', fontsize=8, ha='left', style='italic')
    
    ax.text(5.5, 2.5, '✅ Consistency:', fontsize=9, ha='left', fontweight='bold', color='darkgreen')
    ax.text(5.7, 2.2, 'Query projects to SAME latent space as reference', fontsize=8, ha='left')
    ax.text(5.7, 1.9, 'T-cells (query) → Same region as T-cells (ref)', fontsize=8, ha='left', style='italic')
    
    # Result
    result_box = FancyBboxPatch((0.5, 0.3), 9, 1.5,
                               boxstyle="round,pad=0.1",
                               edgecolor='darkgreen', facecolor='lightgreen', linewidth=3, alpha=0.5)
    ax.add_patch(result_box)
    ax.text(5, 1.6, '🎉 PROJECTION COMPLETE (< 1 second!)', 
           fontsize=12, ha='center', fontweight='bold', color='darkgreen')
    ax.text(5, 1.2, '✅ Query data batch-corrected AND biology-preserved', 
           fontsize=10, ha='center')
    ax.text(5, 0.8, '✅ Can repeat for unlimited query batches - NO RETRAINING NEEDED!', 
           fontsize=10, ha='center')
    ax.text(5, 0.4, '⚡ Speed: 1000x faster than retraining, Same quality results', 
           fontsize=9, ha='center', style='italic', color='blue')
    
    plt.tight_layout()
    return fig


def create_latent_space_diagram():
    """Visualize how reference and query data project to same latent space."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Simulate some data
    np.random.seed(42)
    n_per_type = 50
    
    # Cell type centers (before correction)
    centers_bio = {
        'T-cells': [2, 3],
        'B-cells': [-2, 2],
        'Monocytes': [0, -2]
    }
    
    # Batch offsets (before correction)
    batch_offsets = {
        'batch1': [0, 0],
        'batch2': [1.5, 0.5],
        'batch3': [-1, 0.8],
        'smartseq2': [0.5, -1]
    }
    
    colors_bio = {'T-cells': 'red', 'B-cells': 'blue', 'Monocytes': 'green'}
    markers_batch = {'batch1': 'o', 'batch2': 's', 'batch3': '^', 'smartseq2': 'D'}
    
    # Generate data
    ref_data = []
    query_data = []
    
    for cell_type, center in centers_bio.items():
        for batch, offset in batch_offsets.items():
            if batch == 'smartseq2':
                # Query batch
                points = np.random.randn(n_per_type, 2) * 0.3 + center + offset
                for p in points:
                    query_data.append((p, cell_type, batch))
            else:
                # Reference batches
                points = np.random.randn(n_per_type, 2) * 0.3 + center + offset
                for p in points:
                    ref_data.append((p, cell_type, batch))
    
    # Panel 1: Before correction (gene expression space)
    ax1.set_title('Before Correction\n(Gene Expression Space)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Gene PC1', fontsize=10)
    ax1.set_ylabel('Gene PC2', fontsize=10)
    
    for point, cell_type, batch in ref_data:
        ax1.scatter(point[0], point[1], c=colors_bio[cell_type], 
                   marker=markers_batch[batch], alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
    
    for point, cell_type, batch in query_data:
        ax1.scatter(point[0], point[1], c=colors_bio[cell_type],
                   marker=markers_batch[batch], alpha=0.8, s=50, edgecolors='orange', linewidth=2)
    
    ax1.text(0, 5.5, '❌ Batch effects visible', ha='center', fontsize=9, color='red', fontweight='bold')
    ax1.text(0, 5, 'Cell types separated by batch', ha='center', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-5, 6)
    
    # Panel 2: After correction - Reference
    ax2.set_title('After Correction (Reference)\n(Latent Space Z)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Latent Dim 1', fontsize=10)
    ax2.set_ylabel('Latent Dim 2', fontsize=10)
    
    # Corrected: remove batch offsets
    for point, cell_type, batch in ref_data:
        corrected = point - batch_offsets[batch]
        ax2.scatter(corrected[0], corrected[1], c=colors_bio[cell_type],
                   marker='o', alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
    
    ax2.text(0, 5.5, '✅ Batches mixed', ha='center', fontsize=9, color='green', fontweight='bold')
    ax2.text(0, 5, 'Cell types preserved', ha='center', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-5, 6)
    
    # Panel 3: After correction - Reference + Query
    ax3.set_title('After Correction (Ref + Query)\n(Same Latent Space Z)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Latent Dim 1', fontsize=10)
    ax3.set_ylabel('Latent Dim 2', fontsize=10)
    
    # Reference corrected
    for point, cell_type, batch in ref_data:
        corrected = point - batch_offsets[batch]
        ax3.scatter(corrected[0], corrected[1], c=colors_bio[cell_type],
                   marker='o', alpha=0.4, s=30, edgecolors='black', linewidth=0.5, label='_nolegend_')
    
    # Query corrected (projects to same space!)
    for point, cell_type, batch in query_data:
        corrected = point - batch_offsets[batch]
        ax3.scatter(corrected[0], corrected[1], c=colors_bio[cell_type],
                   marker='D', alpha=0.9, s=60, edgecolors='orange', linewidth=2, label='_nolegend_')
    
    ax3.text(0, 5.5, '✅ Query integrated seamlessly!', ha='center', fontsize=9, color='green', fontweight='bold')
    ax3.text(0, 5, 'Same latent space, batch-free', ha='center', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-5, 5)
    ax3.set_ylim(-5, 6)
    
    # Create custom legend
    legend_elements = [
        mpatches.Patch(color='red', label='T-cells'),
        mpatches.Patch(color='blue', label='B-cells'),
        mpatches.Patch(color='green', label='Monocytes'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                  markersize=8, label='Reference (batch1/2/3)', markeredgecolor='black'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='gray',
                  markersize=10, label='Query (smartseq2)', markeredgecolor='orange', markeredgewidth=2)
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, 
              fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.05))
    
    plt.tight_layout()
    return fig


def main():
    """Generate all diagrams."""
    print("🎨 Generating encoder mechanism diagrams...")
    
    # Training phase
    print("   1. Training phase diagram...")
    fig1 = create_training_phase_diagram()
    fig1.savefig('training_phase_diagram.png', dpi=300, bbox_inches='tight')
    print("      ✅ Saved: training_phase_diagram.png")
    
    # Projection phase
    print("   2. Projection phase diagram...")
    fig2 = create_projection_phase_diagram()
    fig2.savefig('projection_phase_diagram.png', dpi=300, bbox_inches='tight')
    print("      ✅ Saved: projection_phase_diagram.png")
    
    # Latent space
    print("   3. Latent space integration diagram...")
    fig3 = create_latent_space_diagram()
    fig3.savefig('latent_space_diagram.png', dpi=300, bbox_inches='tight')
    print("      ✅ Saved: latent_space_diagram.png")
    
    print("\n🎉 All diagrams generated successfully!")
    print("\n📊 Diagrams explain:")
    print("   • How encoder learns from reference data (training phase)")
    print("   • How frozen encoder projects query data (projection phase)")
    print("   • How data integrates in latent space (batch-free, biology-rich)")
    
    plt.show()


if __name__ == "__main__":
    main()
