import os
from typing import List, Optional
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')


def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    train_accuracies: List[float],
    val_accuracies: List[float],
    save_path: Optional[str] = None,
    model_name: str = 'Model'
):
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'{model_name} - Training and Validation Loss',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accuracies, 'b-',
             label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, val_accuracies, 'r-', label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title(f'{model_name} - Training and Validation Accuracy',
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico de treinamento salvo em: {save_path}")

    plt.close()


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    model_name: str = 'Model',
    normalize: bool = False
):
    if normalize:
        conf_matrix = conf_matrix.astype(
            'float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title_suffix = ' (Normalized)'
    else:
        fmt = 'd'
        title_suffix = ''

    plt.figure(figsize=(12, 10))

    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
        linewidths=0.5,
        linecolor='gray'
    )

    plt.title(f'{model_name} - Confusion Matrix{title_suffix}',
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Matriz de confusão salva em: {save_path}")

    plt.close()


def plot_per_class_metrics(
    metrics_obj,
    save_path: Optional[str] = None,
    model_name: str = 'Model'
):
    class_names = metrics_obj.class_names
    x = np.arange(len(class_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 6))

    bars1 = ax.bar(x - 1.5*width, metrics_obj.precision_per_class, width,
                   label='Precision', color='#3498db', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x - 0.5*width, metrics_obj.recall_per_class, width,
                   label='Recall', color='#2ecc71', edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + 0.5*width, metrics_obj.f1_per_class, width,
                   label='F1-Score', color='#e74c3c', edgecolor='black', linewidth=0.5)
    bars4 = ax.bar(x + 1.5*width, metrics_obj.specificity_per_class, width,
                   label='Specificity', color='#f39c12', edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name} - Per-Class Metrics',
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)

    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=7)

    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)
    autolabel(bars4)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Métricas por classe salvas em: {save_path}")

    plt.close()


def plot_model_comparison(
    results: dict,
    save_path: Optional[str] = None
):
    model_names = list(results.keys())
    metrics_names = ['accuracy', 'precision', 'recall', 'f1', 'mcc']
    metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC']

    x = np.arange(len(metrics_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 7))

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

    for i, model_name in enumerate(model_names):
        values = [results[model_name].get(metric, 0)
                  for metric in metrics_names]
        offset = (i - len(model_names)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model_name,
                      color=colors[i % len(colors)], edgecolor='black', linewidth=0.5)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8, rotation=0)

    ax.set_xlabel('Metrics', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title('Model Comparison - Test Set Performance',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_labels, fontsize=11)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparação de modelos salva em: {save_path}")

    plt.close()


def quick_visualize_all(
    metrics_obj,
    train_losses: List[float],
    val_losses: List[float],
    train_accuracies: List[float],
    val_accuracies: List[float],
    save_dir: str = '.',
    model_name: str = 'Model'
):
    os.makedirs(save_dir, exist_ok=True)

    model_name_clean = model_name.lower().replace(' ', '_').replace('-', '_')

    plot_training_history(
        train_losses, val_losses, train_accuracies, val_accuracies,
        save_path=os.path.join(
            save_dir, f'{model_name_clean}_training_history.png'),
        model_name=model_name
    )

    plot_confusion_matrix(
        metrics_obj.conf_matrix,
        metrics_obj.class_names,
        save_path=os.path.join(
            save_dir, f'{model_name_clean}_confusion_matrix.png'),
        model_name=model_name,
        normalize=False
    )

    plot_confusion_matrix(
        metrics_obj.conf_matrix,
        metrics_obj.class_names,
        save_path=os.path.join(
            save_dir, f'{model_name_clean}_confusion_matrix_normalized.png'),
        model_name=model_name,
        normalize=True
    )

    plot_per_class_metrics(
        metrics_obj,
        save_path=os.path.join(
            save_dir, f'{model_name_clean}_per_class_metrics.png'),
        model_name=model_name
    )

    print(f"\n✓ Todas as visualizações de {model_name} geradas em: {save_dir}")
