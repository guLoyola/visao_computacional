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


def plot_training_progress_comparison(
    training_histories: dict,
    metric: str = 'mAP50-95',
    save_path: Optional[str] = None
):
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {'mobilenetv3': '#2ecc71',
              'efficientnet-b0': '#e67e22', 'resnet50': '#3498db'}
    model_display_names = {
        'mobilenetv3': 'MobileNetV3',
        'efficientnet-b0': 'EfficientNet-B0',
        'resnet50': 'ResNet50'
    }

    for model_name, history in training_histories.items():
        if metric in history:
            progress = np.linspace(0, 100, len(history[metric]))
            display_name = model_display_names.get(model_name, model_name)
            color = colors.get(model_name, '#95a5a6')
            ax.plot(progress, history[metric], label=display_name,
                    color=color, linewidth=2)

    ax.set_xlabel('Progresso do Treino (%)', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'Comparação de modelos - {metric} ao longo do treino',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico de progresso de treino salvo em: {save_path}")

    plt.close()


def create_comparison_table(
    results: dict,
    save_path: Optional[str] = None
):
    model_display_names = {
        'mobilenetv3': 'MobileNetV3',
        'efficientnet-b0': 'EfficientNet-B0',
        'resnet50': 'ResNet50'
    }

    metrics_display = {
        'mAP@50': 'mAP@50',
        'mAP@50-95': 'mAP@50-95',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1': 'F1-score',
        'stability': 'Estabilidade (oscilações)',
        'overfitting': 'Overfitting (gap treino/val)',
        'efficiency': 'Eficiência',
        'inference_time': 'Tempo de inferência (ms/img)',
        'model_size': 'Tamanho do modelo (MB)',
        'generalization': 'Generalização'
    }

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')

    table_data = []
    header = ['Critério'] + \
        [model_display_names.get(m, m) for m in results.keys()]
    table_data.append(header)

    for metric_key, metric_label in metrics_display.items():
        row = [metric_label]
        for model_name in results.keys():
            value = results[model_name].get(metric_key, 0)
            if isinstance(value, (int, float)):
                row.append(f'{value:.4f}')
            else:
                row.append(str(value))
        table_data.append(row)

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25] + [0.25] * len(results))

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')

    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if j == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
                table[(i, j)].set_text_props(weight='bold')
            else:
                table[(i, j)].set_facecolor('white')

    plt.title('Comparação de Modelos - Métricas Detalhadas',
              fontsize=16, fontweight='bold', pad=20)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Tabela de comparação salva em: {save_path}")

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
