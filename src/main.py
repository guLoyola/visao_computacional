from pipeline.metrics import ClassificationMetrics
from pipeline.renet_50 import ResNet50
from pipeline.mobile_net_v3 import MobileNetV3_Large
from pipeline.efficient_net import EfficientNet_B0, EfficientNet_B1
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np
from datetime import datetime
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def get_next_run_number(results_dir='results/runs'):
    os.makedirs(results_dir, exist_ok=True)

    existing_runs = [d for d in os.listdir(
        results_dir) if d.startswith('run_')]
    if not existing_runs:
        return 1

    run_numbers = []
    for run in existing_runs:
        try:
            num = int(run.split('_')[1])
            run_numbers.append(num)
        except (IndexError, ValueError):
            continue

    return max(run_numbers) + 1 if run_numbers else 1


def setup_run_directory(run_number):
    run_dir = f'results/runs/run_{run_number}'

    dirs = {
        'base': run_dir,
        'models': os.path.join(run_dir, 'models'),
        'metrics': os.path.join(run_dir, 'metrics'),
        'visualizations': os.path.join(run_dir, 'visualizations'),
        'logs': os.path.join(run_dir, 'logs')
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return dirs


def get_data_transforms(input_size=224):
    return {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }


def train_model(model, model_name, train_loader, val_loader, dirs,
                num_epochs=10, learning_rate=0.0001, device='cuda'):
    print(f"\n{'='*70}")
    print(f"TREINANDO {model_name}")
    print(f"{'='*70}\n")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss_avg = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss_avg)
        val_accuracies.append(val_acc)

        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss_avg:.4f} | '
              f'Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%')

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            model_path = os.path.join(
                dirs['models'], f'best_{model_name.lower().replace(" ", "_")}.pth')
            torch.save(model.state_dict(), model_path)
            print(f'  ✓ Melhor modelo salvo! (Val Loss: {best_val_loss:.4f})')

    print(f"\n✓ Treinamento de {model_name} completo!")

    return model, train_losses, val_losses, train_accuracies, val_accuracies


def evaluate_model(model, test_loader, class_names, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    metrics = ClassificationMetrics(
        y_true=np.array(all_labels),
        y_pred=np.array(all_preds),
        y_pred_proba=np.array(all_probs),
        class_names=class_names
    )

    return metrics


def save_run_config(dirs, config):
    config_path = os.path.join(dirs['base'], 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"✓ Configuração salva em: {config_path}")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("PIPELINE DE TREINAMENTO - DATASET MISTO")
    print("="*70 + "\n")
    config = {
        'dataset_path': 'data/mixed_dataset',
        'num_epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.0001,
        'models_to_train': ['EfficientNet-B0', 'MobileNetV3-Large', 'ResNet50'],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = str(device)
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU: Not available (running on CPU)")
    print()

    run_number = get_next_run_number()
    dirs = setup_run_directory(run_number)

    print(f"✓ Run #{run_number} criada em: {dirs['base']}\n")
    save_run_config(dirs, config)
    print("Carregando datasets...")

    data_transforms = get_data_transforms(input_size=224)

    train_dataset = ImageFolder(
        os.path.join(config['dataset_path'], 'train'),
        transform=data_transforms['train']
    )
    val_dataset = ImageFolder(
        os.path.join(config['dataset_path'], 'val'),
        transform=data_transforms['val']
    )
    test_dataset = ImageFolder(
        os.path.join(config['dataset_path'], 'test'),
        transform=data_transforms['test']
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                            shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'],
                             shuffle=False, num_workers=4)

    class_names = train_dataset.classes
    num_classes = len(class_names)

    print(f"✓ Datasets carregados!")
    print(f"  Classes ({num_classes}): {class_names}")
    print(f"  Train: {len(train_dataset)} amostras")
    print(f"  Val: {len(val_dataset)} amostras")
    print(f"  Test: {len(test_dataset)} amostras\n")

    available_models = {
        'EfficientNet-B0': lambda: EfficientNet_B0(num_classes=num_classes),
        'EfficientNet-B1': lambda: EfficientNet_B1(num_classes=num_classes),
        'MobileNetV3-Large': lambda: MobileNetV3_Large(num_classes=num_classes),
        'ResNet50': lambda: ResNet50(num_classes=num_classes)
    }

    results = {}

    for model_name in config['models_to_train']:
        if model_name not in available_models:
            print(f"⚠️  Modelo {model_name} não encontrado, pulando...")
            continue

        print(f"\n{'='*70}")
        print(f"MODELO: {model_name}")
        print(f"{'='*70}")

        model = available_models[model_name]()

        model, train_losses, val_losses, train_accs, val_accs = train_model(
            model=model,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            dirs=dirs,
            num_epochs=config['num_epochs'],
            learning_rate=config['learning_rate'],
            device=device_str
        )

        print(f"\nAvaliando {model_name} no conjunto de teste...")
        metrics = evaluate_model(model, test_loader, class_names, device_str)

        metrics.print_summary()

        metrics_file = os.path.join(
            dirs['metrics'], f'{model_name.lower().replace(" ", "_")}_metrics.txt')
        metrics.save_to_txt(metrics_file)

        results[model_name] = {
            'accuracy': metrics.accuracy,
            'precision': metrics.precision_weighted,
            'recall': metrics.recall_weighted,
            'f1': metrics.f1_weighted,
            'mcc': metrics.mcc,
            'kappa': metrics.kappa
        }

        print(f"\n✓ {model_name} completo!")

    print(f"\n{'='*70}")
    print("RESUMO FINAL DA RUN")
    print(f"{'='*70}\n")

    print(f"Run: #{run_number}")
    print(f"Diretório: {dirs['base']}")
    print(f"Modelos treinados: {len(results)}\n")

    print("Resultados por modelo:")
    print("-" * 70)
    for model_name, model_results in results.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy  : {model_results['accuracy']:.4f}")
        print(f"  Precision : {model_results['precision']:.4f}")
        print(f"  Recall    : {model_results['recall']:.4f}")
        print(f"  F1-Score  : {model_results['f1']:.4f}")
        print(f"  MCC       : {model_results['mcc']:.4f}")

    summary_path = os.path.join(dirs['base'], 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\n{'='*70}")
    print("✓ PIPELINE COMPLETO!")
    print(f"{'='*70}")
    print(f"\nResultados salvos em: {dirs['base']}")
    print(f"  - Modelos: {dirs['models']}")
    print(f"  - Métricas: {dirs['metrics']}")
    print(f"  - Visualizações: {dirs['visualizations']}")
    print(f"  - Resumo: {summary_path}\n")
