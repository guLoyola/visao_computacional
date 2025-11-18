import random
import shutil
import os

out_dir = os.path.join('data', 'mixed_dataset')

"""
Mistura os datasets GuavaDiseaseDataset e BananaLSD criando um novo dataset combinado.
Divide em: 70% treino, 20% validação, 10% teste

Estrutura final:
mixed_dataset/
    ├── train/ (70%)
    │   ├── cordana/
    │   ├── sigatoka/
    │   ├── pestalotiopsis/
    │   ├── healthy_banana/
    │   ├── anthracnose/
    │   ├── fruit_fly/
    │   └── healthy_guava/
    ├── val/ (20%)
    └── test/ (10%)
"""

random.seed(42)

banana_root = os.path.join('data', 'shifatearman', 'bananalsd',
                           'versions', '1', 'BananaLSD')

guava_base = os.path.join('data', 'asadullahgalib', 'guava-disease-dataset',
                          'versions', '6', 'GuavaDiseaseDataset', 'GuavaDiseaseDataset')

os.makedirs(out_dir, exist_ok=True)

banana_classes = {
    'cordana': 'cordana',
    'sigatoka': 'sigatoka',
    'pestalotiopsis': 'pestalotiopsis',
    'healthy': 'healthy_banana'
}

guava_classes = {
    'Anthracnose': 'anthracnose',
    'fruit_fly': 'fruit_fly',
    'healthy_guava': 'healthy_guava'
}

splits = ['train', 'val', 'test']
split_ratios = {'train': 0.7, 'val': 0.2, 'test': 0.1}

print("Iniciando a mistura dos datasets com divisão 70/20/10...\n")

all_classes = list(banana_classes.values()) + list(guava_classes.values())
for split in splits:
    for class_name in all_classes:
        class_dir = os.path.join(out_dir, split, class_name)
        os.makedirs(class_dir, exist_ok=True)

print("=== Processando Banana Dataset (OriginalSet + AugmentedSet) ===")

for banana_orig, banana_dest in banana_classes.items():
    all_images = []

    original_path = os.path.join(banana_root, 'OriginalSet', banana_orig)
    if os.path.exists(original_path):
        for img_file in os.listdir(original_path):
            if img_file.lower().endswith(('.jpeg', '.jpg', '.png')):
                all_images.append(('original', os.path.join(
                    original_path, img_file), img_file))

    augmented_path = os.path.join(banana_root, 'AugmentedSet', banana_orig)
    if os.path.exists(augmented_path):
        for img_file in os.listdir(augmented_path):
            if img_file.lower().endswith(('.jpeg', '.jpg', '.png')):
                all_images.append(('augmented', os.path.join(
                    augmented_path, img_file), img_file))

    random.shuffle(all_images)

    total = len(all_images)
    train_end = int(total * split_ratios['train'])
    val_end = train_end + int(total * split_ratios['val'])

    splits_images = {
        'train': all_images[:train_end],
        'val': all_images[train_end:val_end],
        'test': all_images[val_end:]
    }

    for split, split_images in splits_images.items():
        dest_path = os.path.join(out_dir, split, banana_dest)

        for source_type, src_file, img_file in split_images:
            dest_file = os.path.join(
                dest_path, f'banana_{source_type}_{img_file}')
            shutil.copy2(src_file, dest_file)

        print(f'✓ {banana_dest} ({split}): {len(split_images)} imagens')

print("\n=== Processando Guava Dataset ===")
for guava_orig, guava_dest in guava_classes.items():
    for split in splits:
        src_path = os.path.join(guava_base, split, guava_orig)
        dest_path = os.path.join(out_dir, split, guava_dest)

        if os.path.exists(src_path):
            # Copiar arquivos
            images = [f for f in os.listdir(src_path)
                      if f.lower().endswith(('.jpeg', '.jpg', '.png'))]

            for img_file in images:
                src_file = os.path.join(src_path, img_file)
                dest_file = os.path.join(dest_path, f'guava_{img_file}')
                shutil.copy2(src_file, dest_file)

            print(f'✓ {guava_dest} ({split}): {len(images)} imagens')

print(f"\n✅ Dataset misto criado com sucesso em: {out_dir}")

print("\n" + "="*60)
print("ESTATÍSTICAS DO DATASET MISTO")
print("="*60)

total_per_split = {}
for split in splits:
    split_dir = os.path.join(out_dir, split)
    print(f"\n{split.upper()}:")
    split_total = 0

    for class_name in sorted(os.listdir(split_dir)):
        class_path = os.path.join(split_dir, class_name)
        if os.path.isdir(class_path):
            num_images = len([f for f in os.listdir(class_path)
                              if f.lower().endswith(('.jpeg', '.jpg', '.png'))])
            print(f"  {class_name:20s}: {num_images:4d} imagens")
            split_total += num_images

    total_per_split[split] = split_total
    print(f"  {'TOTAL':20s}: {split_total:4d} imagens")

grand_total = sum(total_per_split.values())
print(f"\n{'='*60}")
print("PROPORÇÕES:")
print(f"{'='*60}")
for split in splits:
    proportion = (total_per_split[split] / grand_total) * 100
    print(
        f"  {split:6s}: {total_per_split[split]:4d} imagens ({proportion:.1f}%)")
print(f"  TOTAL : {grand_total:4d} imagens (100.0%)")
