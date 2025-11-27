import random
import os
from PIL import Image

out_dir = os.path.join('data', 'mixed_dataset')
TARGET_SIZE = (224, 224)

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
    original_images = []
    augmented_images = []

    original_path = os.path.join(banana_root, 'OriginalSet', banana_orig)
    if os.path.exists(original_path):
        for img_file in os.listdir(original_path):
            if img_file.lower().endswith(('.jpeg', '.jpg', '.png')):
                original_images.append(os.path.join(original_path, img_file))

    augmented_path = os.path.join(banana_root, 'AugmentedSet', banana_orig)
    if os.path.exists(augmented_path):
        for img_file in os.listdir(augmented_path):
            if img_file.lower().endswith(('.jpeg', '.jpg', '.png')):
                augmented_images.append(os.path.join(augmented_path, img_file))

    random.shuffle(original_images)

    total = len(original_images)
    train_end = int(total * split_ratios['train'])
    val_end = train_end + int(total * split_ratios['val'])

    train_originals = original_images[:train_end]
    val_originals = original_images[train_end:val_end]
    test_originals = original_images[val_end:]

    dest_train = os.path.join(out_dir, 'train', banana_dest)
    for src_file in train_originals:
        img = Image.open(src_file).convert('RGB')
        img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        img_file = os.path.basename(src_file)
        dest_file = os.path.join(dest_train, f'banana_original_{img_file}')
        img.save(dest_file)

    for src_file in augmented_images:
        img = Image.open(src_file).convert('RGB')
        img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        img_file = os.path.basename(src_file)
        dest_file = os.path.join(dest_train, f'banana_augmented_{img_file}')
        img.save(dest_file)

    print(
        f'✓ {banana_dest} (train): {len(train_originals) + len(augmented_images)} imagens')

    dest_val = os.path.join(out_dir, 'val', banana_dest)
    for src_file in val_originals:
        img = Image.open(src_file).convert('RGB')
        img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        img_file = os.path.basename(src_file)
        dest_file = os.path.join(dest_val, f'banana_original_{img_file}')
        img.save(dest_file)

    print(f'✓ {banana_dest} (val): {len(val_originals)} imagens')

    dest_test = os.path.join(out_dir, 'test', banana_dest)
    for src_file in test_originals:
        img = Image.open(src_file).convert('RGB')
        img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        img_file = os.path.basename(src_file)
        dest_file = os.path.join(dest_test, f'banana_original_{img_file}')
        img.save(dest_file)

    print(f'✓ {banana_dest} (test): {len(test_originals)} imagens')

print("\n=== Processando Guava Dataset ===")
for guava_orig, guava_dest in guava_classes.items():
    train_src = os.path.join(guava_base, 'train', guava_orig)
    train_dest = os.path.join(out_dir, 'train', guava_dest)

    if os.path.exists(train_src):
        train_images = [f for f in os.listdir(train_src)
                        if f.lower().endswith(('.jpeg', '.jpg', '.png'))]

        for img_file in train_images:
            src_file = os.path.join(train_src, img_file)
            img = Image.open(src_file).convert('RGB')
            img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
            dest_file = os.path.join(train_dest, f'guava_{img_file}')
            img.save(dest_file)

        print(f'✓ {guava_dest} (train): {len(train_images)} imagens')

    val_src = os.path.join(guava_base, 'val', guava_orig)
    val_dest = os.path.join(out_dir, 'val', guava_dest)

    if os.path.exists(val_src):
        val_images = [f for f in os.listdir(val_src)
                      if f.lower().endswith(('.jpeg', '.jpg', '.png'))]

        for img_file in val_images:
            src_file = os.path.join(val_src, img_file)
            img = Image.open(src_file).convert('RGB')
            img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
            dest_file = os.path.join(val_dest, f'guava_{img_file}')
            img.save(dest_file)

        print(f'✓ {guava_dest} (val): {len(val_images)} imagens')

    test_src = os.path.join(guava_base, 'test', guava_orig)
    test_dest = os.path.join(out_dir, 'test', guava_dest)

    if os.path.exists(test_src):
        test_images = [f for f in os.listdir(test_src)
                       if f.lower().endswith(('.jpeg', '.jpg', '.png'))]

        for img_file in test_images:
            src_file = os.path.join(test_src, img_file)
            img = Image.open(src_file).convert('RGB')
            img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
            dest_file = os.path.join(test_dest, f'guava_{img_file}')
            img.save(dest_file)

        print(f'✓ {guava_dest} (test): {len(test_images)} imagens')

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
