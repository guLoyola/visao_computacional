import shutil
from pathlib import Path

DATASET_BASE = Path(__file__).parent.parent.parent / "data" / \
    "muhammad0subhan" / "fruit-and-vegetable-disease-healthy-vs-rotten"
DATASET_DIR = DATASET_BASE / "versions" / \
    "1" / "Fruit And Vegetable Diseases Dataset"

KEEP_VEGETABLES = [
    "Carrot__Healthy",
    "Carrot__Rotten",
    "Tomato__Healthy",
    "Tomato__Rotten",
    "Potato__Healthy",
    "Potato__Rotten"
]


def clean_dataset():
    for veg in KEEP_VEGETABLES:
        print(f"  - {veg}")

    all_folders = [f for f in DATASET_DIR.iterdir() if f.is_dir()]

    removed_count = 0
    removed_size = 0
    kept_count = 0

    print(f"\n{'='*60}")
    print("Processando pastas...")
    print(f"{'='*60}\n")

    for folder in all_folders:
        folder_name = folder.name

        if folder_name in KEEP_VEGETABLES:
            image_count = len(list(folder.glob("*.*")))
            kept_count += 1
            print(f" MANTIDO: {folder_name} ({image_count} imagens)")
        else:
            folder_size = sum(
                f.stat().st_size for f in folder.rglob("*") if f.is_file())
            removed_size += folder_size

            # Remover a pasta
            shutil.rmtree(folder)
            removed_count += 1
            print(
                f" REMOVIDO: {folder_name} ({folder_size / (1024*1024):.2f} MB)")

    print(f"\n{'='*60}")
    print("Resumo da limpeza:")
    print(f"{'='*60}")
    print(f"Pastas mantidas:  {kept_count}")
    print(f"Pastas removidas: {removed_count}")
    print(f"Espa�o liberado:  {removed_size / (1024*1024*1024):.2f} GB")
    print(f"\n Limpeza conclu�da com sucesso!")

    print(f"\n{'='*60}")
    print("Estat�sticas finais:")
    print(f"{'='*60}")
    for veg in KEEP_VEGETABLES:
        veg_path = DATASET_DIR / veg
        if veg_path.exists():
            image_count = len(list(veg_path.glob("*.*")))
            print(f"{veg}: {image_count} imagens")


if __name__ == "__main__":
    clean_dataset()
