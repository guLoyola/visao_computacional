import kagglehub
import os

os.makedirs('./data', exist_ok=True)

os.environ['KAGGLE_DATA_DIR'] = './data'

print("Baixando datasets...")

banana_path = kagglehub.dataset_download('shifatearman/bananalsd')
print(f"BananaLSD baixado em: {banana_path}")

guava_path = kagglehub.dataset_download('asadullahgalib/guava-disease-dataset')
print(f"Guava Disease baixado em: {guava_path}")

print("\nDownloads concluídos!")