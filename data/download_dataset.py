import kagglehub
import certifi
import os

# Solução para problema de certificados SSL
# Configura o caminho dos certificados do certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# Download latest version
path = kagglehub.dataset_download("muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten")

print("Path to dataset files:", path)