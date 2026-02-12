
import os
import urllib.request
import tarfile
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path("c:/Users/Rian/Desktop/SERASA_DOC") # Hardcoded for safety locally
MODELS_DIR = PROJECT_ROOT / "src/infrastructure/ocr/models"
EASYOCR_DIR = MODELS_DIR / "easyocr"
PADDLE_DIR = MODELS_DIR / "paddle"

def download_file(url, dest_path):
    print(f"Downloading {url}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"Saved to {dest_path}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def setup_paddle_v3():
    print("\n--- Downloading PaddleOCR v3 (English) ---")
    
    # Mapping: local_folder -> url
    # Note: PaddleOCR expects specific folder names usually, but we can pass explicit paths to the engine.
    models = {
        "en_PP-OCRv3_det_infer": "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar",
        "en_PP-OCRv3_rec_infer": "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar",
        "ch_ppocr_mobile_v2.0_cls_infer": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar"
    }

    PADDLE_DIR.mkdir(parents=True, exist_ok=True)
    
    for name, url in models.items():
        tar_path = PADDLE_DIR / (name + ".tar")
        extract_path = PADDLE_DIR / name
        
        if not extract_path.exists():
            if download_file(url, tar_path):
                print(f"Extracting {tar_path}...")
                with tarfile.open(tar_path) as tar:
                    tar.extractall(path=PADDLE_DIR)
                os.remove(tar_path)
        else:
            print(f"Model {name} already exists.")

if __name__ == "__main__":
    setup_paddle_v3()
