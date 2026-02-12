
import os
import shutil
import urllib.request
import tarfile
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
EASYOCR_DIR = MODELS_DIR / "easyocr"
PADDLE_DIR = MODELS_DIR / "paddleocr"

def download_file(url, dest_path):
    print(f"Downloading {url}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"Saved to {dest_path}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def setup_easyocr():
    print("\n--- Setting up EasyOCR models ---")
    EASYOCR_DIR.mkdir(parents=True, exist_ok=True)
    
    # EasyOCR automatically downloads to ~/.EasyOCR usually.
    # We will use the library to download to our specific dir.
    try:
        import easyocr
        print(f"Downloading EasyOCR 'en' model to {EASYOCR_DIR}...")
        # verify_ssl=False sometimes needed if certs are old, but try default first
        reader = easyocr.Reader(['en'], download_enabled=True, model_storage_directory=str(EASYOCR_DIR), verbose=True)
        print("EasyOCR models ready.")
    except ImportError:
        print("easyocr not installed run pip install easyocr")
    except Exception as e:
        print(f"Failed to setup EasyOCR: {e}")

def setup_paddleocr():
    print("\n--- Setting up PaddleOCR models ---")
    PADDLE_DIR.mkdir(parents=True, exist_ok=True)

    # PaddleOCR v3/v4 models (lightweight English)
    # Detection: en_PP-OCRv3_det_infer
    # Classification: ch_ppocr_mobile_v2.0_cls_infer
    # Recognition: en_PP-OCRv4_rec_infer (or v3)
    
    models = {
        "det": "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar",
        "rec": "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar",
        "cls": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar"
    }

    for name, url in models.items():
        tar_name = url.split("/")[-1]
        tar_path = PADDLE_DIR / tar_name
        extract_dir = PADDLE_DIR / name

        if not extract_dir.exists():
            download_file(url, tar_path)
            if tar_path.exists():
                print(f"Extracting {tar_name}...")
                with tarfile.open(tar_path) as tar:
                    tar.extractall(path=PADDLE_DIR)
                
                # Rename the extracted folder to simple name (det, rec, cls)
                # The tar usually extracts to a folder name like "en_PP-OCRv3_det_infer"
                extracted_name = tar_name.replace(".tar", "")
                if (PADDLE_DIR / extracted_name).exists():
                     # Move contents or rename
                     if extract_dir.exists():
                         shutil.rmtree(extract_dir)
                     shutil.move(str(PADDLE_DIR / extracted_name), str(extract_dir))
                     print(f"Moved to {extract_dir}")
                
                # Cleanup tar
                os.remove(tar_path)
        else:
            print(f"Model {name} already exists at {extract_dir}")

def main():
    MODELS_DIR.mkdir(exist_ok=True)
    setup_easyocr()
    setup_paddleocr()
    print("\nAll models downloaded to /models. Ready for Docker build.")

if __name__ == "__main__":
    main()
