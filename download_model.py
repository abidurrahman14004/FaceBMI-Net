"""
Optional script to download model file from external storage.
Use this if model is stored in cloud storage (S3, Google Drive, etc.)
"""
import os
import requests
from pathlib import Path

def download_model(model_url=None, save_path="models/hybrid_model_v2.pth"):
    """
    Download model file from URL if not present locally.
    
    Args:
        model_url: URL to download model from (set via environment variable MODEL_URL)
        save_path: Local path to save model
    """
    if os.path.exists(save_path):
        print(f"Model already exists at {save_path}")
        return True
    
    # Get URL from environment or parameter
    url = model_url or os.environ.get('MODEL_URL')
    if not url:
        print("No model URL provided. Set MODEL_URL environment variable or pass model_url parameter.")
        return False
    
    print(f"Downloading model from {url}...")
    try:
        # Create directory if it doesn't exist
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rDownloaded: {percent:.1f}%", end='', flush=True)
        
        print(f"\nModel downloaded successfully to {save_path}")
        return True
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

if __name__ == '__main__':
    download_model()
