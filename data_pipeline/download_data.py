"""
Download PaySim dataset using Kaggle API.
"""

import os
import logging
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_paysim_data():
    """
    Download the PaySim dataset using Kaggle API.
    """
    # Create data/raw directory if it doesn't exist
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = data_dir / "PS_20174392719_1491204439457_log.csv"
    
    # Check if file already exists and is valid
    if output_path.exists():
        file_size = os.path.getsize(output_path)
        if file_size > 1000:  # Valid file should be larger than 1KB
            logger.info(f"Dataset already exists at {output_path}")
            return str(output_path)
    
    try:
        # Initialize Kaggle API
        api = KaggleApi()
        api.authenticate()
        
        logger.info("Downloading PaySim dataset from Kaggle...")
        
        # Download the dataset
        api.dataset_download_files('ealaxi/paysim1', path=str(data_dir), unzip=True)
        
        # Find the CSV file in the downloaded files
        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            # Rename to expected filename
            downloaded_file = csv_files[0]
            if downloaded_file.name != output_path.name:
                downloaded_file.rename(output_path)
            
            logger.info(f"Successfully downloaded dataset to {output_path}")
            
            # Log file size
            file_size = os.path.getsize(output_path)
            logger.info(f"File size: {file_size / (1024*1024):.2f} MB")
            
            return str(output_path)
        else:
            logger.error("No CSV files found in downloaded dataset")
            raise Exception("Dataset download failed - no CSV files found")
            
    except Exception as e:
        logger.error(f"Failed to download from Kaggle: {e}")
        raise Exception(f"Unable to download authentic PaySim dataset: {e}")

if __name__ == "__main__":
    download_paysim_data()