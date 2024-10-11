import os
import sys
import asyncio
import aiohttp
import h5py
import torch
import numpy as np
from PIL import Image
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor
from tqdm.asyncio import tqdm
import pickle
import io
from huggingface_hub import HfApi, HfFolder, hf_hub_download, create_repo
from huggingface_hub import login
login(token="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")

# Add the tokenizer's directory to the system path
sys.path.insert(0, '1d-tokenizer')
from modeling.titok import TiTok  # Ensure this path is correct

# Configuration Constants
DATASET_NAME = 'laion/aesthetics_v2_4.75'
SPLIT = 'train'
pp = 11
HDF5_PATH = f'laion_encoded{pp}.hdf5'
CHECKPOINT_PATH = f'checkpoint{pp}.pkl'
HUGGINGFACE_REPO = f'irotem98/laion_encoded{pp}'
BATCH_SIZE = 1000
MAX_CONCURRENT_DOWNLOADS = 50  # Adjust based on your network and system
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'mps'
TOKENIZER_MODEL = "yucornetto/tokenizer_titok_l32_imagenet"
UPLOAD_THRESHOLD = 1000  # Upload after 10,000 rows processed

# Image dimension thresholds
MIN_WIDTH = 32
MAX_WIDTH = 1500
MIN_HEIGHT = 32
MAX_HEIGHT = 1500
MIN_ASPECT_RATIO = 0.5
MAX_ASPECT_RATIO = 2.0

# Initialize the TiTok Tokenizer
def initialize_tokenizer(device):
    tokenizer = TiTok.from_pretrained(TOKENIZER_MODEL)
    tokenizer.eval()
    tokenizer.requires_grad_(False)
    tokenizer.to(device)
    return tokenizer

# Function to encode an image using the tokenizer
def encode_image(image, tokenizer, device):
    try:
        image = torch.from_numpy(np.array(image).astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
        encoded_tokens = tokenizer.encode(image.to(device))[1]["min_encoding_indices"]
        return encoded_tokens.cpu().numpy().astype(np.int16).squeeze()
    except Exception as e:
        return None

# Asynchronous function to download an image
async def download_image(session, url):
    try:
        async with session.get(url, timeout=10) as response:
            if response.status != 200:
                return None
            data = await response.read()
            return data
    except:
        return None

# Function to process a single row: download, validate, resize, encode
async def process_row(session, row, tokenizer, device):
    url = row['URL']
    text = row['TEXT']
    width = row['WIDTH']
    height = row['HEIGHT']

    # Filter based on WIDTH, HEIGHT, and aspect ratio before downloading
    if not (MIN_WIDTH <= width <= MAX_WIDTH and MIN_HEIGHT <= height <= MAX_HEIGHT):
        return None

    aspect_ratio = width / height
    if not (MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO):
        return None

    # Filter out empty texts or texts longer than 150 characters
    if not text or len(text) > 150:
        return None

    # Download the image
    data = await download_image(session, url)
    if not data:
        return None

    try:
        image = Image.open(io.BytesIO(data)).convert('RGB')
    except:
        return None

    # Resize the image to 256x256
    try:
        image = image.resize((256, 256))
    except:
        return None

    # Encode the image
    encoded = encode_image(image, tokenizer, device)
    return encoded

# Function to save the current progress as a checkpoint
def save_checkpoint(checkpoint_path, index):
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(index, f)

# Function to load the checkpoint
def load_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)
    return 0

# Function to download a checkpoint from Hugging Face
def download_checkpoint_from_hf(repo_id, checkpoint_filename, local_path):
    try:
        hf_hub_download(repo_id, checkpoint_filename, local_dir=local_path)
    except Exception as e:
        print(f"Checkpoint not found on Hugging Face: {e}")
        return 0  # Continue without the checkpoint

import os
import h5py

# Function to initialize or load the HDF5 file
def initialize_hdf5(hdf5_path):
    if os.path.exists(hdf5_path):
        try:
            # Attempt to open the file
            hdf5_file = h5py.File(hdf5_path, 'a')
        except OSError as e:
            print(f"Error opening HDF5 file: {e}")
            print("File may be corrupted. Deleting and starting fresh.")
            os.remove(hdf5_path)  # Delete the corrupted file
            hdf5_file = h5py.File(hdf5_path, 'w')
            hdf5_file.create_dataset('encoded', shape=(0, 32), maxshape=(None, 32), dtype='int16', chunks=True)
    else:
        # Create a new file if it doesn't exist
        hdf5_file = h5py.File(hdf5_path, 'w')
        hdf5_file.create_dataset('encoded', shape=(0, 32), maxshape=(None, 32), dtype='int16', chunks=True)

    return hdf5_file

# Function to create a dataset repository if it doesn't exist
def create_hf_repo(repo_id):
    api = HfApi()
    token = HfFolder.get_token()
    if not token:
        print("Hugging Face token not found. Please login using `huggingface-cli login`.")
        return
    try:
        # Create the repo if it doesn't exist
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True, token=token)
        print(f"Dataset repository {repo_id} created (or already exists).")
    except Exception as e:
        print(f"Error creating the repository: {e}")

# Function to upload the HDF5 file to Hugging Face Hub
def upload_to_hf(hdf5_path, repo_id):
    api = HfApi()
    token = HfFolder.get_token()
    if not token:
        print("Hugging Face token not found. Please login using `huggingface-cli login`.")
        return
    try:
        # Upload the file to the Hugging Face dataset repository
        print(f"Uploading {hdf5_path} to Hugging Face repo {repo_id}...")
        api.upload_file(
            path_or_fileobj=hdf5_path,
            path_in_repo=os.path.basename(hdf5_path),
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
        print(f"Upload completed: {hdf5_path}")
    except Exception as e:
        print(f"Error uploading the file: {e}")

# Main asynchronous function
async def main():
    # Initialize tokenizer
    tokenizer = initialize_tokenizer(DEVICE)

    # Load the dataset in streaming mode
    dataset = load_dataset(DATASET_NAME, split=SPLIT, streaming=True)

    # Convert the dataset to an iterator
    dataset_iterator = iter(dataset)

    # Create the Hugging Face repository if it doesn't exist
    create_hf_repo(HUGGINGFACE_REPO)

    # Try to download the checkpoint from Hugging Face
    checkpoint_downloaded = download_checkpoint_from_hf(HUGGINGFACE_REPO, CHECKPOINT_PATH, '.')

    # Load checkpoint, either from local file or downloaded one
    start_index = load_checkpoint(CHECKPOINT_PATH)

    # Initialize HDF5 file
    hdf5_file = initialize_hdf5(HDF5_PATH)
    encoded_dataset = hdf5_file['encoded']
    current_index = start_index
    last_upload_count = current_index  # Track the last count when we uploaded

    # Create an asynchronous HTTP session
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_DOWNLOADS)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Initialize progress bar
        total_rows = 956_000_000  # Total rows in the dataset
        pbar = tqdm(total=total_rows, initial=current_index, desc="Processing Images")

        # Skip already processed rows
        if start_index > 0:
            for _ in range(start_index):
                try:
                    next(dataset_iterator)  # Use the iterator instead of dataset directly
                except StopIteration:
                    print("Reached end of dataset during checkpoint skip.")
                    hdf5_file.close()
                    return

        batch = []
        for row in dataset_iterator:  # Use iterator here
            batch.append(row)
            if len(batch) >= BATCH_SIZE:
                # Process the current batch
                tasks = [process_row(session, r, tokenizer, DEVICE) for r in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Filter out failed encodings
                encoded = [res for res in results if isinstance(res, np.ndarray)]
                if encoded:
                    encoded = np.vstack(encoded)
                    # Append to HDF5
                    encoded_dataset.resize((current_index + encoded.shape[0], 32))
                    encoded_dataset[current_index:current_index + encoded.shape[0]] = encoded
                    current_index += encoded.shape[0]
                    pbar.update(len(batch))

                # Save checkpoint
                save_checkpoint(CHECKPOINT_PATH, current_index)

                # Check if we've processed more than 10,000 rows since the last upload
                if current_index - last_upload_count >= UPLOAD_THRESHOLD:
                    hdf5_file.flush()  # Ensure data is written to the file
                    hdf5_file.close()  # Close before upload
                    upload_to_hf(HDF5_PATH, HUGGINGFACE_REPO)
                    hdf5_file = initialize_hdf5(HDF5_PATH)  # Reopen after upload
                    encoded_dataset = hdf5_file['encoded']  # Reinitialize the dataset object
                    last_upload_count = current_index  # Reset the upload counter

                # Clear the batch
                batch = []

        # Process any remaining rows
        if batch:
            tasks = [process_row(session, r, tokenizer, DEVICE) for r in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            encoded = [res for res in results if isinstance(res, np.ndarray)]
            if encoded:
                encoded = np.vstack(encoded)
                encoded_dataset.resize((current_index + encoded.shape[0], 32))
                encoded_dataset[current_index:current_index + encoded.shape[0]] = encoded
                current_index += encoded.shape[0]
                pbar.update(len(batch))
            save_checkpoint(CHECKPOINT_PATH, current_index)

        pbar.close()
        hdf5_file.close()

    # Final upload after all processing
    upload_to_hf(HDF5_PATH, HUGGINGFACE_REPO)
    print("Upload to Hugging Face Hub completed.")

if __name__ == '__main__':
    asyncio.run(main())
