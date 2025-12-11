"""
Google Drive Image Analysis with Claude
========================================
Analyzes images from Google Drive without permanent local storage.
Downloads images temporarily, analyzes with Claude, then cleans up.
"""

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import os
import io
import sys
import tempfile
import json
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))
from src.utils.claude_assistant import ClaudeAssistant
from src.utils.load_env import load_env

# Google Drive API setup
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def authenticate_drive():
    """Authenticate and return Google Drive service"""
    if not os.path.exists('credentials.json'):
        print("❌ credentials.json not found!")
        print("\nSetup instructions:")
        print("1. Go to https://console.cloud.google.com/")
        print("2. Create a new project or select existing")
        print("3. Enable Google Drive API")
        print("4. Create OAuth 2.0 credentials (Desktop app)")
        print("5. Download credentials.json to this directory")
        return None
    
    flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
    creds = flow.run_local_server(port=0)
    service = build('drive', 'v3', credentials=creds)
    return service


def list_images_in_folder(service, folder_id):
    """List all image files in Google Drive folder"""
    query = f"'{folder_id}' in parents and (mimeType contains 'image/')"
    results = service.files().list(
        q=query,
        fields="files(id, name, mimeType, size)",
        pageSize=100
    ).execute()
    
    items = results.get('files', [])
    return items


def download_image_to_memory(service, file_id):
    """Download image to memory buffer"""
    request = service.files().get_media(fileId=file_id)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    
    done = False
    while not done:
        status, done = downloader.next_chunk()
    
    buffer.seek(0)
    return buffer


def analyze_images_from_drive(folder_id, max_images=10, save_results=True):
    """
    Analyze images from Google Drive folder using Claude
    
    Args:
        folder_id: Google Drive folder ID
        max_images: Maximum number of images to analyze
        save_results: Whether to save analysis results to JSON
    """
    load_env()
    
    print("="*60)
    print("Google Drive Image Analysis with Claude")
    print("="*60)
    
    # Authenticate
    print("\n1. Authenticating with Google Drive...")
    service = authenticate_drive()
    if not service:
        return
    
    # Initialize Claude
    print("2. Initializing Claude assistant...")
    try:
        assistant = ClaudeAssistant()
        print(f"   ✓ Using model: {assistant.model}")
    except Exception as e:
        print(f"   ❌ Claude initialization failed: {e}")
        return
    
    # List images
    print(f"\n3. Listing images from folder...")
    images = list_images_in_folder(service, folder_id)
    
    if not images:
        print("   ❌ No images found in folder!")
        print("   Make sure:")
        print("   - Folder ID is correct")
        print("   - Folder is shared with your Google account")
        print("   - Folder contains image files")
        return
    
    print(f"   ✓ Found {len(images)} images")
    
    # Analyze images
    results = []
    images_to_analyze = min(len(images), max_images)
    
    print(f"\n4. Analyzing {images_to_analyze} images with Claude...")
    print("-"*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, image in enumerate(images[:max_images], 1):
            print(f"\n[{i}/{images_to_analyze}] {image['name']}")
            
            try:
                # Download to temp file
                temp_path = os.path.join(temp_dir, image['name'])
                print(f"   Downloading... ({int(image.get('size', 0))/1024:.1f} KB)")
                
                request = service.files().get_media(fileId=image['id'])
                with io.FileIO(temp_path, 'wb') as fh:
                    downloader = MediaIoBaseDownload(fh, request)
                    done = False
                    while not done:
                        status, done = downloader.next_chunk()
                
                print("   Analyzing with Claude...")
                
                # Analyze with Claude
                analysis = assistant.analyze_image_for_annotation(
                    temp_path,
                    task_description="Identify objects, regions, and semantic segments for autonomous driving/urban scene understanding"
                )
                
                # Store result
                result = {
                    'filename': image['name'],
                    'file_id': image['id'],
                    'size_kb': int(image.get('size', 0)) / 1024,
                    'analysis': analysis
                }
                results.append(result)
                
                # Show preview
                preview = analysis[:200] + "..." if len(analysis) > 200 else analysis
                print(f"   ✓ Analysis complete")
                print(f"   Preview: {preview}")
                
                # File deleted automatically when leaving temp_dir context
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                results.append({
                    'filename': image['name'],
                    'file_id': image['id'],
                    'error': str(e)
                })
    
    # Save results
    if save_results and results:
        output_file = 'gdrive_claude_analysis.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n{'='*60}")
        print(f"✅ Analysis complete!")
        print(f"   Results saved to: {output_file}")
        print(f"   Images analyzed: {len([r for r in results if 'error' not in r])}/{len(results)}")
        print(f"   Errors: {len([r for r in results if 'error' in r])}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze Google Drive images with Claude')
    parser.add_argument('folder_id', help='Google Drive folder ID')
    parser.add_argument('--max-images', type=int, default=10, 
                       help='Maximum number of images to analyze (default: 10)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results to file')
    
    args = parser.parse_args()
    
    analyze_images_from_drive(
        args.folder_id,
        max_images=args.max_images,
        save_results=not args.no_save
    )
