"""
Google Drive → Claude Analysis (Memory Only)
=============================================
Downloads images to RAM only, never touches disk.
Analyzes with Claude, then immediately discards.
"""

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import os
import io
import sys
import json
import base64
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from src.utils.load_env import load_env
import anthropic

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def authenticate_drive():
    """Authenticate with Google Drive"""
    if not os.path.exists('credentials.json'):
        print("❌ credentials.json not found!")
        print("Get it from: https://console.cloud.google.com/")
        return None
    
    flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
    creds = flow.run_local_server(port=0)
    return build('drive', 'v3', credentials=creds)


def download_to_memory(service, file_id):
    """Download image directly to RAM (BytesIO)"""
    request = service.files().get_media(fileId=file_id)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    
    done = False
    while not done:
        status, done = downloader.next_chunk()
    
    buffer.seek(0)
    return buffer


def encode_image_from_memory(image_buffer, format='JPEG'):
    """Convert image buffer to base64 for Claude"""
    # Read image from buffer
    img = Image.open(image_buffer)
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Encode to base64
    output_buffer = io.BytesIO()
    img.save(output_buffer, format=format)
    output_buffer.seek(0)
    
    return base64.standard_b64encode(output_buffer.read()).decode('utf-8')


def analyze_with_claude_vision(image_base64, filename):
    """Analyze image using Claude Vision API"""
    load_env()
    client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    
    response = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_base64
                    }
                },
                {
                    "type": "text",
                    "text": f"""Analyze this image for semantic segmentation.

Identify:
1. All distinct objects and regions
2. Boundaries between different semantic areas
3. Suggested class labels for autonomous driving/urban scenes
4. Any challenging areas for segmentation

Image: {filename}"""
                }
            ]
        }]
    )
    
    return response.content[0].text


def analyze_gdrive_images_memory_only(folder_id, max_images=10):
    """
    Analyze Google Drive images using only RAM - NO disk storage
    
    Process:
    1. Download image to RAM (BytesIO)
    2. Convert to base64 in memory
    3. Send to Claude Vision API
    4. Collect results
    5. All data discarded (no files saved)
    """
    print("="*60)
    print("Google Drive → Claude (Memory-Only Analysis)")
    print("="*60)
    
    # Authenticate
    print("\n1. Connecting to Google Drive...")
    service = authenticate_drive()
    if not service:
        return
    
    # List images
    print("2. Listing images...")
    query = f"'{folder_id}' in parents and mimeType contains 'image/'"
    results = service.files().list(
        q=query,
        fields="files(id, name, mimeType, size)",
        pageSize=max_images
    ).execute()
    
    images = results.get('files', [])
    
    if not images:
        print("   ❌ No images found!")
        print("   - Check folder ID")
        print("   - Ensure folder has image files")
        return
    
    print(f"   ✓ Found {len(images)} images")
    
    # Analyze each image
    print(f"\n3. Analyzing (memory-only, no disk writes)...")
    print("-"*60)
    
    analyses = []
    
    for i, img_file in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] {img_file['name']}")
        
        try:
            # Download to RAM only
            print("   ⬇ Downloading to memory...")
            size_mb = int(img_file.get('size', 0)) / (1024*1024)
            print(f"   📊 Size: {size_mb:.2f} MB")
            
            image_buffer = download_to_memory(service, img_file['id'])
            
            # Convert to base64 in memory
            print("   🔄 Encoding in memory...")
            image_base64 = encode_image_from_memory(image_buffer, format='JPEG')
            
            # Clear buffer immediately
            image_buffer.close()
            
            # Analyze with Claude
            print("   🤖 Analyzing with Claude...")
            analysis = analyze_with_claude_vision(image_base64, img_file['name'])
            
            # Clear base64 data
            del image_base64
            
            # Store result
            result = {
                'filename': img_file['name'],
                'file_id': img_file['id'],
                'size_mb': size_mb,
                'analysis': analysis
            }
            analyses.append(result)
            
            # Show preview
            preview = analysis[:150].replace('\n', ' ')
            print(f"   ✅ Complete")
            print(f"   📝 Preview: {preview}...")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            analyses.append({
                'filename': img_file['name'],
                'error': str(e)
            })
    
    # Save results (only the text analysis, not images)
    print(f"\n{'='*60}")
    output_file = 'gdrive_analysis_results.json'
    with open(output_file, 'w') as f:
        json.dump(analyses, f, indent=2)
    
    successful = len([a for a in analyses if 'error' not in a])
    print(f"✅ Analysis complete!")
    print(f"   Success: {successful}/{len(analyses)}")
    print(f"   Results: {output_file}")
    print(f"   💾 No images saved to disk (memory-only processing)")
    
    return analyses


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze Google Drive images with Claude (memory-only, no disk storage)'
    )
    parser.add_argument('folder_id', help='Google Drive folder ID')
    parser.add_argument('--max-images', type=int, default=10,
                       help='Maximum images to analyze (default: 10)')
    
    args = parser.parse_args()
    
    print(f"\n⚠️  Note: Images processed in RAM only")
    print(f"   No image files will be saved to disk\n")
    
    analyze_gdrive_images_memory_only(args.folder_id, args.max_images)
