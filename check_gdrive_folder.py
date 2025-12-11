"""
Check Google Drive Folder Contents
===================================
Lists all files in a Google Drive folder to see what's available
"""

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import os

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def authenticate_drive():
    """Authenticate and return Google Drive service"""
    if not os.path.exists('credentials.json'):
        print("❌ credentials.json not found!")
        return None
    
    flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
    creds = flow.run_local_server(port=0)
    service = build('drive', 'v3', credentials=creds)
    return service

def list_folder_contents(service, folder_id):
    """List everything in the folder"""
    query = f"'{folder_id}' in parents"
    results = service.files().list(
        q=query,
        fields="files(id, name, mimeType, size)",
        pageSize=100
    ).execute()
    
    items = results.get('files', [])
    return items

# Main
service = authenticate_drive()
if service:
    folder_id = "1IiEz1khC8RRNmNJD2T4HWaPv6J5fLC-Y"
    
    print(f"\n📁 Folder contents:")
    print("="*60)
    
    items = list_folder_contents(service, folder_id)
    
    if not items:
        print("❌ Folder is empty or not accessible")
    else:
        print(f"Found {len(items)} items:\n")
        for item in items:
            size = f"{int(item.get('size', 0))/1024:.1f} KB" if 'size' in item else "N/A"
            mime_type = item['mimeType']
            is_folder = "📁" if 'folder' in mime_type else "📄"
            print(f"{is_folder} {item['name']}")
            print(f"   Type: {mime_type}")
            print(f"   Size: {size}")
            print(f"   ID: {item['id']}")
            print()
