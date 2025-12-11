"""
Environment Configuration Loader
================================
Loads environment variables from .env file
"""

import os
from pathlib import Path


def load_env():
    """Load environment variables from .env file"""
    env_file = Path(__file__).parent.parent.parent / '.env'
    
    if not env_file.exists():
        print(f"⚠ No .env file found at {env_file}")
        return
    
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Parse KEY=VALUE
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Don't overwrite existing env vars
                if key not in os.environ:
                    os.environ[key] = value
                    print(f"✓ Loaded {key} from .env")


if __name__ == '__main__':
    load_env()
    
    # Test if API key is loaded
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if api_key and api_key != 'your-api-key-here':
        print(f"\n✅ API key loaded successfully (length: {len(api_key)})")
    else:
        print("\n⚠ Please set your API key in .env file")
