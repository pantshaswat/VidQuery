import os
from pathlib import Path
import hashlib

def get_video_id(video_path: str) -> str:
    """Extract video ID from file path or filename"""
    path = Path(video_path)
    filename = path.stem  # Get filename without extension
    
    # If filename is already a UUID (from your upload system), use it
    if len(filename) == 36 and filename.count('-') == 4:
        return filename
    
    # Otherwise, generate a consistent ID from the full path
    # This is what was causing the mismatch
    return hashlib.md5(video_path.encode()).hexdigest()[:12]

def get_collection_names(video_id: str) -> dict:
    """Get collection names for a given video ID"""
    return {
        'clip': f'video-{video_id}-clip',
        'caption': f'video-{video_id}-captions',
        'audio': f'video-{video_id}-audio'
    }