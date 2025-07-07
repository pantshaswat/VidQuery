from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
import os
import tempfile
import shutil
from typing import List, Dict, Any, Optional
from pathlib import Path
import uuid
import json
from datetime import datetime

from audio_processor import process_video_for_audio_captions
from video_processor import process_video_hybrid_embeddings
from search_engine import hybrid_search, audio_search, get_available_videos, delete_video_collections
from video_utils import get_video_id

app = FastAPI(title="Video Search Engine", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Video metadata storage (in production, use a database)
VIDEO_METADATA = {}

class SearchRequest(BaseModel):
    query: str
    video_id: Optional[str] = None
    top_k: int = 5

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    query: str
    search_type: str
    video_id: Optional[str] = None
    total_results: int

class VideoMetadata(BaseModel):
    video_id: str
    filename: str
    original_name: str
    file_path: str
    upload_date: str
    file_size: int
    duration: Optional[float] = None
    status: str = "processing"

def save_video_metadata(video_id: str, metadata: VideoMetadata):
    """Save video metadata to JSON file"""
    metadata_file = UPLOAD_DIR / "metadata.json"
    
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            data = json.load(f)
    else:
        data = {}
    
    data[video_id] = metadata.dict()
    
    with open(metadata_file, 'w') as f:
        json.dump(data, f, indent=2)

def load_video_metadata():
    """Load video metadata from JSON file"""
    metadata_file = UPLOAD_DIR / "metadata.json"
    
    if not metadata_file.exists():
        return {}
    
    with open(metadata_file, 'r') as f:
        return json.load(f)

def get_video_duration(video_path: str) -> Optional[float]:
    """Get video duration using ffprobe"""
    try:
        import subprocess
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', 
            '-show_format', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return float(data['format']['duration'])
    except Exception as e:
        print(f"Error getting video duration: {e}")
    return None

@app.get("/")
async def root():
    return {
        "message": "Video Search Engine API v2.0 - Enhanced with Streaming", 
        "version": "2.0.0",
        "endpoints": {
            "upload": "/upload-video",
            "search_visual": "/search/video",
            "search_audio": "/search/audio",
            "videos": "/videos",
            "stream": "/videos/{video_id}/stream",
            "thumbnail": "/videos/{video_id}/thumbnail"
        }
    }

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """Upload and process video with enhanced metadata tracking"""
    
    # Validate file format
    allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid video format. Supported formats: {', '.join(allowed_extensions)}"
        )
    
    # Generate unique video ID
    video_id = str(uuid.uuid4())
    filename = f"{video_id}{file_extension}"
    file_path = UPLOAD_DIR / filename
    
    # Save uploaded file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get file size and duration
        file_size = file_path.stat().st_size
        duration = get_video_duration(str(file_path))
        
        # Create metadata
        metadata = VideoMetadata(
            video_id=video_id,
            filename=filename,
            original_name=file.filename,
            file_path=str(file_path),
            upload_date=datetime.now().isoformat(),
            file_size=file_size,
            duration=duration,
            status="processing"
        )
        
        # Save metadata
        save_video_metadata(video_id, metadata)
        
        # Process video for audio captions (Whisper)
        print(f"Processing audio for video {video_id}...")
        try:
            process_video_for_audio_captions(str(file_path))
        except Exception as e:
            print(f"Audio processing failed: {e}")
            metadata.status = "audio_processing_failed"
            save_video_metadata(video_id, metadata)
        
        # Process video for visual search (CLIP + BLIP)
        print(f"Processing video frames for video {video_id}...")
        try:
            process_video_hybrid_embeddings(str(file_path))
            metadata.status = "completed"
        except Exception as e:
            print(f"Video processing failed: {e}")
            metadata.status = "video_processing_failed"
        
        # Update final status
        save_video_metadata(video_id, metadata)
        
        return {
            "message": "Video uploaded and processed successfully",
            "video_id": video_id,
            "filename": file.filename,
            "file_size": file_size,
            "duration": duration,
            "status": metadata.status
        }
    
    except Exception as e:
        # Clean up file if processing fails
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/search/video", response_model=SearchResponse)
async def search_video(request: SearchRequest):
    """Search video using visual content with enhanced debugging"""
    
    try:
        print(f"=== SEARCH DEBUG START ===")
        print(f"Query: {request.query}")
        print(f"Video ID: {request.video_id}")
        print(f"Top K: {request.top_k}")
        
        # Check if video exists in metadata
        metadata = load_video_metadata()
        if request.video_id and request.video_id not in metadata:
            print(f"ERROR: Video {request.video_id} not found in metadata")
            raise HTTPException(status_code=404, detail="Video not found in metadata")
        
        # Check video processing status
        if request.video_id:
            video_status = metadata[request.video_id].get('status', 'unknown')
            print(f"Video processing status: {video_status}")
            if video_status != 'completed':
                print(f"WARNING: Video processing not completed. Status: {video_status}")
        
        # Check Qdrant collections
        from qdrant_client import QdrantClient
        from video_utils import get_collection_names
        
        qdrant = QdrantClient("localhost", port=6333, prefer_grpc=False, check_compatibility=False)
        
        if request.video_id:
            collections = get_collection_names(request.video_id)
            existing_collections = [c.name for c in qdrant.get_collections().collections]
            
            print(f"Expected collections: {collections}")
            print(f"Existing collections: {existing_collections}")
            
            # Check if required collections exist
            missing_collections = []
            for coll_type, coll_name in collections.items():
                if coll_name not in existing_collections:
                    missing_collections.append(coll_name)
            
            if missing_collections:
                print(f"ERROR: Missing collections: {missing_collections}")
                return SearchResponse(
                    results=[],
                    query=request.query,
                    search_type="video_visual",
                    video_id=request.video_id,
                    total_results=0
                )
            
            # Check collection point counts
            for coll_type, coll_name in collections.items():
                try:
                    info = qdrant.get_collection(coll_name)
                    print(f"Collection {coll_name}: {info.points_count} points")
                    if info.points_count == 0:
                        print(f"WARNING: Collection {coll_name} has no points")
                except Exception as e:
                    print(f"ERROR checking collection {coll_name}: {e}")
        
        # Perform the actual search
        print(f"Calling hybrid_search...")
        results = hybrid_search(request.query, video_id=request.video_id, top_k=request.top_k)
        print(f"Raw search results: {len(results)} items")
        
        # Debug raw results
        for i, (timestamp, data) in enumerate(results):
            print(f"Result {i}: timestamp={timestamp}, score={data['score']}, sources={data['sources']}")
        
        formatted_results = []
        for timestamp, data in results:
            formatted_results.append({
                "timestamp": timestamp,
                "score": data['score'],
                "sources": data['sources'],
                "caption": data['payload'].get('caption', ''),
                "video_path": data['payload'].get('video_path', ''),
                "video_id": data['payload'].get('video_id', ''),
                "frame_url": f"/videos/{data['payload'].get('video_id', '')}/frame/{timestamp}"
            })
        
        print(f"Formatted results: {len(formatted_results)} items")
        print(f"=== SEARCH DEBUG END ===")
        
        return SearchResponse(
            results=formatted_results,
            query=request.query,
            search_type="video_visual",
            video_id=request.video_id,
            total_results=len(formatted_results)
        )
    
    except Exception as e:
        print(f"SEARCH ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Visual search failed: {str(e)}")

@app.post("/search/audio", response_model=SearchResponse)
async def search_audio(request: SearchRequest):
    """Search video using audio transcriptions with enhanced response format"""
    
    try:
        results = audio_search(request.query, video_id=request.video_id, top_k=request.top_k)
        
        formatted_results = []
        for result in results:
            formatted_results.append({
                "timestamp": result.payload['timestamp'],
                "score": result.score,
                "transcription": result.payload.get('caption', ''),
                "video_path": result.payload.get('video_path', ''),
                "video_id": result.payload.get('video_id', ''),
                "audio_segment_url": f"/videos/{result.payload.get('video_id', '')}/audio/{result.payload['timestamp']}"
            })
        
        return SearchResponse(
            results=formatted_results,
            query=request.query,
            search_type="audio_transcription",
            video_id=request.video_id,
            total_results=len(formatted_results)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio search failed: {str(e)}")

@app.get("/videos")
async def list_videos():
    """List all available videos with metadata"""
    try:
        metadata = load_video_metadata()
        
        videos = []
        for video_id, data in metadata.items():
            videos.append({
                "video_id": video_id,
                "original_name": data.get('original_name', ''),
                "upload_date": data.get('upload_date', ''),
                "file_size": data.get('file_size', 0),
                "duration": data.get('duration', 0),
                "status": data.get('status', 'unknown'),
                "thumbnail_url": f"/videos/{video_id}/thumbnail"
            })
        
        return {
            "videos": videos, 
            "count": len(videos),
            "total_size": sum(v['file_size'] for v in videos)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list videos: {str(e)}")

@app.get("/videos/{video_id}")
async def get_video_info(video_id: str):
    """Get detailed information about a specific video"""
    try:
        metadata = load_video_metadata()
        
        if video_id not in metadata:
            raise HTTPException(status_code=404, detail="Video not found")
        
        video_data = metadata[video_id]
        video_data["thumbnail_url"] = f"/videos/{video_id}/thumbnail"
        video_data["stream_url"] = f"/videos/{video_id}/stream"
        
        return video_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get video info: {str(e)}")

@app.get("/videos/{video_id}/stream")
async def stream_video(video_id: str, request: Request):
    """Stream video with support for range requests"""
    try:
        metadata = load_video_metadata()
        
        if video_id not in metadata:
            raise HTTPException(status_code=404, detail="Video not found")
        
        video_path = Path(metadata[video_id]['file_path'])
        
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Video file not found")
        
        # Handle range requests for video streaming
        range_header = request.headers.get('Range')
        
        if range_header:
            # Parse range header
            range_match = range_header.replace('bytes=', '').split('-')
            start = int(range_match[0]) if range_match[0] else 0
            end = int(range_match[1]) if range_match[1] else None
            
            file_size = video_path.stat().st_size
            
            if end is None:
                end = file_size - 1
            
            content_length = end - start + 1
            
            def generate_chunk():
                with open(video_path, 'rb') as f:
                    f.seek(start)
                    remaining = content_length
                    while remaining > 0:
                        chunk_size = min(8192, remaining)
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        remaining -= len(chunk)
                        yield chunk
            
            return StreamingResponse(
                generate_chunk(),
                status_code=206,
                headers={
                    'Content-Range': f'bytes {start}-{end}/{file_size}',
                    'Accept-Ranges': 'bytes',
                    'Content-Length': str(content_length),
                    'Content-Type': 'video/mp4'
                }
            )
        else:
            # Regular file response
            return FileResponse(
                video_path,
                media_type='video/mp4',
                headers={'Accept-Ranges': 'bytes'}
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stream video: {str(e)}")

@app.get("/videos/{video_id}/thumbnail")
async def get_video_thumbnail(video_id: str):
    """Generate and return video thumbnail"""
    try:
        metadata = load_video_metadata()
        
        if video_id not in metadata:
            raise HTTPException(status_code=404, detail="Video not found")
        
        video_path = Path(metadata[video_id]['file_path'])
        thumbnail_path = UPLOAD_DIR / f"{video_id}_thumbnail.jpg"
        
        # Generate thumbnail if it doesn't exist
        if not thumbnail_path.exists():
            import subprocess
            cmd = [
                'ffmpeg', '-i', str(video_path), '-ss', '00:00:01.000',
                '-vframes', '1', '-y', str(thumbnail_path)
            ]
            result = subprocess.run(cmd, capture_output=True)
            
            if result.returncode != 0:
                raise HTTPException(status_code=500, detail="Failed to generate thumbnail")
        
        return FileResponse(
            thumbnail_path,
            media_type='image/jpeg',
            headers={'Cache-Control': 'public, max-age=3600'}
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get thumbnail: {str(e)}")

@app.get("/videos/{video_id}/frame/{timestamp}")
async def get_video_frame(video_id: str, timestamp: float):
    """Extract a specific frame from video at given timestamp"""
    try:
        metadata = load_video_metadata()
        
        if video_id not in metadata:
            raise HTTPException(status_code=404, detail="Video not found")
        
        video_path = Path(metadata[video_id]['file_path'])
        frame_path = UPLOAD_DIR / f"{video_id}_frame_{timestamp}.jpg"
        
        # Generate frame if it doesn't exist
        if not frame_path.exists():
            import subprocess
            cmd = [
                'ffmpeg', '-i', str(video_path), '-ss', str(timestamp),
                '-vframes', '1', '-y', str(frame_path)
            ]
            result = subprocess.run(cmd, capture_output=True)
            
            if result.returncode != 0:
                raise HTTPException(status_code=500, detail="Failed to extract frame")
        
        return FileResponse(
            frame_path,
            media_type='image/jpeg',
            headers={'Cache-Control': 'public, max-age=3600'}
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get frame: {str(e)}")

@app.delete("/videos/{video_id}")
async def delete_video(video_id: str):
    """Delete a video and all its associated data"""
    try:
        metadata = load_video_metadata()
        
        if video_id not in metadata:
            raise HTTPException(status_code=404, detail="Video not found")
        
        video_data = metadata[video_id]
        
        # Delete video file
        video_path = Path(video_data['file_path'])
        if video_path.exists():
            video_path.unlink()
        
        # Delete thumbnail
        thumbnail_path = UPLOAD_DIR / f"{video_id}_thumbnail.jpg"
        if thumbnail_path.exists():
            thumbnail_path.unlink()
        
        # Delete frame cache
        for frame_file in UPLOAD_DIR.glob(f"{video_id}_frame_*.jpg"):
            frame_file.unlink()
        
        # Delete from search collections
        delete_video_collections(video_id)
        
        # Remove from metadata
        del metadata[video_id]
        
        # Save updated metadata
        metadata_file = UPLOAD_DIR / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            "message": f"Video {video_id} deleted successfully",
            "deleted_files": [
                str(video_path),
                str(thumbnail_path),
                "frame_cache",
                "search_collections"
            ]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete video: {str(e)}")

@app.get("/videos/{video_id}/search-results/{timestamp}")
async def get_search_context(video_id: str, timestamp: float, context_duration: float = 5.0):
    """Get search context around a specific timestamp"""
    try:
        # This would integrate with your search engine to get surrounding context
        # For now, return placeholder data
        return {
            "video_id": video_id,
            "timestamp": timestamp,
            "context_start": max(0, timestamp - context_duration),
            "context_end": timestamp + context_duration,
            "transcription": "Context transcription would go here...",
            "visual_description": "Visual context description would go here..."
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get search context: {str(e)}")

@app.get("/collections/status")
async def get_collections_status():
    """Get status of Qdrant collections with enhanced details"""
    from qdrant_client import QdrantClient
    
    try:
        qdrant = QdrantClient("localhost", port=6333, prefer_grpc=False, check_compatibility=False)
        collections = qdrant.get_collections().collections
        
        status = {}
        total_points = 0
        
        for collection in collections:
            try:
                info = qdrant.get_collection(collection.name)
                points_count = info.points_count
                total_points += points_count
                
                status[collection.name] = {
                    "points_count": points_count,
                    "status": info.status,
                    "config": {
                        "vector_size": info.config.params.vectors.size if hasattr(info.config.params, 'vectors') else None,
                        "distance": info.config.params.vectors.distance.value if hasattr(info.config.params, 'vectors') else None
                    }
                }
            except Exception as e:
                status[collection.name] = {
                    "error": str(e),
                    "status": "error"
                }
        
        return {
            "collections": status,
            "total_collections": len(collections),
            "total_points": total_points,
            "qdrant_status": "connected"
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "qdrant_status": "disconnected"
        }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "components": {
            "api": "running",
            "storage": "available" if UPLOAD_DIR.exists() else "unavailable",
            "video_count": len(load_video_metadata())
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)