# pyright: basic
# pyright: reportAttributeAccessIssue=false

import argparse
import logging
import os
import sys
from typing import Dict, Any
from dotenv import load_dotenv
from kimiconfig import Config
cfg = Config(use_dataclasses=True)
from rich.console import Console
from rich.logging import RichHandler
from rich.pretty import pretty_repr
from rich.traceback import install as install_rich_traceback 
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.api.types import IncludeEnum
from typing import Dict, Union, List, Optional
import uvicorn
from functools import lru_cache

# Initialize config
APP_NAME = "music2db_server"
HOME_DIR = os.path.expanduser("~")
DEFAULT_CONFIG_FILE = os.path.join(
    os.getenv("XDG_CONFIG_HOME", os.path.join(HOME_DIR, ".config")), 
    APP_NAME, 
    "config.yaml")

load_dotenv()

# Logging setup
logging.basicConfig(
    level=logging.NOTSET,
    format="%(message)s",
    datefmt="%X",
    handlers=[RichHandler(console=Console(), markup=True)],
)
parent_logger = logging.getLogger(APP_NAME)
log = logging.getLogger(f"{APP_NAME}.main")
install_rich_traceback(show_locals=True)

# Initialize FastAPI
app = FastAPI(
    title="Music2DB Server",
    description="FastAPI-based server for embedding-based music track indexing and similarity search",
    version="0.2.3",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Model for input JSON validation
class Track(BaseModel):
    file_path: str
    metadata: Dict[str, Union[str, int, bool, float, ]]  # Only strings and integers

app = FastAPI()
model: SentenceTransformer
client: chromadb.HttpClient  # type: ignore
collection: chromadb.Collection

def _init_entities():
    global app, model, client, collection
    model = SentenceTransformer(cfg.model.name)
    client = chromadb.HttpClient(host=cfg.chromadb.host, port=cfg.chromadb.port)
    collection = client.get_or_create_collection(cfg.chromadb.collection_name)


# Function to generate tag string from metadata
def generate_tag_string(file_path: str, metadata: Dict[str, Union[str, int, bool, float]]) -> str:
    """
    Generate a tag string from file path and metadata.
    Extracts meaningful information from the file path and combines it with metadata.
    """
    # Split path into components and clean them
    path_parts = file_path.replace('\\', '/').split('/')
    
    # Extract meaningful parts (artist, album, song)
    meaningful_parts = []
    for part in path_parts:
        # Skip empty parts and common directory names
        if not part or part.lower() in {'music', 'songs', 'tracks', 'mp3', 'flac'}:
            continue
        
        # Handle "year - album" format
        if ' - ' in part:
            year, album = part.split(' - ', 1)
            if year.isdigit():
                meaningful_parts.append(album)
            else:
                meaningful_parts.append(part)
        # Handle "number. song" format
        elif '. ' in part:
            _, song = part.rsplit('. ', 1)
            # Remove extension if present
            song = os.path.splitext(song)[0]
            meaningful_parts.append(song)
        else:
            # Remove extension if present
            part = os.path.splitext(part)[0]
            meaningful_parts.append(part)
    
    # Combine path information with metadata
    path_info = ' '.join(meaningful_parts)
    metadata_str = ', '.join(f"{key}: {value}" for key, value in metadata.items())
    
    return f"{path_info} {metadata_str}".strip()


async def _check_existing_track(file_path: str, metadata: Dict[str, Union[str, int, bool, float, ]]) -> tuple[bool, bool]:
    """
    Checks if track exists and compares metadata.
    
    Returns:
        tuple[bool, bool]: (exists, needs_update)
        - exists: True if record exists
        - needs_update: True if record exists but metadata differs
    """
    existing = collection.get(ids=[file_path])
    
    if not existing["ids"]:
        return False, False
        
    existing_metadata = existing["metadatas"][0] # type: ignore
    
    # Compare metadata
    metadata_changed = existing_metadata != metadata
    
    return True, metadata_changed


# Endpoint for adding a track
@app.post("/add_track/",
    response_model=dict,
    summary="Add a single track",
    response_description="Track addition status")
async def add_track(track: Track):
    """
    Add a single track to the database with its metadata and generate embeddings.

    Parameters:
    - **file_path**: Full path to the music file
    - **metadata**: Dictionary containing track metadata

    Example request body:
    ```json
    {
        "file_path": "/music/artist/album/track.mp3",
        "metadata": {
            "title": "Song Name",
            "artist": "Artist Name",
            "album": "Album Name",
            "year": 2024,
            "genre": "Rock"
        }
    }
    ```

    Returns:
    - **message**: Success or error message
    """
    log.debug(f"{track=}")
    
    exists, needs_update = await _check_existing_track(track.file_path, track.metadata)
    
    if exists and not needs_update:
        raise HTTPException(
            status_code=400, 
            detail=f"Track with file_path '{track.file_path}' already exists with same metadata"
        )
    
    if exists and needs_update:
        # Delete old record
        collection.delete(ids=[track.file_path])
        log.info(f"Deleted existing track '{track.file_path}' with different metadata")
    
    # Generate tag string including file path information
    tags = generate_tag_string(track.file_path, track.metadata)
    
    # Generate embedding on server
    embedding = model.encode(tags).tolist()

    # Add to ChromaDB
    collection.add(
        embeddings=[embedding],
        metadatas=[track.metadata],
        ids=[track.file_path]
    )

    status_msg = "updated" if exists else "added"
    return {"message": f"Track '{track.file_path}' {status_msg} successfully"}


@app.post("/add_tracks/",
    response_model=dict,
    summary="Add multiple tracks",
    response_description="Batch addition status")
async def add_tracks(tracks: list[Track]):
    """
    Add multiple tracks in a single batch operation.

    Parameters:
    - **tracks**: List of Track objects containing file_path and metadata

    Example request body:
    ```json
    [
        {
            "file_path": "/music/track1.mp3",
            "metadata": { ... }
        },
        {
            "file_path": "/music/track2.mp3",
            "metadata": { ... }
        }
    ]
    ```

    Returns:
    - **message**: Summary of added and updated tracks
    """
    added_count = 0
    updated_count = 0
    
    for track in tracks:
        exists, needs_update = await _check_existing_track(track.file_path, track.metadata)
        
        if exists and not needs_update:
            continue
            
        if exists and needs_update:
            collection.delete(ids=[track.file_path])
            updated_count += 1
            
        tags = generate_tag_string(track.file_path, track.metadata)
        embedding = model.encode(tags).tolist()
        collection.add(
            embeddings=[embedding],
            metadatas=[track.metadata],
            ids=[track.file_path]
        )
        
        if not exists:
            added_count += 1
            
    return {
        "message": f"Added {added_count} new tracks, updated {updated_count} existing tracks"
    }


# Endpoint for checking list of existing tracks (optional)
@app.get("/list_tracks/",
    response_model=dict,
    summary="List all tracks",
    response_description="List of all track paths")
async def list_tracks():
    """
    Get a list of all track file paths in the collection.

    Returns:
    - **tracks**: List of file paths
    """
    all_tracks = collection.get()
    return {"tracks": all_tracks["ids"]}


@app.get("/health/",
    response_model=dict,
    summary="Health check",
    response_description="Server health status")
async def health_check():
    """
    Check if the server is running and healthy.

    Returns:
    - **status**: Server status message
    """
    return {"status": "Server is running"}


# New endpoint for searching tracks by tag string
@lru_cache(maxsize=1000)
def _generate_embedding(tags: str) -> list[float]:
    return model.encode(tags).tolist()


class SearchParams(BaseModel):
    tags: str = Field(..., min_length=1, max_length=200, description="Search query string")
    limit: int = Field(default=5, ge=1, le=100, description="Maximum number of results")
    max_distance: float = Field(
        default=1.2, 
        ge=0.0, 
        le=2.0, 
        description="Maximum cosine distance (0.0 to 2.0)"
    )

    @field_validator('tags')
    @classmethod
    def validate_tags(cls, v: str) -> str:
        if v.strip() == "":
            raise ValueError("tags cannot be empty or just whitespace")
        return v.strip()


@app.get("/search_tracks/",
    response_model=List[str],
    summary="Search similar tracks",
    response_description="List of similar track paths")
async def search_tracks(
    tags: str = Query(..., description="Search query string (e.g., 'rock upbeat energetic')"),
    limit: int = Query(default=5, ge=1, le=100, description="Maximum number of results to return"),
    max_distance: float = Query(
        default=0.5, 
        ge=0.0, 
        le=2.0, 
        description="Maximum cosine distance for matches (0.0 to 2.0)"
    )
):
    """
    Search for similar tracks using semantic similarity.

    Parameters:
    - **tags**: Search query string (e.g., "rock upbeat energetic")
    - **limit**: Maximum number of results to return (1-100, default: 5)
    - **max_distance**: Maximum cosine distance for matches (0.0-2.0, default: 0.5)

    Returns:
    - List of file paths to similar tracks

    Raises:
    - 400: Invalid parameter values
    - 422: Validation error
    """
    # Generate embedding for tag string
    embedding = _generate_embedding(tags)

    # Query ChromaDB with increased limit to account for filtering
    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(limit * 2, 100),  # Request more results but cap at 100
        include=[IncludeEnum.distances, IncludeEnum.metadatas]
    )
    log.info(pretty_repr(results))

    # Filter results by distance
    filtered_paths = []
    for path, distance in zip(results["ids"][0], results["distances"][0]): # type: ignore
        if distance <= max_distance:
            filtered_paths.append(path)
        if len(filtered_paths) >= limit:
            break

    return filtered_paths


@app.delete("/clear_collection/",
    response_model=dict,
    summary="Clear collection",
    response_description="Collection clearing status")
async def clear_collection():
    """
    Delete all tracks from the collection.
    
    Warning: This operation cannot be undone.

    Returns:
    - **message**: Summary of deleted tracks
    """
    global collection

    # Get current count of tracks
    all_tracks = collection.get()
    count = len(all_tracks["ids"])
    
    # Delete all records
    client.delete_collection(cfg.chromadb.collection_name)
    collection = client.create_collection(cfg.chromadb.collection_name)
    
    log.info(f"Cleared collection '{cfg.chromadb.collection_name}', deleted {count} tracks")
    return {"message": f"Collection cleared, {count} tracks deleted"}


@app.get("/collection_stats/",
    response_model=dict,
    summary="Get collection statistics",
    response_description="Database collection statistics")
async def collection_stats():
    """
    Get detailed statistics about the track collection.

    Returns:
    ```json
    {
        "total_tracks": 1000,
        "total_size_mb": 15.26,
        "embedding_dimensions": 384,
        "metadata_stats": {
            "artist": {
                "count": 1000,
                "coverage_percent": 100.0,
                "unique_values_count": 150,
                "types": ["str"]
            },
            "album": {
                "count": 950,
                "coverage_percent": 95.0,
                "unique_values_count": 200,
                "types": ["str"]
            }
        }
    }
    ```
    """
    # Get all tracks with metadata
    results = collection.get(
        include=[IncludeEnum.metadatas, IncludeEnum.embeddings]
    )
    
    if not results["ids"]:
        return {
            "total_tracks": 0,
            "total_size": 0,
            "metadata_stats": {},
            "message": "Collection is empty"
        }

    # Calculate basic stats
    total_tracks = len(results["ids"])
    
    # Calculate embedding size (approximate)
    # Each embedding is a list of floats (32 bits = 4 bytes each)
    embedding_dimension = len(results["embeddings"][0]) if isinstance(results["embeddings"][0], list) else results["embeddings"][0].shape[0]  # type: ignore
    embedding_size = embedding_dimension * 4 * total_tracks / (1024 * 1024)  # Convert to MB
    
    # Analyze metadata fields
    metadata_fields = {}
    metadatas = results.get("metadatas", [])
    if not metadatas:
        return {
            "total_tracks": total_tracks,
            "total_size_mb": round(embedding_size, 2),
            "metadata_stats": {},
            "embedding_dimensions": embedding_dimension,
            "message": "No metadata found in collection"
        }

    for metadata in metadatas:
        for key, value in metadata.items():
            if key not in metadata_fields:
                metadata_fields[key] = {
                    "count": 0,
                    "unique_values": set(),
                    "types": set()
                }
            metadata_fields[key]["count"] += 1
            metadata_fields[key]["unique_values"].add(str(value))
            metadata_fields[key]["types"].add(type(value).__name__)
    
    # Convert sets to lists for JSON serialization
    metadata_stats = {}
    for key, stats in metadata_fields.items():
        metadata_stats[key] = {
            "count": stats["count"],
            "coverage_percent": round(stats["count"] / total_tracks * 100, 1),
            "unique_values_count": len(stats["unique_values"]),
            "types": list(stats["types"])
        }
    
    return {
        "total_tracks": total_tracks,
        "total_size_mb": round(embedding_size, 2),
        "metadata_stats": metadata_stats,
        "embedding_dimensions": embedding_dimension
    }


def _init_logs():
    for logger_name in cfg.logging.loggers.suppress:
        logging.getLogger(logger_name).setLevel(getattr(logging, cfg.logging.loggers.suppress_level))

    parent_logger.setLevel(cfg.logging.level)
    if cfg.logging.level == "DEBUG":
        cfg.print_config()


def _parse_args():
    parser = argparse.ArgumentParser(prog=APP_NAME, description="Music2DB Server")
    parser.add_argument(
        "-c",
        "--config",
        dest="config_file",
        default=DEFAULT_CONFIG_FILE,
        help="Configuration file location.",
    )
    return parser.parse_known_args()


def _init_config(files: list[str], unknown_args: list[str]):
    """
    Initializes the configuration by loading configuration files and passed arguments.

    Args:
        files (List[str]): List of config files.
        unknown_args (List[str]): List of arguments (unknown for argparse).
    """
    cfg.load_files(files)
    cfg.load_args(unknown_args)

    # # add some/override config here if needed
    # cfg.update('runtime.blablabla', True)
    # cfg.update('religion.buddhism.name', 'Gautama Siddharta')


def main():
    args, unknown_args = _parse_args()
    _init_config([args.config_file], unknown_args)
    _init_logs()
    _init_entities()

    log.info(f"Starting {APP_NAME}")
    uvicorn.run(app, host=cfg.app.host, port=cfg.app.port)  


# Run server
if __name__ == "__main__":
    sys.exit(main())
