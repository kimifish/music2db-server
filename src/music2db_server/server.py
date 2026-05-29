# pyright: basic
# pyright: reportAttributeAccessIssue=false

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
from kimiconfig import Config
cfg = Config(use_dataclasses=True)
from rich.traceback import install as install_rich_traceback
from fastapi import FastAPI, HTTPException, Query, Depends
from pydantic import BaseModel, Field
import chromadb
import httpx
from typing import Dict, Union, List, Optional
import uvicorn
from functools import lru_cache
from rapidfuzz import process, fuzz
from .logging_setup import get_logger, setup_logging

try:
    from music2db_server import __version__
except ImportError:
    __version__ = "0.2.0"


# Initialize config
APP_NAME = "music2db_server"
HOME_DIR = os.path.expanduser("~")
SYSTEM_CONFIG_DIR = Path("/etc") / APP_NAME
XDG_CONFIG_DIR = Path(os.getenv("XDG_CONFIG_HOME", os.path.join(HOME_DIR, ".config"))) / APP_NAME
LOCAL_CONFIG_DIR = Path.cwd() / "config"
ACTIVE_CONFIG_FILES: list[Path] = []

load_dotenv()

log = get_logger(__name__)
install_rich_traceback(show_locals=True)

class Track(BaseModel):
    file_path: str
    metadata: Dict[str, Union[str, int, bool, float, ]]  # Only strings and integers

# Initialize FastAPI
# Use only one FastAPI instance
app = FastAPI(
    title="Music2DB Server",
    description="FastAPI-based server for embedding-based music track indexing and similarity search",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

client: chromadb.HttpClient  # type: ignore
collection: chromadb.Collection


class EmbeddingsServiceError(RuntimeError):
    """Raised when the external embeddings service fails."""


class EmbeddingsResponse(BaseModel):
    model: str
    input_type: str
    dimensions: int
    embeddings: list[list[float]]


def _get_embeddings_settings() -> tuple[str, Optional[str], bool, float]:
    """Returns embeddings service settings from config."""
    if not hasattr(cfg, "embeddings"):
        log.error("`config` Embeddings configuration missing in config.yaml")
        sys.exit(1)

    embeddings_cfg = cfg.embeddings
    if not hasattr(embeddings_cfg, "base_url"):
        log.error("`config` Embeddings base_url missing in config.yaml")
        sys.exit(1)

    base_url = str(embeddings_cfg.base_url).rstrip("/")
    model_name = str(embeddings_cfg.model) if hasattr(embeddings_cfg, "model") else None
    normalize = bool(getattr(embeddings_cfg, "normalize", True))
    timeout_seconds = float(getattr(embeddings_cfg, "timeout_seconds", 30))
    return base_url, model_name, normalize, timeout_seconds


def _request_embeddings(texts: list[str], input_type: str) -> EmbeddingsResponse:
    """Fetches embeddings from the external embeddings service."""
    if not texts:
        raise ValueError("texts must not be empty")

    base_url, model_name, normalize, timeout_seconds = _get_embeddings_settings()
    payload: dict[str, Any] = {
        "texts": texts,
        "input_type": input_type,
        "normalize": normalize,
    }
    if model_name:
        payload["model"] = model_name

    log.debug("`http` POST %s/v1/embeddings texts=%s input_type=%s", base_url, len(texts), input_type)

    try:
        with httpx.Client(timeout=timeout_seconds) as http_client:
            response = http_client.post(f"{base_url}/v1/embeddings", json=payload)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        try:
            error_payload = exc.response.json()
            detail = error_payload.get("error", {}).get("message", detail)
        except ValueError:
            pass
        raise EmbeddingsServiceError(
            f"Embeddings service returned {exc.response.status_code}: {detail}"
        ) from exc
    except httpx.HTTPError as exc:
        raise EmbeddingsServiceError(f"Embeddings service request failed: {exc}") from exc

    return EmbeddingsResponse.model_validate(response.json())


def _generate_embedding(text: str, input_type: str) -> list[float]:
    """Generates a single embedding via the external embeddings service."""
    response = _request_embeddings([text], input_type)
    return response.embeddings[0]


def _generate_embeddings(texts: list[str], input_type: str) -> list[list[float]]:
    """Generates embeddings for multiple texts via the external embeddings service."""
    return _request_embeddings(texts, input_type).embeddings


def _get_embeddings_health() -> dict[str, Any]:
    """Checks embeddings service health and basic model compatibility."""
    base_url, model_name, _, timeout_seconds = _get_embeddings_settings()

    try:
        log.debug("`http` GET %s/health", base_url)
        with httpx.Client(timeout=timeout_seconds) as http_client:
            response = http_client.get(f"{base_url}/health")
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPError as exc:
        raise EmbeddingsServiceError(f"Embeddings health check failed: {exc}") from exc

    loaded_model = payload.get("model")
    if model_name and loaded_model and loaded_model != model_name:
        raise EmbeddingsServiceError(
            f"Configured embeddings model '{model_name}' does not match loaded model '{loaded_model}'"
        )

    return payload

def _init_entities():
    """Initializes the ChromaDB client/collection and validates embeddings service."""
    global client, collection

    embeddings_health = _get_embeddings_health()
    log.info(
        "`startup` Embeddings service ready: %s (%s)",
        embeddings_health.get("model", "unknown-model"),
        _get_embeddings_settings()[0],
    )

    if not hasattr(cfg, 'chromadb') or not hasattr(cfg.chromadb, 'host') or not hasattr(cfg.chromadb, 'port'):
         log.error("`config` ChromaDB configuration missing in config.yaml")
         sys.exit(1)

    client = chromadb.HttpClient(host=cfg.chromadb.host, port=cfg.chromadb.port)
    # Use get_or_create_collection to ensure it exists
    # Add a check if cfg.chromadb.collection_name exists
    if not hasattr(cfg.chromadb, 'collection_name'):
         log.error("`config` ChromaDB collection name missing in config.yaml")
         sys.exit(1)
    collection = client.get_or_create_collection(cfg.chromadb.collection_name)


# Modified function to generate tag string, focusing on descriptive tags
def generate_tag_string(file_path: str, metadata: Dict[str, Any]) -> str:
    """
    Generate a tag string from metadata, focusing on descriptive tags for embedding.
    Includes genre and tags from metadata. Optionally includes title/artist if available.
    """
    tag_parts = []

    # Include descriptive tags and genre
    if 'genre' in metadata and isinstance(metadata['genre'], str):
        tag_parts.append(metadata['genre'])
    if 'tags' in metadata and isinstance(metadata['tags'], str):
        # Clean up LastFM tags prefix if present
        tags_content = metadata['tags'].replace('LastFM tags:', '').strip()
        if tags_content:
            tag_parts.append(tags_content)

    # Optionally include title and artist for context in embedding, but they are
    # also stored separately for exact filtering.
    if 'title' in metadata and isinstance(metadata['title'], str):
         tag_parts.append(metadata['title'])
    if 'artist' in metadata and isinstance(metadata['artist'], str):
         tag_parts.append(metadata['artist'])

    # Combine parts into a single string
    return ", ".join(filter(None, tag_parts)).strip()


async def _check_existing_track(file_path: str, metadata: Dict[str, Any]) -> tuple[bool, bool]:
    """
    Checks if track exists and compares metadata.

    Returns:
        tuple[bool, bool]: (exists, needs_update)
        - exists: True if record exists
        - needs_update: True if record exists but metadata differs
    """
    # Use try-except for more robust handling of non-existent IDs
    try:
        # Use include=[] for efficiency as we only need to check existence and metadata
        existing = collection.get(ids=[file_path], include=['metadatas'])
        if not existing["ids"]:
            return False, False

        existing_metadata = existing["metadatas"][0] # type: ignore

        # Compare metadata. Note: Deep comparison might be needed for complex metadata
        # For simplicity, a direct comparison is used here.
        metadata_changed = existing_metadata != metadata

        return True, metadata_changed
    except Exception as e:
        log.error("`state` Error checking existing track %s: %s", file_path, e)
        # Assume it doesn't exist or needs update in case of error
        # Returning True, True will cause it to be deleted and re-added
        return False, True


# Endpoint for adding a track
@app.post("/add_track/",
    response_model=dict,
    summary="Add a single track",
    response_description="Track addition status")
async def add_track(track: Track):
    """
    Add a single track to the database with its metadata and generate embeddings.

    Parameters:
    - **file_path**: Full path to the music file (used as ID)
    - **metadata**: Dictionary containing track metadata (artist, title, album, genre, year, tags, etc.)

    Example request body:
    ```json
    {
        "file_path": "/music/artist/album/track.mp3",
        "metadata": {
            "title": "Song Name",
            "artist": "Artist Name",
            "album": "Album Name",
            "year": 2024,
            "genre": "Rock",
            "tags": "upbeat, energetic, driving"
        }
    }
    ```

    Returns:
    - **message**: Success or error message
    """
    log.info("`api` Adding track: %s", track.file_path)

    try:
        exists, needs_update = await _check_existing_track(track.file_path, track.metadata)

        if exists and not needs_update:
            log.info("`state` Track '%s' already exists with same metadata, skipping.", track.file_path)
            return {"message": f"Track '{track.file_path}' already exists with same metadata"}

        if exists and needs_update:
            # Delete old record before adding the new one
            collection.delete(ids=[track.file_path])
            log.info("`state` Deleted existing track '%s' with different metadata for update.", track.file_path)

        # Generate tag string focusing on descriptive elements for embedding
        tags_for_embedding = generate_tag_string(track.file_path, track.metadata)

        # Generate embedding on server
        # Ensure the input to encode is not empty if no descriptive tags are found
        if not tags_for_embedding:
             log.warning("`state` No descriptive tags found for '%s', using file path for embedding.", track.file_path)
             tags_for_embedding = track.file_path # Fallback to file path if no descriptive tags

        # Add a check for empty tags_for_embedding even after fallback
        if not tags_for_embedding:
             log.error("`state` Could not generate embedding string for '%s'. Skipping.", track.file_path)
             raise HTTPException(status_code=400, detail=f"Could not generate embedding string for '{track.file_path}'")


        embedding = _generate_embedding(tags_for_embedding, "passage")

        # Add to ChromaDB. Store full metadata for filtering.
        collection.add(
            embeddings=[embedding],
            metadatas=[track.metadata], # Store the original, full metadata
            ids=[track.file_path]
        )

        status_msg = "updated" if exists else "added"
        log.info("`state` Track '%s' %s successfully.", track.file_path, status_msg)
        return {"message": f"Track '{track.file_path}' {status_msg} successfully"}

    except EmbeddingsServiceError as e:
        log.error("`http` Error adding track '%s': %s", track.file_path, e)
        raise HTTPException(status_code=503, detail=f"Error adding track: {e}")
    except Exception as e:
        log.error("`state` Error adding track '%s': %s", track.file_path, e)
        # Return a more specific error if it's an HTTPException already
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error adding track: {e}")


@app.post("/add_tracks/",
    response_model=dict,
    summary="Add multiple tracks",
    response_description="Batch addition status")
async def add_tracks(tracks: list[Track]):
    """
    Add multiple tracks in a single batch operation.
    This implementation processes tracks sequentially, checking and adding/updating each.
    For true batching with ChromaDB's add method, you would collect
    all embeddings, metadatas, and ids first before calling collection.add() once.

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
    log.info("`api` Adding batch of %s tracks", len(tracks))

    added_count = 0
    updated_count = 0
    skipped_count = 0
    errors = []

    tracks_to_add: list[Track] = []
    tracks_to_delete: list[str] = []

    for track in tracks:
        try:
            exists, needs_update = await _check_existing_track(track.file_path, track.metadata)

            if exists and not needs_update:
                skipped_count += 1
                continue

            if exists and needs_update:
                tracks_to_delete.append(track.file_path)
                updated_count += 1

            tags_for_embedding = generate_tag_string(track.file_path, track.metadata)
            if not tags_for_embedding:
                 log.warning("`state` No descriptive tags found for '%s' in batch, using file path for embedding.", track.file_path)
                 tags_for_embedding = track.file_path # Fallback

            # Add a check for empty tags_for_embedding even after fallback
            if not tags_for_embedding:
                 log.error("`state` Could not generate embedding string for '%s' in batch. Skipping.", track.file_path)
                 errors.append({"file_path": track.file_path, "error": "Could not generate embedding string"})
                 continue


            tracks_to_add.append(
                Track(
                    file_path=track.file_path,
                    metadata={**track.metadata, "_embedding_text": tags_for_embedding},
                )
            )

            if not exists:
                added_count += 1

        except Exception as e:
            log.error("`state` Error processing track '%s' in batch: %s", track.file_path, e)
            errors.append({"file_path": track.file_path, "error": str(e)})

    try:
        embeddings: list[list[float]] = []
        if tracks_to_add:
            embedding_texts = [track.metadata.pop("_embedding_text") for track in tracks_to_add]
            embeddings = _generate_embeddings(embedding_texts, "passage")

        if tracks_to_delete:
            collection.delete(ids=tracks_to_delete)

        if tracks_to_add:
            collection.add(
                embeddings=embeddings,
                metadatas=[track.metadata for track in tracks_to_add],
                ids=[track.file_path for track in tracks_to_add],
            )
    except Exception as e:
        log.error("`state` Error committing batch operation: %s", e)
        errors.append({"file_path": "<batch>", "error": str(e)})

    message = f"Batch process finished: Added {added_count} new tracks, updated {updated_count} existing tracks, skipped {skipped_count} tracks."
    if errors:
        message += f" Encountered {len(errors)} errors."
        # You might want to return the list of errors as well
        # return {"message": message, "errors": errors}

    return {"message": message}


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
    log.info("`api` Listing all tracks")

    try:
        # Fetch only IDs to be efficient
        all_tracks = collection.get(include=[])
        return {"tracks": all_tracks["ids"]}
    except Exception as e:
        log.error("`state` Error listing tracks: %s", e)
        raise HTTPException(status_code=500, detail=f"Error listing tracks: {e}")


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
    log.info("`api` Health check requested")

    # You could add checks for ChromaDB connection here
    try:
        client.heartbeat()
        chromadb_status = "ok"
    except Exception:
        chromadb_status = "error"

    try:
        embeddings_health = _get_embeddings_health()
        embeddings_status = "ok"
        embedding_model = embeddings_health.get("model")
    except EmbeddingsServiceError:
        embeddings_status = "error"
        embedding_model = None

    return {
        "status": "Server is running",
        "chromadb": chromadb_status,
        "embeddings": embeddings_status,
        "embedding_model": embedding_model,
    }


# New endpoint for searching tracks by tag string with metadata filtering
@lru_cache(maxsize=1000)
def _get_cached_query_embedding(tags: str) -> list[float]:
    """Caches query embeddings to reduce repeated remote calls."""
    return _generate_embedding(tags, "query")


class SearchParams(BaseModel):
    """Parameters for the track search endpoint."""
    tags: str = Field(
        default="", # Make tags optional for metadata-only search
        max_length=200,
        description="Search query string for semantic search (e.g., 'rock upbeat energetic'). Empty for metadata-only search."
    )
    limit: int = Field(default=5, ge=1, le=100, description="Maximum number of results")
    max_distance: float = Field(
        default=0.7, # Adjusted default max_distance, may need tuning
        ge=0.0,
        le=2.0,
        description="Maximum cosine distance for semantic matches (0.0 to 2.0). Only applies if 'tags' is provided."
    )
    # Optional metadata filters. Use Optional and None as default for flexibility.
    artist: Optional[str] = Field(default=None, description="Filter by artist name")
    album: Optional[str] = Field(default=None, description="Filter by album name")
    # Add other metadata fields you might want to filter by

    # No need for a validator for 'tags' if it's optional now


def _fuzzy_match(query: str, choices: List[str], threshold: int = 80) -> Optional[str]:
    """
    Performs fuzzy matching to find the best match for a query in a list of choices.

    Parameters:
    - query: The string to search for.
    - choices: The list of strings to search within.
    - threshold: The minimum score (0-100) for a match to be considered.

    Returns:
    - The best matching string from choices, or None if no match above the threshold is found.
    """
    if not query or not choices:
        return None
    # Use extractOne for the single best match
    match = process.extractOne(query, choices, scorer=fuzz.WRatio)
    if match and match[1] >= threshold:
        log.debug("`state` Fuzzy match found for '%s': '%s' with score %s", query, match[0], match[1])
        return match[0]
    log.debug("`state` No fuzzy match found for '%s' above threshold %s", query, threshold)
    return None


@app.get("/search_tracks/",
    response_model=List[Dict[str, Any]], # Return full metadata for results
    summary="Search similar tracks with optional metadata filtering",
    response_description="List of similar track paths and their metadata")
async def search_tracks(params: SearchParams = Depends()): # Use Depends to inject SearchParams
    """
    Search for similar tracks using semantic similarity and/or filter by metadata.

    Parameters:
    - **tags**: Search query string for semantic search (e.g., "rock upbeat energetic"). Optional.
    - **limit**: Maximum number of results to return (1-100, default: 5).
    - **max_distance**: Maximum cosine distance for semantic matches (0.0-2.0, default: 0.7). Only applies if 'tags' is provided.
    - **artist**: Optional filter by artist name. Fuzzy matching is applied to find the best match in the database.
    - **album**: Optional filter by album name. Fuzzy matching is applied to find the best match in the database.
    - Add other optional metadata filter parameters as needed.

    Returns:
    - List of dictionaries, each containing the file path and metadata of a matching track.

    Raises:
    - 400: Invalid parameter values
    - 422: Validation error
    - 500: Internal server error
    """
    log.debug("`api` Search query params: %s", params)

    # Build the where clause for metadata filtering with fuzzy matching
    where_clause = {}
    if params.artist is not None:
        # Get unique artist names and perform fuzzy matching
        unique_artists = _get_unique_metadata_values("artist")
        matched_artist = _fuzzy_match(params.artist, unique_artists)
        if matched_artist:
            where_clause["artist"] = matched_artist
            log.info("`api` Using fuzzy matched artist: '%s' for search.", matched_artist)
        else:
            # If no fuzzy match, use the original query or skip filter
            # For now, let's use the original query for exact match attempt by ChromaDB
            where_clause["artist"] = params.artist
            log.info("`api` No fuzzy match for artist '%s', attempting exact match.", params.artist)

    if params.album is not None:
        # Get unique album names and perform fuzzy matching
        unique_albums = _get_unique_metadata_values("album")
        matched_album = _fuzzy_match(params.album, unique_albums)
        if matched_album:
            where_clause["album"] = matched_album
            log.info("`api` Using fuzzy matched album: '%s' for search.", matched_album)
        else:
            # If no fuzzy match, use the original query
            where_clause["album"] = params.album
            log.info("`api` No fuzzy match for album '%s', attempting exact match.", params.album)

    # Add other metadata filters here as needed

    # Perform the query
    try:
        if params.tags:
            # Semantic search with optional metadata filtering
            embedding = _get_cached_query_embedding(params.tags)
            raw_results = collection.query(
                query_embeddings=[embedding],
                n_results=min(params.limit * 5, 100), # Request more results to filter by distance and metadata
                include=["distances", "metadatas"],
                where=where_clause if where_clause else None # Apply metadata filter if not empty
            )
            # Normalize the structure for easier processing
            if raw_results and raw_results.get("ids") and raw_results["ids"]:
                 results_ids = raw_results["ids"][0]
                 results_metadatas = raw_results.get("metadatas", [[]])[0] # type: ignore
                 results_distances = raw_results["distances"][0] # type: ignore
            else:
                 results_ids = []
                 results_metadatas = []
                 results_distances = [] # Keep empty list for consistency

        elif where_clause:
            # Metadata-only search
            raw_results = collection.get(
                where=where_clause,
                limit=params.limit, # Apply limit directly for get
                include=["metadatas"]
            )
            # Normalize the structure and add dummy distances
            if raw_results and raw_results.get("ids") and raw_results["ids"]:
                results_ids = raw_results["ids"]
                results_metadatas = raw_results["metadatas"]
                results_distances = [0.0] * len(results_ids) # Dummy distances for metadata-only search
            else:
                results_ids = []
                results_metadatas = []
                results_distances = [] # Keep empty list for consistency

        else:
            # No tags and no metadata filters
            raise HTTPException(status_code=400, detail="Either 'tags' or at least one metadata filter must be provided.")

        log.info("`api` Raw search results count: %s", len(results_ids))
        # log.debug(f"Raw search results: {pretty_repr(raw_results)}") # Avoid logging potentially large raw_results

        # Filter results by distance if semantic search was performed
        # and collect results in the desired format (list of dicts)
        filtered_results = []
        # Iterate over the normalized results lists
        for i in range(len(results_ids)):
             # For semantic search, check distance (only if tags were provided)
             if params.tags:
                distance = results_distances[i]
                if distance > params.max_distance:
                    continue # Skip if distance is too large

             # Collect the result
             result_item = {
                 "file_path": results_ids[i],
                 "metadata": results_metadatas[i] if results_metadatas else {},
             }
             filtered_results.append(result_item)

             # Stop if we reached the limit
             if len(filtered_results) >= params.limit:
                 break

        return filtered_results

    except EmbeddingsServiceError as e:
        # Handle errors from embedding generation
        log.error("`http` Search failed: %s", e)
        raise HTTPException(status_code=503, detail=f"Search failed: {e}")
    except Exception as e:
        log.error("`state` An unexpected error occurred during search: %s", e)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@app.delete("/clear_collection/",
    response_model=dict,
    summary="Clear collection",
    response_description="Collection clearing status",
    include_in_schema=False)
async def clear_collection(confirm: bool = Query(False, description="Explicitly confirm collection deletion")):
    """
    Delete all tracks from the collection.

    Warning: This operation cannot be undone.

    Returns:
    - **message**: Summary of deleted tracks
    """
    log.info("`api` Clearing entire collection")

    global collection

    if not getattr(getattr(cfg, "admin", object()), "clear_collection_enabled", False):
        raise HTTPException(status_code=403, detail="clear_collection is disabled by configuration")

    if not confirm:
        raise HTTPException(status_code=400, detail="Set confirm=true to clear the collection")

    try:
        # Get current count of tracks before deletion attempt
        try:
            # Use the client to count directly
            count = client.get_collection(cfg.chromadb.collection_name).count()
        except Exception as e: # Collection might not exist or other error
            log.warning("`state` Could not get collection count: %s", e)
            # Attempt to delete anyway, in case the error was not about non-existence
            count = 0

        # Delete the collection and recreate it
        try:
            client.delete_collection(cfg.chromadb.collection_name)
            # Recreate the collection immediately after deletion
            collection = client.create_collection(cfg.chromadb.collection_name)
            log.info("`state` Cleared collection '%s', deleted %s tracks (approximate count before deletion).", cfg.chromadb.collection_name, count)
            return {"message": f"Collection cleared, {count} tracks deleted (approximate count before deletion)."}
        except Exception as e:
            log.error("`state` Error deleting or recreating collection: %s", e)
            # If deletion/recreation fails, try to get the count again for the error message
            try:
                current_count = client.get_collection(cfg.chromadb.collection_name).count()
                error_msg = f"Error clearing collection. Current track count: {current_count}. Error: {e}"
            except Exception:
                error_msg = f"Error clearing collection. Could not get current track count. Error: {e}"
            raise HTTPException(status_code=500, detail=error_msg)


    except Exception as e:
        log.error("`state` An unexpected error occurred during clear_collection: %s", e)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


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
        },
        "message": "Collection statistics"
    }
    ```
    """
    log.info("`metrics` Getting collection statistics")

    try:
        count = collection.count()

        if count == 0:
             return {
                "total_tracks": 0,
                "total_size_mb": 0.0,
                "metadata_stats": {},
                "embedding_dimensions": 0,
                "message": "Collection is empty"
            }

        # Fetch a sample of data to get embedding dimension and metadata info
        sample_results = collection.get(
            limit=10, # Fetch a small sample
            include=["metadatas", "embeddings"]
        )

        total_tracks = count # Use the actual count

        # Calculate embedding size (approximate)
        embedding_dimension = 0
        embedding_size = 0.0
        
        # Check if embeddings exist and are not empty
        if "embeddings" in sample_results and len(sample_results["embeddings"]) > 0: # type: ignore
            # Handle both list of floats and numpy array embeddings
            first_embedding = sample_results["embeddings"][0] # type: ignore
            if isinstance(first_embedding, list):
                embedding_dimension = len(first_embedding)
            elif hasattr(first_embedding, 'shape'):  # Check if it's a numpy array or similar
                embedding_dimension = first_embedding.shape[0]
            # Assuming float32 (4 bytes)
            embedding_size = embedding_dimension * 4 * total_tracks / (1024 * 1024)  # Convert to MB
        else:
            log.warning("`metrics` No embeddings found in sample to calculate size and dimension.")


        metadata_fields = {}
        metadatas = sample_results.get("metadatas", [])

        if metadatas is not None:
            for metadata in metadatas:
                if metadata:
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

        metadata_stats = {}
        for key, stats in metadata_fields.items():
            metadata_stats[key] = {
                "sample_count": stats["count"],
                "unique_values_in_sample": len(stats["unique_values"]),
                "types": list(stats["types"])
            }
        metadata_stats["_note"] = "Metadata stats are based on a sample of 10 tracks for performance. For full stats, a more extensive process is needed."


        return {
            "total_tracks": total_tracks,
            "total_size_mb": round(embedding_size, 2),
            "metadata_stats": metadata_stats,
            "embedding_dimensions": embedding_dimension,
            "message": "Collection statistics (metadata stats based on sample)"
        }

    except Exception as e:
        log.error("`metrics` Error getting collection stats: %s", e)
        raise HTTPException(status_code=500, detail=f"Error getting collection stats: {e}")


def _get_unique_metadata_values(key: str) -> List[Any]:
    """
    Get a list of all unique values for a specific metadata key across all tracks.

    Parameters:
    - **key**: The metadata key to retrieve unique values for.

    Returns:
    - List of unique metadata values.
    """
    log.info("`api` Getting unique values for metadata key: %s", key)

    try:
        # Fetch all tracks, including only the specified metadata key
        # This might be inefficient for very large collections.
        # A more efficient approach might involve a dedicated metadata index
        # or a different ChromaDB query if available.
        all_tracks = collection.get(
            include=["metadatas"]
        )
        # Handle case where results might be None (e.g., empty collection)
        if all_tracks is None:
            all_tracks = {"metadatas": []}
        # Ensure all_tracks is a dictionary
        if not isinstance(all_tracks, dict):
            all_tracks = {"metadatas": []}
        # Ensure 'ids' and 'metadatas' are present
        if "ids" not in all_tracks:
            all_tracks["ids"] = []
        if "metadatas" not in all_tracks:
            all_tracks["metadatas"] = []

        unique_values = set()
        metadatas = all_tracks.get("metadatas")
        if metadatas: # Add check for None
            for metadata in metadatas:
                if metadata and key in metadata:
                    value = metadata[key]
                    unique_values.add(value)

        # Convert set to list and sort for consistent output
        sorted_values = sorted(list(unique_values), key=str) # Sort by string representation

        log.info("`api` Found %s unique values for key '%s'.", len(sorted_values), key)
        return sorted_values

    except Exception as e:
        log.error("`api` Error getting metadata list for key '%s': %s", key, e)
        # Re-raise the exception to be handled by the calling function
        raise

@app.get("/get_metadata_list/",
    response_model=dict,
    summary="Get unique values for a metadata key",
    response_description="List of unique values for specified metadata key")
async def get_metadata_list(key: str = Query(..., description="Metadata key (e.g., 'artist', 'album', etc.)")):
    """
    Get a list of all unique values for a specified metadata key.

    Parameters:
    - **key**: Metadata key to get values for (e.g., 'artist', 'album', 'genre', etc.)

    Returns:
    - **values**: List of unique values for the specified key
    - **count**: Total number of unique values
    """
    try:
        unique_values = _get_unique_metadata_values(key)
        return {
            "values": unique_values,
            "count": len(unique_values),
            "message": f"Found {len(unique_values)} unique values for '{key}'"
        }
    except Exception as e:
        # Handle exceptions raised by _get_unique_metadata_values
        raise HTTPException(status_code=500, detail=f"Error getting metadata list: {e}")


def _default_logging_config() -> dict[str, Any]:
    return {
        "level": "INFO",
        "console": True,
        "file_enabled": False,
        "file": "logs/music2db-server.log",
        "show_time": True,
        "time_format": "%H:%M:%S",
        "show_level": False,
        "show_path": False,
        "logs_width": 140,
        "tags_width": 16,
        "tag_filter_mode": "any",
        "unknown_tags": "hide",
        "show_all_tags_errors": True,
        "show_all_tags_warnings": True,
        "level_decor": {
            "notset": {"symbol": "█"},
            "debug": {"symbol": "█"},
            "info": {"symbol": "█"},
            "warning": {"symbol": "█"},
            "error": {"symbol": "█"},
            "critical": {"symbol": "█"},
        },
        "loggers": {
            "uvicorn": "WARNING",
            "uvicorn.error": "WARNING",
            "uvicorn.access": "WARNING",
            "starlette": "WARNING",
            "fastapi": "WARNING",
            "httpx": "WARNING",
            "httpcore": "WARNING",
            "urllib3.connectionpool": "WARNING",
        },
        "tags": {
            "startup": {"show": True, "icon": "S", "tag_color": "#5f875f", "icon_color": "#ffffff"},
            "config": {"show": True, "icon": "cfg", "tag_color": "#5f5f87", "icon_color": "#ffffff"},
            "api": {"show": True, "icon": "api", "tag_color": "#005f87", "icon_color": "#ffffff"},
            "http": {"show": False, "icon": "http", "tag_color": "#444444", "icon_color": "#ffffff"},
            "metrics": {"show": False, "icon": "M", "tag_color": "#008787", "icon_color": "#ffffff"},
            "state": {"show": True, "icon": "st", "tag_color": "#875f00", "icon_color": "#ffffff"},
        },
    }


def _config_search_dirs() -> list[Path]:
    return [SYSTEM_CONFIG_DIR, XDG_CONFIG_DIR, LOCAL_CONFIG_DIR]


def _discover_config_files(filename: str) -> list[str]:
    return [str(path / filename) for path in _config_search_dirs() if (path / filename).exists()]


def _resolve_logging_config_file() -> Path | None:
    explicit_path = getattr(getattr(cfg, "app", object()), "logging_config", None)
    if explicit_path:
        return Path(str(explicit_path)).expanduser()

    for config_file in reversed(ACTIVE_CONFIG_FILES):
        logging_path = config_file.parent / "logging.yaml"
        if logging_path.exists():
            return logging_path

    for candidate in reversed(_config_search_dirs()):
        logging_path = candidate / "logging.yaml"
        if logging_path.exists():
            return logging_path

    return None


def _merge_logging_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_logging_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_logging_config() -> dict[str, Any]:
    config = _default_logging_config()
    logging_path = _resolve_logging_config_file()

    if logging_path and logging_path.exists():
        import yaml

        with logging_path.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}
        config = _merge_logging_config(config, loaded)

    return config


def _init_logs() -> None:
    """Initializes logging configuration."""
    logging_config = _load_logging_config()
    setup_logging(logging_config)
    log.info(
        "`config` Logging configured from %s",
        str(_resolve_logging_config_file() or "<defaults>"),
    )

    if str(logging_config.get("level", "INFO")).upper() == "DEBUG":
        log.debug("`config` Active config files: %s", [str(path) for path in ACTIVE_CONFIG_FILES])


def _parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(prog=APP_NAME, description="Music2DB Server")
    parser.add_argument(
        "-c",
        "--config",
        dest="config_file",
        default=None,
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
    global ACTIVE_CONFIG_FILES
    ACTIVE_CONFIG_FILES = [Path(file).expanduser() for file in files]
    cfg.load_files(files)
    cfg.load_args(unknown_args)


def _resolve_config_files(config_file: str | None) -> list[str]:
    if config_file:
        return [str(Path(config_file).expanduser())]
    return _discover_config_files("config.yaml")

    # # add some/override config here if needed
    # cfg.update('runtime.blablabla', True)
    # cfg.update('religion.buddhism.name', 'Gautama Siddharta')


def main():
    """Main function to initialize and run the FastAPI application."""
    args, unknown_args = _parse_args()
    config_files = _resolve_config_files(args.config_file)
    _init_config(config_files, unknown_args)
    _init_logs() # Initialize logs after config is loaded
    _init_entities() # Initialize entities after config and logs

    log.info("`startup` Starting %s", APP_NAME)
    # Use the 'app' instance that was initialized first
    # Add a check if cfg.app, cfg.app.host, and cfg.app.port exist
    if not hasattr(cfg, 'app') or not hasattr(cfg.app, 'host') or not hasattr(cfg.app, 'port'):
         log.error("`config` App host or port configuration missing in config.yaml")
         sys.exit(1)

    uvicorn.run(
        app,
        host=cfg.app.host,
        port=cfg.app.port,
        log_level=str(_load_logging_config().get("loggers", {}).get("uvicorn", "warning")).lower(),
    )


# Run server
if __name__ == "__main__":
    log.info("`startup` Starting %s", APP_NAME)
    sys.exit(main())
