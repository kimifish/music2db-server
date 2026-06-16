# pyright: basic
# pyright: reportAttributeAccessIssue=false

import argparse
import sys
from typing import Dict, Any
from dotenv import load_dotenv
from rich.traceback import install as install_rich_traceback
from fastapi import FastAPI, HTTPException, Query, Depends
from pydantic import BaseModel, Field, ValidationError
import chromadb
from typing import Dict, Union, List, Optional
import uvicorn
from functools import lru_cache
from rapidfuzz import process, fuzz
from .config_loader import get_active_config_files, load_settings, load_logging_config, resolve_config_files, resolve_logging_config_file, set_active_config_files
from .embeddings import (
    EmbeddingsServiceError,
    generate_embedding,
    generate_embeddings,
    get_embeddings_health,
    get_embeddings_settings,
)
from .logging_setup import get_logger, setup_logging
from .settings import Settings

try:
    from music2db_server import __version__
except ImportError:
    __version__ = "0.2.0"


# Initialize config
APP_NAME = "music2db_server"

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
settings: Settings


def _get_embeddings_settings() -> tuple[str, Optional[str], bool, float]:
    return get_embeddings_settings(settings, log)


def _generate_embedding(text: str, input_type: str) -> list[float]:
    return generate_embedding(settings, log, text, input_type)


def _generate_embeddings(texts: list[str], input_type: str) -> list[list[float]]:
    return generate_embeddings(settings, log, texts, input_type)


def _get_embeddings_health() -> dict[str, Any]:
    return get_embeddings_health(settings, log)

def _init_entities():
    """Initializes the ChromaDB client/collection and validates embeddings service."""
    global client, collection

    embeddings_health = _get_embeddings_health()
    log.info(
        "`startup` Embeddings service ready: %s (%s)",
        embeddings_health.get("model", "unknown-model"),
        _get_embeddings_settings()[0],
    )

    client = chromadb.HttpClient(host=settings.chromadb.host, port=settings.chromadb.port)
    collection = client.get_or_create_collection(settings.chromadb.collection_name)


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


@app.delete("/delete_track/",
    response_model=dict,
    summary="Delete a single track",
    response_description="Track deletion status")
async def delete_track(file_path: str = Query(..., description="Track file_path used as the ChromaDB ID")):
    """
    Delete a single track from the collection by its file_path ID.

    This endpoint is idempotent: deleting an already absent track returns success
    with deleted=false so clients can safely sync removed local files.
    """
    log.info("`api` Deleting track: %s", file_path)

    try:
        existing = collection.get(ids=[file_path], include=[])
        if not existing.get("ids"):
            log.info("`state` Track '%s' not found for deletion.", file_path)
            return {
                "message": f"Track '{file_path}' was not found",
                "file_path": file_path,
                "deleted": False,
            }

        collection.delete(ids=[file_path])
        log.info("`state` Track '%s' deleted successfully.", file_path)
        return {
            "message": f"Track '{file_path}' deleted successfully",
            "file_path": file_path,
            "deleted": True,
        }
    except Exception as e:
        log.error("`state` Error deleting track '%s': %s", file_path, e)
        raise HTTPException(status_code=500, detail=f"Error deleting track: {e}")


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

    if not settings.admin.clear_collection_enabled:
        raise HTTPException(status_code=403, detail="clear_collection is disabled by configuration")

    if not confirm:
        raise HTTPException(status_code=400, detail="Set confirm=true to clear the collection")

    try:
        # Get current count of tracks before deletion attempt
        try:
            # Use the client to count directly
            count = client.get_collection(settings.chromadb.collection_name).count()
        except Exception as e: # Collection might not exist or other error
            log.warning("`state` Could not get collection count: %s", e)
            # Attempt to delete anyway, in case the error was not about non-existence
            count = 0

        # Delete the collection and recreate it
        try:
            client.delete_collection(settings.chromadb.collection_name)
            # Recreate the collection immediately after deletion
            collection = client.create_collection(settings.chromadb.collection_name)
            log.info("`state` Cleared collection '%s', deleted %s tracks (approximate count before deletion).", settings.chromadb.collection_name, count)
            return {"message": f"Collection cleared, {count} tracks deleted (approximate count before deletion)."}
        except Exception as e:
            log.error("`state` Error deleting or recreating collection: %s", e)
            # If deletion/recreation fails, try to get the count again for the error message
            try:
                current_count = client.get_collection(settings.chromadb.collection_name).count()
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


def _resolve_logging_config_file():
    return resolve_logging_config_file(settings, APP_NAME)


def _load_logging_config() -> dict[str, Any]:
    return load_logging_config(settings, APP_NAME)


def _init_logs() -> None:
    """Initializes logging configuration."""
    logging_config = _load_logging_config()
    setup_logging(logging_config)
    log.info(
        "`config` Logging configured from %s",
        str(_resolve_logging_config_file() or "<defaults>"),
    )

    if str(logging_config.get("level", "INFO")).upper() == "DEBUG":
        log.debug("`config` Active config files: %s", [str(path) for path in get_active_config_files()])


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
    return parser.parse_args()


def _init_config(files: list[str]):
    """
    Initializes the configuration by loading configuration files.

    Args:
        files (List[str]): List of config files.
    """
    global settings
    if not files:
        print(f"No config.yaml found for {APP_NAME}. Checked /etc, XDG config, and ./config", file=sys.stderr)
        raise SystemExit(1)

    set_active_config_files(files)
    try:
        settings = load_settings(files)
    except ValidationError as exc:
        print(f"Invalid configuration for {APP_NAME}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def _resolve_config_files(config_file: str | None) -> list[str]:
    return resolve_config_files(APP_NAME, config_file)


def main():
    """Main function to initialize and run the FastAPI application."""
    args = _parse_args()
    config_files = _resolve_config_files(args.config_file)
    _init_config(config_files)
    _init_logs() # Initialize logs after config is loaded
    _init_entities() # Initialize entities after config and logs

    log.info("`startup` Starting %s", APP_NAME)

    uvicorn.run(
        app,
        host=settings.app.host,
        port=settings.app.port,
        log_level=str(_load_logging_config().get("loggers", {}).get("uvicorn", "warning")).lower(),
    )


# Run server
if __name__ == "__main__":
    log.info("`startup` Starting %s", APP_NAME)
    sys.exit(main())
