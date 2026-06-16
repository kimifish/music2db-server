# Music2DB Client Contract

This document defines the compatibility contract between `music2db-client` and `music2db-server`.

The client must treat this document as the source of truth when it is refactored.

## Base URL

Default server URL:

```text
http://<music2db-server-host>:5005
```

Current home deployment commonly uses:

```text
http://kimihome.lan:5005
```

The client must make the base URL configurable.

## Authentication

No authentication is required.

The client must not send API keys, bearer tokens, cookies, or custom auth headers unless a future contract version explicitly requires them.

## Transport

- Protocol: HTTP
- Request bodies: JSON
- Response bodies: JSON
- Recommended timeout: at least 30 seconds for batch indexing requests
- Client should handle transient `503` responses from external embeddings service failures

## Track Object

Track objects sent by the client must have this shape:

```json
{
  "file_path": "relative/or/absolute/path/to/track.mp3",
  "metadata": {
    "artist": "Artist Name",
    "title": "Track Title",
    "album": "Album Name",
    "genre": "Rock",
    "year": "2024",
    "tags": "upbeat, energetic",
    "length": 245
  }
}
```

### `file_path`

- Required.
- Type: string.
- Used as the ChromaDB document ID.
- Must be stable across repeated scans.
- Current client behavior sends paths relative to the configured music root. This is acceptable and preferred for portability.

### `metadata`

- Required.
- Type: object.
- Values must be JSON primitives accepted by the server model:
  - string
  - integer
  - float
  - boolean
- Do not send arrays, nested objects, or null values.
- Client should remove metadata keys with `None`/null values before sending.

Recommended metadata keys:

```text
artist, title, album, genre, year, tags, length
```

The server currently uses `genre`, `tags`, `title`, and `artist` to build the text sent to the embeddings service. Other fields are stored for filtering or display.

## Health Check

### Request

```http
GET /health/
```

### Success Response

Status: `200 OK`

```json
{
  "status": "Server is running",
  "chromadb": "ok",
  "embeddings": "ok",
  "embedding_model": "intfloat/multilingual-e5-small"
}
```

### Client Requirements

- Client must require `status == "Server is running"`.
- Client should treat `chromadb != "ok"` as not ready.
- Client should treat `embeddings != "ok"` as not ready for indexing or semantic search.
- Client should log `embedding_model` when present.

## Add Single Track

### Request

```http
POST /add_track/
Content-Type: application/json
```

Body:

```json
{
  "file_path": "Artist/Album/Track.mp3",
  "metadata": {
    "artist": "Artist",
    "title": "Track",
    "album": "Album",
    "genre": "Rock",
    "tags": "tag1, tag2",
    "length": 245
  }
}
```

### Success Response

Status: `200 OK`

```json
{
  "message": "Track 'Artist/Album/Track.mp3' added successfully"
}
```

Existing identical metadata may return:

```json
{
  "message": "Track 'Artist/Album/Track.mp3' already exists with same metadata"
}
```

Existing changed metadata may return:

```json
{
  "message": "Track 'Artist/Album/Track.mp3' updated successfully"
}
```

### Error Responses

- `400`: invalid or empty embedding text after fallback
- `422`: request validation error
- `503`: embeddings service failed
- `500`: ChromaDB or unexpected server error

## Add Tracks Batch

### Request

```http
POST /add_tracks/
Content-Type: application/json
```

Body:

```json
[
  {
    "file_path": "Artist/Album/Track 1.mp3",
    "metadata": {
      "artist": "Artist",
      "title": "Track 1",
      "album": "Album",
      "genre": "Rock",
      "tags": "tag1, tag2",
      "length": 245
    }
  },
  {
    "file_path": "Artist/Album/Track 2.mp3",
    "metadata": {
      "artist": "Artist",
      "title": "Track 2"
    }
  }
]
```

### Success Response

Status: `200 OK`

```json
{
  "message": "Batch processing complete: 10 added, 2 updated, 3 skipped, 0 errors",
  "added": 10,
  "updated": 2,
  "skipped": 3,
  "errors": []
}
```

If per-track errors occur, `errors` contains objects like:

```json
{
  "file_path": "Artist/Album/Broken.mp3",
  "error": "error message"
}
```

### Client Requirements

- Preferred indexing endpoint.
- Client should batch requests. Current safe default: 100 tracks per request.
- Client must send a JSON array directly, not wrapped in an object.
- Client should treat HTTP `200` with non-empty `errors` as partial success.
- Client should log `added`, `updated`, `skipped`, and error count.
- Client should retry whole batches only when it can tolerate idempotent reprocessing.

## Search Tracks

### Request

```http
GET /search_tracks/
```

Query parameters:

| Name | Type | Required | Default | Constraints | Description |
| --- | --- | --- | --- | --- | --- |
| `tags` | string | no | `""` | max length 200 | Semantic query text. Empty is allowed only when a metadata filter is present. |
| `limit` | integer | no | `5` | `1..100` | Maximum number of returned results. |
| `max_distance` | float | no | `0.7` | `0.0..2.0` | Maximum semantic distance. Only applies when `tags` is non-empty. |
| `artist` | string | no | null | | Metadata filter with server-side fuzzy matching. |
| `album` | string | no | null | | Metadata filter with server-side fuzzy matching. |

At least one of these must be provided:

- non-empty `tags`
- `artist`
- `album`

### Examples

Semantic search:

```http
GET /search_tracks/?tags=upbeat%20rock&limit=20&max_distance=0.9
```

Metadata-only search:

```http
GET /search_tracks/?artist=Radiohead&limit=20
```

Combined search:

```http
GET /search_tracks/?tags=melancholic&artist=Radiohead&album=OK%20Computer&limit=10
```

### Success Response

Status: `200 OK`

```json
[
  {
    "file_path": "Artist/Album/Track.mp3",
    "metadata": {
      "artist": "Artist",
      "title": "Track",
      "album": "Album",
      "genre": "Rock",
      "tags": "tag1, tag2",
      "length": 245
    }
  }
]
```

Empty result:

```json
[]
```

### Client Requirements

- Client must consume `file_path`, not `path`.
- Client must not require a `distance` field. The current server response does not include distance.
- Client should display metadata when available.
- Client should support empty result arrays.

### Known Legacy Client Incompatibility

The current legacy `music2db-client/src/music2db_client/search_by_tags.py` expects:

```json
{
  "path": "...",
  "distance": 0.42
}
```

That shape is not part of this contract. Refactored clients must use `file_path` and `metadata`.

## List Tracks

### Request

```http
GET /list_tracks/
```

### Success Response

Status: `200 OK`

```json
{
  "tracks": [
    "Artist/Album/Track 1.mp3",
    "Artist/Album/Track 2.mp3"
  ]
}
```

### Client Requirements

- Client may use this endpoint to compare indexed IDs with local files.
- Returned values are track IDs, currently equal to submitted `file_path` values.

## Delete Single Track

### Request

```http
DELETE /delete_track/?file_path=Artist%2FAlbum%2FTrack.mp3
```

Query parameters:

| Name | Type | Required | Description |
| --- | --- | --- | --- |
| `file_path` | string | yes | Track ID to delete. Must exactly match the `file_path` previously sent by the client. |

### Success Response

Status: `200 OK`

Deleted record:

```json
{
  "message": "Track 'Artist/Album/Track.mp3' deleted successfully",
  "file_path": "Artist/Album/Track.mp3",
  "deleted": true
}
```

Already absent record:

```json
{
  "message": "Track 'Artist/Album/Track.mp3' was not found",
  "file_path": "Artist/Album/Track.mp3",
  "deleted": false
}
```

### Error Responses

- `422`: missing or invalid `file_path` query parameter
- `500`: ChromaDB or unexpected server error

### Client Requirements

- Client should call this endpoint for indexed files that no longer exist locally or should no longer be indexed.
- Client must URL-encode `file_path` because it can contain `/`, spaces, and other special characters.
- Client should treat both `deleted: true` and `deleted: false` as successful synchronization states.
- For renamed or moved files, client should delete the old `file_path` and add the new `file_path`.

## Metadata List

### Request

```http
GET /get_metadata_list/?key=artist
```

Query parameters:

| Name | Type | Required | Description |
| --- | --- | --- | --- |
| `key` | string | yes | Metadata key to enumerate. Examples: `artist`, `album`, `genre`. |

### Success Response

Status: `200 OK`

```json
{
  "values": ["Artist A", "Artist B"],
  "count": 2,
  "message": "Found 2 unique values for 'artist'"
}
```

### Client Requirements

- Client may use this endpoint for autocomplete/filter UIs.
- Values can be strings, numbers, or booleans depending on stored metadata.

## Collection Stats

### Request

```http
GET /collection_stats/
```

### Success Response

Status: `200 OK`

```json
{
  "total_tracks": 1000,
  "total_size_mb": 1.46,
  "metadata_stats": {
    "artist": {
      "sample_count": 10,
      "unique_values_in_sample": 7,
      "types": ["str"]
    },
    "_note": "Metadata stats are based on a sample of 10 tracks for performance. For full stats, a more extensive process is needed."
  },
  "embedding_dimensions": 384,
  "message": "Collection statistics (metadata stats based on sample)"
}
```

Empty collection:

```json
{
  "total_tracks": 0,
  "total_size_mb": 0.0,
  "metadata_stats": {},
  "embedding_dimensions": 0,
  "message": "Collection is empty"
}
```

### Client Requirements

- Client may use this endpoint for diagnostics only.
- Client must not rely on metadata stats being complete; they are sample-based.

## Clear Collection

### Request

```http
DELETE /clear_collection/?confirm=true
```

### Availability

- Hidden from OpenAPI schema.
- Disabled by default via server config.
- Returns `403` when disabled.
- Returns `400` when `confirm=true` is missing.

### Client Requirements

- Normal clients must not call this endpoint.
- Maintenance tools may call it only when explicitly requested by an operator.

## Error Handling

The client must handle these common status codes:

| Status | Meaning | Client behavior |
| --- | --- | --- |
| `200` | Success | Parse JSON body. |
| `400` | Bad request | Show server `detail`; do not retry unchanged. |
| `403` | Admin operation disabled | Show message; do not retry. |
| `422` | Validation error | Treat as client bug or bad metadata serialization. |
| `500` | Server/ChromaDB error | Log and allow manual retry. |
| `503` | Embeddings service unavailable or failed | Retry later with backoff. |

FastAPI error responses generally have this shape:

```json
{
  "detail": "error message"
}
```

Validation errors use FastAPI's standard `422` structure.

## Indexing Semantics

- Re-sending a track with identical metadata is idempotent and should be treated as skipped.
- Re-sending a track with changed metadata updates the existing ChromaDB record.
- Deleting a track by `file_path` is idempotent; already absent records return `200 OK` with `deleted: false`.
- The client is responsible for detecting deleted, moved, or renamed local files and requesting deletion of stale `file_path` IDs.
- The server generates embeddings; the client must not send embeddings.
- The server uses `input_type=passage` for indexing and `input_type=query` for search through the external embeddings service.
- If embeddings model or ChromaDB collection is reset, the client should perform a full rescan/reindex.

## Compatibility Rules For Client Refactor

The refactored client must preserve these behaviors:

- Health check before scanning.
- Batch upload to `/add_tracks/`.
- Per-track deletion through `/delete_track/` for stale indexed IDs.
- Stable `file_path` IDs relative to configured music root.
- Metadata values serialized as flat JSON primitives.
- Ignore directories containing `.ignore` if preserving current client behavior.
- Skip symlinks if preserving current client behavior.
- Configurable server URL and port.
- Graceful handling of empty search results and partial batch errors.

The refactored client must change these legacy behaviors:

- Search result parser must stop expecting `path`.
- Search result parser must stop expecting `distance`.
- Search result parser must use `file_path` and optionally display `metadata`.

## Contract Version

Contract version: `2026-05-29`

Compatible server baseline: `music2db-server 0.3.4+`
