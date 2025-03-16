
# Music2DB Server

Version: 0.2.3

A FastAPI-based server that provides embedding-based music track indexing and similarity search using ChromaDB and Sentence Transformers.

## Features

- Store music tracks with associated metadata in ChromaDB
- Generate embeddings for music tracks based on their metadata
- Search similar tracks using semantic similarity
- RESTful API with FastAPI
- Configurable via YAML files
- Prometheus metrics for monitoring
- Efficient batch operations
- Caching for improved performance

## Requirements

- Python 3.10 or higher
- ChromaDB server instance
- UV package manager (recommended)

## Installation

### Quick Install (recommended)

1. Clone the repository:

    ```bash
    git clone <repository-url>
    cd music2db-server
    ```

2. Run the installation script:

    ```bash
    ./install.sh
    ```

3. Start the service:

    ```bash
    systemctl --user enable music2db-server
    systemctl --user start music2db-server
    ```

### Manual Installation

1. Install the package:

    ```bash
    pip install -e .
    ```

2. Create configuration:

    ```bash
    mkdir -p ~/.config/music2db_server
    cp config.yaml ~/.config/music2db_server/
    ```

3. Install systemd service:

    ```bash
    mkdir -p ~/.config/systemd/user/
    cp packaging/music2db-server.service ~/.config/systemd/user/
    systemctl --user daemon-reload
    ```

## Configuration

Create a configuration file at `~/.config/music2db_server/config.yaml` or specify a custom path using the `-c` flag:

```yaml
app:
  host: "0.0.0.0"
  port: 5005

logging:
  level: "INFO"
  loggers:
    suppress:
      - "httpx"
      - "httpcore"
      - "mutagen"
      - "uvicorn"
    suppress_level: "WARNING"

model:
  name: "all-MiniLM-L6-v2"

chromadb:
  host: "localhost"
  port: 8000
  collection_name: "music_collection"
```

## Usage

1. Start the server:

    ```bash
    python src/server.py
    # Or with custom config:
    python src/server.py -c /path/to/config.yaml
    ```

2. API Endpoints:

    - `POST /add_track/`: Add a single track

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

    - `POST /add_tracks/`: Add multiple tracks (batch operation)
  
        ```json
        [
            {
                "file_path": "...",
                "metadata": { ... }
            }
        ]
        ```

    - `GET /search_tracks/?tags=rock%20upbeat&limit=5`: Search similar tracks

        ```json
        {
            "tracks": [
                {
                    "path": "/music/path/to/track.mp3",
                    "metadata": { ... },
                    "similarity_score": 0.95
                }
            ]
        }
        ```

    - `GET /list_tracks/`: List all tracked files
    - `GET /health/`: Server health check
    - `DELETE /clear_collection/`: Clear all tracks (use with caution)

## Monitoring

The server exposes Prometheus metrics at `/metrics` endpoint:

- `music2db_requests_total`: Total requests by endpoint
- `music2db_request_latency_seconds`: Request latency by endpoint

## Development

1. Install development dependencies:

    ```bash
    uv pip install -e ".[dev]"
    ```

2. Run tests:

    ```bash
    pytest tests/
    ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
