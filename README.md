
# Music2DB Server

Version: 0.3.4

A FastAPI-based server that provides embedding-based music track indexing and similarity search using ChromaDB and an external embeddings service.

## Features

- Store music tracks with associated metadata in ChromaDB
- Generate embeddings for music tracks via external HTTP embeddings API
- Search similar tracks using semantic similarity
- RESTful API with FastAPI
- Configurable via YAML files
- Efficient batch operations
- Cached query embeddings for repeated searches

## Requirements

- Python 3.11 or higher
- ChromaDB server instance
- `serv4-embeddings` HTTP service
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
    cp config/config.yaml ~/.config/music2db_server/config.yaml
    cp config/logging.yaml ~/.config/music2db_server/logging.yaml
    ```

3. Install systemd service:

    ```bash
    mkdir -p ~/.config/systemd/user/
    cp packaging/music2db-server.service ~/.config/systemd/user/
    systemctl --user daemon-reload
    ```

## Configuration

Configuration files are loaded in this order, with later files overriding earlier ones:

1. `/etc/music2db_server/config.yaml`
2. `$XDG_CONFIG_HOME/music2db_server/config.yaml` or `~/.config/music2db_server/config.yaml`
3. `./config/config.yaml` relative to the current working directory

You can also specify a custom config path with `-c`.

Default `config/config.yaml`:

```yaml
app:
  host: "0.0.0.0"
  port: 5005

embeddings:
  base_url: "http://127.0.0.1:8098"
  model: "intfloat/multilingual-e5-small"
  normalize: true
  timeout_seconds: 30

chromadb:
  host: "localhost"
  port: 8000
  collection_name: "music_collection"

admin:
  clear_collection_enabled: false
```

Embeddings are requested from `serv4-embeddings` over HTTP. For E5 models the server uses `input_type=passage` while indexing and `input_type=query` while searching.

Logging is loaded with the same precedence from `logging.yaml` in `/etc/music2db_server/`, `$XDG_CONFIG_HOME/music2db_server/` and local `./config/`. A custom path can still be set via `app.logging_config`.

## Usage

1. Start the server:

    ```bash
    music2db-server
    # Or with custom config:
    music2db-server -c /path/to/config.yaml
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
        [
            {
                "file_path": "/music/path/to/track.mp3",
                "metadata": { ... }
            }
        ]
        ```

    - `GET /list_tracks/`: List all tracked files
    - `DELETE /delete_track/?file_path=...`: Delete one tracked file by its `file_path` ID; already absent tracks return success with `deleted: false`
    - `GET /health/`: Server health check
    - `DELETE /clear_collection/?confirm=true`: Hidden maintenance endpoint, available only when `admin.clear_collection_enabled=true`

    Clients can use `GET /list_tracks/` to compare indexed IDs with local files and call `DELETE /delete_track/` for files that were removed, moved, or renamed locally.

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
