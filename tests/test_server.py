import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
import types

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

chromadb_stub = types.ModuleType("chromadb")
chromadb_stub.HttpClient = object
chromadb_stub.Collection = object
sys.modules.setdefault("chromadb", chromadb_stub)

rapidfuzz_stub = types.ModuleType("rapidfuzz")
rapidfuzz_stub.process = SimpleNamespace(extractOne=lambda *args, **kwargs: None)
rapidfuzz_stub.fuzz = SimpleNamespace(WRatio=object())
sys.modules.setdefault("rapidfuzz", rapidfuzz_stub)

from music2db_server import server
from music2db_server import config_loader
from music2db_server import embeddings as embeddings_module
from music2db_server.settings import AdminSettings, AppSettings, ChromaDBSettings, EmbeddingsSettings, Settings


def test_generate_embedding_uses_external_service(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "model": "intfloat/multilingual-e5-small",
                "input_type": "query",
                "dimensions": 2,
                "embeddings": [[0.1, 0.2]],
            }

    class FakeClient:
        def __init__(self, timeout):
            captured["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json):
            captured["url"] = url
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr(embeddings_module.httpx, "Client", FakeClient)
    monkeypatch.setattr(
        server,
        "settings",
        Settings(
            chromadb=ChromaDBSettings(host="localhost"),
            embeddings=EmbeddingsSettings(
                base_url="http://127.0.0.1:8098",
                model="intfloat/multilingual-e5-small",
                normalize=True,
                timeout_seconds=12,
            ),
        ),
        raising=False,
    )

    embedding = server._generate_embedding("hello", "query")

    assert embedding == [0.1, 0.2]
    assert captured == {
        "timeout": 12.0,
        "url": "http://127.0.0.1:8098/v1/embeddings",
        "json": {
            "texts": ["hello"],
            "input_type": "query",
            "normalize": True,
            "model": "intfloat/multilingual-e5-small",
        },
    }


def test_clear_collection_is_hidden_and_forbidden_when_disabled(monkeypatch):
    monkeypatch.setattr(
        server,
        "settings",
        Settings(
            chromadb=ChromaDBSettings(host="localhost", collection_name="music_collection"),
            admin=AdminSettings(clear_collection_enabled=False),
        ),
        raising=False,
    )

    client = TestClient(server.app)

    assert "/clear_collection/" not in client.get("/openapi.json").json()["paths"]

    response = client.delete("/clear_collection/?confirm=true")

    assert response.status_code == 403
    assert response.json()["detail"] == "clear_collection is disabled by configuration"


def test_delete_track_deletes_existing_track(monkeypatch):
    class FakeCollection:
        def __init__(self):
            self.deleted_ids = None

        def get(self, ids, include):
            assert ids == ["Artist/Album/Track.mp3"]
            assert include == []
            return {"ids": ids}

        def delete(self, ids):
            self.deleted_ids = ids

    fake_collection = FakeCollection()
    monkeypatch.setattr(server, "collection", fake_collection, raising=False)

    client = TestClient(server.app)
    response = client.delete("/delete_track/", params={"file_path": "Artist/Album/Track.mp3"})

    assert response.status_code == 200
    assert response.json() == {
        "message": "Track 'Artist/Album/Track.mp3' deleted successfully",
        "file_path": "Artist/Album/Track.mp3",
        "deleted": True,
    }
    assert fake_collection.deleted_ids == ["Artist/Album/Track.mp3"]


def test_delete_track_is_idempotent_for_missing_track(monkeypatch):
    class FakeCollection:
        def __init__(self):
            self.delete_called = False

        def get(self, ids, include):
            return {"ids": []}

        def delete(self, ids):
            self.delete_called = True

    fake_collection = FakeCollection()
    monkeypatch.setattr(server, "collection", fake_collection, raising=False)

    client = TestClient(server.app)
    response = client.delete("/delete_track/", params={"file_path": "missing.mp3"})

    assert response.status_code == 200
    assert response.json() == {
        "message": "Track 'missing.mp3' was not found",
        "file_path": "missing.mp3",
        "deleted": False,
    }
    assert fake_collection.delete_called is False


def test_delete_track_returns_500_for_collection_error(monkeypatch):
    class FakeCollection:
        def get(self, ids, include):
            raise RuntimeError("chromadb failed")

    monkeypatch.setattr(server, "collection", FakeCollection(), raising=False)

    client = TestClient(server.app)
    response = client.delete("/delete_track/", params={"file_path": "broken.mp3"})

    assert response.status_code == 500
    assert response.json()["detail"] == "Error deleting track: chromadb failed"


def test_delete_track_is_in_openapi(monkeypatch):
    client = TestClient(server.app)

    assert "/delete_track/" in client.get("/openapi.json").json()["paths"]


def test_collection_stats_collects_all_metadata_fields(monkeypatch):
    class FakeCollection:
        def count(self):
            return 2

        def get(self, limit, include):
            assert limit == 10
            assert include == ["metadatas", "embeddings"]
            return {
                "metadatas": [
                    {"artist": "A", "album": "X", "year": 2024},
                    {"artist": "B", "album": "X", "year": 2025},
                ],
                "embeddings": [[0.1, 0.2, 0.3]],
            }

    monkeypatch.setattr(server, "collection", FakeCollection(), raising=False)

    result = asyncio.run(server.collection_stats())

    assert result["total_tracks"] == 2
    assert result["embedding_dimensions"] == 3
    assert result["metadata_stats"]["artist"] == {
        "sample_count": 2,
        "unique_values_in_sample": 2,
        "types": ["str"],
    }
    assert result["metadata_stats"]["album"] == {
        "sample_count": 2,
        "unique_values_in_sample": 1,
        "types": ["str"],
    }
    assert result["metadata_stats"]["year"] == {
        "sample_count": 2,
        "unique_values_in_sample": 2,
        "types": ["int"],
    }


def test_resolve_config_files_uses_etc_xdg_local_order(monkeypatch, tmp_path):
    etc_dir = tmp_path / "etc"
    xdg_dir = tmp_path / "xdg"
    local_dir = tmp_path / "local"
    for directory in (etc_dir, xdg_dir, local_dir):
        directory.mkdir()
        (directory / "config.yaml").write_text("app: {}\n", encoding="utf-8")

    monkeypatch.setattr(config_loader, "config_search_dirs", lambda app_name, cwd=None: [etc_dir, xdg_dir, local_dir])

    assert server._resolve_config_files(None) == [
        str(etc_dir / "config.yaml"),
        str(xdg_dir / "config.yaml"),
        str(local_dir / "config.yaml"),
    ]


def test_resolve_logging_config_file_prefers_companion_of_explicit_config(monkeypatch, tmp_path):
    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()
    config_path = custom_dir / "config.yaml"
    logging_path = custom_dir / "logging.yaml"
    config_path.write_text("app: {}\n", encoding="utf-8")
    logging_path.write_text("level: DEBUG\n", encoding="utf-8")

    monkeypatch.setattr(config_loader, "ACTIVE_CONFIG_FILES", [config_path])
    monkeypatch.setattr(
        server,
        "settings",
        Settings(
            app=AppSettings(),
            chromadb=ChromaDBSettings(host="localhost"),
        ),
        raising=False,
    )
    monkeypatch.setattr(config_loader, "config_search_dirs", lambda app_name, cwd=None: [])

    assert server._resolve_logging_config_file() == logging_path
