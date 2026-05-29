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

    monkeypatch.setattr(server.httpx, "Client", FakeClient)
    monkeypatch.setattr(
        server,
        "cfg",
        SimpleNamespace(
            embeddings=SimpleNamespace(
                base_url="http://127.0.0.1:8098",
                model="intfloat/multilingual-e5-small",
                normalize=True,
                timeout_seconds=12,
            )
        ),
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
        "cfg",
        SimpleNamespace(
            admin=SimpleNamespace(clear_collection_enabled=False),
            chromadb=SimpleNamespace(collection_name="music_collection"),
        ),
    )

    client = TestClient(server.app)

    assert "/clear_collection/" not in client.get("/openapi.json").json()["paths"]

    response = client.delete("/clear_collection/?confirm=true")

    assert response.status_code == 403
    assert response.json()["detail"] == "clear_collection is disabled by configuration"


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

    monkeypatch.setattr(server, "_config_search_dirs", lambda: [etc_dir, xdg_dir, local_dir])

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

    monkeypatch.setattr(server, "ACTIVE_CONFIG_FILES", [config_path])
    monkeypatch.setattr(server, "cfg", SimpleNamespace(app=SimpleNamespace()))
    monkeypatch.setattr(server, "_config_search_dirs", lambda: [])

    assert server._resolve_logging_config_file() == logging_path
