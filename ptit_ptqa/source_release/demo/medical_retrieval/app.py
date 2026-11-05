from __future__ import annotations

import os
import hashlib
import json
import sqlite3
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.release.config import load_release_config
from src.release.service import ReplayService
from src.retrieval.concept_graph import ConceptLexicon, retrieve
from src.retrieval.replay_graph import build_replay_graph

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).resolve().parent / "static"


class RunRequest(BaseModel):
    mode: str = "replay"
    query_id: int | None = None
    query: str | None = Field(default=None, max_length=12000)


def create_app(config_path: str | Path | None = None) -> FastAPI:
    selected = Path(config_path or os.getenv("R2AI_RELEASE_CONFIG", ROOT / "configs/release_exp143.json"))
    config = load_release_config(selected)
    service = ReplayService(config)
    app = FastAPI(title="R2AI Evidence Review", version=config.release_id)
    app.state.release_config = config
    app.state.replay_service = service
    graph_path = Path(os.getenv("R2AI_GRAPH_PATH", str(ROOT / "data/reference/concept_graph_v1.sqlite"))).resolve()
    lexicon_path = ROOT / "configs/medical_concepts_v1.json"
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok", "release_id": config.release_id,
            "default_mode": "replay", "live_ready": False,
            "live_message": "Live inference is intentionally disabled in the review release.",
            "counts": config.payload["counts"], "models": config.payload["models"],
            "public_metrics": config.payload["public_metrics"],
            "private_score": None,
            "fingerprints": {"submission": config.expected_hash("submission"),
                             "predictions": config.expected_hash("predictions")},
        }

    @app.get("/api/queries")
    def queries(search: str = "", offset: int = 0, limit: int = Query(50, ge=1, le=100)) -> dict:
        return service.list_queries(search=search, offset=offset, limit=limit)

    def graph_assets() -> tuple[Path, ConceptLexicon, dict]:
        manifest_path = graph_path.with_suffix(".manifest.json")
        if not graph_path.is_file() or not manifest_path.is_file() or not lexicon_path.is_file():
            raise HTTPException(status_code=503, detail="Experimental graph artifacts are not installed")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            digest = hashlib.sha256(lexicon_path.read_bytes()).hexdigest()
            if manifest.get("lexicon_sha256") != digest or manifest.get("status") != "full":
                raise ValueError("Graph lexicon or build status mismatch")
            return graph_path, ConceptLexicon(lexicon_path), manifest
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=503, detail=f"Experimental graph unavailable: {exc}") from exc

    @app.get("/api/graph/status")
    def graph_status() -> dict:
        try:
            _, lexicon, manifest = graph_assets()
        except HTTPException:
            return {"available": True, "concept_graph_available": False, "mode": "experimental",
                    "message": "Replay evidence graph available; optional concept artifact not installed."}
        return {"available": True, "concept_graph_available": True, "mode": "experimental", "concepts": len(lexicon.concepts),
                "indexed_docs": manifest["indexed_docs"], "mention_edges": manifest["mention_edges"],
                "message": "Concept graph plus replay evidence fallback; neither is a clinical knowledge base."}

    @app.get("/api/graph/queries/{query_id}")
    def graph_query(query_id: int, limit: int = Query(8, ge=1, le=12)) -> dict:
        try:
            official_query = service.get_query(query_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        try:
            path, lexicon, manifest = graph_assets()
        except HTTPException:
            path = lexicon = manifest = None
        if path is not None and lexicon is not None:
            concepts = sorted({mention.concept_id for mention in lexicon.mentions(official_query["query"])})
            if concepts:
                chapter_by_id = {item["id"]: item["chapter"] for item in lexicon.concepts}
                try:
                    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
                        candidates = retrieve(db, lexicon, official_query["query"], limit=limit)
                        for doc in candidates:
                            edges = db.execute(
                                "SELECT concept_id, alias, normalized_offset FROM mentions "
                                "WHERE doc_id=? AND concept_id IN (" + ",".join("?" for _ in concepts) + ") "
                                "ORDER BY normalized_offset", (doc["doc_id"], *concepts)
                            ).fetchall()
                            doc["evidence"] = [{"concept_id": cid, "alias": alias,
                                                "normalized_offset": offset, "source": "corpus_mention"}
                                               for cid, alias, offset in edges]
                except sqlite3.DatabaseError:
                    candidates = []
                if candidates:
                    return {"query_id": query_id, "query": official_query["query"],
                            "mode": "corpus_concepts", "edge_basis": "corpus_mention",
                            "scored": False, "affects_submission": False,
                            "concepts": [{"id": cid, "label": cid.replace("_", " "),
                                          "kind": "medical_concept", "chapter": chapter_by_id[cid]}
                                         for cid in concepts],
                            "documents": candidates, "indexed_docs": manifest["indexed_docs"],
                            "warning": "Corpus co-mention only; not clinical relevance, diagnosis, or Exp143 output."}
        rows = service.get_graph_rows(query_id)
        return build_replay_graph(query_id, official_query["query"], rows, limit=limit)

    @app.post("/api/runs")
    def create_run(request: RunRequest) -> dict:
        try:
            run = service.create_run(mode=request.mode, query_id=request.query_id, query=request.query)
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"run_id": run.run_id, "status": "completed", "mode": run.mode,
                "query_id": run.query_id, "query": run.query,
                "warning": "Cached release result; no new model inference was executed."}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        try:
            run = service.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"run_id": run.run_id, "status": "completed", "stage": "replay_ready",
                "mode": run.mode, "query_id": run.query_id, "query": run.query,
                "elapsed_seconds": 0.0,
                "warning": "Replay timing is not model-inference latency."}

    @app.get("/api/runs/{run_id}/results")
    def results(run_id: str, kind: str, offset: int = 0,
                limit: int = Query(20, ge=1, le=100)) -> dict:
        try:
            return service.get_results(run_id, kind, offset, limit)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/runs/{run_id}/evidence/{chunk_ref}")
    def evidence(run_id: str, chunk_ref: int) -> dict:
        try:
            return service.evidence(run_id, chunk_ref)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/runs/{run_id}/export")
    def export(run_id: str) -> JSONResponse:
        try:
            payload = service.export(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        headers = {"Content-Disposition": f'attachment; filename="r2ai-query-{payload["id"]}.json"'}
        return JSONResponse(payload, headers=headers)

    @app.post("/api/runs/{run_id}/cancel")
    def cancel(run_id: str) -> dict:
        try:
            service.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"run_id": run_id, "status": "completed",
                "message": "Replay completed synchronously; there is no active worker to cancel."}

    return app


app = create_app()
