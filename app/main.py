"""FastAPI application entrypoint for AI Travel Planner."""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.sqlite import SqliteSaver

from app.api.plans import router as plans_router
from app.graph.workflow import create_travel_planner_graph


def create_app(checkpoint_path: Optional[str] = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Manages checkpointer and graph compilation lifecycle.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db_path = getattr(app.state, "checkpoint_path", None)
        if not db_path:
            db_path = os.getenv("CHECKPOINT_DB_PATH", "data/checkpoints.sqlite")

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        with SqliteSaver.from_conn_string(db_path) as checkpointer:
            app.state.checkpointer = checkpointer
            app.state.graph = create_travel_planner_graph(checkpointer=checkpointer)
            yield

    app = FastAPI(
        title="AI Travel Planner",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.checkpoint_path = checkpoint_path

    @app.get("/health")
    def health_check() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        return Response(status_code=204)

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/", include_in_schema=False)
        def serve_index() -> FileResponse:
            return FileResponse(str(static_dir / "index.html"))

        @app.get("/index.html", include_in_schema=False)
        def serve_index_html() -> FileResponse:
            return FileResponse(str(static_dir / "index.html"))

    app.include_router(plans_router)

    return app


app = create_app()
