# StarLink Orchestration

StarLink Orchestration is an open-source resource orchestration system for multi-architecture server clusters, providing a web console and RESTful API to help operations and development teams manage heterogeneous resources such as x86, ARM, and RISC-V uniformly, achieving cross-architecture scheduling, container/database/snapshot governance, and AI hosting capabilities.

## Features

- **Multi-architecture Scheduling**: Records server hardware architectures and automatically matches the most suitable nodes during task scheduling while deducting resource quotas.
- **Centralized Resource Panel**: Minute-level observability capabilities, displaying real-time metrics, health status, alerts, and exports across resource pools/clusters/tags.
- **Monitoring Trend Analysis**: Supports metric trends from 1h to 30d, P50/P95/P99 percentile calculations, and maintenance event annotations.
- **Resource Pool and Policy Integration**: Defines compute/GPU/storage pool quotas and scheduling watermarks, supporting one-click server migration and policy dry-run.
- **Container and Database Governance**: Provides container lifecycle management, real-time logs, database backup/restore drills, and metric recording.
- **Snapshot and Gateway Center**: Unified management of filesystem/block storage/database snapshots, maintaining central gateway configurations, grayscale policies, and rate-limiting templates.
- **One-click AI Hosting**: Upload AIDef documents for validation, deployment, and dashboard integration, with reserved unified interfaces for OpenAI/DeepSeek/local models.
- **AI Workflow Console**: Configure AI operations workflows on web/desktop, integrate with AIDef, track execution history, and trigger scheduling with one click.
- **Port Policy and Auditing**: Security policies support approval dry-run, with complete audit logs recording "who did what when".
- **RESTful API**: Built with FastAPI, API documentation can be viewed at `/docs` after running the service.
- **Modern Console**: Referencing AWS console's visual hierarchy and interaction style, providing a dark glassmorphism interface with quick action entries.
- **Lightweight Deployment**: SQLite persistence, runs by default; can be replaced with external databases.

## Directory Structure

```
backend/
  app/
    main.py          # FastAPI application entry
    database.py      # Database initialization and Session management
    models.py        # SQLAlchemy ORM models
    schemas.py       # Pydantic data models
    services.py      # Scheduling and action business logic
frontend/
  index.html         # Native JS-based console interface
desktop/
  package.json       # Electron cross-platform desktop application configuration
  main.js / preload.js
requirements.txt     # Python runtime dependencies
```

## Quick Start

1. **Install Dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows uses .venv\\Scripts\\activate
   pip install -r requirements.txt
   ```

2. **Start Backend Service**

   ```bash
   uvicorn backend.app.main:app --reload
   ```

   The service listens on `http://127.0.0.1:8000` by default, interactive API documentation can be accessed at `http://127.0.0.1:8000/docs`.

3. **Start Frontend Console**

   Open `frontend/index.html` directly in a browser, or host with any static file server:

   ```bash
   python -m http.server --directory frontend 8080
   ```

   Open `http://127.0.0.1:8080` to experience the StarLink Orchestration console.

## Desktop (macOS / Windows / Linux)

With the Electron package in the `desktop/` directory, you can get a consistent cross-platform native window experience with the web:

```bash
cd desktop
npm install
npm start
```

To generate installation packages for various platforms, execute respectively:

```bash
npm run package:mac   # macOS .app
npm run package:win   # Windows executable
npm run package:linux # Linux AppImage/directory
```

Packaging results output to `desktop/dist/`.

## Common API Examples

- Register server: `POST /servers`
- View server list: `GET /servers`
- Report real-time metrics: `POST /servers/{server_id}/metrics`
- Overview export: `GET /overview?export=csv`
- Query trends: `GET /trends?host_ids=1&host_ids=2&metric=cpu_usage&hours=6`
- Manage resource pools: `POST /pools` / `POST /pools/move`
- Container operations: `POST /api/containers/{id}/actions {"op":"restart"}`
- Database backup: `POST /api/databases/{id}/backup`
- Register AIDef: `POST /api/ai/defs`
- Manage AI workflows: `POST /api/ai/workflows` / `GET /api/ai/workflows/{id}/runs`
- LLM proxy: `POST /api/llm/completions`
- Policy dry-run: `POST /policies/{id}/dry-run`
- Snapshot operations: `POST /api/snapshots/{id}/actions {"op":"rollback"}`

## Testing

The project includes built-in Pytest cases to validate core scheduling logic:

```bash
pytest
```

## License

This project is open-sourced under the Apache License 2.0. Issues and PRs are welcome to improve StarLink Orchestration together.