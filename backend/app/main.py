from __future__ import annotations

import asyncio
import csv
from datetime import datetime, timedelta
from io import StringIO
from typing import Optional

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Query,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session

from . import services
from .database import Base, engine, get_session
from .models import (
    ActionLog,
    ActionType,
    Architecture,
    DatabaseEngine,
    MaintenanceEvent,
    MaintenanceStatus,
    ServerStatus,
    SnapshotTarget,
    TaskStatus,
)
from .schemas import (
    ActionLogRead,
    AuditLogRead,
    AIDefRead,
    AIDefRegisterRequest,
    AIDeployRequest,
    AIWorkflowCreate,
    AIWorkflowRead,
    AIWorkflowRunRead,
    AIWorkflowRunRequest,
    AIWorkflowUpdate,
    ContainerActionRequest,
    ContainerCreate,
    ContainerLogResponse,
    ContainerRead,
    DashboardHealth,
    DashboardSummary,
    DatabaseBackupRequest,
    DatabaseInstanceCreate,
    DatabaseInstanceRead,
    DatabaseRestoreRequest,
    GatewayCreate,
    GatewayRead,
    LLMCompletionRequest,
    LLMCompletionResponse,
    MaintenanceCreate,
    MaintenanceRead,
    MaintenanceUpdate,
    MetricCreate,
    MetricRead,
    OverviewAlert,
    OverviewResponse,
    OverviewServer,
    OverviewSummary,
    PoolCreate,
    PoolMoveRequest,
    PoolRead,
    PoolRebalanceResult,
    PoolUpdate,
    PolicyCreate,
    PolicyDryRunRequest,
    PolicyDryRunResult,
    PolicyRead,
    ServerCreate,
    ServerRead,
    ServerUpdate,
    SnapshotActionRequest,
    SnapshotCreate,
    SnapshotRead,
    TaskCreate,
    TaskRead,
    TrendSeries,
)


def pool_to_read(pool) -> PoolRead:
    capacity = services.compute_pool_capacity(pool)
    return PoolRead(
        id=pool.id,
        name=pool.name,
        type=pool.type,
        quotas=pool.quotas,
        labels=pool.labels,
        description=pool.description,
        created_at=pool.created_at,
        updated_at=pool.updated_at,
        server_count=len(pool.servers),
        capacity=capacity,
    )

app = FastAPI(title="星联调度", version="0.2.0", description="多架构智能资源调度平台")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)


async def simulate_action(action_id: int, action_type: ActionType, delay: int = 2) -> None:
    await asyncio.sleep(delay)
    with get_session() as session:
        action = session.get(ActionLog, action_id)
        if action:
            services.complete_action(session, action, success=True, details="操作成功完成")
            server = action.server
            if action.action_type == ActionType.INSTALLATION and server:
                server.status = ServerStatus.ONLINE
                session.add(server)


async def simulate_ai_workflow_run(run_id: int, delay: int = 2) -> None:
    await asyncio.sleep(delay)
    with get_session() as session:
        run = services.get_ai_workflow_run(session, run_id)
        if run:
            services.complete_ai_workflow_run(
                session,
                run,
                success=True,
                result={"message": "工作流已成功完成"},
            )


# ---------------------------------------------------------------------------
# Server APIs
# ---------------------------------------------------------------------------


@app.post("/servers", response_model=ServerRead, status_code=201)
def create_server(payload: ServerCreate, session: Session = Depends(get_session)):
    server = services.create_server(session, payload.dict())
    return server


@app.get("/servers", response_model=list[ServerRead])
def list_servers(
    architecture: Optional[Architecture] = Query(None),
    status: Optional[ServerStatus] = Query(None),
    keyword: Optional[str] = Query(None, min_length=1),
    session: Session = Depends(get_session),
):
    return services.list_servers(
        session, architecture=architecture, status=status, keyword=keyword
    )


@app.get("/servers/{server_id}", response_model=ServerRead)
def read_server(server_id: int, session: Session = Depends(get_session)):
    server = services.get_server(session, server_id)
    if not server:
        raise HTTPException(status_code=404, detail="服务器不存在")
    return server


@app.patch("/servers/{server_id}", response_model=ServerRead)
def update_server(server_id: int, payload: ServerUpdate, session: Session = Depends(get_session)):
    server = services.get_server(session, server_id)
    if not server:
        raise HTTPException(status_code=404, detail="服务器不存在")
    data = payload.dict(exclude_unset=True)
    server = services.update_server(session, server, data)
    return server


@app.get("/servers/{server_id}/actions", response_model=list[ActionLogRead])
def list_server_actions(server_id: int, session: Session = Depends(get_session)):
    server = services.get_server(session, server_id)
    if not server:
        raise HTTPException(status_code=404, detail="服务器不存在")
    return [ActionLogRead.from_orm(action) for action in server.actions]


@app.post("/servers/{server_id}/metrics", response_model=MetricRead, status_code=201)
def record_metric(
    server_id: int,
    payload: MetricCreate,
    session: Session = Depends(get_session),
):
    server = services.get_server(session, server_id)
    if not server:
        raise HTTPException(status_code=404, detail="服务器不存在")
    metric = services.record_metric(
        session,
        server,
        {
            **payload.dict(exclude_none=True),
            "recorded_at": payload.recorded_at or datetime.utcnow(),
        },
    )
    return MetricRead.from_orm(metric)


@app.get("/servers/{server_id}/metrics", response_model=list[MetricRead])
def list_metrics(
    server_id: int,
    limit: int = Query(20, ge=1, le=200),
    session: Session = Depends(get_session),
):
    server = services.get_server(session, server_id)
    if not server:
        raise HTTPException(status_code=404, detail="服务器不存在")
    metrics = services.list_metrics(session, server, limit=limit)
    return [MetricRead.from_orm(metric) for metric in metrics]


@app.get("/servers/{server_id}/metrics/latest", response_model=Optional[MetricRead])
def latest_metric(server_id: int, session: Session = Depends(get_session)):
    server = services.get_server(session, server_id)
    if not server:
        raise HTTPException(status_code=404, detail="服务器不存在")
    metric = services.latest_metric(session, server)
    return MetricRead.from_orm(metric) if metric else None


@app.post("/servers/{server_id}/actions/docker", response_model=ActionLogRead)
def launch_docker(
    server_id: int,
    image: str,
    background: BackgroundTasks,
    requested_by: Optional[str] = None,
    session: Session = Depends(get_session),
):
    server = services.get_server(session, server_id)
    if not server:
        raise HTTPException(status_code=404, detail="服务器不存在")
    if server.status != ServerStatus.ONLINE:
        raise HTTPException(status_code=400, detail="服务器不在线")
    action = services.create_action(
        session,
        server=server,
        action_type=ActionType.DOCKER,
        command=f"docker run -d {image}",
        requested_by=requested_by,
        details="容器启动中",
    )
    background.add_task(simulate_action, action.id, ActionType.DOCKER)
    return ActionLogRead.from_orm(action)


@app.post("/servers/{server_id}/actions/install", response_model=ActionLogRead)
def install_system(
    server_id: int,
    os_name: str,
    background: BackgroundTasks,
    requested_by: Optional[str] = None,
    session: Session = Depends(get_session),
):
    server = services.get_server(session, server_id)
    if not server:
        raise HTTPException(status_code=404, detail="服务器不存在")
    action = services.create_action(
        session,
        server=server,
        action_type=ActionType.INSTALLATION,
        command=f"provision --os {os_name}",
        requested_by=requested_by,
        details="系统安装中",
    )
    server.status = ServerStatus.PROVISIONING
    session.add(server)
    session.flush()
    background.add_task(simulate_action, action.id, ActionType.INSTALLATION, 3)
    return ActionLogRead.from_orm(action)


# ---------------------------------------------------------------------------
# Maintenance & tasks
# ---------------------------------------------------------------------------


@app.post("/servers/{server_id}/maintenance", response_model=MaintenanceRead, status_code=201)
def schedule_maintenance(
    server_id: int,
    payload: MaintenanceCreate,
    background: BackgroundTasks,
    session: Session = Depends(get_session),
):
    server = services.get_server(session, server_id)
    if not server:
        raise HTTPException(status_code=404, detail="服务器不存在")
    event = services.schedule_maintenance(
        session,
        server=server,
        title=payload.title,
        description=payload.description,
        scheduled_for=payload.scheduled_for,
        duration_minutes=payload.duration_minutes,
    )
    action = services.create_action(
        session,
        server=server,
        action_type=ActionType.MAINTENANCE,
        command=f"schedule-maintenance {payload.title}",
        requested_by="control-panel",
        details="维护计划创建",
    )
    background.add_task(simulate_action, action.id, ActionType.MAINTENANCE, 1)
    return MaintenanceRead.from_orm(event)


@app.get("/maintenance", response_model=list[MaintenanceRead])
def list_maintenance_events(
    upcoming_only: bool = Query(False), session: Session = Depends(get_session)
):
    events = services.list_maintenance(session, upcoming_only=upcoming_only)
    return [MaintenanceRead.from_orm(event) for event in events]


@app.patch("/maintenance/{event_id}", response_model=MaintenanceRead)
def update_maintenance_event(
    event_id: int,
    payload: MaintenanceUpdate,
    session: Session = Depends(get_session),
):
    event = session.get(MaintenanceEvent, event_id)
    if not event:
        raise HTTPException(status_code=404, detail="维护计划不存在")
    data = payload.dict(exclude_unset=True)
    event = services.update_maintenance(session, event, data)
    server = event.server
    if server:
        if event.status == MaintenanceStatus.IN_PROGRESS:
            server.status = ServerStatus.MAINTENANCE
            session.add(server)
        elif event.status in {
            MaintenanceStatus.COMPLETED,
            MaintenanceStatus.CANCELLED,
        } and server.status == ServerStatus.MAINTENANCE:
            server.status = ServerStatus.ONLINE
            session.add(server)
    return MaintenanceRead.from_orm(event)


@app.post("/tasks", response_model=TaskRead, status_code=201)
def create_task(payload: TaskCreate, session: Session = Depends(get_session)):
    task = services.create_task(session, payload.dict())
    return task


@app.get("/tasks", response_model=list[TaskRead])
def list_tasks(session: Session = Depends(get_session)):
    return services.list_tasks(session)


@app.post("/tasks/{task_id}/start", response_model=TaskRead)
def start_task(task_id: int, session: Session = Depends(get_session)):
    task = services.get_task(session, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    if task.status != TaskStatus.SCHEDULED:
        raise HTTPException(status_code=400, detail="任务未调度，无法开始")
    services.mark_task_running(session, task)
    return task


@app.post("/tasks/{task_id}/dispatch", response_model=TaskRead)
def dispatch_task(task_id: int, session: Session = Depends(get_session)):
    task = services.get_task(session, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    if task.status not in {TaskStatus.PENDING, TaskStatus.FAILED}:
        raise HTTPException(status_code=400, detail="任务状态不允许调度")
    servers = services.list_servers(session)
    server = services.dispatch_task(session, task, servers)
    if not server:
        raise HTTPException(status_code=409, detail="没有可用的服务器")
    return task


@app.post("/tasks/{task_id}/complete", response_model=TaskRead)
def complete_task(task_id: int, success: bool = True, session: Session = Depends(get_session)):
    task = services.get_task(session, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    services.mark_task_completed(session, task, success=success)
    return task


# ---------------------------------------------------------------------------
# Dashboard & observability
# ---------------------------------------------------------------------------


@app.get("/dashboard/summary", response_model=DashboardSummary)
def dashboard_summary(session: Session = Depends(get_session)):
    servers = services.list_servers(session)
    tasks = services.list_tasks(session)
    status_count = {
        ServerStatus.ONLINE: 0,
        ServerStatus.OFFLINE: 0,
        ServerStatus.MAINTENANCE: 0,
        ServerStatus.PROVISIONING: 0,
    }
    for server in servers:
        status_count[server.status] = status_count.get(server.status, 0) + 1

    task_count = {
        TaskStatus.PENDING: 0,
        TaskStatus.RUNNING: 0,
        TaskStatus.SCHEDULED: 0,
        TaskStatus.COMPLETED: 0,
        TaskStatus.FAILED: 0,
    }
    for task in tasks:
        task_count[task.status] = task_count.get(task.status, 0) + 1

    return DashboardSummary(
        total_servers=len(servers),
        online_servers=status_count[ServerStatus.ONLINE],
        offline_servers=status_count[ServerStatus.OFFLINE],
        maintenance_servers=status_count[ServerStatus.MAINTENANCE],
        provisioning_servers=status_count[ServerStatus.PROVISIONING],
        total_tasks=len(tasks),
        running_tasks=task_count[TaskStatus.RUNNING],
        pending_tasks=task_count[TaskStatus.PENDING],
        completed_tasks=task_count[TaskStatus.COMPLETED],
        failed_tasks=task_count[TaskStatus.FAILED],
    )


@app.get("/dashboard/health", response_model=DashboardHealth)
def dashboard_health(session: Session = Depends(get_session)):
    stats = services.compute_dashboard_health(session)
    return DashboardHealth(**stats)


@app.get("/overview", response_model=OverviewResponse)
def overview(
    export: Optional[str] = Query(None, description="json|csv|none"),
    session: Session = Depends(get_session),
):
    payload = services.compute_overview(session)
    response = OverviewResponse(
        summary=OverviewSummary(**payload["summary"]),
        servers=[OverviewServer(**server) for server in payload["servers"]],
        alerts=[OverviewAlert(**alert) for alert in payload["alerts"]],
        bottlenecks=[OverviewAlert(**alert) for alert in payload["bottlenecks"]],
    )
    if export == "csv":
        csv_buffer = StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow([
            "id",
            "name",
            "status",
            "architecture",
            "pool",
            "cluster",
            "zone",
            "cpu",
            "memory",
            "health",
            "gpu",
        ])
        for server in response.servers:
            metrics = server.metrics or {}
            writer.writerow(
                [
                    server.id,
                    server.name,
                    server.status.value,
                    server.architecture.value,
                    server.pool or "-",
                    server.cluster or "-",
                    server.zone or "-",
                    metrics.get("cpu", ""),
                    metrics.get("memory", ""),
                    metrics.get("health", ""),
                    metrics.get("gpu", ""),
                ]
            )
        csv_buffer.seek(0)
        return StreamingResponse(
            csv_buffer,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=overview.csv"},
        )
    if export == "json":
        return JSONResponse(response.dict())
    return response


@app.get("/trends", response_model=list[TrendSeries])
def trends(
    host_ids: list[int] = Query(..., description="主机 ID"),
    metric: str = Query("cpu_usage", description="指标字段"),
    hours: int = Query(6, ge=1, le=720),
    session: Session = Depends(get_session),
):
    since = datetime.utcnow() - timedelta(hours=hours)
    data = services.compute_trends(session, host_ids=host_ids, metric_name=metric, since=since)
    return [TrendSeries(**series) for series in data]


# ---------------------------------------------------------------------------
# Resource pools & policies
# ---------------------------------------------------------------------------


@app.post("/pools", response_model=PoolRead, status_code=201)
def create_pool(payload: PoolCreate, session: Session = Depends(get_session)):
    pool = services.create_pool(session, payload.dict())
    return pool_to_read(pool)


@app.get("/pools", response_model=list[PoolRead])
def list_pools(session: Session = Depends(get_session)):
    pools = services.list_pools(session)
    return [pool_to_read(pool) for pool in pools]


@app.patch("/pools/{pool_id}", response_model=PoolRead)
def update_pool(pool_id: int, payload: PoolUpdate, session: Session = Depends(get_session)):
    pool = services.get_pool(session, pool_id)
    if not pool:
        raise HTTPException(status_code=404, detail="资源池不存在")
    pool = services.update_pool(session, pool, payload.dict(exclude_unset=True))
    return pool_to_read(pool)


@app.post("/pools/{pool_id}/rebalance", response_model=PoolRebalanceResult)
def rebalance_pool(pool_id: int, session: Session = Depends(get_session)):
    pool = services.get_pool(session, pool_id)
    if not pool:
        raise HTTPException(status_code=404, detail="资源池不存在")
    result = services.rebalance_pool(session, pool)
    return PoolRebalanceResult(**result)


@app.post("/pools/move", response_model=ServerRead)
def move_server(payload: PoolMoveRequest, session: Session = Depends(get_session)):
    server = services.get_server(session, payload.server_id)
    if not server:
        raise HTTPException(status_code=404, detail="服务器不存在")
    pool = services.get_pool(session, payload.target_pool_id)
    if not pool:
        raise HTTPException(status_code=404, detail="资源池不存在")
    services.move_server_to_pool(session, server=server, pool=pool)
    return server


@app.post("/policies", response_model=PolicyRead, status_code=201)
def create_policy(payload: PolicyCreate, session: Session = Depends(get_session)):
    policy = services.create_policy(session, payload.dict())
    return PolicyRead.from_orm(policy)


@app.get("/policies", response_model=list[PolicyRead])
def list_policies(session: Session = Depends(get_session)):
    policies = services.list_policies(session)
    return [PolicyRead.from_orm(policy) for policy in policies]


@app.post("/policies/{policy_id}/dry-run", response_model=PolicyDryRunResult)
def policy_dry_run(policy_id: int, payload: PolicyDryRunRequest, session: Session = Depends(get_session)):
    policy = services.get_policy(session, policy_id)
    if not policy:
        raise HTTPException(status_code=404, detail="策略不存在")
    result = services.policy_dry_run(session, policy, labels=payload.target_labels)
    return PolicyDryRunResult(**result)


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------


@app.get("/api/containers", response_model=list[ContainerRead])
def list_containers(host: Optional[int] = Query(None), session: Session = Depends(get_session)):
    containers = services.list_containers(session, host_id=host)
    return [ContainerRead.from_orm(container) for container in containers]


@app.post("/api/containers", response_model=ContainerRead, status_code=201)
def create_container(payload: ContainerCreate, session: Session = Depends(get_session)):
    host = services.get_server(session, payload.host_id)
    if not host:
        raise HTTPException(status_code=404, detail="宿主机不存在")
    container = services.create_container(session, payload.dict())
    return ContainerRead.from_orm(container)


@app.post("/api/containers/{container_id}/actions", response_model=ContainerRead)
def operate_container(
    container_id: int,
    payload: ContainerActionRequest,
    session: Session = Depends(get_session),
):
    container = services.get_container(session, container_id)
    if not container:
        raise HTTPException(status_code=404, detail="容器不存在")
    container = services.container_action(session, container, payload.op, payload.args)
    return ContainerRead.from_orm(container)


@app.get("/api/logs/containers/{container_id}", response_model=ContainerLogResponse)
def container_logs(container_id: int, session: Session = Depends(get_session)):
    container = services.get_container(session, container_id)
    if not container:
        raise HTTPException(status_code=404, detail="容器不存在")
    data = services.container_logs(container)
    return ContainerLogResponse(**data)


# ---------------------------------------------------------------------------
# Database panel
# ---------------------------------------------------------------------------


@app.post("/api/databases", response_model=DatabaseInstanceRead, status_code=201)
def create_database_instance(
    payload: DatabaseInstanceCreate, session: Session = Depends(get_session)
):
    db_instance = services.register_db_instance(session, payload.dict())
    return DatabaseInstanceRead.from_orm(db_instance)


@app.get("/api/databases", response_model=list[DatabaseInstanceRead])
def list_database_instances(
    engine: Optional[DatabaseEngine] = Query(None),
    session: Session = Depends(get_session),
):
    dbs = services.list_db_instances(session, engine=engine)
    return [DatabaseInstanceRead.from_orm(db) for db in dbs]


@app.post("/api/databases/{db_id}/backup", response_model=SnapshotRead, status_code=201)
def backup_database(
    db_id: int,
    payload: DatabaseBackupRequest,
    session: Session = Depends(get_session),
):
    db = services.get_db_instance(session, db_id)
    if not db:
        raise HTTPException(status_code=404, detail="数据库实例不存在")
    snapshot = services.schedule_db_backup(
        session,
        db,
        strategy=payload.strategy,
        target_namespace=payload.target_namespace,
        verify=payload.verify,
    )
    return SnapshotRead.from_orm(snapshot)


@app.post("/api/databases/{db_id}/restore")
def restore_database(
    db_id: int,
    payload: DatabaseRestoreRequest,
    session: Session = Depends(get_session),
):
    db = services.get_db_instance(session, db_id)
    if not db:
        raise HTTPException(status_code=404, detail="数据库实例不存在")
    result = services.restore_database(
        session,
        db,
        snapshot_id=payload.snapshot_id,
        target_namespace=payload.target_namespace,
        approval_ticket=payload.approval_ticket,
    )
    return result


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------


@app.post("/api/snapshots", response_model=SnapshotRead, status_code=201)
def create_snapshot(payload: SnapshotCreate, session: Session = Depends(get_session)):
    snapshot = services.create_snapshot(session, payload.dict())
    return SnapshotRead.from_orm(snapshot)


@app.get("/api/snapshots", response_model=list[SnapshotRead])
def list_snapshots(
    target_type: Optional[SnapshotTarget] = Query(None),
    session: Session = Depends(get_session),
):
    snapshots = services.list_snapshots(session, target_type=target_type)
    return [SnapshotRead.from_orm(snapshot) for snapshot in snapshots]


@app.post("/api/snapshots/{snapshot_id}/actions")
def snapshot_action(
    snapshot_id: int,
    payload: SnapshotActionRequest,
    session: Session = Depends(get_session),
):
    snapshot = services.get_snapshot(session, snapshot_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail="快照不存在")
    result = services.snapshot_action(session, snapshot, op=payload.op, target_pool=payload.target_pool)
    return result


# ---------------------------------------------------------------------------
# Central gateway
# ---------------------------------------------------------------------------


@app.post("/api/gateways", response_model=GatewayRead, status_code=201)
def create_gateway(payload: GatewayCreate, session: Session = Depends(get_session)):
    gateway = services.create_gateway(session, payload.dict())
    return GatewayRead.from_orm(gateway)


@app.get("/api/gateways", response_model=list[GatewayRead])
def list_gateways(session: Session = Depends(get_session)):
    gateways = services.list_gateways(session)
    return [GatewayRead.from_orm(gateway) for gateway in gateways]


# ---------------------------------------------------------------------------
# AI definition & LLM facade
# ---------------------------------------------------------------------------


@app.post("/api/ai/defs", response_model=AIDefRead, status_code=201)
def register_ai_definition(
    payload: AIDefRegisterRequest, session: Session = Depends(get_session)
):
    try:
        definition = services.register_ai_definition(
            session,
            yaml_payload=payload.yaml,
            owner=payload.owner,
            runtime=payload.runtime,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return AIDefRead.from_orm(definition)


@app.get("/api/ai/defs", response_model=list[AIDefRead])
def list_ai_definitions(session: Session = Depends(get_session)):
    definitions = services.list_ai_definitions(session)
    return [AIDefRead.from_orm(definition) for definition in definitions]


@app.post("/api/ai/deploy")
def deploy_ai(payload: AIDeployRequest, session: Session = Depends(get_session)):
    definition = services.get_ai_definition(session, payload.aid)
    if not definition:
        raise HTTPException(status_code=404, detail="AIDef 不存在")
    result = services.deploy_ai_definition(session, definition, target_pool=payload.target_pool)
    return result


@app.post("/api/ai/workflows", response_model=AIWorkflowRead, status_code=201)
def create_ai_workflow(payload: AIWorkflowCreate, session: Session = Depends(get_session)):
    definition = services.get_ai_definition(session, payload.aid)
    if not definition:
        raise HTTPException(status_code=404, detail="AIDef 不存在")
    workflow = services.create_ai_workflow(session, payload.dict())
    return AIWorkflowRead.from_orm(workflow)


@app.get("/api/ai/workflows", response_model=list[AIWorkflowRead])
def list_ai_workflows(session: Session = Depends(get_session)):
    workflows = services.list_ai_workflows(session)
    return [AIWorkflowRead.from_orm(workflow) for workflow in workflows]


@app.get("/api/ai/workflows/{workflow_id}", response_model=AIWorkflowRead)
def get_ai_workflow(workflow_id: int, session: Session = Depends(get_session)):
    workflow = services.get_ai_workflow(session, workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="AI 工作流不存在")
    return AIWorkflowRead.from_orm(workflow)


@app.patch("/api/ai/workflows/{workflow_id}", response_model=AIWorkflowRead)
def update_ai_workflow(
    workflow_id: int,
    payload: AIWorkflowUpdate,
    session: Session = Depends(get_session),
):
    workflow = services.get_ai_workflow(session, workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="AI 工作流不存在")
    data = payload.dict(exclude_unset=True)
    if data:
        workflow = services.update_ai_workflow(session, workflow, data)
    return AIWorkflowRead.from_orm(workflow)


@app.delete("/api/ai/workflows/{workflow_id}", status_code=204)
def delete_ai_workflow(workflow_id: int, session: Session = Depends(get_session)):
    workflow = services.get_ai_workflow(session, workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="AI 工作流不存在")
    services.delete_ai_workflow(session, workflow)
    return JSONResponse(status_code=204, content=None)


@app.post(
    "/api/ai/workflows/{workflow_id}/run",
    response_model=AIWorkflowRunRead,
    status_code=202,
)
def trigger_ai_workflow(
    workflow_id: int,
    payload: AIWorkflowRunRequest,
    background: BackgroundTasks,
    session: Session = Depends(get_session),
):
    workflow = services.get_ai_workflow(session, workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="AI 工作流不存在")
    if not workflow.is_enabled:
        raise HTTPException(status_code=400, detail="工作流已禁用")
    run = services.start_ai_workflow_run(
        session,
        workflow,
        operator=payload.operator,
        context=payload.context,
    )
    background.add_task(simulate_ai_workflow_run, run.id)
    return AIWorkflowRunRead.from_orm(run)


@app.get(
    "/api/ai/workflows/{workflow_id}/runs",
    response_model=list[AIWorkflowRunRead],
)
def list_ai_workflow_runs(workflow_id: int, session: Session = Depends(get_session)):
    workflow = services.get_ai_workflow(session, workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="AI 工作流不存在")
    runs = services.list_ai_workflow_runs(session, workflow)
    return [AIWorkflowRunRead.from_orm(run) for run in runs]


@app.post("/api/llm/completions", response_model=LLMCompletionResponse)
def llm_completions(payload: LLMCompletionRequest):
    result = services.generate_llm_completion(payload.dict())
    return LLMCompletionResponse(**result)


# ---------------------------------------------------------------------------
# Audit logs
# ---------------------------------------------------------------------------


@app.get("/audit/logs", response_model=list[AuditLogRead])
def audit_logs(limit: int = Query(100, ge=1, le=500), session: Session = Depends(get_session)):
    logs = services.list_audit_logs(session, limit=limit)
    return [AuditLogRead.from_orm(log) for log in logs]
