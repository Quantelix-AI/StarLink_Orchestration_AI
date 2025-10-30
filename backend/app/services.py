from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from statistics import mean
from typing import Any, Iterable, Optional, Sequence

import yaml
from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from .models import (
    AIDefinition,
    AIWorkflow,
    AIWorkflowRun,
    ActionLog,
    ActionType,
    Architecture,
    AuditLog,
    Container,
    ContainerState,
    DatabaseEngine,
    DatabaseInstance,
    Gateway,
    GatewayStatus,
    MaintenanceEvent,
    MaintenanceStatus,
    Policy,
    PoolType,
    ResourcePool,
    Server,
    ServerMetric,
    ServerStatus,
    Snapshot,
    SnapshotTarget,
    Task,
    TaskStatus,
    WorkflowRunStatus,
)

ARCH_COMPATIBILITY: dict[Architecture, set[Architecture]] = {
    Architecture.X86_64: {Architecture.X86_64, Architecture.UNIVERSAL},
    Architecture.ARM64: {Architecture.ARM64, Architecture.UNIVERSAL},
    Architecture.RISCV64: {Architecture.RISCV64, Architecture.UNIVERSAL},
    Architecture.UNIVERSAL: {
        Architecture.X86_64,
        Architecture.ARM64,
        Architecture.RISCV64,
        Architecture.UNIVERSAL,
    },
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = (len(sorted_vals) - 1) * (q / 100.0)
    lower = int(idx)
    upper = min(lower + 1, len(sorted_vals) - 1)
    weight = idx - lower
    return round(sorted_vals[lower] * (1 - weight) + sorted_vals[upper] * weight, 2)


def refresh_server_capacity(server: Server) -> None:
    server.available_cpu = max(0, min(server.available_cpu, server.total_cpu))
    server.available_memory = max(0, min(server.available_memory, server.total_memory))
    server.available_storage = max(0, min(server.available_storage, server.total_storage))


def log_audit(
    session: Session,
    *,
    actor: str,
    action: str,
    target: str,
    before: Optional[dict] = None,
    after: Optional[dict] = None,
    message: Optional[str] = None,
) -> AuditLog:
    entry = AuditLog(
        actor=actor,
        action=action,
        target=target,
        before=before,
        after=after,
        message=message,
    )
    session.add(entry)
    session.flush()
    return entry


# ---------------------------------------------------------------------------
# Server & task lifecycle
# ---------------------------------------------------------------------------


def create_server(session: Session, data: dict) -> Server:
    server = Server(
        **data,
        available_cpu=data["total_cpu"],
        available_memory=data["total_memory"],
        available_storage=data["total_storage"],
    )
    session.add(server)
    session.flush()
    log_audit(
        session,
        actor="system",
        action="server.create",
        target=f"server:{server.id}",
        after={"name": server.name, "architecture": server.architecture.value},
    )
    return server


def list_servers(
    session: Session,
    *,
    architecture: Optional[Architecture] = None,
    status: Optional[ServerStatus] = None,
    keyword: Optional[str] = None,
    labels: Optional[dict[str, str]] = None,
) -> list[Server]:
    stmt = select(Server).order_by(Server.created_at.desc())
    if architecture:
        stmt = stmt.where(Server.architecture == architecture)
    if status:
        stmt = stmt.where(Server.status == status)
    if keyword:
        pattern = f"%{keyword}%"
        stmt = stmt.where(Server.name.ilike(pattern))
    if labels:
        for key, value in labels.items():
            stmt = stmt.where(Server.tags.contains({key: value}))
    return list(session.scalars(stmt))


def get_server(session: Session, server_id: int) -> Optional[Server]:
    return session.get(Server, server_id)


def update_server(session: Session, server: Server, data: dict) -> Server:
    before = {
        "status": server.status.value,
        "pool_id": server.pool_id,
        "available_cpu": server.available_cpu,
        "available_memory": server.available_memory,
        "available_storage": server.available_storage,
    }
    for key, value in data.items():
        setattr(server, key, value)
    refresh_server_capacity(server)
    session.add(server)
    session.flush()
    log_audit(
        session,
        actor="system",
        action="server.update",
        target=f"server:{server.id}",
        before=before,
        after={
            "status": server.status.value,
            "pool_id": server.pool_id,
            "available_cpu": server.available_cpu,
            "available_memory": server.available_memory,
            "available_storage": server.available_storage,
        },
    )
    return server


def create_task(session: Session, data: dict) -> Task:
    task = Task(**data)
    session.add(task)
    session.flush()
    log_audit(
        session,
        actor="system",
        action="task.create",
        target=f"task:{task.id}",
        after={"name": task.name, "priority": task.priority},
    )
    return task


def list_tasks(session: Session) -> list[Task]:
    return list(session.scalars(select(Task).order_by(Task.created_at.desc())))


def get_task(session: Session, task_id: int) -> Optional[Task]:
    return session.get(Task, task_id)


def architecture_match(server_arch: Architecture, required: Architecture) -> bool:
    return required in ARCH_COMPATIBILITY.get(server_arch, set())


def _capacity_score(server: Server, task: Task) -> float:
    cpu_headroom = (server.available_cpu - task.required_cpu) / max(server.total_cpu, 1)
    memory_headroom = (server.available_memory - task.required_memory) / max(
        server.total_memory, 1
    )
    storage_headroom = (server.available_storage - task.required_storage) / max(
        server.total_storage, 1
    )
    return max(cpu_headroom, 0) + max(memory_headroom, 0) + max(storage_headroom, 0)


def dispatch_task(session: Session, task: Task, servers: Iterable[Server]) -> Optional[Server]:
    candidates: list[tuple[float, Server]] = []
    now = datetime.utcnow()
    for server in servers:
        if server.status != ServerStatus.ONLINE:
            continue
        if not architecture_match(server.architecture, task.required_architecture):
            continue
        if (
            server.available_cpu >= task.required_cpu
            and server.available_memory >= task.required_memory
            and server.available_storage >= task.required_storage
        ):
            score = _capacity_score(server, task)
            if task.deadline and task.deadline < now:
                score -= 0.5
            adjusted = score - (task.priority - 1) * 0.1
            candidates.append((adjusted, server))

    if not candidates:
        return None

    _, selected = min(candidates, key=lambda item: (item[0], item[1].available_cpu))
    selected.available_cpu -= task.required_cpu
    selected.available_memory -= task.required_memory
    selected.available_storage -= task.required_storage
    task.server = selected
    task.status = TaskStatus.SCHEDULED
    session.add_all([selected, task])
    session.flush()
    log_audit(
        session,
        actor="scheduler",
        action="task.dispatch",
        target=f"task:{task.id}",
        after={"server": selected.name},
    )
    return selected


def mark_task_running(session: Session, task: Task) -> Task:
    task.status = TaskStatus.RUNNING
    session.add(task)
    session.flush()
    log_audit(
        session,
        actor="scheduler",
        action="task.start",
        target=f"task:{task.id}",
        after={"status": task.status.value},
    )
    return task


def mark_task_completed(session: Session, task: Task, success: bool = True) -> Task:
    if task.server:
        server = task.server
        server.available_cpu += task.required_cpu
        server.available_memory += task.required_memory
        server.available_storage += task.required_storage
        refresh_server_capacity(server)
        session.add(server)
    task.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
    session.add(task)
    session.flush()
    log_audit(
        session,
        actor="scheduler",
        action="task.complete",
        target=f"task:{task.id}",
        after={"status": task.status.value},
    )
    return task


def create_action(
    session: Session,
    *,
    server: Server,
    action_type: ActionType,
    command: str,
    requested_by: Optional[str],
    details: Optional[str] = None,
    success: bool = False,
) -> ActionLog:
    action = ActionLog(
        server=server,
        action_type=action_type,
        command=command,
        requested_by=requested_by,
        details=details,
        success=success,
    )
    session.add(action)
    session.flush()
    log_audit(
        session,
        actor=requested_by or "system",
        action=f"action.{action_type.value}",
        target=f"server:{server.id}",
        after={"command": command},
    )
    return action


def complete_action(
    session: Session, action: ActionLog, *, success: bool, details: Optional[str] = None
) -> ActionLog:
    action.success = success
    action.details = details
    action.completed_at = datetime.utcnow()
    session.add(action)
    session.flush()
    return action


# ---------------------------------------------------------------------------
# Metrics & observability
# ---------------------------------------------------------------------------


def record_metric(session: Session, server: Server, data: dict) -> ServerMetric:
    metric = ServerMetric(server=server, **data)
    session.add(metric)
    session.flush()
    return metric


def list_metrics(
    session: Session, server: Server, *, limit: int = 20
) -> Sequence[ServerMetric]:
    stmt = (
        select(ServerMetric)
        .where(ServerMetric.server_id == server.id)
        .order_by(ServerMetric.recorded_at.desc())
        .limit(limit)
    )
    return list(session.scalars(stmt))


def latest_metric(session: Session, server: Server) -> Optional[ServerMetric]:
    stmt = (
        select(ServerMetric)
        .where(ServerMetric.server_id == server.id)
        .order_by(ServerMetric.recorded_at.desc())
        .limit(1)
    )
    return session.scalars(stmt).first()


def schedule_maintenance(
    session: Session,
    *,
    server: Server,
    title: str,
    description: Optional[str],
    scheduled_for: datetime,
    duration_minutes: int,
) -> MaintenanceEvent:
    event = MaintenanceEvent(
        server=server,
        title=title,
        description=description,
        scheduled_for=scheduled_for,
        duration_minutes=duration_minutes,
    )
    session.add(event)
    session.flush()
    log_audit(
        session,
        actor="maintenance",
        action="maintenance.schedule",
        target=f"server:{server.id}",
        after={"title": title, "scheduled_for": scheduled_for.isoformat()},
    )
    return event


def update_maintenance(
    session: Session, event: MaintenanceEvent, data: dict
) -> MaintenanceEvent:
    before = {"status": event.status.value, "description": event.description}
    for key, value in data.items():
        setattr(event, key, value)
    session.add(event)
    session.flush()
    log_audit(
        session,
        actor="maintenance",
        action="maintenance.update",
        target=f"maintenance:{event.id}",
        before=before,
        after={"status": event.status.value, "description": event.description},
    )
    return event


def list_maintenance(
    session: Session, *, upcoming_only: bool = False
) -> list[MaintenanceEvent]:
    stmt = select(MaintenanceEvent).order_by(MaintenanceEvent.scheduled_for.desc())
    if upcoming_only:
        stmt = stmt.where(
            and_(
                MaintenanceEvent.status.in_(
                    [MaintenanceStatus.SCHEDULED, MaintenanceStatus.IN_PROGRESS]
                ),
                MaintenanceEvent.scheduled_for >= datetime.utcnow() - timedelta(minutes=5),
            )
        )
    return list(session.scalars(stmt))


def compute_dashboard_health(session: Session) -> dict:
    servers = list(session.scalars(select(Server)))
    if not servers:
        return {
            "average_cpu": 0.0,
            "average_memory": 0.0,
            "average_storage": 0.0,
            "hottest_server": None,
            "busiest_server": None,
            "metric_servers": 0,
        }

    latest_metrics = []
    for server in servers:
        metric = latest_metric(session, server)
        if metric:
            latest_metrics.append((server, metric))

    if not latest_metrics:
        return {
            "average_cpu": 0.0,
            "average_memory": 0.0,
            "average_storage": 0.0,
            "hottest_server": None,
            "busiest_server": None,
            "metric_servers": 0,
        }

    avg_cpu = mean(metric.cpu_usage for _, metric in latest_metrics)
    avg_mem = mean(metric.memory_usage for _, metric in latest_metrics)
    avg_storage = mean(metric.storage_usage for _, metric in latest_metrics)

    hottest_server = max(
        latest_metrics,
        key=lambda item: item[1].temperature if item[1].temperature is not None else -273,
    )
    busiest_server = max(latest_metrics, key=lambda item: item[1].cpu_usage)

    return {
        "average_cpu": round(avg_cpu, 2),
        "average_memory": round(avg_mem, 2),
        "average_storage": round(avg_storage, 2),
        "hottest_server": hottest_server[0].name
        if hottest_server[1].temperature is not None
        else None,
        "busiest_server": busiest_server[0].name,
        "metric_servers": len(latest_metrics),
    }


def compute_overview(session: Session) -> dict:
    servers = list(session.scalars(select(Server)))
    pools = list(session.scalars(select(ResourcePool)))
    containers = list(session.scalars(select(Container)))
    databases = list(session.scalars(select(DatabaseInstance)))
    gateways = list(session.scalars(select(Gateway)))
    workflows = list(session.scalars(select(AIWorkflow)))

    metrics_map: dict[int, ServerMetric] = {}
    for server in servers:
        metric = latest_metric(session, server)
        if metric:
            metrics_map[server.id] = metric

    alerts = []
    bottlenecks = []
    for server in servers:
        metric = metrics_map.get(server.id)
        if not metric:
            continue
        if metric.cpu_usage >= 85:
            alerts.append(
                {
                    "resource": server.name,
                    "dimension": "cpu_usage%",
                    "value": round(metric.cpu_usage, 2),
                    "threshold": 85.0,
                    "severity": "critical" if metric.cpu_usage > 92 else "warning",
                    "detail": "CPU 使用率接近饱和",
                }
            )
        if metric.slo_violation:
            alerts.append(
                {
                    "resource": server.name,
                    "dimension": "slo",
                    "value": 1,
                    "threshold": 0,
                    "severity": "critical",
                    "detail": "SLO 违规告警",
                }
            )
        bottlenecks.append(
            {
                "resource": server.name,
                "dimension": "mem_used%",
                "value": round(metric.memory_usage, 2),
                "threshold": 80.0,
                "severity": "info",
                "detail": "内存使用水位",
            }
        )

    average_health = (
        mean(metric.health_score for metric in metrics_map.values() if metric.health_score)
        if metrics_map
        else 0.0
    )

    server_payload = []
    for server in servers:
        metric = metrics_map.get(server.id)
        server_payload.append(
            {
                "id": server.id,
                "name": server.name,
                "status": server.status,
                "architecture": server.architecture,
                "pool": server.pool.name if server.pool else None,
                "cluster": server.cluster,
                "zone": server.zone,
                "tags": server.tags or {},
                "metrics": {
                    "cpu": metric.cpu_usage if metric else None,
                    "memory": metric.memory_usage if metric else None,
                    "health": metric.health_score if metric else None,
                    "gpu": metric.gpu_utilization if metric else None,
                }
                if metric
                else None,
            }
        )

    summary = {
        "total_hosts": len(servers),
        "total_pools": len(pools),
        "total_containers": len(containers),
        "total_databases": len(databases),
        "total_gateways": len(gateways),
        "total_ai_workflows": len(workflows),
        "enabled_ai_workflows": sum(1 for wf in workflows if wf.is_enabled),
        "average_health": round(average_health or 0.0, 2),
        "updated_at": datetime.utcnow(),
    }

    return {
        "summary": summary,
        "servers": server_payload,
        "alerts": alerts[:10],
        "bottlenecks": sorted(bottlenecks, key=lambda item: item["value"], reverse=True)[:5],
    }


def compute_trends(
    session: Session,
    *,
    host_ids: Sequence[int],
    metric_name: str,
    since: datetime,
) -> list[dict]:
    if not host_ids:
        return []
    stmt = (
        select(ServerMetric)
        .where(
            and_(
                ServerMetric.server_id.in_(host_ids),
                ServerMetric.recorded_at >= since,
            )
        )
        .order_by(ServerMetric.recorded_at.asc())
    )
    metrics = list(session.scalars(stmt))
    grouped: dict[int, list[ServerMetric]] = defaultdict(list)
    for metric in metrics:
        grouped[metric.server_id].append(metric)

    results: list[dict] = []
    for host_id, samples in grouped.items():
        values = []
        points = []
        for sample in samples:
            value = getattr(sample, metric_name, None)
            if value is None:
                continue
            values.append(value)
            points.append({"timestamp": sample.recorded_at, "value": value})
        if not values:
            continue
        events = []
        maintenance = session.scalars(
            select(MaintenanceEvent)
            .where(
                and_(
                    MaintenanceEvent.server_id == host_id,
                    MaintenanceEvent.scheduled_for >= since - timedelta(hours=1),
                )
            )
            .order_by(MaintenanceEvent.scheduled_for.asc())
        )
        for event in maintenance:
            events.append(
                {
                    "timestamp": event.scheduled_for,
                    "title": event.title,
                    "description": event.description,
                }
            )
        results.append(
            {
                "host_id": host_id,
                "metric": metric_name,
                "points": points,
                "percentiles": {
                    "p50": percentile(values, 50),
                    "p95": percentile(values, 95),
                    "p99": percentile(values, 99),
                },
                "events": events,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Resource pools & policies
# ---------------------------------------------------------------------------


def compute_pool_capacity(pool: ResourcePool) -> dict:
    total_cpu = sum(server.total_cpu for server in pool.servers)
    used_cpu = sum(server.total_cpu - server.available_cpu for server in pool.servers)
    total_memory = sum(server.total_memory for server in pool.servers)
    used_memory = sum(
        server.total_memory - server.available_memory for server in pool.servers
    )
    total_storage = sum(server.total_storage for server in pool.servers)
    used_storage = sum(
        server.total_storage - server.available_storage for server in pool.servers
    )
    return {
        "cpu": {"total": total_cpu, "used": used_cpu},
        "memory": {"total": total_memory, "used": used_memory},
        "storage": {"total": total_storage, "used": used_storage},
    }


def create_pool(session: Session, data: dict) -> ResourcePool:
    pool = ResourcePool(**data)
    session.add(pool)
    session.flush()
    log_audit(
        session,
        actor="system",
        action="pool.create",
        target=f"pool:{pool.id}",
        after={"name": pool.name, "type": pool.type.value},
    )
    return pool


def list_pools(session: Session) -> list[ResourcePool]:
    return list(session.scalars(select(ResourcePool).order_by(ResourcePool.name)))


def get_pool(session: Session, pool_id: int) -> Optional[ResourcePool]:
    return session.get(ResourcePool, pool_id)


def update_pool(session: Session, pool: ResourcePool, data: dict) -> ResourcePool:
    before = {"type": pool.type.value, "quotas": pool.quotas}
    for key, value in data.items():
        setattr(pool, key, value)
    session.add(pool)
    session.flush()
    log_audit(
        session,
        actor="system",
        action="pool.update",
        target=f"pool:{pool.id}",
        before=before,
        after={"type": pool.type.value, "quotas": pool.quotas},
    )
    return pool


def move_server_to_pool(
    session: Session, *, server: Server, pool: Optional[ResourcePool]
) -> Server:
    before_pool = server.pool_id
    server.pool = pool
    session.add(server)
    session.flush()
    log_audit(
        session,
        actor="system",
        action="pool.move_host",
        target=f"server:{server.id}",
        before={"pool_id": before_pool},
        after={"pool_id": server.pool_id},
    )
    return server


def rebalance_pool(session: Session, pool: ResourcePool) -> dict:
    moved_hosts: list[int] = []
    for server in pool.servers:
        metric = latest_metric(session, server)
        if metric and metric.cpu_usage < 20 and pool.type == PoolType.GPU:
            moved_hosts.append(server.id)
    return {
        "pool_id": pool.id,
        "moved_hosts": moved_hosts,
        "notes": "已根据闲置 GPU 节点生成迁移建议" if moved_hosts else "暂无迁移需求",
    }


def match_labels(resource_labels: dict[str, Any], desired: dict[str, Any]) -> bool:
    if not desired:
        return True
    for key, value in desired.items():
        if resource_labels.get(key) != value:
            return False
    return True


def create_policy(session: Session, data: dict) -> Policy:
    policy = Policy(**data)
    session.add(policy)
    session.flush()
    log_audit(
        session,
        actor="system",
        action="policy.create",
        target=f"policy:{policy.id}",
        after={"name": policy.name},
    )
    return policy


def list_policies(session: Session) -> list[Policy]:
    return list(session.scalars(select(Policy).order_by(Policy.created_at.desc())))


def get_policy(session: Session, policy_id: int) -> Optional[Policy]:
    return session.get(Policy, policy_id)


def update_policy(session: Session, policy: Policy, data: dict) -> Policy:
    before = {"rules": policy.rules}
    for key, value in data.items():
        setattr(policy, key, value)
    session.add(policy)
    session.flush()
    log_audit(
        session,
        actor="system",
        action="policy.update",
        target=f"policy:{policy.id}",
        before=before,
        after={"rules": policy.rules},
    )
    return policy


def policy_dry_run(
    session: Session, policy: Policy, *, labels: dict[str, Any]
) -> dict:
    affected_hosts: list[int] = []
    affected_services: set[str] = set()
    servers = list(session.scalars(select(Server)))
    desired = policy.scope.get("labels", {}) if policy.scope else {}
    for server in servers:
        server_labels = server.tags or {}
        if match_labels(server_labels, desired) and match_labels(server_labels, labels):
            affected_hosts.append(server.id)
            if server_labels.get("service"):
                affected_services.add(server_labels["service"])
    return {
        "policy_id": policy.id,
        "affected_hosts": affected_hosts,
        "affected_services": sorted(affected_services),
        "summary": f"预演匹配 {len(affected_hosts)} 台主机",
    }


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------


def create_container(session: Session, data: dict) -> Container:
    container = Container(**data)
    session.add(container)
    session.flush()
    log_audit(
        session,
        actor="containerd",
        action="container.create",
        target=f"container:{container.id}",
        after={"image": container.image, "host_id": container.host_id},
    )
    return container


def list_containers(
    session: Session, *, host_id: Optional[int] = None
) -> list[Container]:
    stmt = select(Container).order_by(Container.updated_at.desc())
    if host_id:
        stmt = stmt.where(Container.host_id == host_id)
    return list(session.scalars(stmt))


def get_container(session: Session, container_id: int) -> Optional[Container]:
    return session.get(Container, container_id)


def container_action(
    session: Session, container: Container, op: str, args: Optional[dict]
) -> Container:
    op = op.lower()
    previous_state = container.state
    if op == "restart":
        container.state = ContainerState.RESTARTING
        container.restarts += 1
        container.logs = (container.logs or "") + "\n[restart] 容器重启中"
    elif op == "start":
        container.state = ContainerState.RUNNING
        container.logs = (container.logs or "") + "\n[start] 容器已启动"
    elif op == "stop":
        container.state = ContainerState.STOPPED
        container.logs = (container.logs or "") + "\n[stop] 容器已停止"
    elif op == "exec" and args and args.get("command"):
        container.logs = (container.logs or "") + f"\n[exec] {args['command']}"
    container.updated_at = datetime.utcnow()
    session.add(container)
    session.flush()
    log_audit(
        session,
        actor="containerd",
        action=f"container.{op}",
        target=f"container:{container.id}",
        before={"state": previous_state.value},
        after={"state": container.state.value},
    )
    return container


def container_logs(container: Container) -> dict:
    return {
        "id": container.id,
        "logs": container.logs or "暂无日志",
        "updated_at": container.updated_at,
    }


# ---------------------------------------------------------------------------
# Database instances
# ---------------------------------------------------------------------------


def register_db_instance(session: Session, data: dict) -> DatabaseInstance:
    db_instance = DatabaseInstance(**data)
    session.add(db_instance)
    session.flush()
    log_audit(
        session,
        actor="db-admin",
        action="db.create",
        target=f"db:{db_instance.id}",
        after={"type": db_instance.type.value, "dsn": db_instance.dsn},
    )
    return db_instance


def list_db_instances(session: Session, *, engine: Optional[DatabaseEngine] = None) -> list[DatabaseInstance]:
    stmt = select(DatabaseInstance).order_by(DatabaseInstance.created_at.desc())
    if engine:
        stmt = stmt.where(DatabaseInstance.type == engine)
    return list(session.scalars(stmt))


def get_db_instance(session: Session, db_id: int) -> Optional[DatabaseInstance]:
    return session.get(DatabaseInstance, db_id)


def update_db_metrics(session: Session, db: DatabaseInstance, metrics: dict) -> DatabaseInstance:
    db.metrics = metrics
    db.updated_at = datetime.utcnow()
    session.add(db)
    session.flush()
    return db


def schedule_db_backup(
    session: Session, db: DatabaseInstance, *, strategy: str, target_namespace: Optional[str], verify: bool
) -> Snapshot:
    snapshot = Snapshot(
        target_type=SnapshotTarget.DATABASE,
        target_id=db.id,
        kind=strategy,
        meta={"namespace": target_namespace, "verify": verify},
        status="scheduled",
    )
    session.add(snapshot)
    db.last_backup_at = datetime.utcnow()
    session.add(db)
    session.flush()
    log_audit(
        session,
        actor="db-admin",
        action="db.backup",
        target=f"db:{db.id}",
        after={"strategy": strategy, "namespace": target_namespace},
    )
    return snapshot


def restore_database(
    session: Session,
    db: DatabaseInstance,
    *,
    snapshot_id: int,
    target_namespace: str,
    approval_ticket: Optional[str],
) -> dict:
    log_audit(
        session,
        actor="db-admin",
        action="db.restore",
        target=f"db:{db.id}",
        after={
            "snapshot_id": snapshot_id,
            "target_namespace": target_namespace,
            "approval": approval_ticket,
        },
    )
    return {
        "db_id": db.id,
        "snapshot_id": snapshot_id,
        "target_namespace": target_namespace,
        "status": "restoring",
    }


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------


def create_snapshot(session: Session, data: dict) -> Snapshot:
    snapshot = Snapshot(**data)
    session.add(snapshot)
    session.flush()
    log_audit(
        session,
        actor="snapshot",
        action="snapshot.create",
        target=f"snapshot:{snapshot.id}",
        after={"target": data.get("target_id"), "kind": data.get("kind")},
    )
    return snapshot


def list_snapshots(session: Session, *, target_type: Optional[SnapshotTarget] = None) -> list[Snapshot]:
    stmt = select(Snapshot).order_by(Snapshot.created_at.desc())
    if target_type:
        stmt = stmt.where(Snapshot.target_type == target_type)
    return list(session.scalars(stmt))


def get_snapshot(session: Session, snapshot_id: int) -> Optional[Snapshot]:
    return session.get(Snapshot, snapshot_id)


def update_snapshot_status(session: Session, snapshot: Snapshot, status: str) -> Snapshot:
    snapshot.status = status
    session.add(snapshot)
    session.flush()
    return snapshot


def snapshot_action(
    session: Session, snapshot: Snapshot, *, op: str, target_pool: Optional[str]
) -> dict:
    log_audit(
        session,
        actor="snapshot",
        action=f"snapshot.{op}",
        target=f"snapshot:{snapshot.id}",
        after={"target_pool": target_pool},
    )
    return {
        "snapshot_id": snapshot.id,
        "operation": op,
        "target_pool": target_pool,
        "status": "processing",
    }


# ---------------------------------------------------------------------------
# Gateways
# ---------------------------------------------------------------------------


def create_gateway(session: Session, data: dict) -> Gateway:
    gateway = Gateway(**data)
    session.add(gateway)
    session.flush()
    log_audit(
        session,
        actor="net-admin",
        action="gateway.create",
        target=f"gateway:{gateway.id}",
        after={"name": gateway.name},
    )
    return gateway


def list_gateways(session: Session) -> list[Gateway]:
    return list(session.scalars(select(Gateway).order_by(Gateway.created_at.desc())))


def get_gateway(session: Session, gateway_id: int) -> Optional[Gateway]:
    return session.get(Gateway, gateway_id)


def update_gateway(session: Session, gateway: Gateway, data: dict) -> Gateway:
    before = {"status": gateway.status.value}
    for key, value in data.items():
        setattr(gateway, key, value)
    session.add(gateway)
    session.flush()
    log_audit(
        session,
        actor="net-admin",
        action="gateway.update",
        target=f"gateway:{gateway.id}",
        before=before,
        after={"status": gateway.status.value},
    )
    return gateway


# ---------------------------------------------------------------------------
# AI Definition & LLM facade
# ---------------------------------------------------------------------------


def register_ai_definition(
    session: Session, *, yaml_payload: str, owner: Optional[str], runtime: Optional[str]
) -> AIDefinition:
    loaded = yaml.safe_load(yaml_payload)
    aid = loaded.get("id") or loaded.get("aid")
    if not aid:
        raise ValueError("AIDef 缺少唯一 id 字段")
    metadata = {
        "identity": loaded.get("identity"),
        "permissions": loaded.get("permissions"),
        "tasks": loaded.get("tasks"),
    }
    definition = AIDefinition(
        aid=aid,
        yaml=yaml_payload,
        owner=owner,
        runtime=runtime,
        details=metadata,
    )
    session.add(definition)
    session.flush()
    log_audit(
        session,
        actor=owner or "ai-admin",
        action="ai.register",
        target=f"ai:{definition.id}",
        after={"aid": aid},
    )
    return definition


def list_ai_definitions(session: Session) -> list[AIDefinition]:
    return list(session.scalars(select(AIDefinition).order_by(AIDefinition.created_at.desc())))


def get_ai_definition(session: Session, aid: str) -> Optional[AIDefinition]:
    stmt = select(AIDefinition).where(AIDefinition.aid == aid)
    return session.scalars(stmt).first()


def deploy_ai_definition(
    session: Session, definition: AIDefinition, *, target_pool: Optional[str]
) -> dict:
    definition.status = "deploying"
    session.add(definition)
    session.flush()
    log_audit(
        session,
        actor="ai-orchestrator",
        action="ai.deploy",
        target=f"ai:{definition.aid}",
        after={"target_pool": target_pool},
    )
    return {
        "aid": definition.aid,
        "target_pool": target_pool,
        "status": definition.status,
    }


def create_ai_workflow(session: Session, data: dict) -> AIWorkflow:
    workflow = AIWorkflow(**data)
    session.add(workflow)
    session.flush()
    log_audit(
        session,
        actor=data.get("owner") or "ai-operator",
        action="ai.workflow.create",
        target=f"ai-workflow:{workflow.id}",
        after={"name": workflow.name, "aid": workflow.aid},
    )
    return workflow


def update_ai_workflow(session: Session, workflow: AIWorkflow, data: dict) -> AIWorkflow:
    before = {
        "name": workflow.name,
        "description": workflow.description,
        "schedule": workflow.schedule,
        "is_enabled": workflow.is_enabled,
    }
    for field, value in data.items():
        setattr(workflow, field, value)
    session.add(workflow)
    session.flush()
    log_audit(
        session,
        actor=data.get("operator") or "ai-operator",
        action="ai.workflow.update",
        target=f"ai-workflow:{workflow.id}",
        before=before,
        after={
            "name": workflow.name,
            "description": workflow.description,
            "schedule": workflow.schedule,
            "is_enabled": workflow.is_enabled,
        },
    )
    return workflow


def list_ai_workflows(session: Session) -> list[AIWorkflow]:
    stmt = select(AIWorkflow).order_by(AIWorkflow.created_at.desc())
    return list(session.scalars(stmt))


def get_ai_workflow(session: Session, workflow_id: int) -> Optional[AIWorkflow]:
    return session.get(AIWorkflow, workflow_id)


def delete_ai_workflow(session: Session, workflow: AIWorkflow) -> None:
    log_audit(
        session,
        actor="ai-operator",
        action="ai.workflow.delete",
        target=f"ai-workflow:{workflow.id}",
        before={"name": workflow.name},
    )
    session.delete(workflow)
    session.flush()


def start_ai_workflow_run(
    session: Session,
    workflow: AIWorkflow,
    *,
    operator: Optional[str],
    context: Optional[dict],
) -> AIWorkflowRun:
    initiated_by = operator or "ai-operator"
    run = AIWorkflowRun(
        workflow=workflow,
        status=WorkflowRunStatus.RUNNING,
        initiated_by=initiated_by,
        context=context,
        started_at=datetime.utcnow(),
    )
    workflow.last_run_status = WorkflowRunStatus.RUNNING.value
    workflow.last_run_at = run.started_at
    session.add(run)
    session.add(workflow)
    session.flush()
    log_audit(
        session,
        actor=initiated_by,
        action="ai.workflow.run",
        target=f"ai-workflow:{workflow.id}",
        after={"run_id": run.id},
    )
    return run


def complete_ai_workflow_run(
    session: Session,
    run: AIWorkflowRun,
    *,
    success: bool,
    result: Optional[dict] = None,
    error: Optional[str] = None,
) -> AIWorkflowRun:
    run.status = (
        WorkflowRunStatus.SUCCEEDED if success else WorkflowRunStatus.FAILED
    )
    run.finished_at = datetime.utcnow()
    run.result = result
    run.error_message = error
    workflow = run.workflow
    if workflow:
        workflow.last_run_status = run.status.value
        workflow.last_run_at = run.finished_at
        session.add(workflow)
    session.add(run)
    session.flush()
    log_audit(
        session,
        actor=run.initiated_by or "ai-operator",
        action="ai.workflow.complete",
        target=f"ai-workflow:{run.workflow_id}",
        after={"status": run.status.value},
    )
    return run


def list_ai_workflow_runs(session: Session, workflow: AIWorkflow) -> list[AIWorkflowRun]:
    stmt = (
        select(AIWorkflowRun)
        .where(AIWorkflowRun.workflow_id == workflow.id)
        .order_by(AIWorkflowRun.started_at.desc())
    )
    return list(session.scalars(stmt))


def get_ai_workflow_run(session: Session, run_id: int) -> Optional[AIWorkflowRun]:
    return session.get(AIWorkflowRun, run_id)


def generate_llm_completion(payload: dict) -> dict:
    messages = payload.get("input", [])
    summary = "".join(message.get("content", "") for message in messages)
    return {
        "provider": payload.get("provider", "mock"),
        "model": payload.get("model", "mock-model"),
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": f"[模拟响应] 已接收 {len(summary)} 字符的上下文",
                }
            }
        ],
        "usage": {"prompt_tokens": len(summary) // 4 + 1, "completion_tokens": 32},
    }


# ---------------------------------------------------------------------------
# Audit inspection
# ---------------------------------------------------------------------------


def list_audit_logs(session: Session, *, limit: int = 100) -> list[AuditLog]:
    stmt = select(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit)
    return list(session.scalars(stmt))
