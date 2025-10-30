from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app import services
from backend.app.database import Base
from backend.app.models import (
    Architecture,
    DatabaseEngine,
    MaintenanceStatus,
    PoolType,
    ServerStatus,
    SnapshotTarget,
    TaskStatus,
    WorkflowRunStatus,
)


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    TestingSessionLocal = sessionmaker(
        bind=engine, autoflush=False, autocommit=False, future=True
    )
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_dispatch_prefers_matching_architecture(session):
    arm_server = services.create_server(
        session,
        {
            "name": "arm-node",
            "architecture": Architecture.ARM64,
            "total_cpu": 32,
            "total_memory": 256,
            "total_storage": 2000,
            "status": ServerStatus.ONLINE,
            "location": "cn-hz-1",
            "tags": {"env": "prod"},
        },
    )
    x86_server = services.create_server(
        session,
        {
            "name": "x86-node",
            "architecture": Architecture.X86_64,
            "total_cpu": 32,
            "total_memory": 256,
            "total_storage": 2000,
            "status": ServerStatus.ONLINE,
            "location": "cn-bj-1",
            "tags": {"env": "prod"},
        },
    )

    universal_task = services.create_task(
        session,
        {
            "name": "multi-arch-workload",
            "required_architecture": Architecture.UNIVERSAL,
            "required_cpu": 8,
            "required_memory": 64,
            "required_storage": 400,
        },
    )

    matched_server = services.dispatch_task(session, universal_task, [arm_server, x86_server])
    assert matched_server is not None
    assert universal_task.server_id == matched_server.id
    assert universal_task.status == TaskStatus.SCHEDULED


def test_dispatch_insufficient_resources(session):
    server = services.create_server(
        session,
        {
            "name": "tiny-node",
            "architecture": Architecture.X86_64,
            "total_cpu": 4,
            "total_memory": 8,
            "total_storage": 50,
            "status": ServerStatus.ONLINE,
            "location": None,
            "tags": {"env": "lab"},
        },
    )

    heavy_task = services.create_task(
        session,
        {
            "name": "big-job",
            "required_architecture": Architecture.X86_64,
            "required_cpu": 16,
            "required_memory": 64,
            "required_storage": 500,
        },
    )

    matched_server = services.dispatch_task(session, heavy_task, [server])
    assert matched_server is None
    assert heavy_task.status == TaskStatus.PENDING


def test_metric_reporting_and_dashboard_health(session):
    server = services.create_server(
        session,
        {
            "name": "metric-node",
            "architecture": Architecture.X86_64,
            "total_cpu": 32,
            "total_memory": 128,
            "total_storage": 500,
            "status": ServerStatus.ONLINE,
            "location": "lab",
            "tags": {"env": "prod"},
        },
    )

    services.record_metric(
        session,
        server,
        {
            "cpu_usage": 50.0,
            "memory_usage": 60.0,
            "storage_usage": 30.0,
            "network_in_mbps": 120.0,
            "network_out_mbps": 90.0,
            "temperature": 42.0,
            "recorded_at": datetime.utcnow(),
        },
    )

    stats = services.compute_dashboard_health(session)
    assert stats["average_cpu"] == 50.0
    assert stats["average_memory"] == 60.0
    assert stats["metric_servers"] == 1
    assert stats["busiest_server"] == "metric-node"


def test_schedule_and_update_maintenance(session):
    server = services.create_server(
        session,
        {
            "name": "maint-node",
            "architecture": Architecture.ARM64,
            "total_cpu": 16,
            "total_memory": 64,
            "total_storage": 300,
            "status": ServerStatus.ONLINE,
            "location": None,
            "tags": {"env": "prod"},
        },
    )

    event = services.schedule_maintenance(
        session,
        server=server,
        title="Firmware upgrade",
        description="升级 BMC 固件",
        scheduled_for=datetime.utcnow() + timedelta(hours=1),
        duration_minutes=90,
    )

    assert event.status == MaintenanceStatus.SCHEDULED

    updated = services.update_maintenance(
        session,
        event,
        {"status": MaintenanceStatus.IN_PROGRESS},
    )

    assert updated.status == MaintenanceStatus.IN_PROGRESS


def test_compute_overview_contains_alerts(session):
    server = services.create_server(
        session,
        {
            "name": "alert-node",
            "architecture": Architecture.X86_64,
            "total_cpu": 16,
            "total_memory": 64,
            "total_storage": 500,
            "status": ServerStatus.ONLINE,
            "location": "zone-a",
            "tags": {"role": "web", "env": "prod"},
        },
    )
    services.record_metric(
        session,
        server,
        {
            "cpu_usage": 95.0,
            "memory_usage": 82.0,
            "storage_usage": 40.0,
            "slo_violation": True,
            "recorded_at": datetime.utcnow(),
        },
    )
    overview = services.compute_overview(session)
    assert overview["summary"]["total_hosts"] == 1
    assert any(alert["resource"] == "alert-node" for alert in overview["alerts"])


def test_policy_dry_run(session):
    pool = services.create_pool(
        session,
        {
            "name": "prod-compute",
            "type": PoolType.COMPUTE,
            "description": "prod pool",
        },
    )
    server = services.create_server(
        session,
        {
            "name": "policy-node",
            "architecture": Architecture.X86_64,
            "total_cpu": 16,
            "total_memory": 64,
            "total_storage": 500,
            "status": ServerStatus.ONLINE,
            "location": "dc1",
            "pool_id": pool.id,
            "tags": {"role": "web", "env": "prod"},
        },
    )
    policy = services.create_policy(
        session,
        {
            "name": "sg-web-prod",
            "scope": {"labels": {"role": "web"}},
            "rules": [{"action": "allow", "port": "80"}],
        },
    )
    result = services.policy_dry_run(session, policy, labels={"env": "prod"})
    assert server.id in result["affected_hosts"]


def test_register_ai_definition_requires_id(session):
    with pytest.raises(ValueError):
        services.register_ai_definition(
            session,
            yaml_payload="version: 1.0",
            owner="ops",
            runtime="container",
        )


def test_database_backup_creates_snapshot(session):
    db = services.register_db_instance(
        session,
        {
            "type": DatabaseEngine.MYSQL,
            "name": "db-main",
            "dsn": "mysql://root@localhost:3306/main",
        },
    )
    snapshot = services.schedule_db_backup(
        session,
        db,
        strategy="full",
        target_namespace="drill",
        verify=True,
    )
    assert snapshot.target_type == SnapshotTarget.DATABASE
    assert snapshot.meta["namespace"] == "drill"


def test_ai_workflow_lifecycle(session):
    yaml_payload = """
id: ops-copilot
identity:
  name: Ops
"""
    definition = services.register_ai_definition(
        session,
        yaml_payload=yaml_payload,
        owner="ai-team",
        runtime="mock",
    )

    workflow = services.create_ai_workflow(
        session,
        {
            "name": "cpu-guard",
            "aid": definition.aid,
            "description": "自动缓解 CPU 告警",
            "schedule": "cron(*/5 * * * *)",
            "triggers": [{"type": "metric_threshold", "expr": "cpu > 80"}],
            "playbook": [{"tool": "run_shell", "args": {"script": "echo heal"}}],
            "tags": {"env": "prod"},
            "is_enabled": True,
        },
    )

    run = services.start_ai_workflow_run(
        session,
        workflow,
        operator="tester",
        context={"incident": "cpu_spike"},
    )
    assert run.status == WorkflowRunStatus.RUNNING

    services.complete_ai_workflow_run(
        session,
        run,
        success=True,
        result={"action": "restart_service"},
    )

    assert run.status == WorkflowRunStatus.SUCCEEDED
    assert workflow.last_run_status == WorkflowRunStatus.SUCCEEDED.value

    runs = services.list_ai_workflow_runs(session, workflow)
    assert len(runs) == 1
    assert runs[0].context == {"incident": "cpu_spike"}
