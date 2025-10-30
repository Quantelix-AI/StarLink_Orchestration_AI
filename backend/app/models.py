from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from .database import Base


class Architecture(str, enum.Enum):
    X86_64 = "x86_64"
    ARM64 = "arm64"
    RISCV64 = "riscv64"
    UNIVERSAL = "universal"


class ServerStatus(str, enum.Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    PROVISIONING = "provisioning"


class TaskStatus(str, enum.Enum):
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ActionType(str, enum.Enum):
    DOCKER = "docker_launch"
    INSTALLATION = "system_install"
    MAINTENANCE = "maintenance"


class PoolType(str, enum.Enum):
    COMPUTE = "compute"
    GPU = "gpu"
    STORAGE = "storage"
    NETWORK = "network"


class ContainerState(str, enum.Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    RESTARTING = "restarting"
    DEGRADED = "degraded"


class DatabaseEngine(str, enum.Enum):
    MYSQL = "mysql"
    MONGODB = "mongodb"


class SnapshotTarget(str, enum.Enum):
    SERVER = "server"
    VOLUME = "volume"
    DATABASE = "database"


class GatewayStatus(str, enum.Enum):
    ACTIVE = "active"
    DRAINING = "draining"
    DISABLED = "disabled"


class PolicyEnforcement(str, enum.Enum):
    ALLOW = "allow"
    DENY = "deny"


class MaintenanceStatus(str, enum.Enum):
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class WorkflowRunStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Server(Base):
    __tablename__ = "servers"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(120), nullable=False, unique=True)
    architecture = Column(Enum(Architecture), nullable=False)
    total_cpu = Column(Integer, nullable=False)
    total_memory = Column(Integer, nullable=False)
    total_storage = Column(Integer, nullable=False)
    available_cpu = Column(Integer, nullable=False)
    available_memory = Column(Integer, nullable=False)
    available_storage = Column(Integer, nullable=False)
    status = Column(Enum(ServerStatus), nullable=False, default=ServerStatus.OFFLINE)
    location = Column(String(255), nullable=True)
    cluster = Column(String(120), nullable=True)
    zone = Column(String(120), nullable=True)
    pool_id = Column(Integer, ForeignKey("resource_pools.id"), nullable=True, index=True)
    tags = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    tasks = relationship("Task", back_populates="server", cascade="all, delete")
    actions = relationship("ActionLog", back_populates="server", cascade="all, delete")
    metrics = relationship(
        "ServerMetric",
        back_populates="server",
        cascade="all, delete-orphan",
        order_by="desc(ServerMetric.recorded_at)",
    )
    maintenance_events = relationship(
        "MaintenanceEvent",
        back_populates="server",
        cascade="all, delete-orphan",
        order_by="desc(MaintenanceEvent.scheduled_for)",
    )
    containers = relationship(
        "Container",
        back_populates="host",
        cascade="all, delete-orphan",
        order_by="desc(Container.updated_at)",
    )

    pool = relationship("ResourcePool", back_populates="servers")


class Task(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(120), nullable=False)
    required_architecture = Column(Enum(Architecture), nullable=False)
    required_cpu = Column(Integer, nullable=False)
    required_memory = Column(Integer, nullable=False)
    required_storage = Column(Integer, nullable=False)
    status = Column(Enum(TaskStatus), nullable=False, default=TaskStatus.PENDING)
    priority = Column(Integer, nullable=False, default=1)
    deadline = Column(DateTime, nullable=True)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    server_id = Column(Integer, ForeignKey("servers.id"), nullable=True)

    server = relationship("Server", back_populates="tasks")


class ActionLog(Base):
    __tablename__ = "action_logs"

    id = Column(Integer, primary_key=True)
    server_id = Column(Integer, ForeignKey("servers.id"), nullable=False)
    action_type = Column(Enum(ActionType), nullable=False)
    command = Column(String(255), nullable=False)
    requested_by = Column(String(120), nullable=True)
    success = Column(Boolean, default=False, nullable=False)
    details = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)

    server = relationship("Server", back_populates="actions")


class ServerMetric(Base):
    __tablename__ = "server_metrics"

    id = Column(Integer, primary_key=True)
    server_id = Column(Integer, ForeignKey("servers.id"), nullable=False, index=True)
    cpu_usage = Column(Float, nullable=False)
    memory_usage = Column(Float, nullable=False)
    storage_usage = Column(Float, nullable=False)
    node_load1 = Column(Float, nullable=True)
    node_load5 = Column(Float, nullable=True)
    node_load15 = Column(Float, nullable=True)
    network_in_mbps = Column(Float, nullable=True)
    network_out_mbps = Column(Float, nullable=True)
    gpu_utilization = Column(Float, nullable=True)
    gpu_memory_utilization = Column(Float, nullable=True)
    slo_violation = Column(Boolean, default=False, nullable=False)
    health_score = Column(Float, nullable=True)
    temperature = Column(Float, nullable=True)
    recorded_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    server = relationship("Server", back_populates="metrics")


class MaintenanceEvent(Base):
    __tablename__ = "maintenance_events"

    id = Column(Integer, primary_key=True)
    server_id = Column(Integer, ForeignKey("servers.id"), nullable=False, index=True)
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    scheduled_for = Column(DateTime, nullable=False, index=True)
    duration_minutes = Column(Integer, nullable=False, default=60)
    status = Column(Enum(MaintenanceStatus), nullable=False, default=MaintenanceStatus.SCHEDULED)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    server = relationship("Server", back_populates="maintenance_events")


class ResourcePool(Base):
    __tablename__ = "resource_pools"
    __table_args__ = (UniqueConstraint("name", name="uq_pool_name"),)

    id = Column(Integer, primary_key=True)
    name = Column(String(120), nullable=False)
    type = Column(Enum(PoolType), nullable=False, default=PoolType.COMPUTE)
    quotas = Column(JSON, nullable=True)
    labels = Column(JSON, nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    servers = relationship(
        "Server",
        back_populates="pool",
        cascade="all, save-update",
        order_by="Server.name",
    )


class Container(Base):
    __tablename__ = "containers"

    id = Column(Integer, primary_key=True)
    host_id = Column(Integer, ForeignKey("servers.id"), nullable=False, index=True)
    name = Column(String(160), nullable=False)
    image = Column(String(200), nullable=False)
    state = Column(Enum(ContainerState), nullable=False, default=ContainerState.RUNNING)
    cpu_percent = Column(Float, nullable=True)
    memory_percent = Column(Float, nullable=True)
    restarts = Column(Integer, nullable=False, default=0)
    health = Column(String(120), nullable=True)
    compose_stack = Column(String(120), nullable=True)
    labels = Column(JSON, nullable=True)
    logs = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    host = relationship("Server", back_populates="containers")


class DatabaseInstance(Base):
    __tablename__ = "db_instances"

    id = Column(Integer, primary_key=True)
    type = Column(Enum(DatabaseEngine), nullable=False)
    name = Column(String(120), nullable=False)
    dsn = Column(String(255), nullable=False)
    role = Column(String(80), nullable=True)
    status = Column(String(80), nullable=False, default="unknown")
    rpo_seconds = Column(Integer, nullable=True)
    rto_seconds = Column(Integer, nullable=True)
    metrics = Column(JSON, nullable=True)
    tags = Column(JSON, nullable=True)
    last_backup_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )


class Snapshot(Base):
    __tablename__ = "snapshots"

    id = Column(Integer, primary_key=True)
    target_type = Column(Enum(SnapshotTarget), nullable=False)
    target_id = Column(Integer, nullable=False, index=True)
    kind = Column(String(80), nullable=False)
    schedule_id = Column(String(80), nullable=True)
    cron = Column(String(120), nullable=True)
    retention = Column(String(80), nullable=True)
    hooks = Column(JSON, nullable=True)
    status = Column(String(80), nullable=False, default="created")
    meta = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )


class Gateway(Base):
    __tablename__ = "gateways"

    id = Column(Integer, primary_key=True)
    name = Column(String(120), nullable=False, unique=True)
    config = Column(JSON, nullable=False)
    status = Column(Enum(GatewayStatus), nullable=False, default=GatewayStatus.ACTIVE)
    labels = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )


class Policy(Base):
    __tablename__ = "policies"

    id = Column(Integer, primary_key=True)
    name = Column(String(160), nullable=False, unique=True)
    scope = Column(JSON, nullable=False)
    rules = Column(JSON, nullable=False)
    audit = Column(Boolean, default=True, nullable=False)
    change_window = Column(String(120), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )


class AIDefinition(Base):
    __tablename__ = "ai_definitions"

    id = Column(Integer, primary_key=True)
    aid = Column(String(120), nullable=False, unique=True)
    yaml = Column(Text, nullable=False)
    status = Column(String(80), nullable=False, default="registered")
    owner = Column(String(120), nullable=True)
    runtime = Column(String(120), nullable=True)
    details = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    workflows = relationship(
        "AIWorkflow", back_populates="definition", cascade="all, delete-orphan"
    )


class AIWorkflow(Base):
    __tablename__ = "ai_workflows"

    id = Column(Integer, primary_key=True)
    name = Column(String(160), nullable=False, unique=True)
    aid = Column(String(120), ForeignKey("ai_definitions.aid"), nullable=False)
    description = Column(Text, nullable=True)
    schedule = Column(String(120), nullable=True)
    triggers = Column(JSON, nullable=True)
    playbook = Column(JSON, nullable=True)
    tags = Column(JSON, nullable=True)
    is_enabled = Column(Boolean, nullable=False, default=True)
    last_run_status = Column(String(80), nullable=True)
    last_run_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    definition = relationship("AIDefinition", back_populates="workflows")
    runs = relationship(
        "AIWorkflowRun",
        back_populates="workflow",
        cascade="all, delete-orphan",
    )


class AIWorkflowRun(Base):
    __tablename__ = "ai_workflow_runs"

    id = Column(Integer, primary_key=True)
    workflow_id = Column(Integer, ForeignKey("ai_workflows.id"), nullable=False)
    status = Column(Enum(WorkflowRunStatus), nullable=False, default=WorkflowRunStatus.PENDING)
    initiated_by = Column(String(120), nullable=True)
    context = Column(JSON, nullable=True)
    result = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    finished_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    workflow = relationship("AIWorkflow", back_populates="runs")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True)
    actor = Column(String(120), nullable=False)
    action = Column(String(160), nullable=False)
    target = Column(String(160), nullable=False)
    before = Column(JSON, nullable=True)
    after = Column(JSON, nullable=True)
    message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
