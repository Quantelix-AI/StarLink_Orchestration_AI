from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, validator

from .models import (
    ActionType,
    Architecture,
    ContainerState,
    DatabaseEngine,
    GatewayStatus,
    MaintenanceStatus,
    PoolType,
    ServerStatus,
    SnapshotTarget,
    TaskStatus,
    WorkflowRunStatus,
)


# -----------------------------
# Core server & task schemas
# -----------------------------


class ServerBase(BaseModel):
    name: str = Field(..., description="服务器名称")
    architecture: Architecture = Field(..., description="架构")
    total_cpu: int = Field(..., ge=1, description="CPU 核心总数")
    total_memory: int = Field(..., ge=1, description="内存（GB）")
    total_storage: int = Field(..., ge=1, description="存储（GB）")
    location: Optional[str] = Field(None, description="数据中心位置")
    cluster: Optional[str] = Field(None, description="集群名称")
    zone: Optional[str] = Field(None, description="可用区/机房")
    pool_id: Optional[int] = Field(None, description="所属资源池 ID")
    tags: Optional[dict[str, Any]] = Field(default=None, description="标签信息")


class ServerCreate(ServerBase):
    status: ServerStatus = ServerStatus.OFFLINE


class ServerUpdate(BaseModel):
    status: Optional[ServerStatus]
    available_cpu: Optional[int]
    available_memory: Optional[int]
    available_storage: Optional[int]
    location: Optional[str]
    cluster: Optional[str]
    zone: Optional[str]
    pool_id: Optional[int]
    tags: Optional[dict[str, Any]]


class ServerRead(ServerBase):
    id: int
    available_cpu: int
    available_memory: int
    available_storage: int
    status: ServerStatus
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class TaskBase(BaseModel):
    name: str
    required_architecture: Architecture
    required_cpu: int = Field(..., ge=1)
    required_memory: int = Field(..., ge=1)
    required_storage: int = Field(..., ge=1)
    metadata: Optional[dict[str, Any]] = None
    priority: int = Field(1, ge=1, le=5, description="优先级，1-5 越大越重要")
    deadline: Optional[datetime] = Field(None, description="任务截止时间")


class TaskCreate(TaskBase):
    pass


class TaskRead(TaskBase):
    id: int
    status: TaskStatus
    server_id: Optional[int]
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class ActionLogRead(BaseModel):
    id: int
    action_type: ActionType
    command: str
    requested_by: Optional[str]
    success: bool
    details: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]

    class Config:
        orm_mode = True


# -----------------------------
# Metrics, dashboard & overview
# -----------------------------


class MetricCreate(BaseModel):
    cpu_usage: float = Field(..., ge=0, le=100, description="CPU 使用率 %")
    memory_usage: float = Field(..., ge=0, le=100, description="内存使用率 %")
    storage_usage: float = Field(..., ge=0, le=100, description="存储使用率 %")
    node_load1: Optional[float] = Field(None, description="1 分钟负载")
    node_load5: Optional[float] = Field(None, description="5 分钟负载")
    node_load15: Optional[float] = Field(None, description="15 分钟负载")
    network_in_mbps: Optional[float] = Field(None, ge=0, description="入站 Mbps")
    network_out_mbps: Optional[float] = Field(None, ge=0, description="出站 Mbps")
    gpu_utilization: Optional[float] = Field(None, ge=0, le=100, description="GPU 利用率 %")
    gpu_memory_utilization: Optional[float] = Field(
        None, ge=0, le=100, description="GPU 显存利用率 %"
    )
    slo_violation: Optional[bool] = Field(False, description="SLO 是否触发")
    health_score: Optional[float] = Field(None, ge=0, le=100, description="健康度评分")
    temperature: Optional[float] = Field(None, ge=-40, le=120, description="温度 °C")
    recorded_at: Optional[datetime] = Field(None, description="记录时间")


class MetricRead(MetricCreate):
    id: int
    recorded_at: datetime

    class Config:
        orm_mode = True


class DashboardSummary(BaseModel):
    total_servers: int
    online_servers: int
    offline_servers: int
    maintenance_servers: int
    provisioning_servers: int
    total_tasks: int
    running_tasks: int
    pending_tasks: int
    completed_tasks: int
    failed_tasks: int


class DashboardHealth(BaseModel):
    average_cpu: float
    average_memory: float
    average_storage: float
    hottest_server: Optional[str]
    busiest_server: Optional[str]
    metric_servers: int


class OverviewSummary(BaseModel):
    total_hosts: int
    total_pools: int
    total_containers: int
    total_databases: int
    total_gateways: int
    average_health: float
    updated_at: datetime


class OverviewServer(BaseModel):
    id: int
    name: str
    status: ServerStatus
    architecture: Architecture
    pool: Optional[str]
    cluster: Optional[str]
    zone: Optional[str]
    tags: dict[str, Any] | None
    metrics: dict[str, Any] | None


class OverviewAlert(BaseModel):
    resource: str
    dimension: str
    value: float
    threshold: float
    severity: str
    detail: str


class OverviewResponse(BaseModel):
    summary: OverviewSummary
    servers: list[OverviewServer]
    alerts: list[OverviewAlert]
    bottlenecks: list[OverviewAlert]


class TrendEvent(BaseModel):
    timestamp: datetime
    title: str
    description: Optional[str] = None


class TrendPoint(BaseModel):
    timestamp: datetime
    value: float


class TrendSeries(BaseModel):
    host_id: int
    metric: str
    points: list[TrendPoint]
    percentiles: dict[str, float]
    events: list[TrendEvent]


# -----------------------------
# Resource pools & policies
# -----------------------------


class PoolBase(BaseModel):
    name: str
    type: PoolType = PoolType.COMPUTE
    quotas: Optional[dict[str, Any]] = None
    labels: Optional[dict[str, Any]] = None
    description: Optional[str] = None


class PoolCreate(PoolBase):
    pass


class PoolUpdate(BaseModel):
    type: Optional[PoolType] = None
    quotas: Optional[dict[str, Any]] = None
    labels: Optional[dict[str, Any]] = None
    description: Optional[str] = None


class PoolRead(PoolBase):
    id: int
    created_at: datetime
    updated_at: datetime
    server_count: int
    capacity: dict[str, Any]

    class Config:
        orm_mode = True


class PoolRebalanceResult(BaseModel):
    pool_id: int
    moved_hosts: list[int]
    notes: str


class PoolMoveRequest(BaseModel):
    server_id: int
    target_pool_id: int


class PolicyBase(BaseModel):
    name: str
    scope: dict[str, Any]
    rules: list[dict[str, Any]]
    audit: bool = True
    change_window: Optional[str] = None


class PolicyCreate(PolicyBase):
    pass


class PolicyRead(PolicyBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class PolicyDryRunRequest(BaseModel):
    target_labels: dict[str, Any]
    description: Optional[str] = None


class PolicyDryRunResult(BaseModel):
    policy_id: int
    affected_hosts: list[int]
    affected_services: list[str]
    summary: str


# -----------------------------
# Containers & Docker panels
# -----------------------------


class ContainerBase(BaseModel):
    host_id: int
    name: str
    image: str
    state: ContainerState = ContainerState.RUNNING
    cpu_percent: Optional[float] = Field(None, ge=0, le=100)
    memory_percent: Optional[float] = Field(None, ge=0, le=100)
    restarts: int = 0
    health: Optional[str] = None
    compose_stack: Optional[str] = None
    labels: Optional[dict[str, Any]] = None
    logs: Optional[str] = None


class ContainerCreate(ContainerBase):
    pass


class ContainerRead(ContainerBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class ContainerActionRequest(BaseModel):
    op: str = Field(..., description="start|stop|restart|logs|exec")
    args: Optional[dict[str, Any]] = None


class ContainerLogResponse(BaseModel):
    id: int
    logs: str
    updated_at: datetime


# -----------------------------
# Database panel schemas
# -----------------------------


class DatabaseInstanceBase(BaseModel):
    type: DatabaseEngine
    name: str
    dsn: str
    role: Optional[str] = None
    status: Optional[str] = Field("unknown", description="实例状态")
    rpo_seconds: Optional[int] = Field(None, ge=0)
    rto_seconds: Optional[int] = Field(None, ge=0)
    metrics: Optional[dict[str, Any]] = None
    tags: Optional[dict[str, Any]] = None


class DatabaseInstanceCreate(DatabaseInstanceBase):
    pass


class DatabaseInstanceRead(DatabaseInstanceBase):
    id: int
    last_backup_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class DatabaseBackupRequest(BaseModel):
    strategy: str = Field(..., description="full|incremental|binlog|oplog")
    target_namespace: Optional[str] = Field(None, description="演练命名空间")
    verify: bool = Field(True, description="是否校验备份")


class DatabaseRestoreRequest(BaseModel):
    snapshot_id: int
    target_namespace: str
    approval_ticket: Optional[str] = None


# -----------------------------
# AI 一键托管 schemas
# -----------------------------


class AIDefRegisterRequest(BaseModel):
    yaml: str = Field(..., description="AIDef YAML 内容")
    owner: Optional[str] = None
    runtime: Optional[str] = None


class AIDefRead(BaseModel):
    id: int
    aid: str
    status: str
    owner: Optional[str]
    runtime: Optional[str]
    details: Optional[dict[str, Any]]
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class AIDeployRequest(BaseModel):
    aid: str
    target_pool: Optional[str] = None


class AIWorkflowBase(BaseModel):
    name: str = Field(..., description="工作流名称")
    aid: str = Field(..., description="关联的 AIDef 标识")
    description: Optional[str] = Field(None, description="工作流描述")
    schedule: Optional[str] = Field(None, description="定时表达式")
    triggers: Optional[list[dict[str, Any]]] = Field(
        default=None, description="触发器配置"
    )
    playbook: Optional[list[dict[str, Any]]] = Field(
        default=None, description="执行步骤"
    )
    tags: Optional[dict[str, Any]] = Field(default=None, description="标签")
    is_enabled: bool = Field(True, description="是否启用")


class AIWorkflowCreate(AIWorkflowBase):
    pass


class AIWorkflowUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    schedule: Optional[str] = None
    triggers: Optional[list[dict[str, Any]]] = None
    playbook: Optional[list[dict[str, Any]]] = None
    tags: Optional[dict[str, Any]] = None
    is_enabled: Optional[bool] = None


class AIWorkflowRead(AIWorkflowBase):
    id: int
    last_run_status: Optional[str]
    last_run_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class AIWorkflowRunRequest(BaseModel):
    context: Optional[dict[str, Any]] = Field(default=None, description="运行上下文")
    operator: Optional[str] = Field(default=None, description="执行人")


class AIWorkflowRunRead(BaseModel):
    id: int
    workflow_id: int
    status: WorkflowRunStatus
    initiated_by: Optional[str]
    context: Optional[dict[str, Any]]
    result: Optional[dict[str, Any]]
    error_message: Optional[str]
    started_at: datetime
    finished_at: Optional[datetime]

    class Config:
        orm_mode = True


class LLMChatMessage(BaseModel):
    role: str
    content: str


class LLMCompletionRequest(BaseModel):
    provider: str
    model: str
    input: list[LLMChatMessage]
    params: Optional[dict[str, Any]] = None
    metadata: Optional[dict[str, Any]] = None


class LLMCompletionResponse(BaseModel):
    provider: str
    model: str
    choices: list[dict[str, Any]]
    usage: dict[str, Any]


# -----------------------------
# Snapshot schemas
# -----------------------------


class SnapshotBase(BaseModel):
    target_type: SnapshotTarget
    target_id: int
    kind: str
    schedule_id: Optional[str] = None
    cron: Optional[str] = None
    retention: Optional[str] = None
    hooks: Optional[dict[str, Any]] = None
    meta: Optional[dict[str, Any]] = None


class SnapshotCreate(SnapshotBase):
    pass


class SnapshotRead(SnapshotBase):
    id: int
    status: str
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class SnapshotActionRequest(BaseModel):
    op: str = Field(..., description="rollback|clone|replicate")
    target_pool: Optional[str] = None


# -----------------------------
# Gateway schemas
# -----------------------------


class GatewayBase(BaseModel):
    name: str
    config: dict[str, Any]
    status: GatewayStatus = GatewayStatus.ACTIVE
    labels: Optional[dict[str, Any]] = None


class GatewayCreate(GatewayBase):
    pass


class GatewayRead(GatewayBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


# -----------------------------
# Audit schemas
# -----------------------------


class AuditLogRead(BaseModel):
    id: int
    actor: str
    action: str
    target: str
    before: Optional[dict[str, Any]]
    after: Optional[dict[str, Any]]
    message: Optional[str]
    created_at: datetime

    class Config:
        orm_mode = True


# -----------------------------
# Validation helpers
# -----------------------------


class MaintenanceBase(BaseModel):
    title: str = Field(..., description="维护标题")
    description: Optional[str] = Field(None, description="详细描述")
    scheduled_for: datetime
    duration_minutes: int = Field(60, ge=1, le=1440)

    @validator("scheduled_for")
    def validate_future(cls, value: datetime) -> datetime:
        if value.tzinfo:
            value = value.astimezone(tz=None).replace(tzinfo=None)
        return value


class MaintenanceCreate(MaintenanceBase):
    pass


class MaintenanceUpdate(BaseModel):
    status: Optional[MaintenanceStatus] = None
    description: Optional[str] = None


class MaintenanceRead(MaintenanceBase):
    id: int
    status: MaintenanceStatus
    server_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True
