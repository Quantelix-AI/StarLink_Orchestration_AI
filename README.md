# 星联调度（StarLink Orchestration）

星联调度是一个面向多架构服务器集群的开源资源编排系统，提供 Web 控制台与 RESTful API，帮助运维与研发团队统一管理 x86、ARM、RISC-V 等异构资源，实现跨架构调度、容器/数据库/快照治理以及 AI 托管能力。

## 功能特性

- **多架构调度**：对服务器记录硬件架构，任务调度时自动匹配最合适的节点并扣减资源配额。
- **集中资源面板**：分钟级可观测能力，展示跨资源池/集群/标签的实时指标、健康度、告警与导出。
- **监控趋势分析**：支持 1h~30d 的指标趋势、P50/P95/P99 百分位计算与维护事件标注。
- **资源池与策略联动**：定义计算池/GPU 池/存储池配额与调度水位，支持服务器一键迁移与策略预演。
- **容器与数据库治理**：提供容器生命周期管理、实时日志、数据库备份/恢复演练及指标记录。
- **快照与网关中心**：统一管理文件系统/块存储/数据库快照，维护中心网关配置、灰度策略与限流模板。
- **AI 一键托管**：上传 AIDef 文档即可校验、部署并挂接观测面板，预留 OpenAI/DeepSeek/本地模型统一接口。
- **AI 工作流控制台**：在 Web/桌面端配置 AI 运维工作流、联动 AIDef、跟踪运行历史并一键触发调度。
- **端口策略与审计**：安全策略支持审批预演、审计日志完整记录“谁在什么时候做了什么”。
- **RESTful API**：使用 FastAPI 构建，接口文档可在运行服务后通过 `/docs` 查看。
- **现代化控制台**：参考 AWS 控制台的视觉层级与交互样式，提供深色玻璃态界面与快速动作入口。
- **轻量部署**：SQLite 持久化，默认即可运行；也可替换为外部数据库。

## 目录结构

```
backend/
  app/
    main.py          # FastAPI 应用入口
    database.py      # 数据库初始化与 Session 管理
    models.py        # SQLAlchemy ORM 模型
    schemas.py       # Pydantic 数据模型
    services.py      # 调度与动作业务逻辑
frontend/
  index.html         # 基于原生 JS 的控制台界面
desktop/
  package.json       # Electron 跨平台桌面应用配置
  main.js / preload.js
requirements.txt     # Python 运行依赖
```

## 快速开始

1. **安装依赖**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows 使用 .venv\\Scripts\\activate
   pip install -r requirements.txt
   ```

2. **启动后端服务**

   ```bash
   uvicorn backend.app.main:app --reload
   ```

   服务默认监听 `http://127.0.0.1:8000`，可访问 `http://127.0.0.1:8000/docs` 查看交互式接口文档。

3. **启动前端控制台**

   直接用浏览器打开 `frontend/index.html`，或使用任意静态文件服务器托管：

   ```bash
   python -m http.server --directory frontend 8080
   ```

   打开 `http://127.0.0.1:8080` 即可体验星联调度控制台。

## 桌面端（macOS / Windows / Linux）

借助 `desktop/` 目录下的 Electron 封装，可获得与 Web 一致的跨平台原生窗口体验：

```bash
cd desktop
npm install
npm start
```

如需生成各平台安装包，可分别执行：

```bash
npm run package:mac   # macOS .app
npm run package:win   # Windows 可执行文件
npm run package:linux # Linux AppImage/目录
```

打包结果输出至 `desktop/dist/`。

## 常用 API 示例

- 注册服务器：`POST /servers`
- 查看服务器列表：`GET /servers`
- 上报实时指标：`POST /servers/{server_id}/metrics`
- 概览导出：`GET /overview?export=csv`
- 查询趋势：`GET /trends?host_ids=1&host_ids=2&metric=cpu_usage&hours=6`
- 管理资源池：`POST /pools` / `POST /pools/move`
- 容器操作：`POST /api/containers/{id}/actions {"op":"restart"}`
- 数据库备份：`POST /api/databases/{id}/backup`
- 注册 AIDef：`POST /api/ai/defs`
- 管理 AI 工作流：`POST /api/ai/workflows` / `GET /api/ai/workflows/{id}/runs`
- LLM 代理：`POST /api/llm/completions`
- 策略预演：`POST /policies/{id}/dry-run`
- 快照操作：`POST /api/snapshots/{id}/actions {"op":"rollback"}`

## 测试

项目内置 Pytest 用例验证核心调度逻辑：

```bash
pytest
```

## 许可证

本项目以 Apache License 2.0 开源，欢迎提交 Issue 与 PR 共同完善星联调度。
