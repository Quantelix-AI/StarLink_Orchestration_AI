# 星联调度桌面端

该目录包含基于 Electron 封装的跨平台桌面应用，可直接加载 `frontend/index.html` 并提供一致的控制台体验，支持 macOS、Windows 与 Linux。

## 快速开始

```bash
cd desktop
npm install
npm start
```

上述命令会在开发模式下启动 Electron 并打开本地前端控制台。

## 打包发行

Electron Packager 已包含在开发依赖中，可针对不同平台生成安装包（需在对应平台打包以确保签名与原生依赖一致）。

```bash
# macOS (.app)
npm run package:mac

# Windows (.exe)
npm run package:win

# Linux (AppImage/二进制目录)
npm run package:linux
```

输出位于 `desktop/dist/`。

## 与后端联动

桌面端默认直接加载 `frontend/index.html`，其 API 请求与浏览器版本保持一致。若需自定义 API 地址，可在前端页面中设置 `apiBase` 或者扩展 `preload.js` 暴露配置面板。
