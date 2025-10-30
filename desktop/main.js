const { app, BrowserWindow, ipcMain, nativeTheme, shell } = require("electron");
const path = require("node:path");

function createWindow() {
  const win = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1280,
    minHeight: 720,
    show: false,
    backgroundColor: "#020617",
    title: "星联调度控制台",
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });

  win.once("ready-to-show", () => {
    win.show();
  });

  const indexPath = path.join(__dirname, "..", "frontend", "index.html");
  win.loadFile(indexPath);
}

app.whenReady().then(() => {
  nativeTheme.themeSource = "dark";
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

ipcMain.handle("app:get-platform", () => process.platform);
ipcMain.handle("app:open-external", (_event, url) => {
  if (typeof url === "string" && url.startsWith("http")) {
    return shell.openExternal(url);
  }
  return false;
});
