const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("xinglianDesktop", {
  platform: () => ipcRenderer.invoke("app:get-platform"),
  openExternal: (url) => ipcRenderer.invoke("app:open-external", url),
});
