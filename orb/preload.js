// Narrow bridge between the orb page and the main process.
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('oracle', {
  backendInfo: () => ipcRenderer.invoke('backend-info'),
  setVisible: (visible) => ipcRenderer.send('orb-visible', !!visible),
  setHitbox: (rect) => ipcRenderer.send('orb-hitbox', rect),
  setFront: (front) => ipcRenderer.send('orb-front', !!front),
  reportMuted: (value) => ipcRenderer.send('orb-muted', !!value),
  // Messages from the tray/hotkeys to forward to the backend socket.
  onCommand: (callback) => ipcRenderer.on('command', (_e, msg) => callback(msg)),
  onDebugEvent: (callback) => ipcRenderer.on('debug-event', (_e, msg) => callback(msg)),
});
