// ORACLE orb - Electron main process (README 4.3, "Orb window").
//
// Starts the Python backend (oracle_server.py) as a child process, reads its
// ORACLE_READY line for the WebSocket port + per-launch token, and shows the
// Golden Lens orb: a transparent, click-through, always-on-top window that
// never takes focus. The renderer owns the WebSocket; this process handles
// the window, tray and global hotkeys.
//
//   npm start                      dev: runs ../oracle_server.py with python
//   ORACLE_BACKEND=<exe>           run a packaged backend exe instead
//   ORACLE_PYTHON=<python.exe>     pick the Python interpreter
//   ORACLE_ORB_ASLEEP=hide         hide the orb while asleep (default: it stays on screen)

const { app, BrowserWindow, Tray, Menu, ipcMain, globalShortcut, screen, nativeImage, dialog } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const readline = require('readline');

const ORB_W = 460;
const ORB_H = 540;
// The orb stays on screen (behind other windows) between commands, dimmed
// to an ember while asleep - the owner's choice.
const HIDE_WHEN_ASLEEP = process.env.ORACLE_ORB_ASLEEP === 'hide';
// Ctrl+Space is VS Code's suggest shortcut, so the talk hotkeys add Alt.
// Ctrl+Alt+O is also the Start-menu shortcut's hotkey (install_shortcut.ps1):
// it starts ORACLE when it isn't running, and talks when it is.
const TALK_HOTKEYS = ['Control+Alt+O', 'Control+Alt+Space'];
const MUTE_HOTKEY = 'Control+Alt+M';
const CHAT_HOTKEY = 'Control+Alt+C';

let backend = null;
let backendInfo = null;
let orb = null;
let tray = null;
let quitting = false;
let muted = false;

// Development: ORACLE_DEV_PROFILE=1 runs a separate instance alongside the
// real one (own profile, so no single-instance clash), ORACLE_NO_VOICE=1
// starts the backend without the microphone, ORACLE_OPEN_CHAT=1 opens the
// chat window at start.
if (process.env.ORACLE_DEV_PROFILE) {
  app.setPath('userData', path.join(app.getPath('temp'), `oracle-dev-${process.pid}`));
}

// A second launch (e.g. the Ctrl+Alt+O shortcut while running) just talks.
if (!app.requestSingleInstanceLock()) {
  app.exit(0);
}
app.on('second-instance', () => sendToBackend({ type: 'talk' }));

// The orb sits behind every other window (the owner's choice). Electron has
// no "send to back", so this calls SetWindowPos(HWND_BOTTOM) via koffi.
const HWND_BOTTOM = 1;
const HWND_NOTOPMOST = -2;
const SWP_NOSIZE = 0x0001, SWP_NOMOVE = 0x0002, SWP_NOACTIVATE = 0x0010;
let setWindowPos = null;
try {
  const user32 = require('koffi').load('user32.dll');
  setWindowPos = user32.func('bool __stdcall SetWindowPos(intptr_t hWnd, intptr_t after, int X, int Y, int cx, int cy, uint32_t flags)');
} catch (e) {
  console.error('koffi unavailable; the orb can\'t be sent behind other windows:', e.message);
}

function orbHwnd() {
  const handle = orb.getNativeWindowHandle();
  return handle.length >= 8 ? handle.readBigInt64LE(0) : BigInt(handle.readInt32LE(0));
}

function sendOrbToBack() {
  if (!orb || orb.isDestroyed() || !setWindowPos) return;
  orb.setAlwaysOnTop(false);
  setWindowPos(orbHwnd(), HWND_NOTOPMOST, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE | SWP_NOACTIVATE);
  setWindowPos(orbHwnd(), HWND_BOTTOM, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE | SWP_NOACTIVATE);
}

function startBackend() {
  const root = path.join(__dirname, '..');
  const exe = process.env.ORACLE_BACKEND;
  const cmd = exe || process.env.ORACLE_PYTHON || 'python';
  // The backend polls our PID and exits when we're gone, crash included.
  const args = (exe ? [] : [path.join(root, 'oracle_server.py')]).concat('--parent-pid', String(process.pid));
  if (process.env.ORACLE_NO_VOICE) args.push('--no-voice');
  if (process.env.ORACLE_DEV_PROFILE) args.push('--no-background');  // the real instance does these

  backend = spawn(cmd, args, {
    cwd: root,
    stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8' },
    windowsHide: true,
  });

  return new Promise((resolve, reject) => {
    readline.createInterface({ input: backend.stdout }).on('line', (line) => {
      if (line.startsWith('ORACLE_READY ')) {
        backendInfo = JSON.parse(line.slice('ORACLE_READY '.length));
        resolve(backendInfo);
      } else {
        console.log('[backend]', line);
      }
    });
    backend.stderr.on('data', (d) => process.stderr.write('[backend] ' + d));
    backend.on('error', reject);
    backend.on('exit', (code) => {
      if (!backendInfo) {
        reject(new Error(`backend exited with code ${code} before it was ready`));
      } else if (!quitting) {
        console.error(`backend exited with code ${code}; quitting`);
        app.quit();
      }
    });
  });
}

function createOrb() {
  const { workArea } = screen.getPrimaryDisplay();
  orb = new BrowserWindow({
    width: ORB_W,
    height: ORB_H,
    x: Math.round(workArea.x + (workArea.width - ORB_W) / 2),
    y: Math.round(workArea.y + (workArea.height - ORB_H) / 2),
    transparent: true,
    backgroundColor: '#00000000',
    frame: false,
    resizable: false,
    movable: false,
    minimizable: false,
    maximizable: false,
    alwaysOnTop: false, // sits behind other windows; see sendOrbToBack
    skipTaskbar: true,
    focusable: false, // never steals focus from what you're doing
    hasShadow: false,
    show: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      sandbox: true,
      backgroundThrottling: false, // keep the socket responsive while hidden
    },
  });
  // Click-through everywhere except where the renderer says (Yes/No buttons).
  orb.setIgnoreMouseEvents(true, { forward: true });
  orb.webContents.on('console-message', (event) => console.log('[orb]', event.message));
  orb.loadFile(path.join(__dirname, 'orb.html'));
}

// ---- chat window (../index.html; README 4.3 "chat window moved to Electron") ----
// Created on first use and hidden, not destroyed, on close, so reopening it
// is instant and keeps the conversation on screen.
let chat = null;

function openChat() {
  if (!chat || chat.isDestroyed()) {
    chat = new BrowserWindow({
      width: 1100,
      height: 760,
      minWidth: 720,
      minHeight: 480,
      frame: false,               // the page draws its own title bar
      backgroundColor: '#060a10',
      title: 'ORACLE',
      show: false,
      icon: path.join(__dirname, '..', 'jarvis.ico'),
      webPreferences: {
        preload: path.join(__dirname, 'chat_preload.js'),
        contextIsolation: true,
        sandbox: true,
      },
    });
    chat.on('close', (e) => {
      if (!quitting) {
        e.preventDefault();
        chat.hide();
      }
    });
    chat.webContents.on('console-message', (event) => console.log('[chat]', event.message));
    chat.loadFile(path.join(__dirname, '..', 'index.html'));
    chat.once('ready-to-show', () => { chat.show(); chat.focus(); });
    return;
  }
  if (chat.isMinimized()) chat.restore();
  chat.show();
  chat.focus();
}

ipcMain.on('open-chat', () => openChat());
ipcMain.on('chat-window', (_e, action) => {
  if (!chat || chat.isDestroyed()) return;
  if (action === 'minimize') chat.minimize();
  else if (action === 'maximize') (chat.isMaximized() ? chat.unmaximize() : chat.maximize());
  else if (action === 'hide') chat.hide();
});
ipcMain.handle('choose-folder', async () => {
  const result = await dialog.showOpenDialog(chat && !chat.isDestroyed() ? chat : undefined, {
    title: 'Choose your projects folder',
    properties: ['openDirectory'],
  });
  return result.canceled ? null : result.filePaths[0];
});

function sendToBackend(msg) {
  if (orb && !orb.isDestroyed()) orb.webContents.send('command', msg);
}

// A 32x32 ring drawn pixel by pixel: gold when the mic is live, grey when muted.
function trayIcon() {
  const size = 32;
  const buf = Buffer.alloc(size * size * 4);
  const [r, g, b] = muted ? [120, 110, 100] : [255, 184, 56];
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const d = Math.hypot(x - 15.5, y - 15.5);
      const ring = Math.max(0, 1 - Math.abs(d - 11.5) / 2.2);
      const core = Math.max(0, 1 - Math.max(0, d - 3.5) / 1.2);
      const a = Math.min(1, ring + core);
      const i = (y * size + x) * 4;
      // createFromBitmap expects BGRA on Windows.
      buf[i] = b; buf[i + 1] = g; buf[i + 2] = r; buf[i + 3] = Math.round(a * 255);
    }
  }
  return nativeImage.createFromBitmap(buf, { width: size, height: size });
}

function refreshTray() {
  if (!tray) return;
  tray.setImage(trayIcon());
  tray.setToolTip(muted ? 'ORACLE - microphone muted' : 'ORACLE - listening for you');
  tray.setContextMenu(Menu.buildFromTemplate([
    { label: 'Talk to ORACLE', accelerator: TALK_HOTKEYS[0], enabled: !muted, click: () => sendToBackend({ type: 'talk' }) },
    { label: 'Open chat', accelerator: CHAT_HOTKEY, click: () => openChat() },
    { label: 'Mute microphone', type: 'checkbox', checked: muted, accelerator: MUTE_HOTKEY, click: () => sendToBackend({ type: 'mute', value: !muted }) },
    { label: 'Learn my voice…', enabled: !muted, click: () => sendToBackend({ type: 'enroll' }) },
    { type: 'separator' },
    { label: 'Quit ORACLE', click: () => app.quit() },
  ]));
}

ipcMain.handle('backend-info', () => ({ ...backendInfo, hideWhenAsleep: HIDE_WHEN_ASLEEP }));

ipcMain.on('orb-visible', (_e, visible) => {
  if (!orb || orb.isDestroyed()) return;
  if (visible && !orb.isVisible()) {
    orb.showInactive();
    if (!confirmPending) sendOrbToBack();
  } else if (!visible && orb.isVisible()) {
    orb.hide();
  }
});

// Exception to "behind everything": a pending Yes/No comes to the front,
// or it would time out unseen (and count as No).
let confirmPending = false;
ipcMain.on('orb-front', (_e, front) => {
  if (!orb || orb.isDestroyed()) return;
  confirmPending = !!front;
  if (confirmPending) {
    orb.setAlwaysOnTop(true, 'screen-saver');
    if (!orb.isVisible()) orb.showInactive();
  } else {
    sendOrbToBack();
  }
});

// Clickable Yes/No on a click-through window. Relying on forwarded hover
// events is unreliable on Windows (forwarding stops after the window has
// been hidden and shown), so while a confirmation is up we poll the cursor
// and make the window clickable only while it's over the box.
let hitbox = null;      // {x, y, w, h} in window CSS px, or null
let hitTimer = null;
let clickable = false;

function setClickable(on) {
  if (clickable === on || !orb || orb.isDestroyed()) return;
  clickable = on;
  orb.setIgnoreMouseEvents(!on, { forward: true });
}

function pollHitbox() {
  if (!hitbox || !orb || orb.isDestroyed() || !orb.isVisible()) return setClickable(false);
  const c = screen.getCursorScreenPoint();
  const b = orb.getContentBounds();
  const x = c.x - b.x, y = c.y - b.y;
  setClickable(x >= hitbox.x && x <= hitbox.x + hitbox.w && y >= hitbox.y && y <= hitbox.y + hitbox.h);
}

ipcMain.on('orb-hitbox', (_e, rect) => {
  hitbox = rect;
  if (rect && !hitTimer) hitTimer = setInterval(pollHitbox, 30);
  if (!rect) {
    clearInterval(hitTimer);
    hitTimer = null;
    setClickable(false);
  }
});

ipcMain.on('orb-muted', (_e, value) => {
  muted = !!value;
  refreshTray();
});

app.whenReady().then(async () => {
  try {
    await startBackend();
  } catch (e) {
    console.error('Could not start the ORACLE backend:', e.message);
    app.quit();
    return;
  }

  if (app.isPackaged) app.setLoginItemSettings({ openAtLogin: true });

  createOrb();
  if (process.env.ORACLE_OPEN_CHAT) openChat();
  if (process.env.ORACLE_ORB_SELFTEST) {
    // Puts up a fake confirmation (its reply is ignored by the backend) and
    // logs the window's clickable state, for testing the Yes/No hit-testing.
    orb.webContents.once('did-finish-load', () => setTimeout(() => {
      orb.webContents.send('debug-event', {
        type: 'confirm', id: 'selftest', tier: 'confirm',
        action: 'Self-test: click Yes', details: 'This confirmation is a test.',
      });
      orb.webContents.executeJavaScript(`
        for (const t of ['pointerdown', 'mousedown', 'mouseup', 'click'])
          document.addEventListener(t, (e) => console.log('[selftest] dom ' + t + ' on ' + (e.target.id || e.target.tagName)), true);
      `);
      orb.on('focus', () => console.log('[selftest] window focus'));
      setInterval(async () => {
        const yes = await orb.webContents.executeJavaScript(
          "JSON.stringify(document.getElementById('confirm-yes').getBoundingClientRect())");
        console.log('[selftest] ' + JSON.stringify({
          hitbox, clickable, bounds: orb.getContentBounds(), yes: JSON.parse(yes),
          scale: screen.getPrimaryDisplay().scaleFactor,
        }));
      }, 500);
    }, 3000));
  }
  tray = new Tray(trayIcon());
  tray.on('click', () => sendToBackend({ type: 'talk' }));
  refreshTray();

  // If Windows already owns Ctrl+Alt+O for the Start-menu shortcut, this
  // registration fails and the second-instance handler covers it instead.
  for (const key of TALK_HOTKEYS) globalShortcut.register(key, () => sendToBackend({ type: 'talk' }));
  globalShortcut.register(MUTE_HOTKEY, () => sendToBackend({ type: 'mute', value: !muted }));
  if (!globalShortcut.register(CHAT_HOTKEY, () => openChat())) {
    console.error(`${CHAT_HOTKEY} is taken by another app; open the chat from the tray instead.`);
  }
});

// Closing the orb never quits ORACLE; Quit is in the tray.
app.on('window-all-closed', () => {});

app.on('before-quit', () => {
  quitting = true;
  globalShortcut.unregisterAll();
  if (backend && backend.exitCode === null) backend.kill();
});
