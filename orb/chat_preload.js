// Bridge for the chat window (chat.html).
//
// The page calls window.pywebview.api.<method>(...), the same API the old
// pywebview window had. Backend calls go over the WebSocket the orb uses
// (oracle_server.py "call" messages); window-only ones (minimise, close,
// folder picker) go to the main process. Backend events (confirmations,
// voice turns, state, audio level) reach the page via oracleChat.onEvent.
const { contextBridge, ipcRenderer } = require('electron');

let ws = null;
let nextId = 1;
const pending = new Map();      // call id -> {resolve, reject}
const queue = [];               // messages sent before the socket opened
const listeners = [];

function send(msg) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
  else queue.push(msg);
}

function call(method, args = {}) {
  return new Promise((resolve, reject) => {
    const id = String(nextId++);
    pending.set(id, { resolve, reject });
    send({ type: 'call', id, method, args });
  });
}

async function connect() {
  const info = await ipcRenderer.invoke('backend-info');
  let delay = 500;
  const open = () => {
    ws = new WebSocket(`ws://127.0.0.1:${info.port}/?token=${encodeURIComponent(info.token)}`);
    ws.onopen = () => {
      delay = 500;
      while (queue.length) ws.send(JSON.stringify(queue.shift()));
    };
    ws.onmessage = (e) => {
      let msg;
      try { msg = JSON.parse(e.data); } catch { return; }
      if (msg.type === 'result') {
        const p = pending.get(msg.id);
        if (!p) return;
        pending.delete(msg.id);
        if (msg.ok) p.resolve(msg.value);
        else p.reject(new Error(msg.error || 'failed'));
        return;
      }
      for (const cb of listeners) {
        try { cb(msg); } catch (err) { console.error(err); }
      }
    };
    ws.onclose = () => {
      // Calls in flight are lost with the connection; fail them rather than hang.
      for (const [, p] of pending) p.reject(new Error('ORACLE disconnected'));
      pending.clear();
      setTimeout(open, delay);
      delay = Math.min(delay * 2, 8000);
    };
  };
  open();
}
connect();

contextBridge.exposeInMainWorld('pywebview', {
  api: {
    // Backend (ChatSession) methods: positional arguments, as the page passes them.
    send_message: (text) => call('send_message', { text }),
    switch_agent: (agent) => call('switch_agent', { agent }),
    new_chat: () => call('new_chat'),
    list_conversations: () => call('list_conversations'),
    load_chat: (conversation_id) => call('load_chat', { conversation_id }),
    rename_conversation: (conversation_id, new_title) => call('rename_conversation', { conversation_id, new_title }),
    delete_conversation: (conversation_id) => call('delete_conversation', { conversation_id }),
    toggle_pin_conversation: (conversation_id) => call('toggle_pin_conversation', { conversation_id }),
    get_projects_root: () => call('get_projects_root'),
    list_projects: () => call('list_projects'),
    open_project: (path) => call('open_project', { path }),
    rename_project: (project_id, new_name) => call('rename_project', { project_id, new_name }),
    delete_project: (project_id) => call('delete_project', { project_id }),
    toggle_pin_project: (project_id) => call('toggle_pin_project', { project_id }),
    get_system_stats: () => call('get_system_stats'),
    start_recording: () => call('start_recording'),
    stop_recording: () => call('stop_recording'),
    // Socket messages rather than calls.
    confirm_reply: (id, ok) => send({ type: 'confirm_reply', id, ok: !!ok }),
    start_voice_turn: () => send({ type: 'talk' }),
    // Window and OS dialogs belong to the main process.
    minimize_window: () => ipcRenderer.send('chat-window', 'minimize'),
    toggle_maximize_window: () => ipcRenderer.send('chat-window', 'maximize'),
    hide_window: () => ipcRenderer.send('chat-window', 'hide'),
    choose_projects_root: async () => {
      const folder = await ipcRenderer.invoke('choose-folder');
      if (!folder) return null;
      return call('set_projects_root', { path: folder });
    },
  },
});

contextBridge.exposeInMainWorld('oracleChat', {
  onEvent: (callback) => listeners.push(callback),
});
