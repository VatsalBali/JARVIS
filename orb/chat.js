// ORACLE chat window: a single centred conversation under a live Golden Lens.
//
// Talks to the backend through window.pywebview.api (orb/chat_preload.js:
// ChatSession calls over the WebSocket) and receives live events through
// window.oracleChat.onEvent. Opened outside Electron (no bridge) it runs on
// sample data, for checking the design in a browser.

(function () {
  const $ = (id) => document.getElementById(id);
  const log = $('log');
  const main = $('main');
  const empty = $('empty');
  const input = $('input');
  const status = $('status');
  const chatList = $('chatList');
  const projectList = $('projectList');
  const menu = $('menu');

  const api = (window.pywebview && window.pywebview.api) || demoApi();
  const events = window.oracleChat || null;

  // ---- live lens ----
  const lens = new GoldenLens($('lens'), { centerY: 58, scale: 0.19, bloomPx: 2 });
  lens.setState('asleep');
  lens.start();
  window.addEventListener('resize', () => lens.resize());
  $('orbSlot').addEventListener('click', () => api.start_voice_turn());

  const STATUS = { asleep: 'ONLINE', wake: 'LISTENING', listening: 'LISTENING', thinking: 'THINKING', speaking: 'SPEAKING', followup: 'LISTENING' };
  function setState(state, label) {
    lens.setState(state === 'followup' ? 'listening' : state);
    lens.lowFps = state === 'asleep';
    status.textContent = label || STATUS[state] || 'ONLINE';
  }

  // ---- greeting ----
  const h = new Date().getHours();
  $('greeting').textContent = (h < 5 ? 'Good night' : h < 12 ? 'Good morning' : h < 18 ? 'Good afternoon' : 'Good evening') + ', Sir.';

  // ---- rendering ----

  const pad = (n) => String(n).padStart(2, '0');
  const clock = (d = new Date()) => `${pad(d.getHours())}:${pad(d.getMinutes())}`;

  function scrollDown() { main.scrollTop = main.scrollHeight; }

  function showConversation() { empty.hidden = true; }

  // Replies are model text, possibly echoing message/email content, so this
  // escapes everything first and only then adds a few safe tags.
  function renderMarkdown(src) {
    const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    const inline = (s) => esc(s)
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/(^|[^*])\*([^*\n]+)\*/g, '$1<em>$2</em>');
    const out = [];
    const blocks = String(src || '').split(/```/);
    blocks.forEach((block, i) => {
      if (i % 2 === 1) {
        out.push('<pre><code>' + esc(block.replace(/^\w*\n/, '')) + '</code></pre>');
        return;
      }
      let list = null;
      for (const raw of block.split('\n')) {
        const line = raw.trimEnd();
        const m = line.match(/^\s*(?:[-*•]|(\d+)[.)])\s+(.*)$/);
        if (m) {
          const kind = m[1] ? 'ol' : 'ul';
          if (!list || list.kind !== kind) { if (list) out.push(`</${list.kind}>`); list = { kind }; out.push(`<${kind}>`); }
          out.push('<li>' + inline(m[2]) + '</li>');
          continue;
        }
        if (list) { out.push(`</${list.kind}>`); list = null; }
        if (line.trim()) out.push('<p>' + inline(line.replace(/^#+\s*/, '')) + '</p>');
      }
      if (list) out.push(`</${list.kind}>`);
    });
    return out.join('');
  }

  function addTurn(role, text, opts = {}) {
    showConversation();
    const turn = document.createElement('div');
    turn.className = 'turn ' + role;
    const who = document.createElement('div');
    who.className = 'who';
    const name = document.createElement('span');
    name.textContent = role === 'user' ? 'YOU' : 'ORACLE';
    const time = document.createElement('time');
    time.textContent = opts.time || clock();
    who.append(name, time);
    if (opts.via) {
      const via = document.createElement('span');
      via.className = 'via';
      via.textContent = opts.via;
      who.append(via);
    }
    const body = document.createElement('div');
    body.className = 'body';
    if (role === 'oracle') body.innerHTML = renderMarkdown(text);
    else body.textContent = text;
    turn.append(who, body);
    log.appendChild(turn);
    scrollDown();
    return turn;
  }

  let thinkingEl = null;
  function showThinking(label) {
    if (!thinkingEl) {
      thinkingEl = addTurn('oracle', '');
      thinkingEl.classList.add('thinking');
    }
    const body = thinkingEl.querySelector('.body');
    body.textContent = label || 'Thinking';
    const dots = document.createElement('span');
    dots.className = 'dots';
    body.appendChild(dots);
    log.appendChild(thinkingEl);   // keep it last, under any approval card
    scrollDown();
  }
  function hideThinking() {
    if (thinkingEl) thinkingEl.remove();
    thinkingEl = null;
  }

  // ---- approval cards (details are tool arguments: textContent only) ----

  function showConfirm(ev) {
    showConversation();
    const card = document.createElement('div');
    card.className = 'confirm' + (ev.tier === 'warn' ? ' warn' : '');
    card.dataset.id = ev.id;
    const title = document.createElement('div');
    title.className = 'confirm-title';
    title.textContent = ev.action;
    const details = document.createElement('pre');
    details.className = 'confirm-details';
    details.textContent = ev.details || '';
    const actions = document.createElement('div');
    actions.className = 'confirm-actions';
    const yes = document.createElement('button');
    yes.className = 'yes';
    yes.type = 'button';
    yes.textContent = 'YES';
    const no = document.createElement('button');
    no.className = 'no';
    no.type = 'button';
    no.textContent = 'NO';
    yes.addEventListener('click', () => { settleConfirm(ev.id, true, 'here'); api.confirm_reply(ev.id, true); });
    no.addEventListener('click', () => { settleConfirm(ev.id, false, 'here'); api.confirm_reply(ev.id, false); });
    actions.append(yes, no);
    if (ev.spoken) {
      const hint = document.createElement('span');
      hint.className = 'confirm-status';
      hint.textContent = 'or say “yes” / “no”';
      actions.append(hint);
    }
    card.append(title, details, actions);
    log.appendChild(card);
    if (thinkingEl) log.appendChild(thinkingEl);
    scrollDown();
  }

  function settleConfirm(id, ok, reason) {
    const card = log.querySelector(`.confirm[data-id="${CSS.escape(id)}"]`);
    if (!card) return;
    const actions = card.querySelector('.confirm-actions');
    if (!actions.querySelector('button')) return;
    const s = document.createElement('span');
    s.className = 'confirm-status';
    s.textContent = reason === 'timeout' ? 'No answer, so nothing was done.'
      : reason === 'cancel' ? 'Cancelled.'
      : (ok ? 'Approved' : 'Declined') + (reason === 'spoken' ? ' by voice.' : '.');
    actions.replaceChildren(s);
  }

  // ---- sending ----

  let busy = false;
  async function send(text) {
    text = (text || '').trim();
    if (!text || busy) return;
    busy = true;
    addTurn('user', text);
    showThinking('Thinking');
    try {
      const reply = await api.send_message(text);
      hideThinking();
      addTurn('oracle', reply || '(no reply)');
    } catch (err) {
      hideThinking();
      addTurn('oracle', 'Something went wrong: ' + (err && err.message ? err.message : err));
    } finally {
      busy = false;
    }
    refreshChats(true);
  }

  function autosize() {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 180) + 'px';
  }
  input.addEventListener('input', autosize);
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      const text = input.value;
      input.value = '';
      autosize();
      send(text);
    }
  });
  $('sendBtn').addEventListener('click', () => {
    const text = input.value;
    input.value = '';
    autosize();
    send(text);
    input.focus();
  });
  document.querySelectorAll('#suggestions button').forEach((b) => b.addEventListener('click', () => send(b.dataset.msg)));

  // Dictation: records with the local Whisper model, then fills the box.
  let recording = false;
  $('micBtn').addEventListener('click', async () => {
    const btn = $('micBtn');
    if (!recording) {
      recording = true;
      btn.classList.add('recording');
      try { await api.start_recording(); } catch (err) { recording = false; btn.classList.remove('recording'); }
      return;
    }
    recording = false;
    btn.classList.remove('recording');
    try {
      const text = await api.stop_recording();
      if (text && !text.startsWith('Error:')) {
        input.value = (input.value ? input.value + ' ' : '') + text;
        autosize();
        input.focus();
      }
    } catch (err) { console.error(err); }
  });

  // ---- drawer: chats, mode, projects ----

  let conversations = [];
  let activeId = null;
  let activeProject = null;

  function toggleDrawer(open) {
    document.body.classList.toggle('drawer-open', open ?? !document.body.classList.contains('drawer-open'));
    if (document.body.classList.contains('drawer-open')) { refreshChats(); refreshProjects(); }
  }
  $('menuBtn').addEventListener('click', () => toggleDrawer());
  $('scrim').addEventListener('click', () => toggleDrawer(false));
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') { closeMenu(); toggleDrawer(false); }
  });

  function resetView() {
    log.replaceChildren();
    thinkingEl = null;
    empty.hidden = false;
  }

  $('newChatBtn').addEventListener('click', async () => {
    await api.new_chat();
    activeId = null;
    activeProject = null;
    resetView();
    toggleDrawer(false);
    input.focus();
  });

  $('agentSelect').addEventListener('change', async (e) => {
    await api.switch_agent(e.target.value);
    activeId = null;
    activeProject = null;
    resetView();
    refreshChats();
  });

  function renderChats(list) {
    chatList.replaceChildren();
    if (!list.length) {
      const e = document.createElement('div');
      e.className = 'list-empty';
      e.textContent = 'No conversations yet.';
      chatList.append(e);
      return;
    }
    for (const c of list) {
      const item = document.createElement('div');
      item.className = 'item' + (c.id === activeId ? ' active' : '');
      item.title = c.title;
      if (c.pinned) {
        const pin = document.createElement('span');
        pin.className = 'pin';
        pin.textContent = '◆';
        item.append(pin);
      }
      const title = document.createElement('span');
      title.className = 'title';
      title.textContent = c.title;
      const more = document.createElement('button');
      more.className = 'more';
      more.type = 'button';
      more.textContent = '⋯';
      more.addEventListener('click', (e) => {
        e.stopPropagation();
        openMenu(more, [
          [c.pinned ? 'Unpin' : 'Pin', async () => { await api.toggle_pin_conversation(c.id); refreshChats(); }],
          ['Rename', () => renameInline(item, title, c)],
          ['Delete', async () => {
            await api.delete_conversation(c.id);
            if (c.id === activeId) { activeId = null; resetView(); }
            refreshChats();
          }, true],
        ]);
      });
      item.append(title, more);
      item.addEventListener('click', () => openChat(c.id));
      chatList.append(item);
    }
  }

  function renameInline(item, titleEl, c) {
    const field = document.createElement('input');
    field.value = c.title;
    field.className = 'title';
    field.style.cssText = 'background:var(--bg-deep);border:1px solid var(--line-strong);border-radius:6px;color:var(--text);padding:2px 6px;font:inherit;outline:0';
    titleEl.replaceWith(field);
    field.focus();
    field.select();
    let done = false;
    const finish = async (save) => {
      if (done) return;
      done = true;
      const name = field.value.trim();
      if (save && name && name !== c.title) await api.rename_conversation(c.id, name);
      refreshChats();
    };
    field.addEventListener('click', (e) => e.stopPropagation());
    field.addEventListener('keydown', (e) => {
      e.stopPropagation();
      if (e.key === 'Enter') finish(true);
      if (e.key === 'Escape') finish(false);
    });
    field.addEventListener('blur', () => finish(true));
  }

  async function openChat(id) {
    const messages = await api.load_chat(id);
    activeId = id;
    activeProject = null;
    resetView();
    for (const m of messages) addTurn(m.role === 'user' ? 'user' : 'oracle', m.content);
    if (!messages.length) empty.hidden = false;
    renderChats(conversations);
    toggleDrawer(false);
  }

  async function refreshChats(selectNewest) {
    try {
      conversations = await api.list_conversations();
      if (selectNewest && conversations.length) activeId = conversations[0].id;
      filterChats();
    } catch (err) { console.error(err); }
  }
  function filterChats() {
    const q = $('searchChats').value.trim().toLowerCase();
    renderChats(q ? conversations.filter((c) => c.title.toLowerCase().includes(q)) : conversations);
  }
  $('searchChats').addEventListener('input', filterChats);

  async function refreshProjects() {
    projectList.replaceChildren();
    let projects = [];
    try {
      if (await api.get_projects_root()) projects = await api.list_projects();
    } catch (err) { console.error(err); }
    if (!projects.length) {
      const e = document.createElement('div');
      e.className = 'list-empty';
      e.textContent = 'No projects folder chosen.';
      projectList.append(e);
      return;
    }
    for (const p of projects) {
      const item = document.createElement('div');
      item.className = 'item' + (p.path === activeProject ? ' active' : '');
      item.title = p.path;
      const title = document.createElement('span');
      title.className = 'title';
      title.textContent = p.name;
      item.append(title);
      item.addEventListener('click', async () => {
        const messages = await api.open_project(p.path);
        activeProject = p.path;
        activeId = null;
        resetView();
        addTurn('oracle', `Working in **${p.name}**. I can read and change its files, with your approval for changes.`);
        for (const m of messages) addTurn(m.role === 'user' ? 'user' : 'oracle', m.content);
        toggleDrawer(false);
      });
      projectList.append(item);
    }
  }
  $('projectsRootBtn').addEventListener('click', async () => {
    if (await api.choose_projects_root()) refreshProjects();
  });

  function openMenu(anchor, options) {
    menu.replaceChildren();
    for (const [label, fn, danger] of options) {
      const b = document.createElement('button');
      b.type = 'button';
      b.textContent = label;
      if (danger) b.className = 'danger';
      b.addEventListener('click', (e) => {
        e.stopPropagation();
        if (danger && b.dataset.armed !== '1') {
          b.dataset.armed = '1';
          b.textContent = 'Click again to delete';
          return;
        }
        closeMenu();
        fn();
      });
      menu.append(b);
    }
    const r = anchor.getBoundingClientRect();
    menu.style.left = Math.min(r.left, window.innerWidth - 170) + 'px';
    menu.style.top = (r.bottom + 4) + 'px';
    menu.hidden = false;
  }
  function closeMenu() { menu.hidden = true; }
  document.addEventListener('click', (e) => { if (!menu.contains(e.target)) closeMenu(); });

  // ---- window controls ----
  $('minBtn').addEventListener('click', () => api.minimize_window());
  $('maxBtn').addEventListener('click', () => api.toggle_maximize_window());
  $('closeBtn').addEventListener('click', () => api.hide_window());

  // ---- live events from the backend ----
  if (events) {
    events.onEvent((ev) => {
      switch (ev.type) {
        case 'state': setState(ev.state, ev.label); break;
        case 'level': lens.setLevel(ev.value); break;
        case 'tool': if (busy) showThinking(ev.label); break;
        case 'confirm': showConfirm(ev); break;
        case 'confirm_closed': settleConfirm(ev.id, ev.ok, ev.reason); break;
        case 'turn':
          // A finished voice exchange belongs to this conversation too.
          if (ev.user) addTurn('user', ev.user, { via: 'VOICE' });
          if (ev.reply) addTurn('oracle', ev.reply, { via: 'VOICE' });
          refreshChats(true);
          break;
      }
    });
  }

  refreshChats();
  input.focus();

  // ---- sample data for previewing outside Electron ----
  function demoApi() {
    const convs = [
      { id: 3, title: 'WhatsApp Mum about dinner', pinned: true },
      { id: 2, title: 'Weather in Prague this week' },
      { id: 1, title: 'Close Chrome and open VS Code' },
    ];
    const wait = (ms, v) => new Promise((r) => setTimeout(() => r(v), ms));
    setTimeout(() => {
      addTurn('user', 'WhatsApp Mum that I\'ll be home by 8');
      showConfirm({ id: 'demo', action: 'Send this WhatsApp message?', details: "To: Mum (+420601234567)\n\nI'll be home by 8.", tier: 'confirm', spoken: true });
      addTurn('oracle', 'Sent to Mum on WhatsApp. Anything else, Sir?\n\n- Reminder set for **7:30**\n- Weather: `14°C`, light rain');
    }, 300);
    return {
      send_message: (t) => wait(900, `You said: **${t}**`),
      list_conversations: () => wait(50, convs),
      load_chat: () => wait(50, [{ role: 'user', content: 'Hello' }, { role: 'jarvis', content: 'Good evening, Sir.' }]),
      new_chat: () => wait(10), switch_agent: () => wait(10),
      rename_conversation: () => wait(10), delete_conversation: () => wait(10), toggle_pin_conversation: () => wait(10),
      get_projects_root: () => wait(10, 'D:/code'), list_projects: () => wait(10, [{ name: 'zombie', path: 'D:/code/zombie' }]),
      open_project: () => wait(10, []), choose_projects_root: () => wait(10, null),
      start_recording: () => wait(10), stop_recording: () => wait(10, ''),
      confirm_reply: () => {}, start_voice_turn: () => setState('listening'),
      minimize_window: () => {}, toggle_maximize_window: () => {}, hide_window: () => {},
    };
  }
})();
