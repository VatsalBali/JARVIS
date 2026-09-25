// Orb page: connects to the backend WebSocket, drives the Golden Lens from
// state/level events, shows captions and the Yes/No confirmation, and tells
// the main process when to show/hide the window and when to accept clicks.
//
// Opened outside Electron (no window.oracle bridge), it runs a demo cycle
// through every state instead, for checking the visuals in a browser.

(function () {
  const stage = document.getElementById('stage');
  const labelEl = document.getElementById('label');
  const textEl = document.getElementById('text');
  const captionsEl = document.getElementById('captions');
  const confirmEl = document.getElementById('confirm');
  const confirmAction = document.getElementById('confirm-action');
  const confirmDetails = document.getElementById('confirm-details');
  const confirmHint = document.getElementById('confirm-hint');
  const yesBtn = document.getElementById('confirm-yes');
  const noBtn = document.getElementById('confirm-no');

  const lens = new GoldenLens(document.getElementById('lens'), { centerY: 210, scale: 0.38 });
  const bridge = window.oracle || null;

  const LABELS = {
    asleep: 'ASLEEP', wake: 'WAKE', listening: 'LISTENING',
    thinking: 'THINKING', speaking: 'SPEAKING', followup: 'FOLLOW-UP',
  };
  const DEFAULT_TEXT = {
    asleep: 'Ctrl+Alt+Space to talk', // replaced by the wake phrase on hello
    wake: 'Yes?',
    listening: '',
    thinking: '',
    speaking: '',
    followup: 'Go on…',
  };
  const MAX_CAPTION_CHARS = 140;

  let ws = null;
  let state = null; // set by the first setState
  let hideWhenAsleep = true;
  let pendingConfirm = null;
  let hideTimer = null;

  function setText(text) {
    text = (text || '').trim();
    if (text.length > MAX_CAPTION_CHARS) text = '…' + text.slice(-MAX_CAPTION_CHARS);
    textEl.textContent = text;
  }

  function show() {
    clearTimeout(hideTimer);
    if (bridge) bridge.setVisible(true);
    lens.lowFps = false;
    lens.start();
    requestAnimationFrame(() => stage.classList.add('visible'));
  }

  function hideSoon() {
    if (pendingConfirm) return;
    if (!hideWhenAsleep) {
      lens.lowFps = true; // faint ember at ~15 fps
      return;
    }
    stage.classList.remove('visible');
    clearTimeout(hideTimer);
    hideTimer = setTimeout(() => {
      lens.stop();
      if (bridge) bridge.setVisible(false);
    }, 400);
  }

  function setState(next, label) {
    if (!LABELS[next]) return;
    const prev = state;
    state = next;
    lens.setState(next);
    labelEl.textContent = label || LABELS[next];
    if (next === 'wake' || (next !== prev && DEFAULT_TEXT[next])) setText(DEFAULT_TEXT[next]);
    if (next === 'asleep') hideSoon();
    else show();
  }

  // ---- confirmation ----

  function showConfirm(msg) {
    pendingConfirm = msg.id;
    confirmEl.classList.toggle('warn', msg.tier === 'warn');
    confirmAction.textContent = msg.action;
    confirmDetails.textContent = msg.details || '';
    confirmHint.textContent = msg.spoken ? 'Say “yes” or “no”, or click.' : '';
    confirmEl.hidden = false;
    captionsEl.classList.add('confirming');
    show();
  }

  function closeConfirm(id) {
    if (id && id !== pendingConfirm) return;
    pendingConfirm = null;
    confirmEl.hidden = true;
    captionsEl.classList.remove('confirming');
    if (bridge) bridge.setInteractive(false);
    if (state === 'asleep') hideSoon();
  }

  function answer(ok) {
    if (!pendingConfirm) return;
    send({ type: 'confirm_reply', id: pendingConfirm, ok });
    closeConfirm();
  }

  yesBtn.addEventListener('click', () => answer(true));
  noBtn.addEventListener('click', () => answer(false));
  // The window is click-through; accept clicks only while over the box.
  confirmEl.addEventListener('mouseenter', () => bridge && bridge.setInteractive(true));
  confirmEl.addEventListener('mouseleave', () => bridge && bridge.setInteractive(false));

  // ---- backend socket ----

  function send(msg) {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
  }

  function handle(msg) {
    switch (msg.type) {
      case 'hello':
        console.log('backend says hello, state=' + msg.state + ', wake phrase=' + msg.wake_phrase);
        if (msg.wake_phrase) DEFAULT_TEXT.asleep = `Say “${msg.wake_phrase}”`;
        setState(msg.state || 'asleep');
        bridge.reportMuted(!!msg.muted);
        break;
      case 'state':
        setState(msg.state, msg.label);
        break;
      case 'level':
        lens.setLevel(msg.value);
        break;
      case 'transcript':
        if (msg.text) setText('“' + msg.text + '”');
        break;
      case 'caption':
        setText(msg.text);
        break;
      case 'tool':
        setText(msg.label);
        break;
      case 'confirm':
        showConfirm(msg);
        break;
      case 'confirm_closed':
        closeConfirm(msg.id);
        break;
      case 'muted':
        bridge.reportMuted(!!msg.value);
        if (state === 'asleep') setText(msg.value ? 'Microphone muted' : DEFAULT_TEXT.asleep);
        break;
    }
  }

  async function connect() {
    const info = await bridge.backendInfo();
    hideWhenAsleep = info.hideWhenAsleep !== false;
    let delay = 500;
    const open = () => {
      ws = new WebSocket(`ws://127.0.0.1:${info.port}/?token=${encodeURIComponent(info.token)}`);
      ws.onopen = () => {
        delay = 500;
        console.log('connected to backend');
      };
      ws.onmessage = (e) => {
        try { handle(JSON.parse(e.data)); } catch (err) { console.error(err); }
      };
      ws.onclose = () => {
        setTimeout(open, delay);
        delay = Math.min(delay * 2, 8000);
      };
    };
    open();
    bridge.onCommand(send);
  }

  // ---- demo (browser preview only) ----

  function demo() {
    hideWhenAsleep = false;
    const steps = [
      ['asleep', null, 1500],
      ['wake', null, 1200],
      ['listening', '“Oracle, what’s on my calendar today?”', 2500],
      ['thinking', 'Checking your calendar…', 2000],
      ['speaking', 'Two meetings today, Sir. The first is at 10:30.', 3000],
      ['followup', null, 1800],
    ];
    let i = 0;
    const next = () => {
      const [s, text, ms] = steps[i++ % steps.length];
      setState(s);
      if (text) setText(text);
      setTimeout(next, ms);
    };
    // Fake audio level while listening/speaking.
    setInterval(() => {
      if (state === 'listening' || state === 'speaking') lens.setLevel(0.25 + Math.random() * 0.6);
    }, 100);
    const params = new URLSearchParams(location.search);
    if (params.get('state')) {
      setState(params.get('state'));
      if (params.get('confirm')) {
        showConfirm({ id: 'demo', tier: params.get('confirm'), action: 'Send this email from Gmail?',
          details: 'To: prof.novak@uni.cz\nSubject: Report\n\nHello Professor, the report is attached.' });
      }
    } else {
      next();
    }
    show();
  }

  window.addEventListener('resize', () => lens.resize());

  if (bridge) connect();
  else demo();
})();
