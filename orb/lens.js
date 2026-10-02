// Golden Lens renderer (README 4.2) on Canvas 2D.
//
// Geometry, dash patterns and state values come from
// design/golden_lens_reference.html (the Component class). Everything is
// drawn in the mockup's own units (rings up to r=560) and scaled to fit.
// Static layers - spokes, traces, pads, fixed rings, sparks - are rendered
// once to offscreen canvases; each frame only draws the rotating rings, the
// core, and composites the cached layers, then applies the bloom.

(function () {
  const GOLD = '#ffb838';
  const AMBER = '#ff8a1e';
  const HOT = '#fff3d1';
  const VOID = '#130a03';
  const DEG = Math.PI / 180;
  const TAU = Math.PI * 2;

  // State table from the mockup (durations in seconds).
  const STATES = {
    // Brighter than the mockup's ember: the orb now stays on screen while asleep.
    asleep:    { ringOp: 0.4,  sparkOp: 0,   circuitOp: 0.15, glowOp: 0.35, coreScale: 0.7, pulse: 1.02, pulseDur: 4,    durA: 90, durB: 70, durC: 60, follows: false },
    wake:      { ringOp: 1,    sparkOp: 1,   circuitOp: 0.9,  glowOp: 1,    coreScale: 1.1,  pulse: 1.08, pulseDur: 1.2,  durA: 40, durB: 30, durC: 20, follows: false },
    listening: { ringOp: 0.85, sparkOp: 0.7, circuitOp: 0.6,  glowOp: 0.8,  coreScale: 1,    pulse: 1.12, pulseDur: 0.7,  durA: 60, durB: 45, durC: 30, follows: true },
    thinking:  { ringOp: 0.75, sparkOp: 0.5, circuitOp: 1,    glowOp: 0.6,  coreScale: 0.9,  pulse: 1.03, pulseDur: 1.6,  durA: 8,  durB: 6,  durC: 3,  follows: false },
    speaking:  { ringOp: 1,    sparkOp: 1,   circuitOp: 0.8,  glowOp: 1,    coreScale: 1.05, pulse: 1.15, pulseDur: 0.45, durA: 50, durB: 38, durC: 24, follows: true },
    // Not in the mockup: the orb dims but stays while waiting for a follow-up.
    followup:  { ringOp: 0.5,  sparkOp: 0.3, circuitOp: 0.3,  glowOp: 0.45, coreScale: 0.8,  pulse: 1.04, pulseDur: 2,    durA: 70, durB: 55, durC: 40, follows: false },
  };
  // Tweened numerically; speeds are tweened as 1/dur so spin changes smoothly.
  const TWEEN_KEYS = ['ringOp', 'sparkOp', 'circuitOp', 'glowOp', 'coreScale', 'pulse', 'pulseDur', 'spinA', 'spinB', 'spinC'];

  function rnd(i) {
    const x = Math.sin(i * 12.9898 + 78.233) * 43758.5453;
    return x - Math.floor(x);
  }

  function targetFor(state) {
    const s = STATES[state] || STATES.asleep;
    return { ...s, spinA: 360 / s.durA, spinB: 360 / s.durB, spinC: 360 / s.durC };
  }

  class GoldenLens {
    constructor(canvas, opts = {}) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.centerY = opts.centerY ?? 220; // CSS px from the top
      this.scale = opts.scale ?? 0.38;    // mockup units -> CSS px
      this.bloomPx = opts.bloomPx ?? 3;
      this.state = 'asleep';
      this.cur = targetFor('asleep');
      this.target = targetFor('asleep');
      this.angles = { a: 0, b: 0, c: 0 };
      this.level = 0;        // smoothed 0-1 audio level
      this.levelTarget = 0;
      this.time = 0;
      this.running = false;
      this.lowFps = false;   // ~15 fps when asleep (README 4.2)
      this.lastFrame = 0;
      this._frame = this._frame.bind(this);
      this.resize();
    }

    setState(state) {
      if (!STATES[state]) return;
      this.state = state;
      this.target = targetFor(state);
    }

    setLevel(value) {
      this.levelTarget = Math.max(this.levelTarget, Math.min(1, Math.max(0, value)));
    }

    start() {
      if (this.running) return;
      this.running = true;
      this.lastFrame = performance.now();
      requestAnimationFrame(this._frame);
    }

    stop() {
      this.running = false;
    }

    // ---- setup ----

    resize() {
      const dpr = window.devicePixelRatio || 1;
      const w = this.canvas.clientWidth || this.canvas.width;
      const h = this.canvas.clientHeight || this.canvas.height;
      this.cssW = w;
      this.cssH = h;
      this.dpr = dpr;
      this.canvas.width = Math.round(w * dpr);
      this.canvas.height = Math.round(h * dpr);
      this.scene = this._offscreen();
      this._buildLayers();
      this._buildEdgeMask();
    }

    _offscreen() {
      const c = document.createElement('canvas');
      c.width = this.canvas.width;
      c.height = this.canvas.height;
      return c;
    }

    // Maps mockup units onto the canvas: origin at the lens centre.
    _baseTransform(ctx) {
      const k = this.dpr * this.scale;
      ctx.setTransform(k, 0, 0, k, (this.cssW / 2) * this.dpr, this.centerY * this.dpr);
    }

    // Stroke widths scale with the lens (as in the SVG), but never below
    // ~0.7 device px so hairlines stay visible at this size.
    _lw(w) {
      return Math.max(w, 0.7 / (this.scale * this.dpr));
    }

    _ringStack(ctx, fn) {
      ctx.save();
      ctx.rotate(-7 * DEG);
      ctx.scale(1, 0.46);
      fn();
      ctx.restore();
    }

    _circle(ctx, r, color, alpha, width, dash, angle = 0, cap = 'butt') {
      ctx.save();
      if (angle) ctx.rotate(angle);
      ctx.beginPath();
      ctx.arc(0, 0, r, 0, TAU);
      ctx.setLineDash(dash || []);
      ctx.lineCap = cap;
      ctx.strokeStyle = color;
      ctx.globalAlpha *= alpha;
      ctx.lineWidth = this._lw(width);
      ctx.stroke();
      ctx.restore();
    }

    _buildLayers() {
      // Circuits: spokes, vertical line, traces with square pads.
      this.circuits = this._offscreen();
      let ctx = this.circuits.getContext('2d');
      this._baseTransform(ctx);
      ctx.strokeStyle = GOLD;
      ctx.lineWidth = this._lw(1);
      ctx.globalAlpha = 0.35;
      ctx.beginPath();
      for (let i = 0; i < 26; i++) {
        const a = (i / 26) * TAU + rnd(i + 50) * 0.2;
        const r0 = 70 + rnd(i + 70) * 60;
        const r1 = 380 + rnd(i + 90) * 380;
        ctx.moveTo(r0 * Math.cos(a), r0 * Math.sin(a) * 0.62);
        ctx.lineTo(r1 * Math.cos(a), r1 * Math.sin(a) * 0.62);
      }
      ctx.stroke();

      ctx.globalAlpha = 0.5;
      ctx.lineWidth = this._lw(1.4);
      ctx.beginPath();
      ctx.moveTo(0, -720);
      ctx.lineTo(0, 720);
      ctx.stroke();

      ctx.globalAlpha = 0.8;
      ctx.lineWidth = this._lw(1.3);
      ctx.lineJoin = 'round';
      ctx.beginPath();
      const pads = [];
      for (let i = 0; i < 46; i++) {
        const a = rnd(i + 200) * TAU;
        const r = 240 + rnd(i + 230) * 330;
        const x = r * Math.cos(a);
        const y = r * Math.sin(a) * 0.5;
        const dir = Math.cos(a) > 0 ? 1 : -1;
        const h1 = 14 + rnd(i + 260) * 50;
        const dg = 8 + rnd(i + 290) * 26;
        const h2 = 10 + rnd(i + 320) * 40;
        const up = rnd(i + 350) > 0.5 ? 1 : -1;
        const x1 = x + dir * h1, x2 = x1 + dir * dg, y2 = y + up * dg, x3 = x2 + dir * h2;
        ctx.moveTo(x, y);
        ctx.lineTo(x1, y);
        ctx.lineTo(x2, y2);
        ctx.lineTo(x3, y2);
        pads.push([x3, y2]);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.fillStyle = GOLD;
      for (const [px, py] of pads) ctx.fillRect(px - 2, py - 2, 4, 4);

      // Rings that don't rotate.
      this.staticRings = this._offscreen();
      ctx = this.staticRings.getContext('2d');
      this._baseTransform(ctx);
      this._ringStack(ctx, () => {
        this._circle(ctx, 560, AMBER, 0.35, 1);
        this._circle(ctx, 360, GOLD, 0.4, 1.2);
        this._circle(ctx, 220, GOLD, 0.5, 1);
        this._circle(ctx, 160, AMBER, 0.6, 6, [3, 5]);
      });
      ctx.save();
      ctx.rotate(-58 * DEG);
      ctx.scale(1, 0.22);
      this._circle(ctx, 380, GOLD, 0.3, 1);
      ctx.restore();

      // Three spark groups that twinkle out of phase.
      this.sparks = [
        this._sparkLayer(0, 70, 40, HOT),
        this._sparkLayer(70, 80, 70, GOLD),
        this._sparkLayer(150, 90, 120, AMBER),
      ];
    }

    _sparkLayer(start, n, spread, color) {
      const c = this._offscreen();
      const ctx = c.getContext('2d');
      this._baseTransform(ctx);
      ctx.fillStyle = color;
      const tilt = -7 * DEG;
      const rings = [160, 220, 290, 360, 420, 500];
      ctx.beginPath();
      for (let i = start; i < start + n; i++) {
        const a = rnd(i) * TAU;
        const ring = rings[Math.floor(rnd(i + 900) * 6)];
        const rr = ring + (rnd(i + 300) - 0.5) * spread;
        const x = rr * Math.cos(a);
        const y = rr * Math.sin(a) * 0.46;
        const xr = x * Math.cos(tilt) - y * Math.sin(tilt);
        const yr = x * Math.sin(tilt) + y * Math.cos(tilt);
        const r = 0.8 + rnd(i + 600) * 2.6;
        ctx.moveTo(xr + r, yr);
        ctx.arc(xr, yr, r, 0, TAU);
      }
      ctx.fill();
      return c;
    }

    // Fades the lens out towards the window edges so spokes and the
    // vertical line don't end in hard cuts over the desktop.
    _buildEdgeMask() {
      this.mask = this._offscreen();
      const ctx = this.mask.getContext('2d');
      const cx = (this.cssW / 2) * this.dpr;
      const cy = this.centerY * this.dpr;
      const r = Math.min(this.cssW / 2, this.centerY) * this.dpr;
      const g = ctx.createRadialGradient(cx, cy, r * 0.55, cx, cy, r * 1.02);
      g.addColorStop(0, 'rgba(0,0,0,1)');
      g.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, this.mask.width, this.mask.height);
    }

    // ---- frame ----

    _frame(now) {
      if (!this.running) return;
      requestAnimationFrame(this._frame);
      const minGap = this.lowFps ? 1000 / 15 : 0;
      if (now - this.lastFrame < minGap) return;
      const dt = Math.min(0.1, (now - this.lastFrame) / 1000);
      this.lastFrame = now;
      this._update(dt);
      this._draw();
    }

    _update(dt) {
      this.time += dt;
      // Exponential approach: settles in ~300-400 ms.
      const k = 1 - Math.exp(-dt / 0.11);
      for (const key of TWEEN_KEYS) this.cur[key] += (this.target[key] - this.cur[key]) * k;
      this.cur.follows = this.target.follows;

      this.angles.a += this.cur.spinA * DEG * dt;
      this.angles.b -= this.cur.spinB * DEG * dt;
      this.angles.c += this.cur.spinC * DEG * dt;

      // Audio level: quick attack, gentle release.
      this.level += (this.levelTarget - this.level) * (1 - Math.exp(-dt / 0.06));
      this.levelTarget *= Math.exp(-dt / 0.25);
    }

    _coreScale() {
      const c = this.cur;
      if (c.follows) {
        // Listening/speaking: the pulse follows the real audio level.
        return c.coreScale * (1 + Math.min(1, this.level * 1.6) * (c.pulse - 1) * 1.6);
      }
      const phase = 0.5 - 0.5 * Math.cos((TAU * this.time) / c.pulseDur);
      return c.coreScale * (1 + (c.pulse - 1) * phase);
    }

    _draw() {
      const { ctx, canvas, cur: c } = this;
      const scene = this.scene.getContext('2d');

      // Sharp layer, drawn into the scene buffer.
      scene.setTransform(1, 0, 0, 1, 0, 0);
      scene.clearRect(0, 0, canvas.width, canvas.height);

      scene.globalAlpha = c.circuitOp;
      scene.drawImage(this.circuits, 0, 0);
      scene.globalAlpha = c.ringOp;
      scene.drawImage(this.staticRings, 0, 0);

      this._baseTransform(scene);
      scene.globalAlpha = c.ringOp;
      this._ringStack(scene, () => {
        this._circle(scene, 500, GOLD, 0.7, 2.5, [260, 60, 40, 90, 140, 70], this.angles.a);
        this._circle(scene, 420, GOLD, 0.55, 10, [1.5, 9], this.angles.b);
        this._circle(scene, 290, HOT, 0.8, 2, [120, 50, 20, 50], this.angles.c, 'round');
      });
      scene.save();
      scene.rotate(62 * DEG);
      scene.scale(1, 0.28);
      this._circle(scene, 330, GOLD, 0.45, 1.5, [200, 40, 10, 40], this.angles.b);
      scene.restore();

      scene.setTransform(1, 0, 0, 1, 0, 0);
      const t = this.time;
      const tw = (dur) => 0.5 - 0.5 * Math.cos((TAU * t) / dur);
      const sparkAlphas = [0.25 + 0.75 * tw(2.4), 1 - 0.8 * tw(3.1), 0.4 + 0.5 * tw(1.7)];
      for (let i = 0; i < 3; i++) {
        scene.globalAlpha = c.sparkOp * sparkAlphas[i];
        if (scene.globalAlpha > 0.005) scene.drawImage(this.sparks[i], 0, 0);
      }

      // Core.
      this._baseTransform(scene);
      scene.globalAlpha = 1;
      const s = this._coreScale();
      scene.scale(s, s);
      this._circle(scene, 58, GOLD, 0.55, 1.5);
      this._circle(scene, 38, GOLD, 0.3, 16);
      this._circle(scene, 38, HOT, 1, 8);
      scene.beginPath();
      scene.arc(0, 0, 22, 0, TAU);
      scene.fillStyle = VOID;
      scene.fill();
      scene.beginPath();
      scene.arc(0, 0, 10, 0, TAU);
      scene.globalAlpha = 0.35;
      scene.fillStyle = GOLD;
      scene.fill();

      // Composite: haze, then bloom (blurred copy under the sharp one),
      // then fade the edges.
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.globalCompositeOperation = 'source-over';

      this._baseTransform(ctx);
      const haze = ctx.createRadialGradient(0, 0, 0, 0, 0, 300);
      haze.addColorStop(0, 'rgba(255,184,56,0.55)');
      haze.addColorStop(0.4, 'rgba(255,138,30,0.15)');
      haze.addColorStop(1, 'rgba(255,138,30,0)');
      ctx.globalAlpha = c.glowOp;
      ctx.fillStyle = haze;
      ctx.beginPath();
      ctx.arc(0, 0, 300, 0, TAU);
      ctx.fill();

      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.globalAlpha = 1;
      if (this.bloomPx > 0) {
        ctx.filter = `blur(${this.bloomPx * this.dpr}px)`;
        ctx.drawImage(this.scene, 0, 0);
        ctx.filter = 'none';
      }
      ctx.drawImage(this.scene, 0, 0);

      ctx.globalCompositeOperation = 'destination-in';
      ctx.drawImage(this.mask, 0, 0);
      ctx.globalCompositeOperation = 'source-over';
    }
  }

  window.GoldenLens = GoldenLens;
})();
