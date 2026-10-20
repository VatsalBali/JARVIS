# ORACLE: Personal AI Assistant (JARVIS-inspired)

> **Handoff doc.** This file describes what ORACLE is today, what it should become, and in what order to build it.
> The capability list in section 3 is the spec. All owner decisions are settled (section 2).

**Status legend:** ✅ Built · 🔧 Built, needs fixing · 🆕 New
**Priority:** **P1** do first · **P2** next · **P3** later / nice to have

---

## 1. What ORACLE is

A desktop assistant for Windows modeled on JARVIS from Iron Man. It's calm, dry-witted, quietly loyal, and it acts instead of only talking. **Target experience:** say "Oracle" and a glowing orb appears in the middle of the screen, listens, answers out loud and fades away, like Alexa. It controls the PC, handles mail and calendar, researches and helps with code. A chat window exists as a secondary interface. (Today it's a pywebview HUD window with click-to-talk; section 4 describes the change.)

**Owner's machine:** Intel Core i7-8650U (4 cores, integrated graphics only), 32 GB RAM, Windows.

### Current stack

| Part | Tech |
|---|---|
| Main brain | `openai/gpt-oss-120b` via Groq (`core.py`, `MODEL`) |
| Coding agent | `qwen/qwen3.6-27b` via Groq (`CODING_MODEL`) |
| Voice (ring click) | Gemini Live, native audio in and out (`gemini_voice.py`) |
| Dictation (text box mic) | faster-whisper `base`, local CPU |
| UI | pywebview HUD (`UI.py` + `index.html`), pystray tray icon, autostart |
| Memory | SQLite at `%LOCALAPPDATA%\ORACLE\oracle_memory.db` |
| Web search | Tavily |
| Mail and calendar | Microsoft Graph (Outlook) via MSAL, Gmail via Google OAuth |
| Packaging | PyInstaller → `JARVIS.exe` (`build.bat`) |

### Environment variables

`GROQ_API_KEY`, `TAVILY_API_KEY`, `GOOGLE_API_KEY` (Gemini), `MS_CLIENT_ID`, `GOOGLE_CLIENT_SECRET_PATH`, `TELEGRAM_BOT_TOKEN` (optional, from @BotFather). Never hardcode keys. The Google client-secret JSON is gitignored and must stay that way.

### Files

```
core.py           brain, 33 tools, DB, email/calendar, STT
gemini_voice.py   ring-click voice turn (Gemini Live)
memory.py         long-term memory: remember / forget / recall, added to every request
telegram_bot.py   Telegram bot: talk to ORACLE from the phone, approvals as buttons
UI.py             pywebview window, JS bridge (Api class), tray, autostart
index.html        HUD front end
chat.html         older chat UI
build.bat / JARVIS.spec   exe build
log_ram.ps1 / register_task.bat   hourly RAM logger (separate utility)
```

---

## 2. Owner decisions (all settled)

- ✅ **How ORACLE addresses the owner:** ORACLE picks naturally between **"Sir"**, **"Vatsal"** and **"V"** (or similar) to suit the moment. For example "Sir" for alerts, confirmations and dry wit, "Vatsal" for normal conversation, "V" when casual. Vary it, never overuse it (at most once per reply, often not at all). Replace `USER_NAME` with an owner-name setting plus this guidance in `SYSTEM_PROMPT`.
- ✅ **Wake word:** decided. Always-on "Oracle" wake word, Alexa-style, with the orb as the main interface (section 4).
- ✅ **Orb look:** decided. **Golden Lens**, board K in the "ORACLE Orb Mockups" canvas (visual spec in 4.2).
- ✅ **Offline brain:** gpt-oss-20b on a USB stick as the fallback (section 3.12).
- ✅ **Messaging:** WhatsApp only (section 3.8).
- ✅ **Smart home:** none. Smart-home control is out of scope.
- ✅ **Confirmation style:** both. Yes/No buttons under the orb, or say "yes" / "confirm" (section 4.1).

---

## 3. Capabilities (the spec)

### 3.1 Presence & personality: *"At your service, sir."*
| Capability | Status | Pri |
|---|---|---|
| JARVIS-style persona (calm, dry wit, loyal, concise) | ✅ | P1 |
| Address the owner as Sir / Vatsal / V, chosen by context (section 2; fix `USER_NAME`) | 🔧 | P1 |
| Time-aware greeting on activation, varied each time (`generate_wake_greeting` exists but is unused) | 🔧 | P2 |
| Morning briefing: time, weather, calendar, unread mail, system health, top news | 🆕 | P2 |
| Proactive remarks ("Sir, your meeting starts in 10 minutes." / "Battery at 15%.") | 🆕 | P2 |

### 3.2 Voice: *"Oracle, are you there?"*
| Capability | Status | Pri |
|---|---|---|
| **Wake word "Oracle"**, always listening, Alexa-style (full spec in section 4) | 🆕 | **P1** |
| Click the ring → speak → spoken reply (Gemini Live), kept as a backup trigger | ✅ | P1 |
| Mic dictation into the chat box (Whisper) | ✅ | P1 |
| Follow-up window: after a reply, keep listening ~6 s without needing "Oracle" again | 🆕 | P1 |
| Interrupt ORACLE mid-sentence by saying "Oracle" | 🆕 | P2 |
| Global hotkey to talk (e.g. Ctrl+Space) and to mute the mic | 🆕 | P2 |
| **Press and hold to speak:** hold Ctrl+Alt+O (or the chat window's lens) and ORACLE listens until you let go; pauses don't end the turn and the reply starts on release. A quick tap stays hands-free | ✅ | P2 |
| Offline voice fallback (Whisper in, Piper out) when Gemini is unreachable | 🆕 | P3 |

### 3.3 Memory: *"You asked me to remind you…"*
| Capability | Status | Pri |
|---|---|---|
| All chats saved and reopenable, pin/rename/delete | ✅ | P1 |
| Voice and typed turns share one history | ✅ | P1 |
| Long-term facts about the owner ("remember I prefer…"), recalled automatically. Saved from the owner's own words only (never from emails, pages or messages), an unrecognised voice needs approval, passwords/PINs/card numbers refused, forgetting needs approval | ✅ | P2 |
| Search past conversations ("what did we decide about X last week?") | 🆕 | P2 |

### 3.4 Diagnostics: *"Power at 400% capacity."* (the suit-status equivalent)
| Capability | Status | Pri |
|---|---|---|
| Live CPU / RAM / disk stats in the HUD and on request | ✅ | P1 |
| Battery, temperature, network status and speed | 🆕 | P2 |
| List top resource-hungry processes, end a process (confirm first) | 🆕 | P2 |
| Warn automatically when RAM/CPU/disk/battery cross thresholds | 🆕 | P2 |
| Absorb the `log_ram.ps1` job: log usage history, answer "how's RAM been today?" | 🆕 | P3 |

### 3.5 PC control: *"Right away, sir."*
| Capability | Status | Pri |
|---|---|---|
| Launch apps by name | ✅ | P1 |
| Open files, URLs, web searches in the browser | ✅ | P1 |
| Close apps, switch/minimise/arrange windows | 🆕 | P2 |
| Volume, mute, brightness | 🆕 | P2 |
| Media control: play/pause/skip, "play some AC/DC" (Spotify) | 🆕 | P2 |
| Screenshot, read and write the clipboard | 🆕 | P2 |
| Lock, sleep, restart, shutdown (always confirm) | 🆕 | P3 |

### 3.6 Files: *"Pulling it up now."*
| Capability | Status | Pri |
|---|---|---|
| List, read (PDF, DOCX, text), create, move files | ✅ | P1 |
| Delete files: switch to the Recycle Bin and require confirmation | 🔧 | P1 |
| Find files by name or content across drives | 🆕 | P2 |
| Summarise or compare documents | 🆕 | P2 |
| Tidy a folder (e.g. sort Downloads by type), preview the plan before moving | 🆕 | P3 |

### 3.7 Research & intel: *"I've run the numbers."*
| Capability | Status | Pri |
|---|---|---|
| Web search with summarised answer (Tavily) | ✅ | P1 |
| Weather (current and forecast) | 🆕 | P1 |
| News headlines on chosen topics | 🆕 | P2 |
| Deeper research: several searches → short report saved to a file | 🆕 | P3 |
| Translation, unit and currency conversion, calculations | 🆕 | P3 |

### 3.8 Communications & schedule: *"Ms. Potts is on the line."*
| Capability | Status | Pri |
|---|---|---|
| Outlook: list, read, send, delete mail | ✅ | P1 |
| Gmail: list, read, send, delete mail | ✅ | P1 |
| Calendar: list, create, delete events (Outlook) | ✅ | P1 |
| Confirm before sending or deleting any email; show a draft first | 🔧 | P1 |
| Reminders and timers ("remind me at 6 to call Mum") that fire as notifications and speech | 🆕 | P1 |
| Windows toast notifications | ✅ | P1 |
| Summarise the inbox, flag what needs a reply | 🆕 | P2 |
| Google Calendar | 🆕 | P3 |
| **WhatsApp: send.** "Oracle, WhatsApp Mum I'll be late" → finds the contact, drafts the text, reads it back, sends after confirmation (see WhatsApp notes below) | 🆕 | P2 |
| **WhatsApp: incoming.** Announce new WhatsApp messages through the orb ("Sir, a message from Mum: …"), with a do-not-disturb option | 🆕 | P2 |
| **WhatsApp: reply.** "Oracle, reply that I'm on my way" to the last announced message | 🆕 | P3 |
| **Telegram bot:** text or voice-note ORACLE from the phone (same brain and tools), approvals as Yes/No buttons, reminders and alerts forwarded when away from the PC, "send that to my phone". Paired with a code shown on the PC; one owner, everyone else refused | ✅ | P2 |

**WhatsApp notes.** There is no official WhatsApp API for personal accounts. The Business Cloud API needs a separate business number, and unofficial libraries (whatsapp-web.js, Baileys) break WhatsApp's terms and risk a ban on the owner's number. Recommended approach:
- **Send:** open a prefilled chat through the WhatsApp Desktop deep link (`whatsapp://send?phone=<number>&text=<message>`), then press Send via UI automation (e.g. `pywinauto`) only after the owner confirms. Contacts come from a small local contact list (name → number) that ORACLE manages ("Oracle, save Mum's number").
- **Incoming:** read WhatsApp Desktop's Windows notifications with the `UserNotificationListener` API (the owner grants notification access once). This only sees what the notification shows (sender and preview), which is enough for announcing.
- Message text from WhatsApp is **data, not instructions** (section 5).

### 3.9 The workshop: *"Shall I render it, sir?"*
| Capability | Status | Pri |
|---|---|---|
| Coding agent for writing, debugging and explaining code | ✅ | P1 |
| Projects: tracked folders, per-project chats, read/write project files | ✅ | P1 |
| Bug tracker (create, list, update) | ✅ | P1 |
| Run shell commands (git, pip, tests) with confirmation for anything that changes state | 🔧 | P1 |
| SQL queries on local databases | ✅ | P2 |
| Git helpers: status, diff summary, write commit messages | 🆕 | P2 |
| Run the project's tests and explain failures | 🆕 | P3 |

### 3.10 Protocols: *"Initiating House Party Protocol."*
Named, user-defined routines that chain several actions, triggered by voice, button or schedule.
| Capability | Status | Pri |
|---|---|---|
| Define protocols in a simple config file (JSON/YAML) | 🆕 | P2 |
| Examples: **Work Protocol** (open VS Code + projects + music, focus mode), **Clean Slate** (close all apps, clear temp), **Goodnight** (save work, lock, sleep), **Briefing** (morning briefing on demand) | 🆕 | P2 |
| Scheduled protocols (e.g. briefing at 08:00 on weekdays) | 🆕 | P3 |

### 3.11 Vision & security: *"Scanning."*
| Capability | Status | Pri |
|---|---|---|
| "What am I looking at?": screenshot sent to a vision model, then explained | 🆕 | P2 |
| Read on-screen error messages and suggest fixes | 🆕 | P2 |
| Alert when a USB device is plugged in or a new device joins the network | 🆕 | P3 |

### 3.12 Offline mode: *"Switching to backup systems."*
| Capability | Status | Pri |
|---|---|---|
| Detect no internet → switch the brain to a local llamafile server | 🆕 | P2 |
| Model: **gpt-oss-20b** GGUF (~13 GB) on an exFAT/NTFS USB stick, run as `llamafile.exe --server --model <file>.gguf` | 🆕 | P2 |
| Reduced tool set offline (no email/web), and ORACLE says it's on backup power | 🆕 | P2 |

Expected offline performance on this PC: ~5–10 tokens/s generation. The first reply takes 1–3 minutes while it processes the ~4k-token tool list; later replies start within seconds thanks to prompt caching. Verify llamafile's OpenAI-compatible `tools` support before building on it.

### 3.13 Interface: *the orb first, the chat second*
| Capability | Status | Pri |
|---|---|---|
| **Voice orb in the centre of the screen**, shown only when woken (section 4) | 🆕 | **P1** |
| Orb animates per state: asleep / wake / listening / thinking / speaking | 🆕 | P1 |
| Captions under the orb: your words live, then ORACLE's reply and the tool in use ("Checking your calendar…") | 🆕 | P1 |
| Chat window (current HUD: chat, sidebar, projects) becomes **secondary**, opened from the tray, a hotkey, or "Oracle, open the chat" | 🔧 | P2 |
| Tray icon, autostart | ✅ | P1 |
| Live widgets in the chat window: system gauges, next event, weather, unread count | 🆕 | P3 |

---

## 4. Voice-first orb: "Oracle" wake word (the main interface) · P1

ORACLE should work like Alexa: always listening for its name, invisible until called. Say **"Oracle"** and a glowing orb appears in the middle of the screen, listens, answers out loud, then fades away. The chat window stays available but is secondary.

**Chosen design: Golden Lens.** See boards "K · Golden Lens" and "Golden Lens: wake-word lifecycle" in the "ORACLE Orb Mockups" canvas. The other boards there are rejected explorations.

### 4.1 How it behaves

| # | State | What the user sees and hears | What happens underneath |
|---|---|---|---|
| 1 | **Asleep** | Nothing on screen (optionally a faint dot). Tray icon shows the mic is live. | Local wake-word model scores the mic stream. **No audio leaves the PC.** |
| 2 | **Wake** | "Oracle" heard → orb blooms in the centre with a soft chime. Playing media ducks. | Open a Gemini Live session. Send the ~1.5 s of audio *after* the wake word too, so "Oracle what time is it" said in one breath works. |
| 3 | **Listening** | Orb core swells with your voice, live transcript under it. | Stream mic to Gemini Live. Its turn detection decides when you've finished. |
| 4 | **Thinking** | Amber arc sweeps. Caption shows the tool in use: "Checking your calendar…" | Tool calls run through `core.AVAILABLE_FUNCTIONS`. |
| 5 | **Speaking** | Core pulses with ORACLE's voice, reply captioned. | Play Gemini audio. Saying "Oracle" interrupts (barge-in). |
| 6 | **Follow-up** | Orb dims but stays ~6 s. Just keep talking, no "Oracle" needed. | Session kept open. Timeout → back to Asleep, orb fades out. |

Rules:
- Risky actions (section 5) show a **Yes / No** under the orb, and saying "yes" or "confirm" also works.
- "Oracle, stop" / "Oracle, never mind" cancels at any point.
- "Oracle, open the chat" opens the chat window with this conversation in it.
- Every voice turn is saved to the same SQLite history as typed chat (already true today).

### 4.2 Visual spec: Golden Lens

A holographic gold lens: tilted rings of light arranged like a gyroscope around a bright core, with radiating lines, circuit traces and twinkling sparks. No blue anywhere.

**Palette**
| Token | Hex | Use |
|---|---|---|
| gold | `#ffb838` | main rings, lines, traces, labels |
| amber | `#ff8a1e` | outer ring, dotted ring, haze |
| hot | `#fff3d1` | core ring, brightest sparks |
| void | `#130a03` | core centre |
| background (preview only) | `#0b0703` → `#050302` | the real window is transparent over the desktop |

**Layers (back to front)**
1. **Haze:** radial amber glow, radius ~300 px.
2. **Lines:** ~26 thin radiating spokes, one vertical line through the centre, ~46 short circuit traces (horizontal → 45° → horizontal) ending in small square pads.
3. **Main ring stack:** tilted −7°, flattened to 46% height for perspective. Rings at radius 160 (dotted), 220, 290 (bright, dashed, rotating), 360, 420 (tick ring, counter-rotating), 500 (long dashes, rotating), 560 (faint). Each rotates at its own speed.
4. **Cross rings:** two extra flat ellipses at +62° and −58° for gyroscope depth.
5. **Sparks:** ~240 dots of 1–3.5 px clustered around the rings, in three groups that twinkle out of phase.
6. **Core:** bright ring (radius 38, 8 px stroke) with a soft wide glow, dark centre, faint outer ring at 58.
7. **Bloom:** soft glow over everything (Gaussian blur ~3.5 px merged with the sharp layer).
8. **Captions:** state label (Chakra Petch, 13 px, letter-spacing 6 px, gold) and text (IBM Plex Mono, 20 px) below the lens.

**States** (values from the mockup; animate between them over ~250–400 ms)
| State | Rings | Sparks | Lines | Haze | Core size | Core pulse | Ring spin |
|---|---|---|---|---|---|---|---|
| Asleep | 10% | 0% | 4% | 12% | 0.45× | barely (4 s) | very slow (60–90 s) |
| Wake | 100% | 100% | 90% | 100% | 1.1× | 1.08× (1.2 s) | medium (20–40 s) |
| Listening | 85% | 70% | 60% | 80% | 1.0× | **follows the mic level** | slow (30–60 s) |
| Thinking | 75% | 50% | 100% | 60% | 0.9× | small (1.6 s) | **fast (3–8 s)** |
| Speaking | 100% | 100% | 80% | 100% | 1.05× | **follows ORACLE's voice level** | medium (24–50 s) |

In the real app, "Asleep" can hide the window entirely instead of showing the faint ember. Make that a setting.

**Implementation notes**
- Render with **Canvas 2D or WebGL (e.g. PixiJS or Three.js)**, not SVG. The bloom and ~300 animated elements should stay smooth without loading the i7-8650U's integrated graphics. Target 60 fps with low CPU/GPU use, and drop to ~15 fps when asleep.
- The mockup's geometry (ring radii, dash patterns, trace generator, spark distribution) is in `design/golden_lens_reference.html` (a copy of the canvas board; the `Component` class at the bottom holds the generators and state values). Use it as the reference.
- Drive the core pulse from the real audio level (`{"type":"level"}` events): the mic while listening, TTS output while speaking.

### 4.3 Technical spec

**Wake word**
- Use **openWakeWord** (free, open source, runs on CPU). Train a custom "Oracle" model with its synthetic-speech training notebook. Alternative: **Picovoice Porcupine** custom keyword (very accurate, needs an access key; check current free-tier terms).
- Tune the threshold and require several consecutive positive frames to cut false wakes. Log wake events with scores for tuning.
- Must stay light enough to run all day on the i7-8650U (target: low single-digit CPU %).
- Auto-pause when the mic is muted, during calls (optional), and via a hotkey or tray toggle.

**Audio pipeline**
- One `sounddevice` input stream (16 kHz mono) feeds a ring buffer. The wake-word model reads from it. After a wake, the same buffer (including the post-wake pre-roll) streams to Gemini Live. Never open the mic twice.
- Duck system volume while listening and speaking (e.g. `pycaw`), then restore it.
- `gemini_voice.py` today does one connect → one exchange → disconnect per ring click. Refactor it into a session that survives the follow-up window and supports barge-in.
- Gemini Live model id is a preview id and rotates. Keep it in config, not code.

**Orb window**
- Replace pywebview for the orb with **Electron**: `transparent: true`, `frame: false`, `alwaysOnTop: true`, `skipTaskbar: true`, `focusable: false` (never steals focus from what you're doing), and click-through (`setIgnoreMouseEvents(true, { forward: true })`) except over the Yes/No buttons.
- Centred on the primary monitor, about 400×500 px incl. captions. Hidden when asleep.
- Render the Golden Lens per 4.2. Animation is driven by state events and the live audio level (0–1) from Python.
- The existing `index.html` chat becomes a second, normal Electron window, opened on demand.

**Python ↔ UI bridge**
- Python backend (`core.py`, `gemini_voice.py`, wake-word loop) runs as its own process and serves a **WebSocket on 127.0.0.1 only**, with a per-launch random token.
- Events to the UI: `{"type":"state","state":"asleep|wake|listening|thinking|speaking|followup"}`, `{"type":"level","value":0.0-1.0}`, `{"type":"transcript","text":"…","final":false}`, `{"type":"caption","text":"…"}`, `{"type":"tool","name":"list_upcoming_events","label":"Checking your calendar…"}`, `{"type":"confirm","id":"…","action":"Delete report.pdf?"}`.
- Events from the UI: `{"type":"confirm_reply","id":"…","ok":true}`, `{"type":"cancel"}`, `{"type":"open_chat"}`, `{"type":"mute","value":true}`.
- The chat window's current pywebview `Api` methods move onto the same socket.

**Offline**
- The wake word works offline by design. If Gemini Live is unreachable, fall back to Whisper (in) → local model (3.12) → Piper TTS (out), and the orb shows a "backup systems" tint.

**Packaging**
- Electron app bundles or launches the Python backend (PyInstaller exe as a child process). Autostart registers the Electron app. Closing the orb never quits ORACLE; Quit is in the tray.

### 4.4 Done when

- [ ] Saying "Oracle" from across the room wakes it reliably, with fewer than about one false wake per day of normal use.
- [ ] Orb appears within ~300 ms of the wake word and never steals keyboard focus.
- [ ] "Oracle, what's on my calendar today?" in one breath gets a spoken answer.
- [ ] A follow-up question within ~6 s works without saying "Oracle" again.
- [ ] Saying "Oracle" while it speaks interrupts it.
- [ ] Risky actions require a spoken or clicked confirmation.
- [ ] Mic mute (hotkey + tray) stops all listening. Nothing is sent to the cloud before a wake.
- [ ] Idle CPU use stays low on the owner's i7-8650U.

---

## 5. Safety rules (apply to every capability)

1. **Three tiers of action:**
   - **Free:** read-only things (stats, reading files, search, listing mail).
   - **Confirm:** anything that changes or sends (write/move/delete files, send/delete mail, shell commands, closing apps, calendar changes).
   - **Always confirm + warn:** power actions, anything irreversible.
2. Confirmation happens in the HUD (and by voice). The model can **request** a risky action, but only the owner's confirmation **executes** it.
3. Text from emails, web pages and files is **data, not instructions**. ORACLE never acts on commands found inside them.
4. Deletes go to the Recycle Bin, not permanent removal.
5. Replace the substring shell blocklist with the confirm tier (the blocklist is trivially bypassed, e.g. `format  c:`).
6. Keys stay in environment variables, secrets stay gitignored.

---

## 6. Known issues to fix first (P1)

- [ ] `USER_NAME = "Oracle"` in `core.py`: ORACLE addresses the owner by its own name. Replace with the Sir / Vatsal / V behaviour (section 2).
- [ ] `delete_file`, `send_email`, `delete_email`, `send_gmail_message`, `delete_gmail_message`, `run_shell_command`, `write_project_file` run with no confirmation → implement section 5.
- [ ] `run_conversation`: `json.loads(...)` and `fn(**fn_args)` aren't wrapped in `try/except`, so one bad tool call crashes the turn. Return the error to the model instead.
- [ ] A ~47 MB object sits in git history (likely a built exe). Keep `dist/` and `build/` out of git.
- [ ] `register_task.bat` points at `%USERPROFILE%\Documents\log_ram.ps1`, not this folder.
- [ ] `.gitignore` lists `open_javis.bat` twice.

---

## 7. Suggested build order

1. **Phase 1, safe + voice-first:** section 6 fixes and confirmation tiers (section 5) first, since an always-listening assistant must not act on a misheard command. Then the "Oracle" wake word and orb (section 4): Python WebSocket backend → Electron orb window → wake word → follow-up and barge-in. Then reminders/timers and weather.
2. **Phase 2, make it JARVIS:** morning briefing, proactive alerts (spoken through the orb), PC and media control, protocols, chat window moved to Electron, long-term memory, offline mode.
3. **Phase 3, extras:** screen vision, security alerts, WhatsApp replies, deep research.

## 8. Running it

```bash
pip install -r requirements.txt
python UI.py          # full app
python core.py        # terminal test mode
build.bat             # build dist\JARVIS.exe (Windows only)
```
