"""Proactive alerts (README 3, "Proactive remarks"): ORACLE speaks up on its
own about things worth interrupting for.

    battery   low (20%) and critical (10%) while unplugged, once per level
              per discharge
    cpu/ram   one program keeping the CPU or memory above 90% for 2 minutes
    disk      the system drive under 10% free, once a day
    calendar  an Outlook event starting in 10 minutes (when Outlook is set up)

AlertMonitor.check() only decides *what* to say; oracle_server decides
*when* (not over a conversation, not muted, not in quiet hours, and only
when the owner is at the PC). Settings: alerts on/off, alerts_quiet
"23:00-07:00", alert_battery_levels "20,10".
"""
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import psutil

import core

SUSTAIN_SEC = 120          # CPU/RAM must stay high this long
HIGH_PERCENT = 90
LOAD_COOLDOWN_SEC = 30 * 60
DISK_FREE_PERCENT = 10
EVENT_LEAD_MIN = 10
# Never named as the culprit: Windows itself.
_NOT_A_CULPRIT = {"system idle process", "system", "registry", "memory compression", "idle"}


@dataclass
class Alert:
    key: str                # dedupe/cooldown identity, e.g. "battery:10"
    speech: str             # what ORACLE says
    title: str              # toast title
    body: str               # toast text
    expires: float = field(default=0.0)  # monotonic time after which it's stale and not spoken

    def stale(self, now: float) -> bool:
        return bool(self.expires) and now > self.expires


def _program_name(proc) -> str:
    """'chrome.exe' -> 'Chrome'."""
    name = os.path.splitext(proc.info.get("name") or proc.name())[0]
    return name[:1].upper() + name[1:]


def _battery_levels() -> list:
    raw = core.get_setting("alert_battery_levels") or "20,10"
    try:
        return sorted({int(x) for x in raw.split(",") if x.strip()}, reverse=True)
    except ValueError:
        return [20, 10]


class AlertMonitor:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self._battery_done = set()      # levels already announced this discharge
        self._high_since = {"cpu": None, "ram": None}
        self._last_load_alert = {"cpu": -1e9, "ram": -1e9}
        self._disk_day = None
        self._events_done = set()
        self._next_calendar = 0.0
        psutil.cpu_percent(None)        # prime: the first reading is meaningless

    def check(self) -> list:
        alerts = []
        for fn in (self._battery, self._load, self._disk, self._calendar):
            try:
                alerts += fn()
            except Exception as e:
                print(f"Alert check {fn.__name__} failed: {e}")
        return alerts

    # ---- battery ----

    def _battery(self) -> list:
        b = psutil.sensors_battery()
        if b is None:
            return []
        if b.power_plugged:
            self._battery_done.clear()   # a new discharge starts fresh
            return []
        pct = round(b.percent)
        due = [lvl for lvl in _battery_levels() if pct <= lvl and lvl not in self._battery_done]
        if not due:
            return []
        self._battery_done.update(lvl for lvl in _battery_levels() if pct <= lvl)
        critical = min(_battery_levels())
        left = ""
        if b.secsleft not in (psutil.POWER_TIME_UNKNOWN, psutil.POWER_TIME_UNLIMITED) and b.secsleft > 0:
            mins = b.secsleft // 60
            left = f", about {mins // 60} hours {mins % 60} minutes left" if mins >= 60 else f", about {mins} minutes left"
        if pct <= critical:
            speech = f"Sir, battery's at {pct} percent{left}. I'd plug in soon."
        else:
            speech = f"Sir, battery's down to {pct} percent{left}."
        return [Alert(f"battery:{min(due)}", speech, "Battery low", f"{pct}% remaining{left}.",
                      self.clock() + 15 * 60)]

    # ---- sustained CPU / memory ----

    def _top(self, by: str):
        """(program, value): the program using the most CPU (percent of the
        whole machine, sampled over 1 s) or memory (bytes). Several processes
        of one program (Chrome) count together. Slow - reading every process
        on Windows takes seconds - so it only runs when an alert fires."""
        procs = []
        for p in psutil.process_iter(["name"]):
            if (p.info["name"] or "").lower().removesuffix(".exe") in _NOT_A_CULPRIT or p.pid == os.getpid():
                continue
            try:
                procs.append((p, p.cpu_times() if by == "cpu" else p.memory_info().rss))
            except psutil.Error:
                continue
        if by == "cpu":
            start = time.perf_counter()
            time.sleep(1.0)
            sampled = []
            for p, before in procs:
                try:
                    after = p.cpu_times()
                except psutil.Error:
                    continue
                busy = (after.user - before.user) + (after.system - before.system)
                sampled.append((p, busy))
            elapsed = (time.perf_counter() - start) * psutil.cpu_count()
            procs = [(p, 100 * busy / elapsed) for p, busy in sampled]
        totals = {}
        for p, value in procs:
            name = _program_name(p)
            totals[name] = totals.get(name, 0) + value
        if not totals:
            return None, 0
        name = max(totals, key=totals.get)
        return name, totals[name]

    def _load(self) -> list:
        now = self.clock()
        readings = {"cpu": psutil.cpu_percent(None), "ram": psutil.virtual_memory().percent}
        alerts = []
        for kind, value in readings.items():
            if value < HIGH_PERCENT:
                self._high_since[kind] = None
                continue
            if self._high_since[kind] is None:
                self._high_since[kind] = now
            if now - self._high_since[kind] < SUSTAIN_SEC or now - self._last_load_alert[kind] < LOAD_COOLDOWN_SEC:
                continue
            self._last_load_alert[kind] = now
            if kind == "cpu":
                name, share = self._top("cpu")
                culprit = f" {name} is the main culprit, at about {share:.0f} percent." if name and share >= 20 else ""
                speech = f"Sir, the processor has been running flat out for a couple of minutes.{culprit}"
                body = f"CPU at {value:.0f}% for 2 minutes." + (f" Top: {name} ({share:.0f}%)." if culprit else "")
            else:
                name, rss = self._top("ram")
                culprit = f" {name} is using the most, {rss / 2**30:.1f} gigabytes." if name else ""
                speech = f"Sir, memory is at {value:.0f} percent.{culprit}"
                body = f"Memory at {value:.0f}%." + (f" Top: {name} ({rss / 2**30:.1f} GB)." if name else "")
            alerts.append(Alert(f"{kind}:high", speech, "Your PC is under strain", body, now + 5 * 60))
        return alerts

    # ---- disk ----

    def _disk(self) -> list:
        today = datetime.now().date()
        if self._disk_day == today:
            return []
        self._disk_day = today
        drive = os.environ.get("SystemDrive", "C:") + "\\"
        usage = psutil.disk_usage(drive)
        free_pct = 100 - usage.percent
        if free_pct >= DISK_FREE_PERCENT:
            return []
        gb = usage.free / 2**30
        return [Alert(f"disk:{today}", f"Sir, the {drive[0]} drive is nearly full, only {gb:.0f} gigabytes left.",
                      "Disk almost full", f"{drive} has {gb:.1f} GB free ({free_pct:.0f}%).", self.clock() + 12 * 3600)]

    # ---- calendar (Outlook) ----

    def _calendar(self) -> list:
        now = self.clock()
        if not os.environ.get("MS_CLIENT_ID") or now < self._next_calendar:
            return []
        self._next_calendar = now + 60
        start = datetime.now(timezone.utc)
        z = lambda d: d.strftime("%Y-%m-%dT%H:%M:%SZ")
        with core.no_interactive_login():
            resp = core._graph_request(
                "GET",
                f"/me/calendarView?startDateTime={z(start)}&endDateTime={z(start + timedelta(minutes=EVENT_LEAD_MIN + 1))}"
                f"&$select=id,subject,start,isAllDay,location&$top=10",
            )
        resp.raise_for_status()
        alerts = []
        for e in resp.json().get("value", []):
            if e.get("isAllDay") or e["id"] in self._events_done:
                continue
            begins = datetime.fromisoformat(e["start"]["dateTime"][:19]).replace(tzinfo=timezone.utc)
            mins = max(0, round((begins - start).total_seconds() / 60))
            if mins > EVENT_LEAD_MIN:
                continue
            self._events_done.add(e["id"])
            subject = e.get("subject") or "an event"
            loc = (e.get("location") or {}).get("displayName")
            when = "now" if mins == 0 else f"in {mins} minute{'s' if mins != 1 else ''}"
            speech = f"Sir, {subject} starts {when}" + (f", in {loc}." if loc else ".")
            alerts.append(Alert(f"event:{e['id']}", speech, "Starting soon",
                                f"{subject} at {begins.astimezone():%H:%M}" + (f" ({loc})" if loc else ""),
                                now + (mins + 2) * 60))
        return alerts


def in_quiet_hours(now: datetime = None) -> bool:
    """alerts_quiet "23:00-07:00" (may wrap midnight); empty or "off" = never."""
    raw = (core.get_setting("alerts_quiet") or "23:00-07:00").strip().lower()
    if raw in ("", "off", "none"):
        return False
    try:
        a, b = (datetime.strptime(x.strip(), "%H:%M").time() for x in raw.split("-"))
    except ValueError:
        return False
    t = (now or datetime.now()).time()
    return a <= t < b if a < b else (t >= a or t < b)
