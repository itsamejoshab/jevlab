# TEMPORARY. Delete this file after the turbo publish watch.
#
# Watches new activity-log and net-error lines once Launch starts.
# Ignores everything already in the logs. Stops itself when the publisher
# logs "finished:" or "publisher crashed:".
#
#   python3 scripts/tmp_watch_turbo_publish.py
#
# Notes: /tmp/jevlab-turbo-publish-notes.md

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/jb/dev/jever-game/jevlab/data")
NOTES = Path("/tmp/jevlab-turbo-publish-notes.md")

LINE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (?P<edition>\S+) (?P<pid>\d+) (?P<source>\S+) (?P<msg>.*)$"
)
MARK = re.compile(r"^\[(?P<i>\d+)/(?P<n>\d+)\]\s*")
HEADER = re.compile(r"^== \S+")

# Lines that are a normal publish, including a win. Anything else matching BAD is a note.
WON = re.compile(r"^-> (won\b|already$|dry-run$)")
BAD: list[tuple[str, re.Pattern[str]]] = [
    ("crash", re.compile(r"publisher crashed", re.I)),
    ("banned", re.compile(r"-> banned:|banned phrase|server did not name the span", re.I)),
    ("rejected", re.compile(r"-> rejected:", re.I)),
    ("failed", re.compile(r"-> failed:|reload failed", re.I)),
    ("missed", re.compile(r"-> missed\b|Jev picked", re.I)),
    ("short", re.compile(r"-> short\b", re.I)),
    ("rate-limit", re.compile(r"rate limited", re.I)),
    ("busy", re.compile(r"\bbusy \(", re.I)),
    ("retry", re.compile(r"retry in \d", re.I)),
    ("round-closed", re.compile(r"round closed", re.I)),
    ("fall-through", re.compile(r"falling through|fall-throughs|leaving the rest|no other winning", re.I)),
    ("skipped", re.compile(r": skipped\b", re.I)),
    ("stopped", re.compile(r"\bstopped\b", re.I)),
    ("lock", re.compile(r"another publisher", re.I)),
    ("auth", re.compile(r"not signed in", re.I)),
    ("empty", re.compile(r"nothing queued", re.I)),
    ("dropped", re.compile(r"dropped the .+ estimate", re.I)),
    ("goal", re.compile(r"goal .+ -> |does not match the site|scored against goal", re.I)),
    ("kept-despite-error", re.compile(r"despite the error", re.I)),
    ("leader", re.compile(r"no longer beats the live leader", re.I)),
    ("rules", re.compile(r"phrase breaks strict rules|question is gone|revision changed|empty golf", re.I)),
    ("borrowed", re.compile(r"estimated from \S+: posting the borrowed score", re.I)),
    ("gamble", re.compile(r"^\s*gamble:", re.I)),
    ("long-shot", re.compile(r"^\s*long shot:", re.I)),
    ("deterministic", re.compile(r"no extra re-rolls", re.I)),
]
END = re.compile(r"finished:|publisher crashed")
COUNTDOWN = re.compile(r"^[1-5]$")


def discover(name: str) -> list[Path]:
    found = []
    main = ROOT / name
    if main.exists():
        found.append(main)
    found.extend(sorted(p for p in ROOT.glob(f"*/{name}") if p.is_file()))
    return found


def slot_of(msg: str) -> tuple[str, str]:
    found = MARK.match(msg)
    if not found:
        return "", msg
    return f"{found['i']}/{found['n']}", msg[found.end() :]


def classify(body: str) -> str | None:
    if COUNTDOWN.match(body) or body.startswith("Turbo is on") or body.startswith("publishing ") or body.startswith("DRY RUN:"):
        return None
    if WON.match(body):
        return None
    if HEADER.match(body) and ": skipped " not in body:
        return None
    for tag, pattern in BAD:
        if pattern.search(body):
            return tag
    return None


class Watch:
    def __init__(self) -> None:
        self.offsets: dict[Path, int] = {}
        self.pending: dict[Path, str] = {}
        self.headers: dict[str, str] = {}
        self.recent: dict[str, list[str]] = {}
        self.started = False
        self.started_at = ""
        self.notes = 0
        self.counts: dict[str, int] = {}
        self.publish_lines = 0
        self.net_lines = 0
        NOTES.write_text(
            "# TEMPORARY turbo publish notes\n\n"
            f"Armed {datetime.now():%Y-%m-%d %H:%M:%S}. Lines already in the logs are ignored.\n"
            "Scan traffic is ignored until the first publish line.\n\n"
        )

    def arm(self, path: Path) -> None:
        if path in self.offsets:
            return
        self.offsets[path] = path.stat().st_size if path.exists() else 0
        self.pending[path] = ""

    def read_new(self, path: Path) -> list[str]:
        if not path.exists():
            return []
        size = path.stat().st_size
        offset = self.offsets.get(path, 0)
        if size < offset:
            offset = 0
        if size == offset:
            return []
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(offset)
            chunk = handle.read()
            self.offsets[path] = handle.tell()
        text = self.pending.get(path, "") + chunk
        parts = text.splitlines(keepends=True)
        if parts and not parts[-1].endswith("\n"):
            self.pending[path] = parts.pop()
        else:
            self.pending[path] = ""
        return [part.strip("\n") for part in parts if part.strip()]

    def add(self, block: str) -> None:
        self.notes += 1
        with NOTES.open("a", encoding="utf-8") as handle:
            handle.write(block)
        first = next(line for line in block.splitlines() if line.startswith("### "))
        print(f"INCIDENT {first.removeprefix('### ')}", flush=True)

    def note_publish(self, ts: str, edition: str, msg: str) -> None:
        self.publish_lines += 1
        slot, body = slot_of(msg.strip())
        key = slot or "-"
        if HEADER.match(body):
            self.headers[key] = body
        tag = classify(body)
        self.recent.setdefault(key, []).append(msg)
        self.recent[key] = self.recent[key][-8:]
        if tag is None:
            return
        self.counts[tag] = self.counts.get(tag, 0) + 1
        header = self.headers.get(key, "")
        context = [line for line in self.recent.get(key, [])[:-1] if line != msg]
        where = f" [{slot}]" if slot else ""
        lines = [
            f"### {self.notes + 1}. {tag} — {ts} {edition}{where}",
            "",
            f"- line: `{msg}`",
        ]
        if header and header not in msg:
            lines.append(f"- phrase: `{header}`")
        if context:
            lines.append("- just before:")
            lines.extend(f"  - `{line}`" for line in context[-4:])
        lines.append("")
        self.add("\n".join(lines) + "\n")

    def note_net(self, raw: str) -> None:
        self.net_lines += 1
        if not self.started:
            return
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            row = {"detail": raw}
        when = datetime.fromtimestamp(float(row.get("at") or time.time())).strftime("%Y-%m-%d %H:%M:%S")
        service = row.get("service") or "?"
        status = row.get("status") or "?"
        host = row.get("host") or "?"
        endpoint = row.get("endpoint") or ""
        detail = row.get("detail") or ""
        tag = f"net:{service}"
        self.counts[tag] = self.counts.get(tag, 0) + 1
        self.add(
            "\n".join(
                [
                    f"### {self.notes + 1}. {tag} — {when}",
                    "",
                    f"- `{service} {status} from {host}{endpoint}`",
                    f"- {detail}" if detail else "- (no detail)",
                    "",
                ]
            )
            + "\n"
        )

    def on_activity(self, raw: str) -> str | None:
        found = LINE.match(raw)
        if not found or found["source"] != "publish":
            return None
        if not self.started:
            self.started = True
            self.started_at = found["ts"]
            print(f"PUBLISH_STARTED {found['ts']} {found['edition']}", flush=True)
            with NOTES.open("a", encoding="utf-8") as handle:
                handle.write(f"Publish lines began {found['ts']} ({found['edition']}).\n\n")
        self.note_publish(found["ts"], found["edition"], found["msg"])
        if END.search(found["msg"]):
            return found["msg"]
        return None

    def finish(self, end_line: str) -> None:
        counts = ", ".join(f"{tag} {n}" for tag, n in sorted(self.counts.items())) or "none"
        summary = (
            "\n## Summary\n\n"
            f"- started: {self.started_at or 'no publish lines seen'}\n"
            f"- ended on: `{end_line}`\n"
            f"- publish lines: {self.publish_lines}\n"
            f"- net errors during publish: {sum(n for tag, n in self.counts.items() if tag.startswith('net:'))}\n"
            f"- notes: {self.notes}\n"
            f"- by tag: {counts}\n"
        )
        with NOTES.open("a", encoding="utf-8") as handle:
            handle.write(summary)
        print(f"PUBLISH_WATCH_DONE notes={self.notes} file={NOTES}", flush=True)
        print(summary, flush=True)


def main() -> None:
    watch = Watch()
    for name in ("activity.log", "net_errors.log"):
        for path in discover(name):
            watch.arm(path)
    armed = ", ".join(str(path.relative_to(ROOT)) for path in watch.offsets) or "no logs yet"
    print(f"READY {armed}", flush=True)
    print(f"NOTES {NOTES}", flush=True)
    end_line = ""
    last_beat = time.monotonic()
    while True:
        for path in discover("activity.log"):
            watch.arm(path)
            for raw in watch.read_new(path):
                ended = watch.on_activity(raw)
                if ended:
                    end_line = ended
        for path in discover("net_errors.log"):
            watch.arm(path)
            for raw in watch.read_new(path):
                watch.note_net(raw)
        if end_line:
            time.sleep(2)
            for path in discover("activity.log"):
                for raw in watch.read_new(path):
                    watch.on_activity(raw)
            for path in discover("net_errors.log"):
                for raw in watch.read_new(path):
                    watch.note_net(raw)
            watch.finish(end_line)
            return
        now = time.monotonic()
        if now - last_beat >= 20:
            last_beat = now
            state = "publishing" if watch.started else "waiting"
            print(f"HEARTBEAT {state} publish_lines={watch.publish_lines} notes={watch.notes}", flush=True)
        time.sleep(0.4)


if __name__ == "__main__":
    main()
