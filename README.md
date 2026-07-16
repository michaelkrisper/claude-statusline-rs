# claude-statusline

[![CI](https://github.com/michaelkrisper/claude-statusline/actions/workflows/ci.yml/badge.svg)](https://github.com/michaelkrisper/claude-statusline/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/michaelkrisper/claude-statusline)](https://github.com/michaelkrisper/claude-statusline/releases/latest)
[![License: MIT](https://img.shields.io/github/license/michaelkrisper/claude-statusline)](LICENSE)
[![Platforms](https://img.shields.io/badge/platform-linux%20%7C%20macos%20%7C%20windows-blue)](https://github.com/michaelkrisper/claude-statusline/releases/latest)
[![Made with Rust](https://img.shields.io/badge/rust-stable-orange?logo=rust)](https://www.rust-lang.org/)

A fast, single-binary status line for [Claude Code](https://claude.com/claude-code) that
doubles as a tiny system monitor: clock, CPU, RAM, free disk space, the active account,
rate-limit consumption — and **predicts when your tokens will run out**, based on your
live burn rate and your usage history.

```
📁 ~/projects/foo 🧠 4% 📟 21% 🎮 7% 🖼 13% 💾 21G 👤 you Fable 5 high 📊 38% ⏳ 66% (~20:29 / 23:52)      🕐 09:48
```

Fields are separated by single spaces; the symbols carry the visual separation. The
clock is pushed to the right edge of the terminal (`TIOCGWINSZ` on `/dev/tty`, with
`$COLUMNS` as fallback; a single space when neither is available). Claude Code renders
the line a few columns narrower than the tty reports, so 3 columns are kept free on
the right — tune with `STATUSLINE_MARGIN` if your setup clips or under-shoots.

| Field | Source |
|---|---|
| `📁 ~/projects/foo` | project directory |
| `🧠 4%` | CPU usage since the previous refresh (`/proc/stat` delta; appears from the second invocation on) |
| `📟 21%` | used RAM, `MemAvailable` vs `MemTotal` from `/proc/meminfo` |
| `🎮 7%` | GPU utilization (first GPU, via `nvidia-smi`, cached — see below) |
| `🖼 13%` | used VRAM (memory.used vs memory.total) |
| `💾 21G` | free space on the filesystem holding the project directory (`statvfs`) |
| `👤 you` | active Claude account — the part before the `@` of the signed-in email |
| `Fable 5 high` | model and effort level |
| `📊 38%` | context window usage of the current session |
| `⏳ 66% (…)` | 5 h rate-limit usage with depletion forecast (see below) |
| `🕐 09:48` | current local time, right-aligned |

The host metrics (`cpu`, `ram`) come from `/proc` and are shown on Linux; `disk` on any
Unix. Fields whose source is unavailable are simply omitted.

`nvidia-smi` takes hundreds of milliseconds (notably on WSL2), so the GPU fields are
never queried inline: at most every 10 s a **detached** background child refreshes
`gpu.csv` in the state dir, and invocations only ever read the cached value — no
status-line refresh ever blocks on the GPU. Without `nvidia-smi` the fields stay
hidden.

Reading the `⏳` (5 h session) segment:

| Display | Meaning |
|---|---|
| `⏳ 66% (~20:29 / 23:52)` | at the current burn rate you hit 100% at ~20:29, window resets 23:52 |
| `⏳ 22% (+8h / 23:52)` | you have headroom: depletion would land ~8 h *past* the reset |
| `⏳ 22% (23:52)` | no rate estimate yet (fresh install, no history) |

The account segment is read live from `~/.claude.json` on every invocation so it
reflects the current login immediately after an account switch. It is omitted when
that file is absent or holds no signed-in account.

## How the prediction works

The status line is invoked by Claude Code on every refresh. Each invocation appends a
usage sample (timestamp, 5h/7d percentage, reset timestamps) to a small log — at most
one sample per 10 s, 12 h retention. From that log it estimates the burn rate as a
**shrinkage blend** of two estimators:

1. **Live rate** — exponentially weighted least-squares regression over the last 45 min
   of samples (15 min half-life, so recent activity dominates). Under 5 min of history
   it falls back to the window average, which is exact early on because a 5 h window
   starts at 0% with your first message.
2. **Personal prior** — whenever a 5 h window closes, its observed average rate is
   harvested into a per-window history (last 20 windows). The prior is the **median**
   of those rates, so a single burst session can't skew it.

The prior starts with the weight of ~20 min of live evidence and fades linearly to zero
once a full 45 min of live data exists: right after you start a session you get a
sensible estimate from your typical behavior; once there is real data, only the live
regression counts. The projected depletion time is then `now + remaining / rate`.

No prediction is shown when the rate is zero (idle) and no prior exists yet.

## Why Rust?

A status line runs on *every* UI refresh — easily thousands of times per session — so
per-invocation cost is the whole game:

- **~1 ms per invocation.** Measured against `/bin/true`, the binary is at the kernel's
  process-spawn floor: parsing the JSON payload, reading the sample log, and the
  regression itself are no longer measurable. An interpreter would pay 30–100 ms
  *before executing its first line* (Python/Node startup), i.e. 30–100× the entire
  budget, on every refresh.
- **~400 KB peak RSS.** Statically linked against musl there is no interpreter heap, no
  GC, no runtime — compared to ~2.1 MB glibc-dynamic and tens of MB for a scripting
  runtime.
- **~500 KB binary, zero dependencies at runtime.** One file, no venv, no node_modules,
  nothing to keep in sync.

Release builds use `opt-level=3`, fat LTO, a single codegen unit, `panic=abort` and
symbol stripping; `.cargo/config.toml` adds `-C target-cpu=native` for local builds
(CI release artifacts are built portable with `target-cpu=x86-64`).

## Install (Claude Code)

### 1. Get the binary

Download the binary for your platform from [Releases](../../releases):

| Platform | Asset |
|---|---|
| Linux x86_64 (static) | `statusline-x86_64-linux-musl` |
| Linux arm64 (static) | `statusline-aarch64-linux-musl` |
| macOS Apple Silicon | `statusline-aarch64-macos` |
| macOS Intel | `statusline-x86_64-macos` |
| Windows x86_64 | `statusline-x86_64-windows.exe` |
| Windows arm64 | `statusline-aarch64-windows.exe` |

Or build from source (Linux shown; on macOS/Windows a plain
`cargo build --release` does the job):

```sh
rustup target add x86_64-unknown-linux-musl
cargo build --release --target x86_64-unknown-linux-musl
```

### 2. Put it somewhere stable

```sh
mkdir -p ~/.claude/statusline
cp target/x86_64-unknown-linux-musl/release/statusline ~/.claude/statusline/
```

### 3. Point Claude Code at it

In `~/.claude/settings.json`:

```json
{
  "statusLine": {
    "type": "command",
    "command": "/home/YOU/.claude/statusline/statusline"
  }
}
```

That's it — the prediction appears automatically once enough samples exist, and gets
sharper after your first completed 5 h window.

## State files

| File | Content |
|---|---|
| `~/.cache/statusline-rs/samples.tsv` | rolling usage samples (12 h) |
| `~/.cache/statusline-rs/rates.tsv` | per-window burn rates of the last 20 closed 5 h windows |
| `~/.cache/statusline-rs/cpu.tsv` | `/proc/stat` jiffies baseline for the CPU-usage delta |
| `~/.cache/statusline-rs/gpu.csv` | cached `nvidia-smi` sample, refreshed in the background every 10 s |

(`$XDG_CACHE_HOME` is honored; on Windows the files live under
`%USERPROFILE%\.cache\statusline-rs`.) Delete both to reset all learned history.

## Versioning & releases

[SemVer](https://semver.org/): breaking output-format changes bump minor (pre-1.0) /
major, everything else patch. A release is cut by bumping `version` in `Cargo.toml`
and pushing a matching tag — CI builds the portable binary and attaches it:

```sh
git tag v0.1.1 && git push origin v0.1.1
```

## Development

```sh
cargo test          # unit tests for regression, blending, harvesting, formatting
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

## License

MIT
