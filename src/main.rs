use chrono::{Local, TimeZone};
use serde_json::Value;
use std::io::Read;
use std::path::PathBuf;

const LOOKBACK: i64 = 45 * 60; // burn-rate estimation window
const HALF_LIFE: f64 = 15.0 * 60.0; // recent samples weigh more
const MIN_SPAN: i64 = 5 * 60; // below this, fall back to window average
const SAMPLE_GAP: i64 = 10;
const KEEP: i64 = 12 * 3600;
const W_PRIOR: f64 = 1200.0; // prior counts like 20 min of live evidence
const HARVEST_MIN_SPAN: i64 = 900; // closed window must span this to yield a rate
const HARVEST_MIN_PCT: f64 = 2.0;
const RATES_KEEP: usize = 20;
const GPU_REFRESH: i64 = 10; // nvidia-smi is re-queried at most this often
const URGENT_SECS: i64 = 15 * 60; // a 5h depletion ETA closer than this is flagged
const URGENT: &str = "\x1b[91m"; // bright-red signal color on the ETA clock time
const FG_RESET: &str = "\x1b[39m"; // reset foreground only, preserving surrounding bold

fn fval(v: &Value, path: &[&str]) -> Option<f64> {
    path.iter().try_fold(v, |acc, k| acc.get(k))?.as_f64()
}

fn ival(v: &Value, path: &[&str]) -> Option<i64> {
    path.iter().try_fold(v, |acc, k| acc.get(k))?.as_i64()
}

fn str_at<'a>(v: &'a Value, path: &[&str]) -> Option<&'a str> {
    path.iter().try_fold(v, |acc, k| acc.get(k))?.as_str()
}

fn fmt_ts(ts: i64, fmt: &str) -> Option<String> {
    Some(Local.timestamp_opt(ts, 0).single()?.format(fmt).to_string())
}

// " (~ETA / RESET)" if depletion lands before the reset, " (+Nh / RESET)" with the
// overshoot past the reset otherwise, " (RESET)" if no rate is known. When the ETA is
// less than URGENT_SECS away, the clock time is painted in the urgent signal color.
fn push_times(out: &mut String, eta: Option<i64>, reset: i64, now: i64, reset_fmt: &str) {
    let eta_s = eta.and_then(|t| {
        if t >= reset {
            return Some(format!("+{}h", ((t - reset + 3599) / 3600).min(9999)));
        }
        let same_day = Local
            .timestamp_opt(t, 0)
            .single()
            .zip(Local.timestamp_opt(now, 0).single())
            .is_some_and(|(a, b)| a.date_naive() == b.date_naive());
        let hm = fmt_ts(t, if same_day { "%H:%M" } else { "%a %H:%M" })?;
        Some(if t - now < URGENT_SECS {
            format!("~{URGENT}{hm}{FG_RESET}")
        } else {
            format!("~{hm}")
        })
    });
    match (eta_s, fmt_ts(reset, reset_fmt)) {
        (Some(e), Some(r)) => out.push_str(&format!(" ({e} / {r})")),
        (None, Some(r)) => out.push_str(&format!(" ({r})")),
        (Some(e), None) => out.push_str(&format!(" ({e})")),
        (None, None) => {}
    }
}

#[derive(Clone, Copy)]
struct Sample {
    ts: i64,
    five: f64,
    five_reset: i64,
    seven: f64,
    seven_reset: i64,
}

// active Claude account, read live from ~/.claude.json on every invocation so it
// reflects the current login immediately after a switch, with no caching. A
// single-pass substring scan rather than a full parse of the ~30 KB file: locate
// the oauthAccount object, then the first emailAddress string inside it.
fn parse_email(data: &str) -> Option<&str> {
    let rest = &data[data.find("\"oauthAccount\"")?..];
    let after = rest[rest.find("\"emailAddress\"")? + 14..].trim_start();
    let after = after.strip_prefix(':')?.trim_start().strip_prefix('"')?;
    Some(&after[..after.find('"')?])
}

fn account_email() -> Option<String> {
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))?;
    let data = std::fs::read_to_string(PathBuf::from(home).join(".claude.json")).ok()?;
    parse_email(&data).map(str::to_string)
}

fn state_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .or_else(|| std::env::var_os("USERPROFILE").map(|h| PathBuf::from(h).join(".cache")))?
        .join("statusline-rs");
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

// cumulative (total, idle) jiffies since boot from /proc/stat's aggregate cpu line
fn parse_cpu_totals(stat: &str) -> Option<(u64, u64)> {
    let mut it = stat.lines().next()?.split_whitespace();
    if it.next()? != "cpu" {
        return None;
    }
    let vals: Vec<u64> = it.map_while(|t| t.parse().ok()).collect();
    if vals.len() < 4 {
        return None;
    }
    // idle + iowait both count as not-working
    Some((
        vals.iter().sum(),
        vals[3] + vals.get(4).copied().unwrap_or(0),
    ))
}

// CPU usage over the interval since the previous invocation, via (total, idle)
// jiffies persisted in the state dir. None on the very first run (no baseline yet)
// and on platforms without /proc.
fn cpu_pct(dir: &std::path::Path) -> Option<i64> {
    let (total, idle) = parse_cpu_totals(&std::fs::read_to_string("/proc/stat").ok()?)?;
    let path = dir.join("cpu.tsv");
    let prev = std::fs::read_to_string(&path).ok().and_then(|s| {
        let mut it = s.split_whitespace();
        Some((
            it.next()?.parse::<u64>().ok()?,
            it.next()?.parse::<u64>().ok()?,
        ))
    });
    std::fs::write(&path, format!("{total} {idle}\n")).ok();
    let (pt, pi) = prev?;
    let dt = total.checked_sub(pt)?;
    let di = idle.saturating_sub(pi).min(dt);
    (dt > 0).then(|| (100.0 * (1.0 - di as f64 / dt as f64)).round() as i64)
}

// used-RAM percentage from /proc/meminfo (MemAvailable vs MemTotal)
fn meminfo_pct(data: &str) -> Option<i64> {
    let kb = |key: &str| -> Option<f64> {
        data.lines()
            .find(|l| l.starts_with(key))?
            .split_whitespace()
            .nth(1)?
            .parse()
            .ok()
    };
    let total = kb("MemTotal:")?;
    let avail = kb("MemAvailable:")?;
    (total > 0.0).then(|| (100.0 * (1.0 - avail / total)).round() as i64)
}

// free bytes available to unprivileged users on the filesystem holding `path`
#[cfg(unix)]
fn disk_free(path: &str) -> Option<u64> {
    let c = std::ffi::CString::new(path).ok()?;
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    (unsafe { libc::statvfs(c.as_ptr(), &mut st) } == 0)
        .then(|| st.f_bavail as u64 * st.f_frsize as u64)
}

#[cfg(not(unix))]
fn disk_free(_path: &str) -> Option<u64> {
    None
}

// first GPU of an nvidia-smi csv line "util, mem.used, mem.total" (MiB) into
// (gpu %, vram %)
fn parse_gpu_csv(data: &str) -> Option<(i64, i64)> {
    let mut f = data.lines().next()?.split(',').map(str::trim);
    let util: i64 = f.next()?.parse().ok()?;
    let used: f64 = f.next()?.parse().ok()?;
    let total: f64 = f.next()?.parse().ok()?;
    (total > 0.0).then(|| (util, (100.0 * used / total).round() as i64))
}

fn mtime_secs(m: &std::fs::Metadata) -> Option<i64> {
    let d = m
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?;
    Some(d.as_secs() as i64)
}

// cached (gpu %, vram %). nvidia-smi is slow (hundreds of ms on WSL2), so it is
// never awaited: at most every GPU_REFRESH seconds a detached child re-queries it
// into gpu.csv.tmp; a later invocation adopts the finished file atomically. The
// empty tmp file doubles as the in-flight marker against spawn stampedes.
fn gpu_stats(dir: &std::path::Path, now: i64) -> Option<(i64, i64)> {
    let path = dir.join("gpu.csv");
    let tmp = dir.join("gpu.csv.tmp");
    let age = |p: &std::path::Path| {
        std::fs::metadata(p)
            .ok()
            .as_ref()
            .and_then(mtime_secs)
            .map(|t| now - t)
    };
    if std::fs::metadata(&tmp).is_ok_and(|m| m.len() > 0) {
        std::fs::rename(&tmp, &path).ok();
    }
    let fresh = |a: Option<i64>| a.is_some_and(|a| a < GPU_REFRESH);
    if !fresh(age(&path))
        && !fresh(age(&tmp))
        && let Ok(f) = std::fs::File::create(&tmp)
    {
        std::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ])
            .stdin(std::process::Stdio::null())
            .stdout(f)
            .stderr(std::process::Stdio::null())
            .spawn()
            .ok();
    }
    parse_gpu_csv(&std::fs::read_to_string(&path).ok()?)
}

// columns of the terminal the status line is rendered in. stdout is a pipe, so
// ask the controlling terminal directly; COLUMNS as a fallback.
#[cfg(unix)]
fn term_width() -> Option<usize> {
    let tty = std::ffi::CString::new("/dev/tty").ok()?;
    let fd = unsafe { libc::open(tty.as_ptr(), libc::O_RDONLY) };
    if fd >= 0 {
        let mut ws: libc::winsize = unsafe { std::mem::zeroed() };
        let r = unsafe { libc::ioctl(fd, libc::TIOCGWINSZ, &mut ws) };
        unsafe { libc::close(fd) };
        if r == 0 && ws.ws_col > 0 {
            return Some(ws.ws_col as usize);
        }
    }
    std::env::var("COLUMNS").ok()?.parse().ok()
}

#[cfg(not(unix))]
fn term_width() -> Option<usize> {
    std::env::var("COLUMNS").ok()?.parse().ok()
}

// terminal cell width of one char: emoji and East Asian wide glyphs take two
// columns, variation selectors and ZWJ none
fn char_width(c: char) -> usize {
    match c as u32 {
        0xFE0E..=0xFE0F | 0x200D => 0,
        0x1100..=0x115F
        | 0x231A..=0x231B
        | 0x23E9..=0x23FA
        | 0x2E80..=0x303E
        | 0x3041..=0x33FF
        | 0xA000..=0xA4CF
        | 0xAC00..=0xD7A3
        | 0xF900..=0xFAFF
        | 0xFE30..=0xFE4F
        | 0xFF00..=0xFF60
        | 0xFFE0..=0xFFE6
        | 0x1F000..=0x1FAFF => 2,
        _ => 1,
    }
}

// visible width of the line: ANSI escape sequences occupy no columns
fn disp_width(s: &str) -> usize {
    let mut w = 0;
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c == '\x1b' {
            for n in it.by_ref() {
                if n.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            w += char_width(c);
        }
    }
    w
}

fn fmt_bytes(b: u64) -> String {
    let g = b as f64 / (1u64 << 30) as f64;
    if g >= 1024.0 {
        format!("{:.1}T", g / 1024.0)
    } else if g >= 10.0 {
        format!("{}G", g.round() as u64)
    } else if g >= 1.0 {
        format!("{g:.1}G")
    } else {
        format!("{}M", (b as f64 / (1u64 << 20) as f64).round() as u64)
    }
}

fn load_samples(path: &PathBuf) -> Vec<Sample> {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .filter_map(|l| {
            let p: Vec<&str> = l.split_whitespace().collect();
            if p.len() != 5 {
                return None;
            }
            Some(Sample {
                ts: p[0].parse().ok()?,
                five: p[1].parse().ok()?,
                five_reset: p[2].parse().ok()?,
                seven: p[3].parse().ok()?,
                seven_reset: p[4].parse().ok()?,
            })
        })
        .collect()
}

fn load_rates(path: &PathBuf) -> Vec<(i64, f64)> {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .filter_map(|l| {
            let p: Vec<&str> = l.split_whitespace().collect();
            if p.len() != 2 {
                return None;
            }
            Some((p[0].parse().ok()?, p[1].parse().ok()?))
        })
        .collect()
}

// summarize closed 5h windows still present in the sample log into per-window
// burn rates (keyed by reset ts); returns whether `rates` changed
fn harvest_rates(samples: &[Sample], cur_reset: i64, rates: &mut Vec<(i64, f64)>) -> bool {
    let mut changed = false;
    let mut i = 0;
    while i < samples.len() {
        let r = samples[i].five_reset;
        let mut j = i;
        while j < samples.len() && samples[j].five_reset == r {
            j += 1;
        }
        if r != cur_reset && !rates.iter().any(|&(k, _)| k == r) {
            let (a, b) = (&samples[i], &samples[j - 1]);
            if b.ts - a.ts >= HARVEST_MIN_SPAN && b.five - a.five >= HARVEST_MIN_PCT {
                rates.push((r, (b.five - a.five) / (b.ts - a.ts) as f64));
                changed = true;
            }
        }
        i = j;
    }
    if rates.len() > RATES_KEEP {
        let cut = rates.len() - RATES_KEEP;
        rates.drain(..cut);
        changed = true;
    }
    changed
}

fn median(mut v: Vec<f64>) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(v[v.len() / 2])
}

// exponentially weighted least-squares slope in %/sec
fn wslope(pts: &[(i64, f64)]) -> Option<f64> {
    if pts.len() < 3 {
        return None;
    }
    let tl = pts.last()?.0;
    let (mut sw, mut sx, mut sy, mut sxx, mut sxy) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for &(t, y) in pts {
        let x = (t - tl) as f64; // <= 0
        let w = (x / HALF_LIFE * std::f64::consts::LN_2).exp();
        sw += w;
        sx += w * x;
        sy += w * y;
        sxx += w * x * x;
        sxy += w * x * y;
    }
    let d = sw * sxx - sx * sx;
    (d.abs() > f64::EPSILON).then(|| (sw * sxy - sx * sy) / d)
}

// projected time (unix secs) when this limit hits 100%. Rate is a shrinkage
// blend: live evidence (weighted regression, or the window average early on)
// weighted by its observed span, pulled toward the historical per-window prior
// weighted at W_PRIOR — the prior dominates early, live data as span grows.
fn eta(
    samples: &[Sample],
    now: i64,
    cur: f64,
    reset: i64,
    window: i64,
    prior: Option<f64>,
    get: impl Fn(&Sample) -> (f64, i64),
) -> Option<i64> {
    let mut pts: Vec<(i64, f64)> = samples
        .iter()
        .filter(|s| {
            let (p, r) = get(s);
            r == reset && p >= 0.0 && s.ts >= now - LOOKBACK
        })
        .map(|s| (s.ts, get(s).0))
        .collect();
    if pts.last().is_none_or(|p| p.0 != now) {
        pts.push((now, cur));
    }
    let span = pts.last()?.0 - pts.first()?.0;
    let (mut num, mut den) = (0f64, 0f64);
    if span >= MIN_SPAN {
        if let Some(r) = wslope(&pts) {
            num += span as f64 * r.max(0.0);
            den += span as f64;
        }
    } else {
        // window starts at 0% on first use, so the window average is a fair early guess
        let elapsed = now - (reset - window);
        if elapsed >= 60 {
            let w = elapsed.min(MIN_SPAN) as f64;
            num += w * (cur / elapsed as f64);
            den += w;
        }
    }
    if let Some(p) = prior {
        // prior fades out as live evidence accumulates; gone at full lookback
        let w = W_PRIOR * (1.0 - span as f64 / LOOKBACK as f64).max(0.0);
        num += w * p.max(0.0);
        den += w;
    }
    if den <= 0.0 {
        return None;
    }
    let rate = num / den;
    if rate <= 0.0 {
        return None;
    }
    Some(now + (((100.0 - cur) / rate).min(1e9) as i64))
}

// separator prefix: empty for the first field, a space once the line has content;
// the per-field symbols carry the visual separation
fn sep(out: &str) -> &'static str {
    if out.is_empty() { "" } else { " " }
}

fn main() {
    let mut buf = String::new();
    std::io::stdin().read_to_string(&mut buf).ok();
    let v: Value = serde_json::from_str(&buf).unwrap_or(Value::Null);

    let mut out = String::new();

    let now = Local::now().timestamp();
    let state = state_dir();
    let cwd = str_at(&v, &["cwd"]).filter(|s| !s.is_empty());

    if let Some(cwd) = cwd {
        out.push_str(&format!("📁 {cwd}"));
    }

    if let Some(p) = state.as_deref().and_then(cpu_pct) {
        out.push_str(&format!("{}🧠 {p}%", sep(&out)));
    }
    if let Some(p) = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .as_deref()
        .and_then(meminfo_pct)
    {
        out.push_str(&format!("{}📟 {p}%", sep(&out)));
    }
    if let Some((gpu, vram)) = state.as_deref().and_then(|d| gpu_stats(d, now)) {
        out.push_str(&format!("{}🎮 {gpu}% 🖼 {vram}%", sep(&out)));
    }
    if let Some(free) = disk_free(cwd.unwrap_or("/")) {
        out.push_str(&format!("{}💾 {}", sep(&out), fmt_bytes(free)));
    }

    if let Some(email) = account_email() {
        let user = email.split('@').next().unwrap_or(&email);
        out.push_str(&format!("{}👤 {user}", sep(&out)));
    }

    if let Some(model) = str_at(&v, &["model", "display_name"]).filter(|s| !s.is_empty()) {
        out.push_str(&format!("{}{model}", sep(&out)));
        if let Some(e) = str_at(&v, &["effort", "level"]) {
            out.push_str(&format!(" {e}"));
        }
    }

    if let Some(p) = fval(&v, &["context_window", "used_percentage"]) {
        out.push_str(&format!("{}📊 {}%", sep(&out), p.round() as i64));
    }

    // projected depletion: sample usage over time, extrapolate burn rate to 100%
    let five = fval(&v, &["rate_limits", "five_hour", "used_percentage"])
        .zip(ival(&v, &["rate_limits", "five_hour", "resets_at"]));
    let seven = fval(&v, &["rate_limits", "seven_day", "used_percentage"])
        .zip(ival(&v, &["rate_limits", "seven_day", "resets_at"]));
    let mut e5 = None;
    if let (Some((fp, fr)), Some(dir)) = (five, state.as_ref()) {
        let spath = dir.join("samples.tsv");
        let rpath = dir.join("rates.tsv");
        let mut samples = load_samples(&spath);
        let mut rates = load_rates(&rpath);
        if harvest_rates(&samples, fr, &mut rates) {
            let body: String = rates.iter().map(|(k, r)| format!("{k} {r}\n")).collect();
            std::fs::write(&rpath, body).ok();
        }
        if samples.last().is_none_or(|l| now - l.ts >= SAMPLE_GAP) {
            let (sp, sr) = seven.unwrap_or((-1.0, 0));
            samples.push(Sample {
                ts: now,
                five: fp,
                five_reset: fr,
                seven: sp,
                seven_reset: sr,
            });
            samples.retain(|s| s.ts >= now - KEEP);
            let body: String = samples
                .iter()
                .map(|s| {
                    format!(
                        "{} {} {} {} {}\n",
                        s.ts, s.five, s.five_reset, s.seven, s.seven_reset
                    )
                })
                .collect();
            std::fs::write(&spath, body).ok();
        }
        let prior = median(rates.iter().map(|&(_, r)| r).collect());
        e5 = eta(&samples, now, fp, fr, 5 * 3600, prior, |s| {
            (s.five, s.five_reset)
        });
    }

    if let Some((p, r)) = five {
        // the 5h window (usage + depletion forecast) is the headline metric, so
        // emphasize it in bold (\x1b[1m); the rest of the line stays default weight
        let mut seg = format!("⏳ {}%", p.round() as i64);
        push_times(&mut seg, e5, r, now, "%H:%M");
        out.push_str(&format!("{}\x1b[1m{seg}\x1b[0m", sep(&out)));
    }

    // clock last, pushed to the right edge when the terminal width is known.
    // Claude Code renders the line a few columns narrower than the tty reports,
    // so a right margin (default 3, tunable via STATUSLINE_MARGIN) is kept free.
    let margin: usize = std::env::var("STATUSLINE_MARGIN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let clock = Local::now().format("🕐 %H:%M").to_string();
    let pad = term_width()
        .map(|w| w.saturating_sub(disp_width(&out) + disp_width(&clock) + margin))
        .filter(|&p| p > 0)
        .unwrap_or(1);
    out.push_str(&format!("{}{clock}", " ".repeat(pad)));

    println!("{out}");
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: i64 = 1_750_000_000;

    fn linear_samples(now: i64, reset: i64, minutes: i64, start: f64, per_min: f64) -> Vec<Sample> {
        (1..=minutes)
            .rev()
            .map(|i| Sample {
                ts: now - i * 60,
                five: start + (minutes - i) as f64 * per_min,
                five_reset: reset,
                seven: -1.0,
                seven_reset: 0,
            })
            .collect()
    }

    #[test]
    fn wslope_exact_on_linear_data() {
        let pts: Vec<(i64, f64)> = (0..10).map(|i| (i * 60, 10.0 + i as f64 * 0.5)).collect();
        let s = wslope(&pts).unwrap();
        assert!((s - 0.5 / 60.0).abs() < 1e-12);
    }

    #[test]
    fn wslope_zero_on_flat_data() {
        let pts: Vec<(i64, f64)> = (0..10).map(|i| (i * 60, 42.0)).collect();
        assert!(wslope(&pts).unwrap().abs() < 1e-12);
    }

    #[test]
    fn wslope_needs_three_points() {
        assert!(wslope(&[(0, 1.0), (60, 2.0)]).is_none());
    }

    #[test]
    fn median_basics() {
        assert_eq!(median(vec![]), None);
        assert_eq!(median(vec![3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(vec![4.0, 1.0]), Some(4.0)); // upper median on even n
    }

    #[test]
    fn eta_regression_matches_linear_burn() {
        // 0.9 %/min for 40 min, 66.0% now -> 100% in (34/0.9) min
        let reset = NOW + 4 * 3600;
        let samples = linear_samples(NOW, reset, 40, 30.0, 0.9);
        let t = eta(&samples, NOW, 66.0, reset, 5 * 3600, None, |s| {
            (s.five, s.five_reset)
        })
        .unwrap();
        let expected = NOW + (34.0 / 0.9 * 60.0) as i64;
        assert!((t - expected).abs() < 5, "eta {t} vs expected {expected}");
    }

    #[test]
    fn eta_none_when_idle_without_prior() {
        let reset = NOW + 4 * 3600;
        let samples = linear_samples(NOW, reset, 40, 66.2, 0.0);
        assert!(
            eta(&samples, NOW, 66.2, reset, 5 * 3600, None, |s| (
                s.five,
                s.five_reset
            ))
            .is_none()
        );
    }

    #[test]
    fn eta_window_average_fallback_under_min_span() {
        // only 2 min of samples, window elapsed 1h at 10% -> 0.1667 %/min
        let reset = NOW + 4 * 3600;
        let samples = linear_samples(NOW, reset, 2, 9.9, 0.05);
        let t = eta(&samples, NOW, 10.0, reset, 5 * 3600, None, |s| {
            (s.five, s.five_reset)
        })
        .unwrap();
        let expected = NOW + (90.0 / (10.0 / 3600.0)) as i64; // 9h
        assert!((t - expected).abs() < 5, "eta {t} vs expected {expected}");
    }

    #[test]
    fn eta_prior_dominates_fresh_window() {
        // no samples, window 5 min old at 2%: blend of window avg (w=300) and prior (w=1200)
        let reset = NOW + 295 * 60;
        let prior = 0.02; // %/sec
        let t = eta(&[], NOW, 2.0, reset, 5 * 3600, Some(prior), |s| {
            (s.five, s.five_reset)
        })
        .unwrap();
        let rate = (300.0 * (2.0 / 300.0) + 1200.0 * prior) / 1500.0;
        let expected = NOW + (98.0 / rate) as i64;
        assert!((t - expected).abs() < 5, "eta {t} vs expected {expected}");
    }

    #[test]
    fn eta_prior_fully_faded_at_lookback() {
        // 45 min of flat live data: prior weight is 0, idle -> no projection
        let reset = NOW + 4 * 3600;
        let samples = linear_samples(NOW, reset, 45, 66.2, 0.0);
        assert!(
            eta(&samples, NOW, 66.2, reset, 5 * 3600, Some(0.02), |s| (
                s.five,
                s.five_reset
            ))
            .is_none()
        );
    }

    #[test]
    fn eta_ignores_other_windows() {
        // samples from a previous window must not feed the regression: with them
        // filtered out only the window-average fallback remains (5% over 1h elapsed)
        let reset = NOW + 4 * 3600;
        let samples = linear_samples(NOW, reset - 7200, 40, 30.0, 0.9);
        let t = eta(&samples, NOW, 5.0, reset, 5 * 3600, None, |s| {
            (s.five, s.five_reset)
        })
        .unwrap();
        let expected = NOW + (95.0 / (5.0 / 3600.0)) as i64;
        assert!((t - expected).abs() < 5, "eta {t} vs expected {expected}");
    }

    #[test]
    fn harvest_closed_window_rate() {
        let cur_reset = NOW + 4 * 3600;
        let old_reset = NOW - 7200;
        // 30 min span, +18% -> 0.01 %/sec
        let mut samples = linear_samples(NOW - 7200 - 1800, old_reset, 30, 20.6, 0.6);
        samples.extend(linear_samples(NOW, cur_reset, 5, 1.0, 0.5));
        let mut rates = Vec::new();
        assert!(harvest_rates(&samples, cur_reset, &mut rates));
        assert_eq!(rates.len(), 1);
        assert_eq!(rates[0].0, old_reset);
        assert!((rates[0].1 - 0.01).abs() < 1e-6, "rate {}", rates[0].1);
    }

    #[test]
    fn harvest_skips_current_window_and_dedups() {
        let cur_reset = NOW + 4 * 3600;
        let old_reset = NOW - 7200;
        let mut samples = linear_samples(NOW - 7200 - 1800, old_reset, 30, 20.6, 0.6);
        samples.extend(linear_samples(NOW, cur_reset, 30, 1.0, 0.5));
        let mut rates = vec![(old_reset, 0.01)];
        // old window already harvested, current one must never be -> no change
        assert!(!harvest_rates(&samples, cur_reset, &mut rates));
        assert_eq!(rates.len(), 1);
    }

    #[test]
    fn harvest_rejects_short_or_flat_windows() {
        let cur_reset = NOW + 4 * 3600;
        // 10 min span (< HARVEST_MIN_SPAN)
        let short = linear_samples(NOW - 7200, NOW - 3600, 10, 20.0, 0.6);
        // 30 min span but only +1.5% (< HARVEST_MIN_PCT)
        let flat = linear_samples(NOW - 7200, NOW - 7000, 30, 20.0, 0.05);
        let mut rates = Vec::new();
        assert!(!harvest_rates(&short, cur_reset, &mut rates));
        assert!(!harvest_rates(&flat, cur_reset, &mut rates));
        assert!(rates.is_empty());
    }

    #[test]
    fn harvest_caps_history_length() {
        let mut rates: Vec<(i64, f64)> = (0..RATES_KEEP as i64 + 5).map(|i| (i, 0.01)).collect();
        assert!(harvest_rates(&[], 999_999, &mut rates));
        assert_eq!(rates.len(), RATES_KEEP);
        assert_eq!(rates[0].0, 5); // oldest entries dropped
    }

    #[test]
    fn push_times_overshoot_in_hours() {
        let mut out = String::new();
        let reset = NOW;
        push_times(&mut out, Some(reset + 7 * 3600 + 1800), reset, NOW, "%H:%M");
        assert!(out.contains("(+8h / "), "got {out:?}"); // ceil(7.5h)
    }

    #[test]
    fn push_times_clock_time_before_reset() {
        let mut out = String::new();
        // 30 min out (>= URGENT_SECS) -> plain clock time, no signal color
        push_times(&mut out, Some(NOW + 1800), NOW + 3600, NOW, "%H:%M");
        assert!(out.contains("(~"), "got {out:?}");
        assert!(out.contains(" / "), "got {out:?}");
        assert!(!out.contains(URGENT), "got {out:?}");
    }

    #[test]
    fn push_times_urgent_eta_is_colored() {
        let mut out = String::new();
        // 10 min out (< URGENT_SECS) -> ETA clock time wrapped in the signal color
        push_times(&mut out, Some(NOW + 600), NOW + 3600, NOW, "%H:%M");
        assert!(
            out.contains(URGENT) && out.contains(FG_RESET),
            "got {out:?}"
        );
        // color must not leak past the ETA into the reset time
        assert!(out.contains(&format!("{FG_RESET} / ")), "got {out:?}");
    }

    #[test]
    fn push_times_reset_only_without_eta() {
        let mut out = String::new();
        push_times(&mut out, None, NOW, NOW, "%H:%M");
        assert!(!out.contains('~') && !out.contains('+'), "got {out:?}");
        assert!(out.starts_with(" (") && out.ends_with(')'), "got {out:?}");
    }

    #[test]
    fn parse_cpu_totals_sums_and_idles() {
        // user nice system idle iowait irq softirq steal
        let stat = "cpu  100 0 50 800 40 5 5 0\ncpu0 50 0 25 400 20 2 3 0\n";
        assert_eq!(parse_cpu_totals(stat), Some((1000, 840)));
        // pre-2.6 kernels: only 4 fields, no iowait
        assert_eq!(parse_cpu_totals("cpu 10 0 10 80\n"), Some((100, 80)));
        assert_eq!(parse_cpu_totals("cpu0 1 2 3 4\n"), None);
        assert_eq!(parse_cpu_totals("cpu 1 2 3\n"), None);
    }

    #[test]
    fn meminfo_pct_uses_available() {
        let m = "MemTotal:       16000000 kB\nMemFree:         1000000 kB\nMemAvailable:    4000000 kB\n";
        assert_eq!(meminfo_pct(m), Some(75));
        assert_eq!(meminfo_pct("MemTotal: 16000000 kB\n"), None);
        assert_eq!(meminfo_pct(""), None);
    }

    #[test]
    fn fmt_bytes_units() {
        assert_eq!(fmt_bytes(500 << 20), "500M");
        assert_eq!(fmt_bytes(5 << 30), "5.0G");
        assert_eq!(fmt_bytes(897 << 30), "897G");
        assert_eq!(fmt_bytes(1536 << 30), "1.5T");
    }

    #[test]
    fn parse_gpu_csv_first_gpu() {
        assert_eq!(parse_gpu_csv("7, 2126, 16303\n"), Some((7, 13)));
        // multi-GPU: first line wins
        assert_eq!(
            parse_gpu_csv("50, 8000, 16000\n10, 1, 16000\n"),
            Some((50, 50))
        );
        assert_eq!(parse_gpu_csv(""), None);
        assert_eq!(parse_gpu_csv("[N/A], 0, 0\n"), None);
        assert_eq!(parse_gpu_csv("7, 2126, 0\n"), None);
    }

    #[test]
    fn disp_width_counts_cells_not_bytes() {
        assert_eq!(disp_width("abc"), 3);
        assert_eq!(disp_width("🧠 3%"), 5); // emoji = 2 cells
        assert_eq!(disp_width("⏳ 41%"), 6); // U+23F3 renders wide
        assert_eq!(disp_width("\x1b[1m⏳ 41%\x1b[0m"), 6); // ANSI = 0 cells
        assert_eq!(disp_width("🕐 09:48"), 8);
        assert_eq!(disp_width("⚙\u{fe0f}"), 1); // variation selector = 0 cells
    }

    #[test]
    fn parse_email_extracts_oauth_account() {
        let j =
            r#"{"a":"b","oauthAccount":{"accountUuid":"u","emailAddress":"me@example.com"},"c":1}"#;
        assert_eq!(parse_email(j), Some("me@example.com"));
        // an emailAddress before oauthAccount must not be picked up
        let j2 = r#"{"emailAddress":"decoy@x.io","oauthAccount":{"emailAddress":"real@x.io"}}"#;
        assert_eq!(parse_email(j2), Some("real@x.io"));
        assert_eq!(parse_email("{}"), None);
        assert_eq!(parse_email(r#"{"oauthAccount":{}}"#), None);
    }

    #[test]
    fn load_samples_skips_malformed_lines() {
        let path = std::env::temp_dir().join("statusline-rs-test-samples.tsv");
        std::fs::write(
            &path,
            "100 1.5 200 2.5 300\ngarbage\n101 1.6 200 -1 0\n1 2 3\n",
        )
        .unwrap();
        let s = load_samples(&path);
        std::fs::remove_file(&path).ok();
        assert_eq!(s.len(), 2);
        assert_eq!(s[1].ts, 101);
        assert_eq!(s[1].seven, -1.0);
    }
}
