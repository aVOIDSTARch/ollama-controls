//! Local Ollama daemon helpers and persisted download directory (`OLLAMA_MODELS`).
//!
//! Ollama reads [`OLLAMA_MODELS_ENV`] when the server **starts**. Changing the saved path only affects
//! processes you start via [`ollama_start_serve`] until you restart a system-managed Ollama with the same env.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Environment variable that sets where models are stored (see Ollama docs).
pub const OLLAMA_MODELS_ENV: &str = "OLLAMA_MODELS";

const SETTINGS_FILE: &str = "settings.json";
const SERVE_PID_FILE: &str = "ollama-serve.pid";

/// Persisted settings under [`config_dir`] / `settings.json`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OllamaControlsSettings {
    /// Directory for model blobs (maps to `OLLAMA_MODELS` when starting [`ollama_start_serve`]).
    pub models_download_path: Option<String>,
}

/// Effective and configured download locations for display / API.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelsPathInfo {
    /// Current process env `OLLAMA_MODELS`, if set.
    pub ollama_models_env: Option<String>,
    /// Path saved in `settings.json` (applied when starting via [`ollama_start_serve`]).
    pub configured_download_path: Option<String>,
    /// Typical default `~/.ollama/models` when `HOME` / `USERPROFILE` is set.
    pub default_home_models_dir: Option<String>,
    /// Shell snippet to export the **configured** path (if any).
    pub shell_export_configured: Option<String>,
}

/// `~/.config/ollama-controls` (or `%APPDATA%\ollama-controls` on Windows).
pub fn config_dir() -> Option<PathBuf> {
    if let Ok(base) = std::env::var("XDG_CONFIG_HOME") {
        let p = PathBuf::from(base).join("ollama-controls");
        return Some(p);
    }
    #[cfg(windows)]
    {
        if let Ok(app) = std::env::var("APPDATA") {
            return Some(PathBuf::from(app).join("ollama-controls"));
        }
    }
    #[cfg(not(windows))]
    {
        if let Ok(home) = std::env::var("HOME") {
            return Some(PathBuf::from(home).join(".config").join("ollama-controls"));
        }
    }
    None
}

fn settings_path() -> Option<PathBuf> {
    Some(config_dir()?.join(SETTINGS_FILE))
}

fn pid_path() -> Option<PathBuf> {
    Some(config_dir()?.join(SERVE_PID_FILE))
}

/// Load persisted settings; missing file returns default.
pub fn load_settings() -> OllamaControlsSettings {
    let Some(path) = settings_path() else {
        return OllamaControlsSettings::default();
    };
    let Ok(bytes) = fs::read(&path) else {
        return OllamaControlsSettings::default();
    };
    serde_json::from_slice(&bytes).unwrap_or_default()
}

fn save_settings_inner(settings: &OllamaControlsSettings) -> Result<(), String> {
    let dir = config_dir().ok_or_else(|| "could not resolve config directory".to_string())?;
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let path = dir.join(SETTINGS_FILE);
    let json = serde_json::to_string_pretty(settings).map_err(|e| e.to_string())?;
    fs::write(path, json).map_err(|e| e.to_string())
}

/// Save download directory for future [`ollama_start_serve`] calls and return updated info.
///
/// Creates the directory (and parents) if missing. Does **not** move existing data; see Ollama docs for relocating `~/.ollama`.
pub fn set_models_download_path(path: &Path) -> Result<(), String> {
    fs::create_dir_all(path).map_err(|e| format!("create_dir_all {}: {e}", path.display()))?;
    let canon = path
        .canonicalize()
        .unwrap_or_else(|_| path.to_path_buf());
    let mut s = load_settings();
    s.models_download_path = Some(canon.to_string_lossy().into_owned());
    save_settings_inner(&s)
}

/// Snapshot of env, file config, and default `~/.ollama/models`.
pub fn models_path_info() -> ModelsPathInfo {
    let ollama_models_env = std::env::var(OLLAMA_MODELS_ENV).ok();
    let configured = load_settings().models_download_path;
    let default_home_models_dir = default_models_dir_home();
    let shell_export_configured = configured.as_ref().map(|p| {
        format!(
            "export {}={}",
            OLLAMA_MODELS_ENV,
            shell_escape_path(p)
        )
    });
    ModelsPathInfo {
        ollama_models_env,
        configured_download_path: configured,
        default_home_models_dir,
        shell_export_configured,
    }
}

fn default_models_dir_home() -> Option<String> {
    #[cfg(windows)]
    {
        std::env::var("USERPROFILE").ok().map(|h| {
            PathBuf::from(h)
                .join(".ollama")
                .join("models")
                .to_string_lossy()
                .into_owned()
        })
    }
    #[cfg(not(windows))]
    {
        std::env::var("HOME").ok().map(|h| {
            PathBuf::from(h)
                .join(".ollama")
                .join("models")
                .to_string_lossy()
                .into_owned()
        })
    }
}

fn shell_escape_path(p: &str) -> String {
    if p.contains('\'') || p.chars().any(char::is_whitespace) {
        format!("'{}'", p.replace('\'', "'\"'\"'"))
    } else {
        p.to_string()
    }
}

/// Starts `ollama serve` detached, applies saved [`OllamaControlsSettings::models_download_path`] as
/// `OLLAMA_MODELS` (plus inherited environment), and records the PID for [`ollama_stop_serve`].
///
/// Returns the child PID. On drop, [`std::process::Child`] would wait on some platforms; the handle is
/// leaked so the daemon keeps running.
pub fn ollama_start_serve() -> Result<u32, String> {
    let settings = load_settings();
    let mut cmd = Command::new("ollama");
    cmd.arg("serve");
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::null());
    cmd.stderr(Stdio::null());
    cmd.envs(std::env::vars_os());
    if let Some(ref p) = settings.models_download_path {
        cmd.env(OLLAMA_MODELS_ENV, p);
    }
    let child = cmd.spawn().map_err(|e| e.to_string())?;
    let pid = child.id();
    std::mem::forget(child);
    if let Some(dir) = config_dir() {
        let _ = fs::create_dir_all(&dir);
        if let Some(pp) = pid_path() {
            let _ = fs::write(pp, pid.to_string());
        }
    }
    Ok(pid)
}

/// Stops a server previously started with [`ollama_start_serve`] (PID file), then best-effort `pkill` / `taskkill`.
pub fn ollama_stop_serve() -> Result<(), String> {
    if let Some(pp) = pid_path() {
        if let Ok(text) = fs::read_to_string(&pp) {
            if let Ok(pid) = text.trim().parse::<u32>() {
                #[cfg(unix)]
                {
                    let _ = Command::new("kill").arg(pid.to_string()).status();
                }
                #[cfg(windows)]
                {
                    let _ = Command::new("taskkill")
                        .args(["/PID", &pid.to_string(), "/F"])
                        .status();
                }
            }
        }
        let _ = fs::remove_file(&pp);
    }
    ollama_stop_serve_fallback()
}

fn ollama_stop_serve_fallback() -> Result<(), String> {
    #[cfg(unix)]
    {
        let st = Command::new("pkill")
            .args(["-f", "ollama serve"])
            .status()
            .map_err(|e| e.to_string())?;
        if st.success() || st.code() == Some(1) {
            return Ok(());
        }
        return Err("could not stop ollama serve (pkill failed)".into());
    }
    #[cfg(windows)]
    {
        let st = Command::new("taskkill")
            .args(["/F", "/IM", "ollama.exe"])
            .status()
            .map_err(|e| e.to_string())?;
        if st.success() || st.code() == Some(128) {
            return Ok(());
        }
        Err("could not stop Ollama (taskkill)".into())
    }
}
