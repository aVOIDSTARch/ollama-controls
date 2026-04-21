//! Remote control of Ollama via its HTTP API (`OLLAMA_HOST`, default `http://127.0.0.1:11434`).
//! Mirrors the operations described in programmatic model management guides.

use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader};

/// Default port for Ollama when only a host is given.
pub const DEFAULT_OLLAMA_PORT: u16 = 11434;

fn trim_slash(s: &str) -> &str {
    s.trim_end_matches('/')
}

/// Build a base URL such as `http://127.0.0.1:11434` from host (and optional port).
pub fn ollama_base_url(host: &str, port: Option<u16>) -> String {
    let host = host.trim();
    if host.starts_with("http://") || host.starts_with("https://") {
        return trim_slash(host).to_string();
    }
    let port = port.unwrap_or(DEFAULT_OLLAMA_PORT);
    format!("http://{}:{}", host.trim_start_matches("http://").trim_start_matches("https://"), port)
}

/// Client for [`Ollama` HTTP API](https://github.com/ollama/ollama/blob/main/docs/api.md).
#[derive(Debug, Clone)]
pub struct OllamaClient {
    base_url: String,
    agent: ureq::Agent,
}

impl OllamaClient {
    pub fn new(base_url: impl Into<String>) -> Self {
        let base_url = trim_slash(&base_url.into()).to_string();
        Self {
            base_url,
            agent: ureq::Agent::new(),
        }
    }

    /// `http://127.0.0.1:11434`
    pub fn localhost() -> Self {
        Self::new(ollama_base_url("127.0.0.1", None))
    }

    /// Uses [`std::env::var`] `OLLAMA_HOST` when set (same as the `ollama` CLI), otherwise [`localhost`](Self::localhost).
    pub fn from_env() -> Self {
        match std::env::var("OLLAMA_HOST") {
            Ok(host) if !host.is_empty() => Self::new(host),
            _ => Self::localhost(),
        }
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }

    /// `GET /api/tags` — list installed models.
    pub fn list_models(&self) -> Result<Vec<ListedModel>, String> {
        let r: TagsResponse = self
            .agent
            .get(&self.url("/api/tags"))
            .call()
            .map_err(|e| e.to_string())?
            .into_json()
            .map_err(|e| e.to_string())?;
        Ok(r.models)
    }

    /// `POST /api/show` — full model metadata (license, modelfile, template, tensors, …).
    pub fn show(&self, name: &str) -> Result<ShowResponse, String> {
        self.agent
            .post(&self.url("/api/show"))
            .send_json(ShowRequest { name: name.to_string() })
            .map_err(|e| e.to_string())?
            .into_json()
            .map_err(|e| e.to_string())
    }

    /// `POST /api/pull` with `stream: false` — wait until pull finishes (no progress lines).
    pub fn pull_blocking(&self, name: &str) -> Result<(), String> {
        self.pull_stream(name, |_line| Ok(()))
    }

    /// `POST /api/pull` with `stream: true` — invokes `on_line` for each NDJSON status object.
    /// Stops with `Err` if a line contains an `error` field.
    pub fn pull_stream<F>(&self, name: &str, mut on_line: F) -> Result<(), String>
    where
        F: FnMut(&PullProgressLine) -> Result<(), String>,
    {
        self.stream_post_json(
            "/api/pull",
            &PullRequest {
                name: name.to_string(),
                stream: true,
            },
            move |line: &PullProgressLine| {
                if let Some(ref err) = line.error {
                    return Err(err.clone());
                }
                on_line(line)
            },
        )
    }

    /// `DELETE /api/delete`
    pub fn delete_model(&self, name: &str) -> Result<(), String> {
        self.agent
            .delete(&self.url("/api/delete"))
            .send_json(DeleteRequest { name: name.to_string() })
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    /// `POST /api/copy`
    pub fn copy_model(&self, source: &str, destination: &str) -> Result<(), String> {
        self.agent
            .post(&self.url("/api/copy"))
            .send_json(CopyRequest {
                source: source.to_string(),
                destination: destination.to_string(),
            })
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    /// `POST /api/create` with streaming. Parses a leading `FROM <model>` line and sends
    /// `from` plus the remainder as `modelfile` (required by Ollama 0.21+). Whitespace before `FROM` is ignored.
    pub fn create_model<F>(&self, name: &str, modelfile: &str, on_line: F) -> Result<(), String>
    where
        F: FnMut(&serde_json::Value) -> Result<(), String>,
    {
        let (from, rest) = split_modelfile_from(modelfile)?;
        self.create_model_inner(
            CreateRequest {
                name: name.to_string(),
                from: Some(from),
                modelfile: rest,
                stream: true,
            },
            on_line,
        )
    }

    /// Create a tag pointing at an existing model (no custom Modelfile). Same as `ollama create X --from Y`.
    pub fn create_from_base<F>(&self, new_name: &str, from_model: &str, on_line: F) -> Result<(), String>
    where
        F: FnMut(&serde_json::Value) -> Result<(), String>,
    {
        self.create_model_inner(
            CreateRequest {
                name: new_name.to_string(),
                from: Some(from_model.to_string()),
                modelfile: None,
                stream: true,
            },
            on_line,
        )
    }

    fn create_model_inner<F>(&self, body: CreateRequest, mut on_line: F) -> Result<(), String>
    where
        F: FnMut(&serde_json::Value) -> Result<(), String>,
    {
        self.stream_ndjson("/api/create", &body, move |v| {
            if let Some(err) = v.get("error").and_then(|e| e.as_str()) {
                return Err(err.to_string());
            }
            on_line(v)
        })
    }

    /// `GET /api/ps` — models loaded in memory.
    pub fn list_running(&self) -> Result<Vec<RunningModel>, String> {
        let r: PsResponse = self
            .agent
            .get(&self.url("/api/ps"))
            .call()
            .map_err(|e| e.to_string())?
            .into_json()
            .map_err(|e| e.to_string())?;
        Ok(r.models)
    }

    /// Unload a model from memory via `POST /api/generate` with `keep_alive: 0` (empty prompt).
    pub fn unload_model(&self, model: &str) -> Result<(), String> {
        self.agent
            .post(&self.url("/api/generate"))
            .send_json(GenerateUnloadRequest {
                model: model.to_string(),
                prompt: String::new(),
                keep_alive: 0,
            })
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Re-pull every installed tag (same idea as `update_all_models.sh` in the blog).
    pub fn update_all_models<F>(&self, mut on_pull: F) -> Result<Vec<String>, String>
    where
        F: FnMut(&str, &PullProgressLine) -> Result<(), String>,
    {
        let names: Vec<String> = self
            .list_models()?
            .into_iter()
            .map(|m| m.name)
            .collect();
        for name in &names {
            self.pull_stream(name, |line| on_pull(name, line))?;
        }
        Ok(names)
    }

    /// Remove every installed model whose name is not in `keep` (blog `cleanup_models.sh` pattern).
    pub fn remove_all_except(&self, keep: &[&str]) -> Result<Vec<String>, String> {
        let mut removed = Vec::new();
        for m in self.list_models()? {
            if keep.contains(&m.name.as_str()) {
                continue;
            }
            self.delete_model(&m.name)?;
            removed.push(m.name);
        }
        Ok(removed)
    }

    fn stream_post_json<Req, Line, F>(&self, path: &str, body: &Req, mut on_line: F) -> Result<(), String>
    where
        Req: Serialize,
        Line: serde::de::DeserializeOwned,
        F: FnMut(&Line) -> Result<(), String>,
    {
        let resp = self
            .agent
            .post(&self.url(path))
            .send_json(body)
            .map_err(|e| e.to_string())?;
        read_ndjson_lines(resp.into_reader(), move |t| {
            let parsed: Line = serde_json::from_str(t).map_err(|e| format!("{e}: {t}"))?;
            on_line(&parsed)
        })
    }

    fn stream_ndjson<Req, F>(&self, path: &str, body: &Req, mut on_line: F) -> Result<(), String>
    where
        Req: Serialize,
        F: FnMut(&serde_json::Value) -> Result<(), String>,
    {
        let resp = self
            .agent
            .post(&self.url(path))
            .send_json(body)
            .map_err(|e| e.to_string())?;
        read_ndjson_lines(resp.into_reader(), move |t| {
            let v: serde_json::Value =
                serde_json::from_str(t).map_err(|e| format!("{e}: {t}"))?;
            on_line(&v)
        })
    }
}

fn read_ndjson_lines<R: std::io::Read, F>(reader: R, mut on_line: F) -> Result<(), String>
where
    F: FnMut(&str) -> Result<(), String>,
{
    let mut buf = BufReader::new(reader);
    let mut line = String::new();
    loop {
        line.clear();
        let n = buf.read_line(&mut line).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        on_line(t)?;
    }
    Ok(())
}

/// Splits Modelfile text into the base model name (for the `from` field) and the rest (parameters, SYSTEM, …).
pub fn split_modelfile_from(content: &str) -> Result<(String, Option<String>), String> {
    let trimmed = content.trim_start();
    if trimmed.is_empty() {
        return Err("modelfile is empty".into());
    }
    let mut lines = trimmed.lines();
    let first = lines.next().unwrap().trim();
    let upper = first.to_ascii_uppercase();
    let prefix = "FROM ";
    if !upper.starts_with(prefix) {
        return Err(format!(
            "modelfile must start with `FROM <model>` (first line: {first:?})"
        ));
    }
    let base = first[prefix.len()..].trim().to_string();
    if base.is_empty() {
        return Err("FROM line must name a model".into());
    }
    let rest: String = lines.map(|l| l.to_string()).collect::<Vec<_>>().join("\n");
    let rest_opt = if rest.trim().is_empty() {
        None
    } else {
        Some(rest + "\n")
    };
    Ok((base, rest_opt))
}

// --- Request bodies ---

#[derive(Serialize)]
struct ShowRequest {
    name: String,
}

#[derive(Serialize)]
struct PullRequest {
    name: String,
    stream: bool,
}

#[derive(Serialize)]
struct DeleteRequest {
    name: String,
}

#[derive(Serialize)]
struct CopyRequest {
    source: String,
    destination: String,
}

#[derive(Serialize)]
struct CreateRequest {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    from: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    modelfile: Option<String>,
    stream: bool,
}

#[derive(Serialize)]
struct GenerateUnloadRequest {
    model: String,
    prompt: String,
    keep_alive: u32,
}

// --- Responses ---

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TagsResponse {
    pub models: Vec<ListedModel>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ListedModel {
    pub name: String,
    pub model: String,
    pub modified_at: String,
    pub size: u64,
    pub digest: String,
    pub details: Option<ModelTagDetails>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelTagDetails {
    #[serde(default)]
    pub parent_model: String,
    pub format: Option<String>,
    pub family: Option<String>,
    pub families: Option<Vec<String>>,
    pub parameter_size: Option<String>,
    pub quantization_level: Option<String>,
}

impl ListedModel {
    pub fn size_gb(&self) -> f64 {
        self.size as f64 / (1024.0_f64.powi(3))
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ShowResponse {
    pub license: Option<String>,
    pub modelfile: Option<String>,
    pub parameters: Option<String>,
    pub template: Option<String>,
    pub details: Option<serde_json::Value>,
    pub model_info: Option<serde_json::Value>,
    pub tensors: Option<serde_json::Value>,
    pub capabilities: Option<Vec<String>>,
    pub modified_at: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PullProgressLine {
    pub status: String,
    #[serde(default)]
    pub digest: Option<String>,
    #[serde(default)]
    pub total: Option<u64>,
    #[serde(default)]
    pub completed: Option<u64>,
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PsResponse {
    pub models: Vec<RunningModel>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RunningModel {
    pub name: String,
    pub model: String,
    pub size: u64,
    pub digest: String,
    pub details: Option<ModelTagDetails>,
    #[serde(default)]
    pub expires_at: Option<String>,
    #[serde(default)]
    pub size_vram: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ollama_base_url_parses() {
        assert_eq!(
            ollama_base_url("192.168.1.5", Some(8080)),
            "http://192.168.1.5:8080"
        );
        assert_eq!(
            ollama_base_url("http://10.0.0.1:11434", None),
            "http://10.0.0.1:11434"
        );
    }

    #[test]
    fn split_modelfile_from_parses_from_line() {
        let (base, rest) =
            split_modelfile_from("FROM gemma3:latest\nSYSTEM \"x\"\n").unwrap();
        assert_eq!(base, "gemma3:latest");
        assert_eq!(rest.as_deref(), Some("SYSTEM \"x\"\n"));
    }
}
