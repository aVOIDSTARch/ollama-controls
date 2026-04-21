//! Control Ollama locally (CLI wrappers) or remotely via [`api::OllamaClient`] on the HTTP API
//! (`OLLAMA_HOST`, default `http://127.0.0.1:11434`), matching common model-management flows
//! (pull, list, show, copy, create, delete, running processes, unload).

pub mod api;
pub mod config;

pub use api::{
    ollama_base_url, split_modelfile_from, ListedModel, ModelTagDetails, OllamaClient, PullProgressLine,
    PsResponse, RunningModel, ShowResponse, TagsResponse, DEFAULT_OLLAMA_PORT,
};

pub use config::{
    config_dir, models_path_info, ollama_start_serve, ollama_stop_serve, set_models_download_path,
    ModelsPathInfo, OllamaControlsSettings, OLLAMA_MODELS_ENV,
};

use serde::Serialize;
use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

/// Same as [`list_models`] — raw lines from `ollama list` (header plus one row per model).
pub fn list_local_downloaded_models() -> Result<Vec<String>, String> {
    list_models()
}

// List all downloaded models
pub fn list_models() -> Result<Vec<String>, String> {
    let output = Command::new("ollama")
        .arg("list")
        .output()
        .map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }
    Ok(
        String::from_utf8_lossy(&output.stdout)
            .to_string()
            .split('\n')
            .map(|s| s.to_string())
            .collect(),
    )
}

/// Identity fields used with `ollama pull` / `ollama show`; optional [`ModelDetails`] from `ollama show`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Model {
    pub name: String,
    pub param_count: String,
    pub variant: String,
    pub quantization: String,
    /// Populated when parsing `ollama show` output into structured sections.
    pub details: Option<ModelDetails>,
}

/// Parsed `ollama show` output: known sections are typed; anything else is preserved under [`ModelDetails::other`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct ModelDetails {
    pub model: Option<ModelSection>,
    pub capabilities: Option<CapabilitiesSection>,
    pub parameters: Option<ParametersSection>,
    pub license: Option<LicenseSection>,
    /// Section name → entries: `(key, Some(value))` for key–value lines, `(tag, None)` for single-token lines.
    pub other: HashMap<String, Vec<(String, Option<String>)>>,
}

/// The `Model` block from `ollama show` (architecture, size, quantization, etc.).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct ModelSection {
    pub architecture: Option<String>,
    /// Model size string from the manifest (e.g. `4.3B`), distinct from sampling [`ParametersSection`].
    pub parameters: Option<String>,
    pub context_length: Option<String>,
    pub embedding_length: Option<String>,
    pub quantization: Option<String>,
    pub extra: HashMap<String, String>,
}

/// Declared capabilities (e.g. `completion`, `vision`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct CapabilitiesSection {
    pub tags: Vec<String>,
    pub extra: HashMap<String, String>,
}

/// Default sampling / generation parameters embedded in the model.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct ParametersSection {
    pub temperature: Option<String>,
    pub top_k: Option<String>,
    pub top_p: Option<String>,
    pub stop: Option<String>,
    pub extra: HashMap<String, String>,
}

/// License block: mostly free-form lines; lines with a key–value split are also captured in [`LicenseSection::fields`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct LicenseSection {
    pub lines: Vec<String>,
    pub fields: HashMap<String, String>,
}

impl Model {
    pub fn new(
        name: String,
        param_count: String,
        variant: String,
        quantization: String,
    ) -> Self {
        Self {
            name,
            param_count,
            variant,
            quantization,
            details: None,
        }
    }

    /// Parses a tag like `name:param-variant-quantization` into identity fields. `details` is left unset.
    pub fn from_str(s: String) -> Result<Self, String> {
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() < 2 {
            return Err("expected name:param-variant-quantization".to_string());
        }
        let parts2: Vec<&str> = parts[1].split('-').collect();
        if parts2.len() < 3 {
            return Err("expected param-variant-quantization after ':'".to_string());
        }
        Ok(Self::new(
            parts[0].to_string(),
            parts2[0].to_string(),
            parts2[1].to_string(),
            parts2[2].to_string(),
        ))
    }

    pub fn get_extended_name(&self) -> String {
        format!(
            "{}:{}-{}-{}",
            self.name, self.param_count, self.variant, self.quantization
        )
    }

    pub fn with_details(mut self, details: ModelDetails) -> Self {
        self.details = Some(details);
        self
    }
}

impl ModelDetails {
    pub fn from_show_output(s: &str) -> Self {
        parse_model_show(s)
    }
}

// Download a model
pub fn download_model(model: &Model) -> Result<(), String> {
    let extended_name = model.get_extended_name();
    let output = Command::new("ollama")
        .arg("pull")
        .arg(extended_name)
        .output()
        .map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }
    Ok(())
}

// List all running models
pub fn list_running_models() -> Result<Vec<String>, String> {
    let output = Command::new("ollama")
        .arg("ps")
        .output()
        .map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }
    Ok(
        String::from_utf8_lossy(&output.stdout)
            .to_string()
            .split('\n')
            .map(|s| s.to_string())
            .collect(),
    )
}

// Inspect model details
pub fn inspect_model(model: &Model) -> Result<String, String> {
    let output = Command::new("ollama")
        .arg("show")
        .arg(&model.name)
        .output()
        .map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

/// Parses `ollama show` text into [`ModelDetails`]. On `Err`, returns default (empty) details.
pub fn to_model_details(results: Result<String, String>) -> ModelDetails {
    match results {
        Ok(s) => parse_model_show(&s),
        Err(_) => ModelDetails::default(),
    }
}

fn split_trait_value_line(line: &str) -> Option<(String, String)> {
    let pos = line.find("  ")?;
    let key = line[..pos].trim_end();
    let value = line[pos..].trim();
    if key.is_empty() {
        return None;
    }
    Some((key.to_string(), value.to_string()))
}

fn parse_model_show(output: &str) -> ModelDetails {
    let mut details = ModelDetails::default();
    let mut current_section: Option<String> = None;

    for line in output.lines() {
        let line = line.trim_end_matches('\r');
        if line.trim().is_empty() {
            continue;
        }

        if line.starts_with("  ") && !line.starts_with("    ") {
            current_section = Some(line.trim().to_string());
            continue;
        }

        if !line.starts_with("    ") {
            continue;
        }

        let Some(section) = current_section.as_ref() else {
            continue;
        };

        let content = line.trim_start();
        apply_show_line(&mut details, section, content);
    }

    details
}

fn apply_show_line(details: &mut ModelDetails, section: &str, content: &str) {
    if let Some((key, value)) = split_trait_value_line(content) {
        if value.is_empty() {
            match section {
                "Capabilities" => push_capability_tag(&mut details.capabilities, key),
                "License" => {
                    let l = details.license.get_or_insert_with(LicenseSection::default);
                    l.lines.push(key);
                }
                _ => push_other(&mut details.other, section, key, None),
            }
            return;
        }

        match section {
            "Model" => {
                let m = details.model.get_or_insert_with(ModelSection::default);
                match key.as_str() {
                    "architecture" => m.architecture = Some(value),
                    "parameters" => m.parameters = Some(value),
                    "context length" => m.context_length = Some(value),
                    "embedding length" => m.embedding_length = Some(value),
                    "quantization" => m.quantization = Some(value),
                    _ => {
                        m.extra.insert(key, value);
                    }
                }
            }
            "Capabilities" => {
                let c = details
                    .capabilities
                    .get_or_insert_with(CapabilitiesSection::default);
                c.extra.insert(key, value);
            }
            "Parameters" => {
                let p = details
                    .parameters
                    .get_or_insert_with(ParametersSection::default);
                match key.as_str() {
                    "temperature" => p.temperature = Some(value),
                    "top_k" => p.top_k = Some(value),
                    "top_p" => p.top_p = Some(value),
                    "stop" => p.stop = Some(value),
                    _ => {
                        p.extra.insert(key, value);
                    }
                }
            }
            "License" => {
                let l = details.license.get_or_insert_with(LicenseSection::default);
                l.fields.insert(key, value);
            }
            other => push_other(&mut details.other, other, key, Some(value)),
        }
        return;
    }

    let token = content.trim();
    if token.is_empty() {
        return;
    }

    match section {
        "Capabilities" => push_capability_tag(&mut details.capabilities, token.to_string()),
        "License" => {
            let l = details.license.get_or_insert_with(LicenseSection::default);
            l.lines.push(token.to_string());
        }
        other => push_other(&mut details.other, other, token.to_string(), None),
    }
}

fn push_capability_tag(slot: &mut Option<CapabilitiesSection>, tag: String) {
    let c = slot.get_or_insert_with(CapabilitiesSection::default);
    c.tags.push(tag);
}

fn push_other(
    other: &mut HashMap<String, Vec<(String, Option<String>)>>,
    section: &str,
    key: String,
    value: Option<String>,
) {
    other
        .entry(section.to_string())
        .or_default()
        .push((key, value));
}

/// Runs `ollama show` and parses into [`ModelDetails`].
pub fn inspect_model_details(model: &Model) -> Result<ModelDetails, String> {
    let text = inspect_model(model)?;
    Ok(parse_model_show(&text))
}

/// Attaches parsed [`ModelDetails`] to a copy of `model`.
pub fn model_with_inspect_details(model: &Model) -> Result<Model, String> {
    let d = inspect_model_details(model)?;
    Ok(Model {
        name: model.name.clone(),
        param_count: model.param_count.clone(),
        variant: model.variant.clone(),
        quantization: model.quantization.clone(),
        details: Some(d),
    })
}

// --- Local CLI parity (`ollama` binary) — for scripts and hosts without HTTP routing to the API ---

/// `ollama rm <name>`
pub fn ollama_rm(name: &str) -> Result<(), String> {
    let output = Command::new("ollama")
        .arg("rm")
        .arg(name)
        .output()
        .map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }
    Ok(())
}

/// `ollama cp <source> <destination>`
pub fn ollama_cp(source: &str, destination: &str) -> Result<(), String> {
    let output = Command::new("ollama")
        .arg("cp")
        .arg(source)
        .arg(destination)
        .output()
        .map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }
    Ok(())
}

/// `ollama pull <name>` — `name` is a full tag (e.g. `llama3.2:3b-instruct-q4_K_M`).
pub fn ollama_pull(name: &str) -> Result<(), String> {
    let output = Command::new("ollama")
        .arg("pull")
        .arg(name)
        .output()
        .map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }
    Ok(())
}

/// `ollama create <name> -f <path>`
pub fn ollama_create_from_file(name: &str, modelfile_path: &Path) -> Result<(), String> {
    let output = Command::new("ollama")
        .arg("create")
        .arg(name)
        .arg("-f")
        .arg(modelfile_path)
        .output()
        .map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }
    Ok(())
}

/// Re-pulls every model returned by [`list_models`] (same pattern as `update_all_models.sh` in the Ollama guides).
pub fn ollama_update_all_installed() -> Result<(), String> {
    let lines = list_models()?;
    for line in lines.iter().skip(1) {
        if line.trim().is_empty() {
            continue;
        }
        let name = line.split_whitespace().next().unwrap_or("");
        if name.is_empty() || name == "NAME" {
            continue;
        }
        ollama_pull(name)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_model_details_parses_sections_and_pairs() {
        let sample = r"  Model
    architecture        gemma3    
    context length      131072    

  Capabilities
    completion    
    vision        

  Parameters
    temperature    1                  
    unknown_param  x

  License
    Some license text
";
        let d = to_model_details(Ok(sample.to_string()));
        let m = d.model.as_ref().unwrap();
        assert_eq!(m.architecture.as_deref(), Some("gemma3"));
        assert_eq!(m.context_length.as_deref(), Some("131072"));

        let caps = d.capabilities.as_ref().unwrap();
        assert!(caps.tags.contains(&"completion".to_string()));
        assert!(caps.tags.contains(&"vision".to_string()));

        let p = d.parameters.as_ref().unwrap();
        assert_eq!(p.temperature.as_deref(), Some("1"));
        assert_eq!(p.extra.get("unknown_param"), Some(&"x".to_string()));

        let l = d.license.as_ref().unwrap();
        assert!(l.lines.iter().any(|s| s.contains("Some license text")));
    }

    #[test]
    fn to_model_details_err_returns_empty() {
        assert_eq!(to_model_details(Err("nope".to_string())), ModelDetails::default());
    }

    #[test]
    fn model_from_str_sets_details_none() {
        let m = Model::from_str("foo:7b-instruct-q4".to_string()).unwrap();
        assert!(m.details.is_none());
        assert_eq!(m.name, "foo");
    }
}
