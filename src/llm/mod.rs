//! Optional local LLM abstraction (ADR-005: the LLM must never define LKOS).
//!
//! The engine is fully functional without any provider. When one is
//! configured, it powers: document summaries, grounded answering (`ask`),
//! and future enrichment. v0.1 ships:
//! - [`NullProvider`] — default; reports "not configured"
//! - [`LlamaCppProvider`] — isolated llama.cpp subprocess, prompt passed via
//!   temp file (avoids Windows command-line length limits), hard timeout,
//!   guaranteed temp-file cleanup.
//!
//! Tests use an in-tree fake provider via the same trait.

use crate::error::{LkosError, Result};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// A local LLM provider.
pub trait LlmProvider: Send + Sync {
    /// Provider name.
    fn name(&self) -> &str;
    /// Generate a completion for a prompt (non-streaming v0.1).
    fn generate(&self, prompt: &str, max_tokens: u32, temperature: f32) -> Result<String>;
}

/// Default provider: LLM features disabled (the engine still works).
pub struct NullProvider;

impl LlmProvider for NullProvider {
    fn name(&self) -> &str {
        "null"
    }

    fn generate(&self, _prompt: &str, _max_tokens: u32, _temperature: f32) -> Result<String> {
        Err(LkosError::Llm(
            "no LLM provider configured; LKOS search/retrieval still works — \
             configure a provider for summaries and `ask`"
                .into(),
        ))
    }
}

/// A fake provider for tests and examples.
pub struct FakeProvider {
    /// Deterministic response builder used by tests.
    pub response: String,
}

impl LlmProvider for FakeProvider {
    fn name(&self) -> &str {
        "fake"
    }

    fn generate(&self, prompt: &str, _max_tokens: u32, _temperature: f32) -> Result<String> {
        Ok(format!(
            "{} [echo of {} chars]",
            self.response,
            prompt.len()
        ))
    }
}

/// llama.cpp subprocess provider (`llama-cli` compatible CLIs).
pub struct LlamaCppProvider {
    /// Path to the CLI binary.
    pub binary: PathBuf,
    /// Path to the GGUF model file.
    pub model: PathBuf,
    /// Extra CLI args (e.g. `-ngl 99`).
    pub extra_args: Vec<String>,
    /// Timeout in seconds.
    pub timeout: Duration,
    cancelled: Arc<AtomicBool>,
}

impl LlamaCppProvider {
    /// Configure a provider.
    pub fn new(binary: impl Into<PathBuf>, model: impl Into<PathBuf>, timeout_secs: u64) -> Self {
        LlamaCppProvider {
            binary: binary.into(),
            model: model.into(),
            extra_args: Vec::new(),
            timeout: Duration::from_secs(timeout_secs),
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Add an extra CLI argument.
    pub fn arg(mut self, a: impl Into<String>) -> Self {
        self.extra_args.push(a.into());
        self
    }

    /// Cooperative cancel flag (checked between spawn and completion).
    pub fn cancel_handle(&self) -> Arc<AtomicBool> {
        self.cancelled.clone()
    }
}

impl LlmProvider for LlamaCppProvider {
    fn name(&self) -> &str {
        "llama-cpp"
    }

    #[allow(clippy::too_many_lines)]
    fn generate(&self, prompt: &str, max_tokens: u32, temperature: f32) -> Result<String> {
        if self.cancelled.load(Ordering::Relaxed) {
            return Err(LkosError::Cancelled);
        }
        // Prompt via temp file: avoids command-line length limits (Windows).
        // The file is removed deterministically when `prompt_file` drops.
        let prompt_file = TempFile::create(prompt)?;

        let mut cmd = std::process::Command::new(&self.binary);
        cmd.arg("-m")
            .arg(&self.model)
            .arg("-f")
            .arg(&prompt_file.0)
            .arg("-n")
            .arg(max_tokens.to_string())
            .arg("--temp")
            .arg(format!("{temperature}"))
            .arg("-no-cnv")
            .args(&self.extra_args)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null());

        let mut child = cmd.spawn().map_err(|e| {
            LkosError::Llm(format!("failed to launch {}: {e}", self.binary.display()))
        })?;

        let mut stdout = child
            .stdout
            .take()
            .ok_or_else(|| LkosError::Llm("no stdout".into()))?;

        // Read stdout on a helper thread while we poll for exit/timeout.
        let reader = std::thread::spawn(move || -> std::io::Result<String> {
            use std::io::Read;
            let mut buf = String::new();
            stdout.read_to_string(&mut buf)?;
            Ok(buf)
        });

        let start = Instant::now();
        let status = loop {
            if let Some(status) = child.try_wait()? {
                break Ok(status);
            }
            if start.elapsed() > self.timeout {
                let _ = child.kill();
                break Err("timeout");
            }
            std::thread::sleep(Duration::from_millis(50));
        };

        // Join exactly once (both on success and on timeout paths). After a
        // kill, stdout closes so the reader thread terminates promptly.
        let output = reader
            .join()
            .unwrap_or_else(|_| Ok(String::new()))
            .unwrap_or_else(|_| String::new());

        match status {
            Ok(s) if s.success() => Ok(output),
            Ok(s) => Err(LkosError::Llm(format!(
                "llama.cpp exited with {s} (prompt file: {})",
                prompt_file.0.display()
            ))),
            Err("timeout") => Err(LkosError::Llm(format!(
                "llama.cpp timed out after {}s",
                self.timeout.as_secs()
            ))),
            Err(e) => Err(LkosError::Llm(format!("llama.cpp failure: {e}"))),
        }
    }
}

/// A temp file removed on drop.
struct TempFile(std::path::PathBuf);

impl TempFile {
    fn create(content: &str) -> Result<TempFile> {
        let dir = std::env::temp_dir().join("lkos-prompts");
        std::fs::create_dir_all(&dir)?;
        let path = dir.join(format!(
            "prompt-{}-{}.txt",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        std::fs::write(&path, content)?;
        Ok(TempFile(path))
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Build a grounded answer prompt (ASCII only — small models, no emoji).
pub fn grounded_prompt(question: &str, context: &str) -> String {
    format!(
        "You are a precise assistant. Answer ONLY using the context below. \
Cite sources as [doc:section]. If the context is insufficient, say: \
I cannot find this in your documents.\n\nCONTEXT:\n{context}\n\nQUESTION: {question}\n\nAnswer in English:",
    )
}
