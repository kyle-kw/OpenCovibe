use crate::agent::pipe_parser::{CodexStdoutParser, PipeStdoutParser};
use crate::models::{ChatDelta, ChatDone, RunEventType};
use crate::process_ext::HideConsole;
use crate::storage;
use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;
use tauri::{AppHandle, Emitter};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

pub type ProcessMap = Arc<Mutex<HashMap<String, Child>>>;

pub fn new_process_map() -> ProcessMap {
    Arc::new(Mutex::new(HashMap::new()))
}

/// Emit a chunk of Claude stdout: append to assistant_text + persist + Tauri emit.
fn emit_claude_stdout(run_id: &str, app: &AppHandle, assistant: &mut String, text: String) {
    assistant.push_str(&text);
    if let Err(e) = storage::events::append_event(
        run_id,
        RunEventType::Stdout,
        serde_json::json!({ "text": text.clone(), "source": "ui_chat" }),
    ) {
        log::warn!("[stream] stdout append failed: {}", e);
    }
    let _ = app.emit(
        "run-event",
        serde_json::json!({ "run_id": run_id, "type": "stdout", "text": text.clone() }),
    );
    let _ = app.emit("chat-delta", ChatDelta { text });
}

/// Streaming UTF-8 decoder for chunk-boundary correctness.
///
/// Concatenates `leftover` (previous incomplete trailing bytes) with `chunk`,
/// returns the longest valid-UTF-8 prefix as a `String` plus any trailing
/// partial multibyte sequence to defer to the next chunk. A naive
/// `String::from_utf8_lossy(&chunk[..n])` per read would emit U+FFFD on every
/// multibyte character that straddles the read boundary (common for CJK /
/// emoji output from `claude --print`).
///
/// On genuinely invalid UTF-8 (a byte error mid-buffer, not just an incomplete
/// tail), the whole combined buffer is lossy-decoded so the stream can make
/// progress instead of accumulating bad bytes forever.
fn decode_utf8_chunk(leftover: Vec<u8>, chunk: &[u8]) -> (String, Vec<u8>) {
    let mut combined = leftover;
    combined.extend_from_slice(chunk);
    match std::str::from_utf8(&combined) {
        Ok(s) => (s.to_string(), Vec::new()),
        Err(e) => {
            let valid_up_to = e.valid_up_to();
            match e.error_len() {
                None => {
                    // Incomplete trailing char — defer to next read.
                    let text = std::str::from_utf8(&combined[..valid_up_to])
                        .expect("valid_up_to bytes form valid UTF-8")
                        .to_string();
                    let tail = combined[valid_up_to..].to_vec();
                    (text, tail)
                }
                Some(_) => {
                    // Truly invalid bytes mid-buffer (corrupted source). Fall
                    // back to lossy decode so we don't get stuck.
                    (String::from_utf8_lossy(&combined).into_owned(), Vec::new())
                }
            }
        }
    }
}

pub async fn run_agent(
    app: AppHandle,
    process_map: ProcessMap,
    run_id: String,
    command: String,
    args: Vec<String>,
    cwd: String,
    agent: String,
) -> Result<(), String> {
    log::debug!(
        "[stream] run_agent: run_id={}, cmd={}, args={:?}, cwd={}, agent={}",
        run_id,
        command,
        args,
        cwd,
        agent
    );

    let emit_run_event = |rt: RunEventType, payload: serde_json::Value| {
        if let Err(e) = storage::events::append_event(&run_id, rt, payload) {
            log::warn!(
                "[stream] failed to append event for run_id={}: {}",
                run_id,
                e
            );
        }
    };

    // Log start
    emit_run_event(
        RunEventType::System,
        serde_json::json!({
            "message": format!("Started {} {}", command, args.join(" ")),
            "source": "ui_chat"
        }),
    );

    let mut child = Command::new(&command)
        .args(&args)
        .current_dir(&cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env("OPENCOVIBE_TASK_ID", &run_id)
        .env("OPENCOVIBE_RUN_ID", &run_id)
        .env_remove("CLAUDECODE") // Allow running inside a Claude Code session
        .hide_console()
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| {
            let msg = if e.kind() == std::io::ErrorKind::NotFound {
                format!(
                    "Command \"{}\" not found. Is {} CLI installed and in your PATH?",
                    command, agent
                )
            } else {
                e.to_string()
            };
            log::error!("[stream] spawn failed: {}", msg);
            msg
        })?;

    let pid = child.id().unwrap_or(0);
    log::debug!("[stream] spawned process: run_id={}, pid={}", run_id, pid);

    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();

    // Store child for stop_run
    {
        let mut map = process_map.lock().await;
        map.insert(run_id.clone(), child);
    }

    let run_id_out = run_id.clone();
    let run_id_err = run_id.clone();
    let app_out = app.clone();
    let agent_clone = agent.clone();

    // Stdout reader
    let stdout_handle = tokio::spawn(async move {
        let mut assistant_text = String::new();
        let is_codex = agent_clone == "codex";

        if is_codex {
            let mut parser = CodexStdoutParser;
            let reader = BufReader::new(stdout);
            let mut lines = reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if let Err(e) = storage::events::append_event(
                    &run_id_out,
                    RunEventType::Stdout,
                    serde_json::json!({ "text": line, "source": "ui_chat" }),
                ) {
                    log::warn!("[stream] stdout append failed: {}", e);
                }
                let _ = app_out.emit(
                    "run-event",
                    serde_json::json!({
                        "run_id": run_id_out,
                        "type": "stdout",
                        "text": line
                    }),
                );

                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if let Ok(payload) = serde_json::from_str::<serde_json::Value>(trimmed) {
                    // Capture thread_id as ConversationRef for Codex resume
                    let type_str = payload.get("type").and_then(|v| v.as_str()).unwrap_or("");
                    if type_str == "thread.started" {
                        if let Some(tid) = payload.get("thread_id").and_then(|v| v.as_str()) {
                            log::debug!("[codex] captured thread_id={} as conversation_ref", tid);
                            let tid_str = tid.to_string();
                            let rid = run_id_out.clone();
                            if let Err(e) = crate::storage::runs::with_meta(&rid, |meta| {
                                meta.conversation_ref =
                                    Some(crate::models::ConversationRef::CodexThread(tid_str));
                                Ok(())
                            }) {
                                log::warn!("[codex] failed to persist conversation_ref: {}", e);
                            }
                        }
                    }

                    // Use PipeStdoutParser trait for structured event → BusEvent
                    let events = parser.parse_line(&run_id_out, &payload);
                    for ev in &events {
                        if let crate::models::BusEvent::MessageDelta { text, .. } = ev {
                            assistant_text.push_str(text);
                            let _ = app_out.emit("chat-delta", ChatDelta { text: text.clone() });
                        }
                    }
                    if events.is_empty() && !type_str.is_empty() {
                        log::debug!("[codex] unhandled event: type={}", type_str);
                    }
                }
            }
        } else {
            // Claude: stdout is the response text. Decode UTF-8 across read
            // boundaries — multibyte characters can straddle a single read(),
            // so leftover trailing bytes are deferred to the next chunk to
            // avoid emitting U+FFFD for valid (just-split) input.
            let mut reader = BufReader::new(stdout);
            let mut buf = vec![0u8; 8192];
            let mut leftover: Vec<u8> = Vec::new();
            loop {
                match reader.read(&mut buf).await {
                    Ok(0) => {
                        // Flush any trailing incomplete bytes at EOF (lossy —
                        // CLI should always end on a char boundary, but a
                        // truncated last char shouldn't be silently dropped).
                        if !leftover.is_empty() {
                            let text = String::from_utf8_lossy(&leftover).into_owned();
                            leftover.clear();
                            emit_claude_stdout(&run_id_out, &app_out, &mut assistant_text, text);
                        }
                        break;
                    }
                    Ok(n) => {
                        let (text, new_leftover) =
                            decode_utf8_chunk(std::mem::take(&mut leftover), &buf[..n]);
                        leftover = new_leftover;
                        if !text.is_empty() {
                            emit_claude_stdout(&run_id_out, &app_out, &mut assistant_text, text);
                        }
                    }
                    Err(_) => break,
                }
            }
        }

        assistant_text
    });

    // Stderr reader
    let app_err = app.clone();
    let stderr_handle = tokio::spawn(async move {
        let mut stderr_text = String::new();
        let reader = BufReader::new(stderr);
        let mut lines = reader.lines();
        while let Ok(Some(line)) = lines.next_line().await {
            stderr_text.push_str(&line);
            stderr_text.push('\n');
            if let Err(e) = storage::events::append_event(
                &run_id_err,
                RunEventType::Stderr,
                serde_json::json!({ "text": line, "source": "ui_chat" }),
            ) {
                log::warn!("[stream] stderr append failed: {}", e);
            }
            let _ = app_err.emit(
                "run-event",
                serde_json::json!({
                    "run_id": run_id_err,
                    "type": "stderr",
                    "text": line
                }),
            );
        }
        stderr_text
    });

    // Wait for stdout/stderr to close (= process exited or pipes broken).
    // This completes without holding the ProcessMap lock.
    let assistant_text = stdout_handle.await.unwrap_or_default();
    let _stderr_text = stderr_handle.await.unwrap_or_default();

    // Short lock: remove child from map, then wait() outside the lock.
    // If stop_process already removed+killed the child, we get None → exit_code -1.
    let removed_child = {
        let mut map = process_map.lock().await;
        map.remove(&run_id)
    };
    let exit_code = if let Some(mut child) = removed_child {
        match child.wait().await {
            Ok(status) => status.code().unwrap_or(1),
            Err(_) => 1,
        }
    } else {
        // Was killed by stop_run
        -1
    };

    // Save assistant event
    if !assistant_text.trim().is_empty() {
        emit_run_event(
            RunEventType::Assistant,
            serde_json::json!({ "text": assistant_text.trim(), "source": "ui_chat" }),
        );
    }

    log::debug!(
        "[stream] process exited: run_id={}, exit_code={}, output_len={}",
        run_id,
        exit_code,
        assistant_text.len()
    );

    // Update run status
    if exit_code == 0 {
        if let Err(e) = storage::runs::update_status(
            &run_id,
            crate::models::RunStatus::Completed,
            Some(0),
            None,
        ) {
            log::warn!("[stream] failed to update status to Completed: {}", e);
        }
    } else if exit_code == -1 {
        if let Err(e) = storage::runs::update_status(
            &run_id,
            crate::models::RunStatus::Stopped,
            None,
            Some("Stopped by user".to_string()),
        ) {
            log::warn!("[stream] failed to update status to Stopped: {}", e);
        }
    } else if let Err(e) = storage::runs::update_status(
        &run_id,
        crate::models::RunStatus::Failed,
        Some(exit_code),
        Some(format!("Exit code {}", exit_code)),
    ) {
        log::warn!("[stream] failed to update status to Failed: {}", e);
    }

    emit_run_event(
        RunEventType::System,
        serde_json::json!({ "message": format!("Process exited with code {}", exit_code), "source": "ui_chat" }),
    );

    let _ = app.emit(
        "chat-done",
        ChatDone {
            ok: exit_code == 0,
            code: exit_code,
            error: None,
        },
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_ascii_only_no_leftover() {
        let (text, leftover) = decode_utf8_chunk(Vec::new(), b"hello world");
        assert_eq!(text, "hello world");
        assert!(leftover.is_empty());
    }

    #[test]
    fn decode_complete_multibyte_no_leftover() {
        // "你好" = 0xE4 0xBD 0xA0 0xE5 0xA5 0xBD (6 bytes)
        let (text, leftover) = decode_utf8_chunk(Vec::new(), "你好".as_bytes());
        assert_eq!(text, "你好");
        assert!(leftover.is_empty());
    }

    #[test]
    fn decode_split_multibyte_defers_tail() {
        // "你好" split at byte 4: first chunk has "你" (3 bytes) + 1 byte of "好"
        let bytes = "你好".as_bytes();
        let (text, leftover) = decode_utf8_chunk(Vec::new(), &bytes[..4]);
        assert_eq!(text, "你", "valid prefix emitted");
        assert_eq!(leftover, vec![bytes[3]], "incomplete tail deferred");

        // Next read brings the remaining 2 bytes of "好"
        let (text2, leftover2) = decode_utf8_chunk(leftover, &bytes[4..]);
        assert_eq!(text2, "好");
        assert!(leftover2.is_empty());
    }

    #[test]
    fn decode_split_at_first_byte_of_multibyte() {
        // First read ends exactly at the start byte of "你"
        let bytes = "ab你cd".as_bytes(); // 2 ASCII + 3 multibyte + 2 ASCII
        let (text, leftover) = decode_utf8_chunk(Vec::new(), &bytes[..3]); // "ab" + first byte of 你
        assert_eq!(text, "ab");
        assert_eq!(leftover, vec![bytes[2]]);

        let (text2, leftover2) = decode_utf8_chunk(leftover, &bytes[3..]);
        assert_eq!(text2, "你cd");
        assert!(leftover2.is_empty());
    }

    #[test]
    fn decode_emoji_split_across_reads() {
        // "🦀" = 0xF0 0x9F 0xA6 0x80 (4 bytes)
        let bytes = "🦀".as_bytes();
        let (text1, leftover1) = decode_utf8_chunk(Vec::new(), &bytes[..2]);
        assert_eq!(text1, "");
        assert_eq!(leftover1, bytes[..2].to_vec());

        let (text2, leftover2) = decode_utf8_chunk(leftover1, &bytes[2..3]);
        assert_eq!(text2, "");
        assert_eq!(leftover2, bytes[..3].to_vec());

        let (text3, leftover3) = decode_utf8_chunk(leftover2, &bytes[3..]);
        assert_eq!(text3, "🦀");
        assert!(leftover3.is_empty());
    }

    #[test]
    fn decode_invalid_bytes_falls_back_to_lossy() {
        // 0xFF is never valid in UTF-8 — error_len is Some, not None.
        let (text, leftover) = decode_utf8_chunk(Vec::new(), &[b'a', 0xFF, b'b']);
        // Lossy decode replaces 0xFF with U+FFFD and consumes the whole buffer.
        assert!(text.contains('a'));
        assert!(text.contains('b'));
        assert!(text.contains('\u{FFFD}'));
        assert!(leftover.is_empty(), "lossy fallback clears leftover");
    }

    #[test]
    fn decode_carries_leftover_from_prior_chunk() {
        // Caller has already stashed the first byte of "你"; this chunk has the rest.
        let bytes = "你".as_bytes();
        let leftover_in = vec![bytes[0]];
        let (text, leftover_out) = decode_utf8_chunk(leftover_in, &bytes[1..]);
        assert_eq!(text, "你");
        assert!(leftover_out.is_empty());
    }
}

pub async fn stop_process(process_map: &ProcessMap, run_id: &str) -> bool {
    log::debug!("[stream] stop_process: run_id={}", run_id);
    // Short lock: remove child, then kill+wait outside the lock.
    let removed = {
        let mut map = process_map.lock().await;
        map.remove(run_id)
    };
    if let Some(mut child) = removed {
        let _ = child.kill().await;
        let _ = child.wait().await;
        log::debug!("[stream] stop_process: killed run_id={}", run_id);
        true
    } else {
        log::debug!("[stream] stop_process: no process for run_id={}", run_id);
        false
    }
}
