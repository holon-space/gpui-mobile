//! Native Android text-input bridge.
//!
//! A hidden `GpuiTextInputView` (a Java `View` hosting the platform IME) drives
//! editing through its `InputConnection` and forwards exact edit commands here
//! over JNI. Those commands are queued and applied to GPUI's
//! `PlatformInputHandler` from the frame callback (the only place the handler
//! can be touched on the main thread). State flows back to Java via
//! [`sync_state_to_java`] so the IME mirror stays in sync with GPUI's buffer.

use std::{
    collections::VecDeque,
    ffi::c_void,
    ops::Range,
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex, OnceLock,
    },
};

use gpui::PlatformInputHandler;
use jni::objects::{JObject, JValue};

use crate::android::jni::{self as jni_helpers, JniExt};

#[derive(Debug, Clone)]
pub enum TextInputCommand {
    ReplaceText {
        range_utf16: Range<usize>,
        text: String,
    },
    SetMarkedText {
        range_utf16: Range<usize>,
        text: String,
        selected_utf16: Range<usize>,
    },
    UnmarkText,
    /// Soft-keyboard Return. Queued alongside the edits so it keeps its place
    /// in the typing order, but delivered to gpui as an `enter` keystroke
    /// rather than as text — a newline written into the buffer would never
    /// reach the keymap, and the editor's `Enter` action would never run.
    KeyEnter,
}

/// An IME call to make from the dedicated JNI thread.
enum ImeCommand {
    ShowKeyboard(crate::KeyboardType),
    HideKeyboard,
    UpdateEditingState {
        text: String,
        selection: (i32, i32, bool),
        marked: (i32, i32),
    },
}

static IME_TX: OnceLock<std::sync::mpsc::Sender<ImeCommand>> = OnceLock::new();

/// Hand an IME call to the dedicated JNI thread.
///
/// These calls must NOT run on `android_main`. By the time a tap reaches the
/// keyboard the gpui draw and input-dispatch frames have consumed most of that
/// thread's stack, and ART refuses any Java upcall whose stack reserve check
/// fails — it throws `StackOverflowError`, which on some devices cannot even be
/// formatted without throwing again. A dedicated thread with its own large stack
/// removes the depth sensitivity instead of hoping the remaining stack is enough.
fn send_ime(command: ImeCommand) {
    let sender = IME_TX.get_or_init(spawn_ime_thread);
    if let Err(e) = sender.send(command) {
        log::error!("IME JNI thread is not accepting commands: {e}");
    }
}

fn spawn_ime_thread() -> std::sync::mpsc::Sender<ImeCommand> {
    let (tx, rx) = std::sync::mpsc::channel::<ImeCommand>();
    let started = std::thread::Builder::new()
        .name("gpui-ime-jni".to_owned())
        .stack_size(8 * 1024 * 1024)
        .spawn(move || {
            // The channel is FIFO and this is its only consumer, so IME calls
            // keep the order the editor issued them in.
            for command in rx {
                match command {
                    ImeCommand::ShowKeyboard(keyboard_type) => call_show_keyboard(keyboard_type),
                    ImeCommand::HideKeyboard => call_hide_keyboard(),
                    ImeCommand::UpdateEditingState {
                        text,
                        selection,
                        marked,
                    } => call_update_editing_state(&text, selection, marked),
                }
            }
            log::error!("IME JNI thread exiting: every sender was dropped");
        });
    if let Err(e) = started {
        log::error!("could not start the IME JNI thread, the keyboard will not work: {e}");
    }
    tx
}

static COMMANDS: OnceLock<Mutex<VecDeque<TextInputCommand>>> = OnceLock::new();
static DIRTY: AtomicBool = AtomicBool::new(false);

fn commands() -> &'static Mutex<VecDeque<TextInputCommand>> {
    COMMANDS.get_or_init(|| Mutex::new(VecDeque::new()))
}

pub fn push(command: TextInputCommand) {
    commands()
        .lock()
        .expect("IME command queue mutex poisoned")
        .push_back(command);
    DIRTY.store(true, Ordering::Release);
    crate::TEXT_INPUT_DIRTY.store(true, Ordering::Release);
    // Deliberately no `request_frame()` here. This runs on the IME's thread, and
    // `AndroidWindow::request_frame` invokes the frame callback synchronously on
    // the caller — which would run the drain, the Java mirror, and a gpui draw
    // off the main thread. The event loop calls `request_frame` itself every
    // iteration, so the flags above are enough for the edit to be picked up.
}

pub fn has_pending() -> bool {
    DIRTY.load(Ordering::Acquire)
}

/// Report IME edits that arrived while no editor was focused.
///
/// Called from the frame callback when the input handler is unset. The queue is
/// left intact in case an editor takes focus later, but the condition is logged:
/// silently holding keystrokes is what makes this look like a keyboard typing
/// into a void. Rate limited because the frame callback runs continuously.
pub fn report_undeliverable_commands() {
    if !has_pending() {
        return;
    }
    let pending = commands().lock().map(|q| q.len()).unwrap_or(0);
    let reported = UNDELIVERABLE_REPORTS.fetch_add(1, Ordering::Relaxed);
    if reported % 120 == 0 {
        log::error!(
            "{pending} IME edit(s) cannot be applied: no input handler is set, so no \
             editor is focused. Keystrokes are being held, not applied."
        );
    }
}

static UNDELIVERABLE_REPORTS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// How far [`drain_into`] got.
pub enum Drained {
    /// The queue ran empty. `edits` reports whether anything was applied.
    Done { edits: bool },
    /// Every command queued before a soft Return was applied. The caller must
    /// dispatch the `enter` keystroke — which cannot happen here, because gpui's
    /// key dispatch takes the input handler this call still holds — and then
    /// drain again for whatever follows the Return.
    AtEnter,
}

pub fn drain_into(input_handler: &mut PlatformInputHandler) -> Drained {
    let mut edits = false;
    loop {
        let command = {
            let mut queue = commands().lock().expect("IME command queue mutex poisoned");
            let command = queue.pop_front();
            if command.is_none() {
                // Cleared while the queue lock is held: `push` sets the flag
                // only after releasing that lock, so a concurrent push cannot
                // have its flag overwritten here.
                DIRTY.store(false, Ordering::Release);
            }
            command
        };
        let Some(command) = command else {
            return Drained::Done { edits };
        };
        if let TextInputCommand::KeyEnter = command {
            return Drained::AtEnter;
        }
        edits = true;
        match command {
            TextInputCommand::ReplaceText { range_utf16, text } => {
                input_handler.replace_text_in_range(Some(range_utf16), &text);
            }
            TextInputCommand::SetMarkedText {
                range_utf16,
                text,
                selected_utf16,
            } => {
                input_handler.replace_and_mark_text_in_range(
                    Some(range_utf16),
                    &text,
                    Some(selected_utf16),
                );
            }
            TextInputCommand::UnmarkText => {
                input_handler.unmark_text();
            }
            TextInputCommand::KeyEnter => unreachable!("returned above"),
        }
    }
}

/// Mirror the editor's text and selection into the Java-side host view.
///
/// Java computes the replacement range of every `commitText` from this mirror,
/// so a wrong or empty mirror silently sends each keystroke to the wrong offset.
/// When the handler cannot be read, leave the mirror alone rather than
/// overwriting it with placeholder state.
pub fn sync_state_to_java(input_handler: &mut PlatformInputHandler) {
    let mut adjusted = None;
    let Some(text) = input_handler.text_for_range(0..usize::MAX, &mut adjusted) else {
        log::error!(
            "sync_state_to_java: text_for_range returned None — Java mirror left unchanged"
        );
        return;
    };
    let Some(selection) = input_handler.selected_text_range(true).map(|selection| {
        (
            selection.range.start as i32,
            selection.range.end as i32,
            selection.reversed,
        )
    }) else {
        log::error!(
            "sync_state_to_java: selected_text_range returned None — Java mirror left unchanged"
        );
        return;
    };
    let marked = input_handler
        .marked_text_range()
        .map(|range| (range.start as i32, range.end as i32))
        .unwrap_or((-1, -1));

    // Reading the handler is Rust-side and must happen here, on the main thread;
    // only the Java upcall is handed off.
    send_ime(ImeCommand::UpdateEditingState {
        text,
        selection,
        marked,
    });
}

fn call_update_editing_state(text: &str, selection: (i32, i32, bool), marked: (i32, i32)) {
    log_jni_failure(
        "updateEditingState",
        jni_helpers::with_env(|env| {
            let class = jni_helpers::find_app_class(env, "dev.gpui.mobile.GpuiTextInputView")?;
            let text = env.new_string(text).e()?;
            env.call_static_method(
                &class,
                jni::jni_str!("updateEditingState"),
                jni::jni_sig!("(Ljava/lang/String;IIIIZ)V"),
                &[
                    JValue::Object(&text),
                    JValue::Int(selection.0),
                    JValue::Int(selection.1),
                    JValue::Int(marked.0),
                    JValue::Int(marked.1),
                    JValue::Bool(selection.2),
                ],
            )
            .e()?;
            Ok(())
        }),
    );
}

pub fn show_keyboard(keyboard_type: crate::KeyboardType) {
    send_ime(ImeCommand::ShowKeyboard(keyboard_type));
}

fn call_show_keyboard(keyboard_type: crate::KeyboardType) {
    log_jni_failure(
        "show_keyboard",
        jni_helpers::with_env(|env| {
            let activity = jni_helpers::activity(env)?;
            let class = jni_helpers::find_app_class(env, "dev.gpui.mobile.GpuiTextInputView")?;
            env.call_static_method(
                &class,
                jni::jni_str!("showKeyboard"),
                jni::jni_sig!("(Landroid/app/Activity;I)V"),
                &[JValue::Object(&activity), JValue::Int(keyboard_type as i32)],
            )
            .e()?;
            Ok(())
        }),
    );
}

pub fn hide_keyboard() {
    send_ime(ImeCommand::HideKeyboard);
}

fn call_hide_keyboard() {
    log_jni_failure(
        "hide_keyboard",
        jni_helpers::with_env(|env| {
            let activity = jni_helpers::activity(env)?;
            let class = jni_helpers::find_app_class(env, "dev.gpui.mobile.GpuiTextInputView")?;
            env.call_static_method(
                &class,
                jni::jni_str!("hideKeyboard"),
                jni::jni_sig!("(Landroid/app/Activity;)V"),
                &[JValue::Object(&activity)],
            )
            .e()?;
            Ok(())
        }),
    );
}

/// Report a failed IME JNI call instead of discarding it.
///
/// These failures are otherwise invisible: the keyboard simply never appears,
/// with nothing in logcat to say why.
fn log_jni_failure(operation: &str, result: Result<(), String>) {
    if let Err(error) = result {
        log::error!("{operation} failed: {error}");
    }
}

fn jrange(start: i32, end: i32) -> Range<usize> {
    start.max(0) as usize..end.max(start).max(0) as usize
}

fn java_string(value: *mut c_void) -> String {
    let value = value as jni::sys::jobject;
    jni_helpers::with_env(|env| {
        let value = unsafe { JObject::from_raw(env, value) };
        Ok(jni_helpers::get_string(env, &value))
    })
    .unwrap_or_default()
}

#[no_mangle]
pub unsafe extern "C" fn Java_dev_gpui_mobile_GpuiTextInputView_nativeReplaceText(
    _env: *mut c_void,
    _class: *mut c_void,
    start: i32,
    end: i32,
    text: *mut c_void,
) {
    let text = java_string(text);
    push(TextInputCommand::ReplaceText {
        range_utf16: jrange(start, end),
        text,
    });
}

#[no_mangle]
pub unsafe extern "C" fn Java_dev_gpui_mobile_GpuiTextInputView_nativeSetMarkedText(
    _env: *mut c_void,
    _class: *mut c_void,
    start: i32,
    end: i32,
    text: *mut c_void,
    selection_start: i32,
    selection_end: i32,
) {
    let text = java_string(text);
    push(TextInputCommand::SetMarkedText {
        range_utf16: jrange(start, end),
        text,
        selected_utf16: jrange(selection_start, selection_end),
    });
}

#[no_mangle]
pub unsafe extern "C" fn Java_dev_gpui_mobile_GpuiTextInputView_nativeUnmarkText(
    _env: *mut c_void,
    _class: *mut c_void,
) {
    push(TextInputCommand::UnmarkText);
}

#[no_mangle]
pub unsafe extern "C" fn Java_dev_gpui_mobile_GpuiTextInputView_nativeKeyEnter(
    _env: *mut c_void,
    _class: *mut c_void,
) {
    push(TextInputCommand::KeyEnter);
}

/// IME-driven selection changes are not propagated: the `gpui` revision this
/// crate pins exposes no selection setter on `PlatformInputHandler` (it was
/// added only in the gpui-ce mobile fork). The cursor follows implicitly from
/// `replace_text_in_range` / `replace_and_mark_text_in_range`, so ordinary
/// typing and composition are unaffected; only explicit IME cursor repositioning
/// is dropped. The symbol must remain so Java's `nativeSetSelection` link
/// resolves.
#[no_mangle]
pub unsafe extern "C" fn Java_dev_gpui_mobile_GpuiTextInputView_nativeSetSelection(
    _env: *mut c_void,
    _class: *mut c_void,
    _start: i32,
    _end: i32,
) {
}
