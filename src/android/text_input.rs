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
}

static COMMANDS: OnceLock<Mutex<VecDeque<TextInputCommand>>> = OnceLock::new();
static DIRTY: AtomicBool = AtomicBool::new(false);

fn commands() -> &'static Mutex<VecDeque<TextInputCommand>> {
    COMMANDS.get_or_init(|| Mutex::new(VecDeque::new()))
}

pub fn push(command: TextInputCommand) {
    if let Ok(mut commands) = commands().lock() {
        commands.push_back(command);
    }
    DIRTY.store(true, Ordering::Release);
    crate::TEXT_INPUT_DIRTY.store(true, Ordering::Release);
    if let Some(platform) = jni_helpers::platform() {
        if let Some(window) = platform.primary_window() {
            window.request_frame();
        }
    }
}

pub fn has_pending() -> bool {
    DIRTY.load(Ordering::Acquire)
}

pub fn drain_into(input_handler: &mut PlatformInputHandler) -> bool {
    let mut drained = false;
    loop {
        let command = commands()
            .lock()
            .ok()
            .and_then(|mut commands| commands.pop_front());
        let Some(command) = command else {
            break;
        };
        drained = true;
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
        }
    }
    if drained {
        DIRTY.store(false, Ordering::Release);
    }
    drained
}

pub fn sync_state_to_java(input_handler: &mut PlatformInputHandler) {
    let mut adjusted = None;
    let text = input_handler
        .text_for_range(0..usize::MAX, &mut adjusted)
        .unwrap_or_default();
    let selection = input_handler
        .selected_text_range(true)
        .map(|selection| {
            (
                selection.range.start as i32,
                selection.range.end as i32,
                selection.reversed,
            )
        })
        .unwrap_or((-1, -1, false));
    let marked = input_handler
        .marked_text_range()
        .map(|range| (range.start as i32, range.end as i32))
        .unwrap_or((-1, -1));

    let _ = jni_helpers::with_env(|env| {
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
    });
}

pub fn show_keyboard(keyboard_type: crate::KeyboardType) {
    let _ = jni_helpers::with_env(|env| {
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
    });
}

pub fn hide_keyboard() {
    let _ = jni_helpers::with_env(|env| {
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
    });
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
