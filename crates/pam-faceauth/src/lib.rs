//! Minimal PAM authentication bridge for the dedicated `faceauth-test` service.
//!
//! The FFI boundary reads only PAM service/user items, resolves a numeric UID, and exchanges the
//! bounded faceauth protocol over a fixed Unix socket. It never opens cameras, loads models,
//! reads templates, or handles passwords.

use std::{
    ffi::{CStr, c_char, c_int, c_void},
    os::unix::net::UnixStream,
    panic::{AssertUnwindSafe, catch_unwind},
    path::Path,
    ptr,
    time::{Duration, Instant},
};

use faceauth_protocol::{
    AuthenticationPurpose, DecisionCode, Request, RequestContext, Response, ServiceName,
    TransactionId,
};
use faceauth_transport::{PeerStream, TransportConfig};

const PAM_SUCCESS: c_int = 0;
const PAM_IGNORE: c_int = 25;
const PAM_SERVICE: c_int = 1;
const TEST_SERVICE: &str = "faceauth-test";
const SOCKET_PATH: &str = "/run/faceauth/auth.sock";
const MAX_RESPONSES: usize = 64;
const MAX_TRANSACTION_DURATION: Duration = Duration::from_secs(20);
const MAX_USERNAME_BYTES: usize = 256;
const PASSWD_BUFFER_BYTES: usize = 16 * 1024;

#[repr(C)]
/// Opaque PAM transaction handle owned by libpam.
pub struct PamHandle {
    _private: [u8; 0],
}

#[link(name = "pam")]
unsafe extern "C" {
    fn pam_get_user(pamh: *mut PamHandle, user: *mut *const c_char, prompt: *const c_char)
    -> c_int;
    fn pam_get_item(pamh: *const PamHandle, item_type: c_int, item: *mut *const c_void) -> c_int;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PamOutcome {
    Accepted,
    Ignore,
}

impl PamOutcome {
    const fn code(self) -> c_int {
        match self {
            Self::Accepted => PAM_SUCCESS,
            Self::Ignore => PAM_IGNORE,
        }
    }
}

/// PAM authentication entry point.
///
/// Every panic, malformed PAM item, daemon failure, timeout, rejection, or biometric mismatch is
/// converted to `PAM_IGNORE`, preserving the administrator-configured password fallback.
#[unsafe(no_mangle)]
pub extern "C" fn pam_sm_authenticate(
    pamh: *mut PamHandle,
    _flags: c_int,
    argc: c_int,
    _argv: *const *const c_char,
) -> c_int {
    catch_unwind(AssertUnwindSafe(|| authenticate(pamh, argc))).unwrap_or(PamOutcome::Ignore).code()
}

/// PAM credential-management entry point. This module creates no credentials.
#[unsafe(no_mangle)]
pub const extern "C" fn pam_sm_setcred(
    _pamh: *mut PamHandle,
    _flags: c_int,
    _argc: c_int,
    _argv: *const *const c_char,
) -> c_int {
    PAM_SUCCESS
}

fn authenticate(pamh: *mut PamHandle, argc: c_int) -> PamOutcome {
    if pamh.is_null() || argc != 0 {
        return PamOutcome::Ignore;
    }
    let Some(service) = pam_service(pamh) else {
        return PamOutcome::Ignore;
    };
    if service != TEST_SERVICE {
        return PamOutcome::Ignore;
    }
    let Some(username) = pam_username(pamh) else {
        return PamOutcome::Ignore;
    };
    let Some(target_uid) = resolve_uid(&username) else {
        return PamOutcome::Ignore;
    };
    let Ok(stream) = UnixStream::connect(Path::new(SOCKET_PATH)) else {
        return PamOutcome::Ignore;
    };
    let Ok(mut peer) = PeerStream::connect(stream, TransportConfig::default()) else {
        return PamOutcome::Ignore;
    };
    if peer.peer().uid != 0 {
        return PamOutcome::Ignore;
    }
    let Ok(service) = ServiceName::parse(service) else {
        return PamOutcome::Ignore;
    };
    let request = Request::Authenticate {
        context: RequestContext {
            transaction_id: TransactionId::generate(),
            target_uid,
            service,
            purpose: AuthenticationPurpose::Test,
        },
    };
    exchange(&mut peer, request)
}

fn exchange(peer: &mut PeerStream, request: Request) -> PamOutcome {
    let transaction_id = match &request {
        Request::Authenticate { context } => context.transaction_id,
        Request::Cancel { .. } => return PamOutcome::Ignore,
    };
    if peer.write_message(request).is_err() {
        return PamOutcome::Ignore;
    }
    let started = Instant::now();
    for _ in 0..MAX_RESPONSES {
        if started.elapsed() >= MAX_TRANSACTION_DURATION {
            return PamOutcome::Ignore;
        }
        let Ok(envelope) = peer.read_message::<Response>() else {
            return PamOutcome::Ignore;
        };
        match envelope.message {
            Response::Started { transaction_id: received }
            | Response::Progress { transaction_id: received, .. }
                if received == transaction_id => {}
            Response::Completed { transaction_id: received, decision }
                if received == transaction_id =>
            {
                return if decision == DecisionCode::Accepted {
                    PamOutcome::Accepted
                } else {
                    PamOutcome::Ignore
                };
            }
            Response::Rejected { transaction_id: Some(received), .. }
                if received == transaction_id =>
            {
                return PamOutcome::Ignore;
            }
            _ => return PamOutcome::Ignore,
        }
    }
    PamOutcome::Ignore
}

fn pam_service(pamh: *mut PamHandle) -> Option<String> {
    let mut item = ptr::null();
    // SAFETY: `pamh` is supplied by libpam for this callback; `item` points to storage for the
    // borrowed PAM item pointer. The result is copied before returning.
    if unsafe { pam_get_item(pamh, PAM_SERVICE, &raw mut item) } != PAM_SUCCESS || item.is_null() {
        return None;
    }
    bounded_c_string(item.cast())
}

fn pam_username(pamh: *mut PamHandle) -> Option<String> {
    let mut user = ptr::null();
    // SAFETY: `pamh` is supplied by libpam and `user` is an out pointer documented by
    // `pam_get_user`. A null prompt requests the configured PAM prompt.
    if unsafe { pam_get_user(pamh, &raw mut user, ptr::null()) } != PAM_SUCCESS || user.is_null() {
        return None;
    }
    bounded_c_string(user)
}

fn bounded_c_string(value: *const c_char) -> Option<String> {
    // SAFETY: PAM and libc return NUL-terminated strings valid for the duration of the callback.
    let bytes = unsafe { CStr::from_ptr(value) }.to_bytes();
    if bytes.is_empty() || bytes.len() > MAX_USERNAME_BYTES {
        return None;
    }
    std::str::from_utf8(bytes).ok().map(ToOwned::to_owned)
}

fn resolve_uid(username: &str) -> Option<u32> {
    if username.as_bytes().contains(&0) || username.len() > MAX_USERNAME_BYTES {
        return None;
    }
    let username = std::ffi::CString::new(username).ok()?;
    // SAFETY: `passwd` is immediately initialized by `getpwnam_r`; zero is a valid initial bit
    // pattern for this C record and all returned pointers refer into `buffer` while inspected.
    let mut passwd: libc::passwd = unsafe { std::mem::zeroed() };
    let mut result = ptr::null_mut();
    let mut buffer = vec![0_u8; PASSWD_BUFFER_BYTES];
    // SAFETY: all pointers reference writable/live storage of the documented sizes. The function
    // is reentrant and does not retain these pointers.
    let status = unsafe {
        libc::getpwnam_r(
            username.as_ptr(),
            &raw mut passwd,
            buffer.as_mut_ptr().cast(),
            buffer.len(),
            &raw mut result,
        )
    };
    if status == 0 && !result.is_null() { Some(passwd.pw_uid) } else { None }
}

#[cfg(test)]
mod tests {
    use std::{os::unix::net::UnixStream, thread};

    use faceauth_protocol::{Envelope, ProgressCode};

    use super::*;

    fn request() -> Result<Request, Box<dyn std::error::Error>> {
        Ok(Request::Authenticate {
            context: RequestContext {
                transaction_id: TransactionId::generate(),
                target_uid: 1000,
                service: ServiceName::parse(TEST_SERVICE)?,
                purpose: AuthenticationPurpose::Test,
            },
        })
    }

    #[test]
    fn only_exact_accepted_terminal_response_succeeds() -> Result<(), Box<dyn std::error::Error>> {
        let request = request()?;
        let transaction_id = match &request {
            Request::Authenticate { context } => context.transaction_id,
            Request::Cancel { .. } => return Err("unexpected cancel request".into()),
        };
        let (client, server) = UnixStream::pair()?;
        let mut client = PeerStream::connect(client, TransportConfig::default())?;
        let mut server = PeerStream::connect(server, TransportConfig::default())?;
        let server_thread = thread::spawn(move || -> Result<(), String> {
            server.read_message::<Request>().map_err(|error| error.to_string())?;
            server
                .write_message(Response::Started { transaction_id })
                .map_err(|error| error.to_string())?;
            server
                .write_message(Response::Progress {
                    transaction_id,
                    progress: ProgressCode::HoldStill,
                })
                .map_err(|error| error.to_string())?;
            server
                .write_message(Response::Completed {
                    transaction_id,
                    decision: DecisionCode::Accepted,
                })
                .map_err(|error| error.to_string())?;
            Ok(())
        });
        assert_eq!(exchange(&mut client, request), PamOutcome::Accepted);
        server_thread.join().map_err(|_| "server thread panicked")??;
        Ok(())
    }

    #[test]
    fn mismatched_or_nonaccepted_responses_preserve_fallback()
    -> Result<(), Box<dyn std::error::Error>> {
        let request = request()?;
        let wrong_transaction = TransactionId::generate();
        let (client, server) = UnixStream::pair()?;
        let mut client = PeerStream::connect(client, TransportConfig::default())?;
        let mut server = PeerStream::connect(server, TransportConfig::default())?;
        let server_thread = thread::spawn(move || -> Result<(), String> {
            server.read_message::<Request>().map_err(|error| error.to_string())?;
            server
                .write_message(Response::Completed {
                    transaction_id: wrong_transaction,
                    decision: DecisionCode::Accepted,
                })
                .map_err(|error| error.to_string())?;
            Ok(())
        });
        assert_eq!(exchange(&mut client, request), PamOutcome::Ignore);
        server_thread.join().map_err(|_| "server thread panicked")??;
        Ok(())
    }

    #[test]
    fn uid_resolution_is_bounded_and_unknown_users_fail() {
        assert!(resolve_uid("root").is_some());
        assert!(resolve_uid("definitely-not-a-faceauth-user").is_none());
        assert!(resolve_uid(&"x".repeat(MAX_USERNAME_BYTES + 1)).is_none());
    }

    #[test]
    fn response_envelope_remains_protocol_versioned() {
        let transaction_id = TransactionId::generate();
        let envelope = Envelope::current(Response::Completed {
            transaction_id,
            decision: DecisionCode::Accepted,
        });
        assert!(matches!(
            envelope.message,
            Response::Completed {
                transaction_id: received,
                decision: DecisionCode::Accepted,
            } if received == transaction_id
        ));
    }
}
