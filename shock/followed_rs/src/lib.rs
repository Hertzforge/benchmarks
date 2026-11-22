//! The law math's `followed` and `followed-hold` behind a plain C ABI, for the
//! shock harness (`shock/laws_rust.py`, ctypes), closed loop, pulse by pulse.
//! Conventions (the research bridge's): records row-major `f64` (`rows * n`);
//! the identification line-major, row-major per line, complex interleaved
//! (`(n/2+1) * nc * nd * 2`); 0 on success, -1 on failure with the message
//! readable by `fl_last_error`.

#![allow(clippy::missing_safety_doc, clippy::too_many_arguments)]

use hertzforge_mimo_shock_law_math as math;
use math::followed::{FollowedLaw, FollowedOptions};
use math::{LawSteps, MatrixStack, PlantModel, PulseInputs, Setup, SrsTarget, C64};
use std::cell::RefCell;
use std::ffi::{c_char, CStr};

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

fn set_err(e: impl Into<String>) -> i32 {
    LAST_ERROR.with(|l| *l.borrow_mut() = e.into());
    -1
}

unsafe fn write_str(s: &str, buf: *mut c_char, len: usize) -> usize {
    let b = s.as_bytes();
    if !buf.is_null() && len > 0 {
        let n = b.len().min(len - 1);
        std::ptr::copy_nonoverlapping(b.as_ptr() as *const c_char, buf, n);
        *buf.add(n) = 0;
    }
    b.len()
}

#[no_mangle]
pub unsafe extern "C" fn fl_last_error(buf: *mut c_char, len: usize) -> usize {
    LAST_ERROR.with(|l| write_str(&l.borrow(), buf, len))
}

fn guarded<F: FnOnce() -> Result<(), String>>(f: F) -> i32 {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => set_err(e),
        Err(p) => {
            let m = p
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| p.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "a panic".into());
            set_err(format!("panic: {m}"))
        }
    }
}

unsafe fn rows_in(p: *const f64, rows: usize, len: usize) -> Vec<Vec<f64>> {
    let s = std::slice::from_raw_parts(p, rows * len);
    s.chunks(len.max(1)).map(|c| c.to_vec()).collect()
}

unsafe fn rows_out(rows: &[Vec<f64>], p: *mut f64) {
    let mut k = 0;
    for r in rows {
        std::ptr::copy_nonoverlapping(r.as_ptr(), p.add(k), r.len());
        k += r.len();
    }
}

pub struct FlLaw {
    law: FollowedLaw,
    nc: usize,
    nd: usize,
    n: usize,
    values: Vec<(&'static str, f64)>,
}

/// A law (`kind` "followed" or "followed-hold") on a setup from arrays:
/// `reference` `nc * n`, `natural` `nf`, `points` `nc * nf` (full level),
/// `frf` the identification, `trusted` `n/2+1` bytes. Null on failure.
#[no_mangle]
pub unsafe extern "C" fn fl_new(
    kind: *const c_char,
    fs: f64,
    n: usize,
    nc: usize,
    nd: usize,
    band_lo: f64,
    band_hi: f64,
    reference: *const f64,
    natural: *const f64,
    nf: usize,
    q: f64,
    points: *const f64,
    frf: *const f64,
    trusted: *const u8,
) -> *mut FlLaw {
    let mut out = std::ptr::null_mut();
    let rc = guarded(|| {
        let kind = CStr::from_ptr(kind).to_str().map_err(|e| e.to_string())?;
        let o = match kind {
            "followed" => FollowedOptions::followed(),
            "followed-hold" => FollowedOptions::followed_hold(),
            k => return Err(format!("no law {k:?}")),
        };
        let lines = n / 2 + 1;
        let f = std::slice::from_raw_parts(frf, lines * nc * nd * 2);
        let data: Vec<C64> = f.chunks(2).map(|c| C64::new(c[0], c[1])).collect();
        let setup = Setup {
            fs_hz: fs,
            record_len: n,
            n_drives: nd,
            n_controls: nc,
            band_lo_hz: band_lo,
            band_hi_hz: band_hi,
            reference: rows_in(reference, nc, n),
            srs: Some(SrsTarget {
                natural_hz: std::slice::from_raw_parts(natural, nf).to_vec(),
                q,
                points: rows_in(points, nc, nf),
            }),
            plant: PlantModel {
                frf: MatrixStack::from_vec(lines, nc, nd, data).map_err(|e| format!("{e:?}"))?,
                trusted: std::slice::from_raw_parts(trusted, lines)
                    .iter()
                    .map(|b| *b != 0)
                    .collect(),
            },
        };
        let law = FollowedLaw::new(setup, o)?;
        out = Box::into_raw(Box::new(FlLaw { law, nc, nd, n, values: Vec::new() }));
        Ok(())
    });
    if rc != 0 {
        std::ptr::null_mut()
    } else {
        out
    }
}

#[no_mangle]
pub unsafe extern "C" fn fl_free(l: *mut FlLaw) {
    if !l.is_null() {
        drop(Box::from_raw(l));
    }
}

/// The first pulse's drives at `level_db` into `out` (`nd * n`).
#[no_mangle]
pub unsafe extern "C" fn fl_seed(l: *mut FlLaw, level_db: f64, out: *mut f64) -> i32 {
    let l = &mut *l;
    guarded(|| {
        let o = l.law.seed(level_db)?;
        l.values = o.values;
        rows_out(&o.drives, out);
        Ok(())
    })
}

/// The pulse just played (`drives` `nd * n`, `responses` `nc * n`), the next
/// pulse's drives into `out`.
#[no_mangle]
pub unsafe extern "C" fn fl_iterate(
    l: *mut FlLaw,
    pulse_index: usize,
    level_db: f64,
    next_level_db: f64,
    drives: *const f64,
    responses: *const f64,
    out: *mut f64,
) -> i32 {
    let l = &mut *l;
    guarded(|| {
        let inputs = PulseInputs {
            pulse_index,
            level_db,
            next_level_db,
            drives: rows_in(drives, l.nd, l.n),
            responses: rows_in(responses, l.nc, l.n),
            srs: None,
        };
        let o = l.law.iterate(&inputs)?;
        l.values = o.values;
        rows_out(&o.drives, out);
        Ok(())
    })
}

/// JSON: the last call's values, the probe checks (level, scale dB,
/// predicted dB, measured dB) and the holds.
#[no_mangle]
pub unsafe extern "C" fn fl_report(l: *mut FlLaw, buf: *mut c_char, len: usize) -> usize {
    let l = &*l;
    let vals: serde_json::Map<String, serde_json::Value> = l
        .values
        .iter()
        .map(|(k, v)| {
            let j = if v.is_finite() { serde_json::json!(v) } else { serde_json::Value::Null };
            (k.to_string(), j)
        })
        .collect();
    let checks: Vec<[f64; 4]> = l.law.probe_checks().iter().map(|c| [c.0, c.1, c.2, c.3]).collect();
    let s = serde_json::json!({"values": vals, "probe_checks": checks, "holds": l.law.holds()}).to_string();
    write_str(&s, buf, len)
}
