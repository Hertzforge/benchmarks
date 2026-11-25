// Replays a Python paired run through the shipped Rust law: same setup, same played pulses.
use hertzforge_mimo_shock_law_math::paired::{PairedLaw, PairedParams};
use hertzforge_mimo_shock_law_math::{LawSteps, MatrixStack, PlantModel, PulseInputs, Setup, C64};
use serde_json::Value;
use std::io::Write;
fn rd(p: &str, n: usize) -> Vec<f64> {
    let b = std::fs::read(p).unwrap(); assert_eq!(b.len(), n * 8);
    b.chunks(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()
}
fn main() {
    let dir = std::env::args().nth(1).unwrap();
    let m: Value = serde_json::from_str(&std::fs::read_to_string(format!("{dir}/meta.json")).unwrap()).unwrap();
    let n = m["n"].as_u64().unwrap() as usize; let nc = m["nc"].as_u64().unwrap() as usize;
    let nd = m["nd"].as_u64().unwrap() as usize; let lines = n / 2 + 1;
    let re = rd(&format!("{dir}/frf_re.bin"), lines * nc * nd); let im = rd(&format!("{dir}/frf_im.bin"), lines * nc * nd);
    let frf = MatrixStack::from_vec(lines, nc, nd, re.iter().zip(&im).map(|(a, b)| C64::new(*a, *b)).collect()).unwrap();
    let reference: Vec<Vec<f64>> = rd(&format!("{dir}/reference.bin"), nc * n).chunks(n).map(|c| c.to_vec()).collect();
    let setup = Setup { fs_hz: m["fs"].as_f64().unwrap(), record_len: n, n_drives: nd, n_controls: nc,
        band_lo_hz: m["band_lo"].as_f64().unwrap(), band_hi_hz: m["band_hi"].as_f64().unwrap(), reference, srs: None,
        plant: PlantModel { frf, trusted: vec![true; lines] } };
    let mut law = PairedLaw::new(setup, PairedParams::diagonal(nc, nd)).unwrap();
    let mut out = std::fs::File::create(format!("{dir}/rust_out.bin")).unwrap();
    let put = |o: &mut std::fs::File, d: &Vec<Vec<f64>>| for r in d { for x in r { o.write_all(&x.to_le_bytes()).unwrap(); } };
    let s = law.seed(m["seed_level"].as_f64().unwrap()).unwrap(); put(&mut out, &s.drives);
    for (i, p) in m["pulses"].as_array().unwrap().iter().enumerate() {
        let dr: Vec<Vec<f64>> = rd(&format!("{dir}/drv_{i}.bin"), nd * n).chunks(n).map(|c| c.to_vec()).collect();
        let rs: Vec<Vec<f64>> = rd(&format!("{dir}/rsp_{i}.bin"), nc * n).chunks(n).map(|c| c.to_vec()).collect();
        let o = law.iterate(&PulseInputs { pulse_index: i, level_db: p["level"].as_f64().unwrap(),
            next_level_db: p["next"].as_f64().unwrap(), drives: dr, responses: rs, srs: None }).unwrap();
        put(&mut out, &o.drives);
    }
}
