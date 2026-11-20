//! Option 2: auto-crop + enhance (segformer + Real-ESRGAN via ONNX Runtime).

use crate::meta::{self, TagJob};
use anyhow::Result;
use image::{imageops::FilterType, DynamicImage, ImageDecoder, ImageReader, RgbImage};
use md5::{Digest, Md5};
use ndarray::{Array4, Ix4};
use once_cell::sync::Lazy;
use ort::{session::Session, value::TensorRef};
use rayon::prelude::*;
use regex::Regex;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Instant;

pub const OUT_W: u32 = 3240;
pub const OUT_H: u32 = 4320;
pub const MAX_MB: f64 = 5.0;
pub const ENHANCE: u32 = 50;
const META_ROOM: usize = 64 * 1024;
const HEAD_PAD: f64 = 0.035;
const HEM_PAD: f64 = 0.06;
const MIN_KEEP: f64 = 0.5;
const SH: usize = 512;
const SW: usize = 384;

// segformer_b2_clothes labels
const HAT: u8 = 1; const HAIR: u8 = 2; const UPPER: u8 = 4; const SKIRT: u8 = 5; const PANTS: u8 = 6;
const DRESS: u8 = 7; const BELT: u8 = 8; const LSHOE: u8 = 9; const RSHOE: u8 = 10; const FACE: u8 = 11;
const BAG: u8 = 16; const SCARF: u8 = 17;
const TOP_GARMENT: [u8; 4] = [UPPER, DRESS, BELT, SCARF];
const ALL_GARMENT: [u8; 6] = [UPPER, SKIRT, PANTS, DRESS, BELT, SCARF];

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

static TOP_PREFIX: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)^(TOP|SHIRT|TUNIC|BLOUSE|TSHIRT|T-SHIRT|KURTI|CROP)").unwrap());

pub struct Models { seg: Session, sr: Option<Session> }

impl Models {
    pub fn load(base: &Path, need_sr: bool) -> Result<Self> {
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let mk = |f: &str, gpu: bool| -> Result<Session> {
            #[allow(unused_mut)]
            let mut b = Session::builder()?;
            #[cfg(all(windows, feature = "directml"))]
            if gpu {
                use ort::execution_providers::DirectMLExecutionProvider;
                b = b.with_execution_providers([DirectMLExecutionProvider::default().build()])?
                    .with_memory_pattern(false)?
                    .with_parallel_execution(false)?;
            }
            let _ = gpu;
            Ok(b.with_intra_threads(threads)?.commit_from_file(base.join("models").join(f))?)
        };
        Ok(Models { seg: mk("segformer_b2_clothes.onnx", false)?, sr: if need_sr { Some(mk("realesr-general-x4v3.onnx", true)?) } else { None } })
    }
}

pub fn load_rgb(path: &Path) -> Result<RgbImage> {
    let mut dec = ImageReader::open(path)?.with_guessed_format()?.into_decoder()?;
    let orient = dec.orientation().unwrap_or(image::metadata::Orientation::NoTransforms);
    let mut img = DynamicImage::from_decoder(dec)?;
    img.apply_orientation(orient); // OpenCV imdecode also honours EXIF rotation
    Ok(img.to_rgb8())
}

/// Per-pixel clothing labels at SH x SW.
fn segment(img: &RgbImage, seg: &mut Session) -> Result<Vec<u8>> {
    let small = image::imageops::resize(img, SW as u32, SH as u32, FilterType::Triangle);
    let mut x = Array4::<f32>::zeros((1, 3, SH, SW));
    for (px, py, p) in small.enumerate_pixels() {
        for c in 0..3 { x[[0, c, py as usize, px as usize]] = (p[c] as f32 / 255.0 - MEAN[c]) / STD[c]; }
    }
    let outputs = seg.run(ort::inputs!["pixel_values" => TensorRef::from_array_view(&x)?])?;
    let l = outputs[0].try_extract_array::<f32>()?.into_dimensionality::<Ix4>()?;
    let (nc, lh, lw) = (l.shape()[1], l.shape()[2], l.shape()[3]);
    let (sy, sx) = (lh as f32 / SH as f32, lw as f32 / SW as f32);
    let mut lab = vec![0u8; SH * SW];
    lab.par_chunks_mut(SW).enumerate().for_each(|(y, row)| {
        let fy = ((y as f32 + 0.5) * sy - 0.5).max(0.0);
        let y0 = (fy.floor() as usize).min(lh - 1);
        let y1 = (y0 + 1).min(lh - 1);
        let wy = fy - y0 as f32;
        for (xx, out) in row.iter_mut().enumerate() {
            let fx = ((xx as f32 + 0.5) * sx - 0.5).max(0.0);
            let x0 = (fx.floor() as usize).min(lw - 1);
            let x1 = (x0 + 1).min(lw - 1);
            let wx = fx - x0 as f32;
            let (mut best, mut bi) = (f32::MIN, 0u8);
            for k in 0..nc {
                let v = (1.0 - wy) * ((1.0 - wx) * l[[0, k, y0, x0]] + wx * l[[0, k, y0, x1]])
                    + wy * ((1.0 - wx) * l[[0, k, y1, x0]] + wx * l[[0, k, y1, x1]]);
                if v > best { best = v; bi = k as u8; }
            }
            *out = bi;
        }
    });
    Ok(lab)
}

fn last_row(lab: &[u8], set: &[u8], frac: f64) -> Option<usize> {
    let cnt: Vec<usize> = (0..SH).map(|y| lab[y * SW..(y + 1) * SW].iter().filter(|v| set.contains(v)).count()).collect();
    let mx = *cnt.iter().max()?;
    if mx == 0 { return None; }
    let thr = (2.0f64).max(frac * mx as f64);
    cnt.iter().rposition(|&c| c as f64 > thr)
}

type Box4 = (u32, u32, u32, u32);

fn crop_box(img: &RgbImage, is_top: bool, seg: &mut Session) -> Result<(Option<Box4>, &'static str)> {
    let lab = segment(img, seg)?;
    let person = |v: u8| v > 0 && v != BAG;
    let row_cnt = |f: &dyn Fn(u8) -> bool, y: usize| lab[y * SW..(y + 1) * SW].iter().filter(|v| f(**v)).count();
    let prow: Vec<usize> = (0..SH).filter(|&y| row_cnt(&|v| person(v), y) > 2).collect();
    if prow.is_empty() { return Ok((None, "no model")); }
    let is_head = |v: u8| v == HAT || v == HAIR || v == FACE;
    let hrow: Vec<usize> = (0..SH).filter(|&y| row_cnt(&is_head, y) > 1).collect();
    let top = *hrow.first().unwrap_or(&prow[0]) as f64;
    if top <= 2.0 { return Ok((None, "closeup")); }
    let shoes = lab.iter().filter(|v| **v == LSHOE || **v == RSHOE).count();
    if shoes > 15 || (*prow.last().unwrap() as f64) < SH as f64 * 0.97 { return Ok((None, "full")); }

    let Some(hem) = last_row(&lab, if is_top { &TOP_GARMENT } else { &ALL_GARMENT }, 0.15) else { return Ok((None, "no garment")) };
    let hem = hem as f64;
    let mh = hem - top;
    let mut y0 = top - HEAD_PAD * mh;
    let y1 = if hem < SH as f64 * 0.95 { hem + HEM_PAD * mh } else { SH as f64 };

    let (ra, rb) = (top.max(0.0) as usize, (y1.min(SH as f64)) as usize);
    let (mut x0, mut x1) = (0.0f64, SW as f64);
    if rb > ra {
        let n = (rb - ra) as f64;
        let cols: Vec<usize> = (0..SW).filter(|&x| (ra..rb).filter(|&y| person(lab[y * SW + x])).count() as f64 / n > 0.02).collect();
        if let (Some(a), Some(b)) = (cols.first(), cols.last()) { x0 = *a as f64; x1 = *b as f64; }
    }
    let mut ch = (y1 - y0).max((x1 - x0) * 1.04 * SH as f64 / SW as f64).max(MIN_KEEP * SH as f64);
    if ch > y1 - y0 { y0 -= (ch - (y1 - y0)) / 2.0; }
    ch = ch.min(SH as f64);
    y0 = y0.min(SH as f64 - ch).max(0.0);
    if ch >= SH as f64 * 0.97 { return Ok((None, "already tight")); }
    let cw = ch * SW as f64 / SH as f64;
    let left = ((x0 + x1) / 2.0 - cw / 2.0).min(SW as f64 - cw).max(0.0);
    let (sy, sx) = (img.height() as f64 / SH as f64, img.width() as f64 / SW as f64);
    let (w, h) = (img.width(), img.height());
    let b = (
        ((left * sx) as u32).min(w - 1), ((y0 * sy) as u32).min(h - 1),
        ((((left + cw) * sx) as u32)).min(w), (((y0 + ch) * sy) as u32).min(h),
    );
    Ok((Some(b), "cropped"))
}

/// Real-ESRGAN 4x in overlapping tiles; output written straight into a u8 image.
fn upscale4(img: &RgbImage, sr: &mut Session) -> Result<RgbImage> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut planar = vec![0f32; 3 * h * w];
    for (i, p) in img.pixels().enumerate() { for c in 0..3 { planar[c * h * w + i] = p[c] as f32 / 255.0; } }
    let mut out = RgbImage::new((w * 4) as u32, (h * 4) as u32);
    let (tile, pad) = (256usize, 16usize);
    for ty in (0..h).step_by(tile) {
        for tx in (0..w).step_by(tile) {
            let (ya, xa) = (ty.saturating_sub(pad), tx.saturating_sub(pad));
            let ts = tile + 2 * pad; // fixed 288x288 input, edge-replicated
            let mut t = Array4::<f32>::zeros((1, 3, ts, ts));
            for c in 0..3 {
                for y in 0..ts {
                    let row = &planar[c * h * w + (ya + y).min(h - 1) * w..][..w];
                    for x in 0..ts { t[[0, c, y, x]] = row[(xa + x).min(w - 1)]; }
                }
            }
            let o = sr.run(ort::inputs!["input" => TensorRef::from_array_view(&t)?])?;
            let r = o[0].try_extract_array::<f32>()?.into_dimensionality::<Ix4>()?;
            let (oy, ox) = ((ty - ya) * 4, (tx - xa) * 4);
            let th = ((ty + tile).min(h) - ty) * 4;
            let tw = ((tx + tile).min(w) - tx) * 4;
            for y in 0..th {
                for x in 0..tw {
                    let px = out.get_pixel_mut((tx * 4 + x) as u32, (ty * 4 + y) as u32);
                    for c in 0..3 { px[c] = (r[[0, c, oy + y, ox + x]].clamp(0.0, 1.0) * 255.0 + 0.5) as u8; }
                }
            }
        }
    }
    Ok(out)
}

// ---------- colour lock: detail (L) from AI, colour + tone from plain resize ----------
const XN: f32 = 0.950456; const ZN: f32 = 1.088754;

fn lut() -> [f32; 256] {
    let mut t = [0f32; 256];
    for (i, v) in t.iter_mut().enumerate() {
        let c = i as f32 / 255.0;
        *v = if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) };
    }
    t
}
fn f_lab(t: f32) -> f32 { if t > 0.008856 { t.cbrt() } else { 7.787 * t + 16.0 / 116.0 } }
fn finv(f: f32) -> f32 { let c = f * f * f; if c > 0.008856 { c } else { (f - 16.0 / 116.0) / 7.787 } }

fn rgb_to_lab(p: &[u8], lt: &[f32; 256]) -> (f32, f32, f32) {
    let (r, g, b) = (lt[p[0] as usize], lt[p[1] as usize], lt[p[2] as usize]);
    let x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / XN;
    let y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
    let z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / ZN;
    let (fx, fy, fz) = (f_lab(x), f_lab(y), f_lab(z));
    (116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz))
}
fn lab_to_rgb(l: f32, a: f32, b: f32, out: &mut [u8]) {
    let fy = (l + 16.0) / 116.0;
    let (fx, fz) = (a / 500.0 + fy, fy - b / 200.0);
    let (x, y, z) = (finv(fx) * XN, finv(fy), finv(fz) * ZN);
    let lin = [3.2404542 * x - 1.5371385 * y - 0.4985314 * z, -0.9692660 * x + 1.8760108 * y + 0.0415560 * z, 0.0556434 * x - 0.2040259 * y + 1.0572252 * z];
    for c in 0..3 {
        let v = lin[c].clamp(0.0, 1.0);
        let g = if v <= 0.0031308 { 12.92 * v } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
        out[c] = (g.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    }
}

fn gauss(src: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    let r = (sigma * 3.0).ceil() as i32;
    let mut k: Vec<f32> = (-r..=r).map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let s: f32 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= s);
    let refl = |i: i32, n: i32| -> usize { (if i < 0 { -i } else if i >= n { 2 * (n - 1) - i } else { i }) as usize };
    let mut tmp = vec![0f32; w * h];
    tmp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let line = &src[y * w..(y + 1) * w];
        for x in 0..w { row[x] = (-r..=r).zip(&k).map(|(d, kv)| line[refl(x as i32 + d, w as i32)] * kv).sum(); }
    });
    let mut out = vec![0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (d, kv) in (-r..=r).zip(&k) {
            let line = &tmp[refl(y as i32 + d, h as i32) * w..][..w];
            for x in 0..w { row[x] += line[x] * kv; }
        }
    });
    out
}

fn colour_lock(ai: &RgbImage, plain: &RgbImage) -> RgbImage {
    let (w, h) = (plain.width() as usize, plain.height() as usize);
    let lt = lut();
    let l_of = |im: &RgbImage| -> Vec<f32> { im.as_raw().par_chunks(3).map(|p| rgb_to_lab(p, &lt).0 * 2.55).collect() };
    let (la, lr) = (l_of(ai), l_of(plain));
    let (ba, br) = (gauss(&la, w, h, 3.0), gauss(&lr, w, h, 3.0));
    let mut out = vec![0u8; w * h * 3];
    out.par_chunks_mut(3).zip(plain.as_raw().par_chunks(3)).enumerate().for_each(|(i, (o, p))| {
        let (_, a, b) = rgb_to_lab(p, &lt);
        let l = (la[i] - ba[i] + br[i]).clamp(0.0, 255.0) / 2.55;
        lab_to_rgb(l, a, b, o);
    });
    RgbImage::from_raw(w as u32, h as u32, out).unwrap()
}

fn fit_3x4(img: RgbImage) -> RgbImage {
    let (w, h) = (img.width(), img.height());
    if w * 4 > h * 3 {
        let nw = h * 3 / 4; let x = (w - nw) / 2;
        image::imageops::crop_imm(&img, x, 0, nw, h).to_image()
    } else {
        let nh = w * 4 / 3; let y = (h - nh) / 2;
        image::imageops::crop_imm(&img, 0, y, w, nh).to_image()
    }
}

fn encode(img: &RgbImage, q: u8) -> Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(4 << 20);
    let mut enc = jpeg_encoder::Encoder::new(&mut buf, q);
    enc.set_sampling_factor(jpeg_encoder::SamplingFactor::R_4_2_0);
    enc.set_optimized_huffman_tables(true);
    enc.encode(img.as_raw(), img.width() as u16, img.height() as u16, jpeg_encoder::ColorType::Rgb)?;
    Ok(buf)
}

pub fn process(img: RgbImage, is_top: bool, strength: u32, max_mb: f64, m: &mut Models) -> Result<(Vec<u8>, String)> {
    let (b, why) = crop_box(&img, is_top, &mut m.seg)?;
    let img = match b {
        Some((l, t, r, bt)) if r > l && bt > t => image::imageops::crop_imm(&img, l, t, r - l, bt - t).to_image(),
        _ => img,
    };
    let img = fit_3x4(img);
    let plain = image::imageops::resize(&img, OUT_W, OUT_H, FilterType::Lanczos3);
    let res = if strength == 0 || img.width() >= OUT_W {
        plain
    } else {
        let big = upscale4(&img, m.sr.as_mut().expect("SR model not loaded"))?;
        let ai = image::imageops::resize(&big, OUT_W, OUT_H, FilterType::Triangle);
        drop(big);
        let mut res = colour_lock(&ai, &plain);
        if strength < 100 {
            let a = strength as f32 / 100.0;
            res.as_mut().par_iter_mut().zip(plain.as_raw().par_iter()).for_each(|(r, p)| *r = (*r as f32 * a + *p as f32 * (1.0 - a)).round() as u8);
        }
        res
    };
    // highest quality that still fits the size limit (sizes shrink monotonically with q)
    let max_bytes = (max_mb * 1024.0 * 1024.0) as usize - META_ROOM;
    let qs: [u8; 15] = [95, 92, 90, 87, 85, 82, 80, 77, 75, 72, 70, 67, 65, 62, 60];
    let (mut lo, mut hi) = (0usize, qs.len() - 1);
    let mut best: Option<Vec<u8>> = None;
    while lo <= hi {
        let mid = (lo + hi) / 2;
        let buf = encode(&res, qs[mid])?;
        if buf.len() <= max_bytes { best = Some(buf); if mid == 0 { break; } hi = mid - 1; } else { lo = mid + 1; }
    }
    let buf = match best { Some(b) => b, None => encode(&res, qs[qs.len() - 1])? };
    let mut why = why.to_string();
    if buf.len() > max_bytes { why += " >LIMIT"; }
    Ok((buf, why))
}

pub fn run_autocrop(base: &Path, src: &Path, dst: &Path, strength: u32, max_mb: f64) -> Result<()> {
    let mut models = Models::load(base, strength > 0)?;
    let mut files: Vec<_> = walkdir::WalkDir::new(src).sort_by_file_name().into_iter().flatten()
        .filter(|e| e.file_type().is_file() && {
            let x = e.path().extension().map(|x| x.to_string_lossy().to_lowercase()).unwrap_or_default();
            x == "jpg" || x == "jpeg" || x == "png"
        }).map(|e| e.into_path()).collect();
    files.sort();
    let mut cache: HashMap<([u8; 16], bool), (Vec<u8>, String)> = HashMap::new(); // same picture for several portals -> once
    let mut tags: Vec<TagJob> = Vec::new();
    let t0 = Instant::now();
    for (n, p) in files.iter().enumerate() {
        let dir = p.parent().unwrap();
        let fname = p.file_name().unwrap().to_string_lossy().to_string();
        let out_dir = dst.join(dir.strip_prefix(src).unwrap_or(dir));
        fs::create_dir_all(&out_dir)?;
        let out = out_dir.join(format!("{}.jpg", p.file_stem().unwrap().to_string_lossy()));
        let is_top = TOP_PREFIX.is_match(&fname);
        let digest: [u8; 16] = Md5::digest(fs::read(p)?).into();
        let key = (digest, is_top);
        if !cache.contains_key(&key) {
            let img = match load_rgb(p) { Ok(i) => i, Err(e) => { println!("SKIP (unreadable) {} {e}", p.display()); continue; } };
            match process(img, is_top, strength, max_mb, &mut models) {
                Ok(r) => { cache.insert(key, r); }
                Err(e) => { println!("SKIP (error) {} {e}", p.display()); continue; }
            }
        }
        let (buf, why) = &cache[&key];
        fs::write(&out, buf)?;
        if meta::PIPELINE_ENABLED {
            let (sku, brand, portal, index) = meta::names(src, dir, &fname);
            tags.push(TagJob { path: out, sku, brand, portal, index });
        }
        println!("[{}/{} {:5.1} min] {:13} {}", n + 1, files.len(), t0.elapsed().as_secs_f64() / 60.0, why, p.strip_prefix(src).unwrap_or(p).display());
        if tags.len() >= 50 { meta::tag_batch(&tags)?; tags.clear(); }
    }
    meta::tag_batch(&tags)?;
    println!("{} images -> {}", files.len(), dst.display());
    Ok(())
}
