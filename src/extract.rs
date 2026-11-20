//! Option 1: input.xlsx + path.csv -> OUTPUT_PORTALWISE/<PORTAL>/<BRAND>/<SKU>/<SKU>_N.jpg
//! Selection rules are a 1:1 port of the Python version.

use crate::meta::{self, TagJob};
use anyhow::{anyhow, Result};
use calamine::{open_workbook, Reader, Xlsx};
use chrono::NaiveDateTime;
use once_cell::sync::Lazy;
use rayon::prelude::*;
use regex::Regex;
use rust_xlsxwriter::Workbook;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

const ROOT: &str = r"\\192.168.0.101\Private\Serein\SereinData\FALGUNI\FINALPICS ,CONSUMPTION SHEET & CAT\";

static EXCLUDE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"CATALOGUE|CAT PIC|^CAT$|^RAW$|^AZ\b|AZ WHITE|^BG$|WHITE BG|=WHITE|^WHITE$|ONMODEL|ON MODEL|NYKAA|NYKKA").unwrap()
});
static RE_FF: Lazy<Regex> = Lazy::new(|| Regex::new(r"\bFF\b").unwrap());
static RE_AAY: Lazy<Regex> = Lazy::new(|| Regex::new(r"\bAAY\b").unwrap());
static RE_DB: Lazy<Regex> = Lazy::new(|| Regex::new(r"\bDB\b").unwrap());
static RE_MAIN: Lazy<Regex> = Lazy::new(|| Regex::new(r"^[-_ ]*MAIN( IMAGE)?$").unwrap());
static RE_NUM: Lazy<Regex> = Lazy::new(|| Regex::new(r"^(?:[_ ]*_\s*(\d+)|\s*\((\d+)\)|[-_ ]*OTHER[-_ ]*(\d+))$").unwrap());
static RE_SRC_GEN: Lazy<Regex> = Lazy::new(|| Regex::new(r"FLIPKART|FILPKART|MYNTRA").unwrap());

const DATE_FORMATS: [&str; 7] = [
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%d-%m-%Y %H.%M", "%d-%m-%Y %H:%M:%S",
    "%d-%m-%Y %H:%M", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M",
];

fn brand_bucket(b: &str) -> Option<&'static str> {
    match b {
        "FASHFUN" | "MURCIA" | "LEIA" | "FYREROSE" => Some("FF"),
        "DRESSBERRY" => Some("DB"),
        "ALL ABOUT YOU" => Some("AAY"),
        _ => None,
    }
}

fn chain(b: &str) -> [&'static str; 4] {
    match b {
        "DB" => ["DB", "AAY", "FF", "GEN"],
        "AAY" => ["AAY", "DB", "FF", "GEN"],
        _ => ["FF", "AAY", "DB", "GEN"],
    }
}

fn bucket_of(parts: &[&str]) -> Option<&'static str> {
    let up: Vec<String> = parts.iter().map(|p| p.trim().to_uppercase()).collect();
    if up.iter().any(|p| EXCLUDE.is_match(p)) { return None; }
    for p in &up {
        if p.contains("FASHFUN") || RE_FF.is_match(p) { return Some("FF"); }
        if RE_AAY.is_match(p) || p.contains("ALL ABOUT YOU") { return Some("AAY"); }
        if RE_DB.is_match(p) { return Some("DB"); }
    }
    Some("GEN")
}

/// `s` is the already upper-cased, trimmed, FSH--stripped stem.
fn index_of(s: &str, sku: &str) -> Option<u32> {
    let r = s.strip_prefix(sku)?.trim();
    if r.is_empty() || RE_MAIN.is_match(r) { return Some(1); }
    let c = RE_NUM.captures(r)?;
    if let Some(m) = c.get(3) { return m.as_str().parse::<u32>().ok().map(|v| v + 1); }
    c.get(1).or(c.get(2))?.as_str().parse().ok()
}

fn to_iso(t: &str) -> Option<String> {
    DATE_FORMATS.iter().find_map(|f| NaiveDateTime::parse_from_str(t.trim(), f).ok())
        .map(|d| d.format("%Y-%m-%d %H:%M:%S").to_string())
}

/// (full path, modified date) for every image row; columns are not read by position.
fn read_path_csv(path: &Path) -> Result<Vec<(String, String)>> {
    let bytes = fs::read(path)?;
    let text = String::from_utf8_lossy(&bytes);
    let text = text.trim_start_matches('\u{feff}');
    let mut rdr = csv::ReaderBuilder::new().has_headers(false).flexible(true).from_reader(text.as_bytes());
    let mut out = Vec::new();
    for rec in rdr.records().flatten() {
        let Some(first) = rec.get(0) else { continue };
        if !first.starts_with(r"\\") { continue; }
        let full = first.trim().to_string();
        let ext = Path::new(&full.replace('\\', "/")).extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
        if !["jpg", "jpeg", "png"].contains(&ext.as_str()) { continue; }
        let dates: Vec<String> = rec.iter().skip(1).filter(|x| !x.is_empty()).filter_map(to_iso).collect();
        let m = dates.get(1).or(dates.first()).cloned().unwrap_or_default();
        out.push((full, m));
    }
    Ok(out)
}

struct Rec { bucket: &'static str, dir: String, idx: u32, mtime: String, path: String }
type Sets = BTreeMap<(&'static str, String), BTreeMap<u32, (String, String)>>;

struct Row { portal: String, brand: String, sku: String, status: String, main_src: String, oth_src: String, count: usize, files: String }

fn write_jpg(src: &str, out: &Path) -> Result<()> {
    let lower = src.to_lowercase();
    if lower.ends_with(".jpg") || lower.ends_with(".jpeg") {
        fs::copy(src, out)?;
        return Ok(());
    }
    let im = image::open(src)?.to_rgba8();
    let mut rgb = image::RgbImage::new(im.width(), im.height());
    for (x, y, p) in im.enumerate_pixels() {
        let a = p[3] as f32 / 255.0;
        let m = |c: u8| (c as f32 * a + 255.0 * (1.0 - a)).round() as u8;
        rgb.put_pixel(x, y, image::Rgb([m(p[0]), m(p[1]), m(p[2])]));
    }
    let mut buf = Vec::new();
    let enc = jpeg_encoder::Encoder::new(&mut buf, 95);
    enc.encode(rgb.as_raw(), rgb.width() as u16, rgb.height() as u16, jpeg_encoder::ColorType::Rgb)?;
    fs::write(out, buf)?;
    Ok(())
}

pub fn run_extract(base: &Path, out_root: &Path, dry: bool) -> Result<()> {
    // ---- input.xlsx ----
    let mut wb: Xlsx<_> = open_workbook(base.join("input.xlsx"))?;
    let range = wb.worksheet_range_at(0).ok_or_else(|| anyhow!("input.xlsx khali hai"))??;
    let mut rows = range.rows();
    let header: Vec<String> = rows.next().ok_or_else(|| anyhow!("header nahi mila"))?.iter().map(|c| c.to_string().trim().to_uppercase()).collect();
    let col = |n: &str| header.iter().position(|h| h == n).ok_or_else(|| anyhow!("input.xlsx me '{n}' column nahi hai"));
    let (ci_sku, ci_portal, ci_brand) = (col("SKU")?, col("PORTAL")?, col("BRAND")?);
    let mut jobs: Vec<(String, String, String)> = Vec::new();
    let mut seen = HashSet::new();
    for r in rows {
        let g = |i: usize| r.get(i).map(|c| c.to_string().trim().to_uppercase()).unwrap_or_default();
        let (portal, brand, sku) = (g(ci_portal), g(ci_brand), g(ci_sku));
        if sku.is_empty() { continue; }
        if seen.insert((portal.clone(), brand.clone(), sku.clone())) { jobs.push((portal, brand, sku)); }
    }
    let skus: HashSet<String> = jobs.iter().map(|j| j.2.clone()).collect();

    // ---- path.csv -> found[sku] ----
    let mut found: HashMap<String, Vec<Rec>> = skus.iter().map(|s| (s.clone(), Vec::new())).collect();
    for (full, mod_) in read_path_csv(&base.join("PATH.csv"))? {
        let rel = full.strip_prefix(ROOT).unwrap_or(&full);
        let parts: Vec<&str> = rel.split('\\').collect();
        let fname = parts[parts.len() - 1];
        let stem = fname.rsplit_once('.').map(|x| x.0).unwrap_or(fname);
        let up = stem.to_uppercase();
        let up = up.trim();
        let s = up.strip_prefix("FSH-").unwrap_or(up);
        // longest SKU that is a prefix of s and gives a valid index
        let mut hit: Option<(String, u32)> = None;
        let ends: Vec<usize> = s.char_indices().map(|(i, _)| i).chain(std::iter::once(s.len())).collect();
        for &e in ends.iter().rev() {
            if e == 0 { continue; }
            if skus.contains(&s[..e]) {
                if let Some(i) = index_of(s, &s[..e]) { hit = Some((s[..e].to_string(), i)); break; }
            }
        }
        let Some((sku, idx)) = hit else { continue };
        let Some(b) = bucket_of(&parts[..parts.len() - 1]) else { continue };
        found.get_mut(&sku).unwrap().push(Rec { bucket: b, dir: parts[..parts.len() - 1].join("\\"), idx, mtime: mod_, path: full.clone() });
    }

    // ---- per job (parallel: network copy is the slow part) ----
    let results: Vec<(Row, Vec<TagJob>)> = jobs.par_iter().map(|(portal, brand, sku)| {
        let mut row = Row { portal: portal.clone(), brand: brand.clone(), sku: sku.clone(), status: String::new(), main_src: String::new(), oth_src: String::new(), count: 0, files: String::new() };
        let Some(bb) = brand_bucket(brand) else { row.status = "UNKNOWN BRAND".into(); return (row, vec![]); };

        let mut sets: Sets = BTreeMap::new();
        for r in found.get(sku).into_iter().flatten() {
            let cur = sets.entry((r.bucket, r.dir.clone())).or_default();
            if cur.get(&r.idx).map_or(true, |c| r.mtime > c.0) { cur.insert(r.idx, (r.mtime.clone(), r.path.clone())); }
        }

        // _1
        let mut main: Option<String> = None;
        let mut main_src = String::new();
        for b in [bb, "GEN"] {
            let c: Vec<(String, String, String)> = sets.iter().filter(|(k, v)| k.0 == b && v.contains_key(&1))
                .map(|(k, v)| (v[&1].0.clone(), v[&1].1.clone(), k.1.clone())).collect();
            if let Some((_, p, d)) = c.into_iter().max() { main = Some(p); main_src = format!("{b}: {d}"); break; }
        }
        // others
        let mut others: Vec<String> = Vec::new();
        let mut oth_src = String::new();
        for b in chain(bb) {
            let mut best: Option<(usize, String, String, Vec<String>)> = None;
            for (k, v) in sets.iter().filter(|(k, _)| k.0 == b) {
                let rest: Vec<(&u32, &(String, String))> = v.iter().filter(|(i, _)| **i >= 2).collect();
                if rest.is_empty() { continue; }
                let mt = rest.iter().map(|x| x.1 .0.clone()).max().unwrap();
                let cand = (rest.len(), mt, k.1.clone(), rest.iter().map(|x| x.1 .1.clone()).collect::<Vec<_>>());
                if best.as_ref().map_or(true, |bst| (cand.0, &cand.1) > (bst.0, &bst.1)) { best = Some(cand); }
            }
            if let Some((_, _, d, files)) = best { others = files; oth_src = format!("{b}: {d}"); break; }
        }

        // AJIO: whole set from one folder
        let same_set = portal == "AJIO";
        if same_set {
            let prio = |b: &str, d: &str| match b {
                "GEN" => if RE_SRC_GEN.is_match(&d.to_uppercase()) { 0 } else { 1 },
                "FF" => 2, "AAY" => 3, _ => 4,
            };
            let mut c: Vec<(i32, String, usize, &(&'static str, String), &BTreeMap<u32, (String, String)>)> = sets.iter()
                .filter(|(_, v)| v.contains_key(&1) && v.len() > 1)
                .map(|(k, v)| (prio(k.0, &k.1), v.values().map(|x| x.0.clone()).max().unwrap(), v.len(), k, v)).collect();
            c.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)).then(b.2.cmp(&a.2)));
            if let Some((_, _, _, k, v)) = c.first() {
                main = Some(v[&1].1.clone());
                others = v.iter().filter(|(i, _)| **i >= 2).map(|(_, x)| x.1.clone()).collect();
                main_src = format!("{}: {}", k.0, k.1);
                oth_src = main_src.clone();
            }
        }

        let files: Vec<String> = main.iter().cloned().chain(others.iter().cloned()).collect();
        let mut status = "OK".to_string();
        if main.is_none() { status = "NO MAIN IMAGE".into(); }
        else if same_set && main_src == oth_src { status = "OK (AJIO SAME SET)".into(); }
        else if !main_src.starts_with(bb) { status = "MAIN NOT FROM BRAND FOLDER".into(); }
        if files.is_empty() { status = "NOT FOUND".into(); }
        if others.is_empty() && main.is_some() { status += " / ONLY MAIN"; }

        let dest = out_root.join(portal).join(brand).join(sku);
        let mut written = 0usize;
        let mut tags = Vec::new();
        if !files.is_empty() && !dry {
            let _ = fs::create_dir_all(&dest);
            for (n, src) in files.iter().enumerate() {
                let n = n + 1;
                let out = dest.join(format!("{sku}_{n}.jpg"));
                match write_jpg(src, &out) {
                    Ok(()) => {
                        written += 1;
                        if meta::PIPELINE_ENABLED {
                            tags.push(TagJob { path: out, sku: sku.clone(), brand: brand.clone(), portal: portal.clone(), index: n as u32 });
                        }
                    }
                    Err(e) => status += &format!(" / ERR {src}: {e}"),
                }
            }
            // leftover numbered files from an older, longer run
            let re = Regex::new(&format!(r"^{}_(\d+)\.jpg$", regex::escape(sku))).unwrap();
            if let Ok(rd) = fs::read_dir(&dest) {
                for e in rd.flatten() {
                    let name = e.file_name().to_string_lossy().to_string();
                    if let Some(c) = re.captures(&name) {
                        if c[1].parse::<usize>().unwrap_or(0) > files.len() { let _ = fs::remove_file(e.path()); }
                    }
                }
            }
        }
        println!("{portal} {brand} {sku} {status} {}", files.len());
        row.status = status; row.main_src = main_src; row.oth_src = oth_src;
        row.count = if dry { files.len() } else { written };
        row.files = files.join(" | ");
        (row, tags)
    }).collect();

    // ---- tagging (batched) ----
    if !dry {
        let all: Vec<TagJob> = results.iter().flat_map(|r| r.1.clone()).collect();
        meta::tag_batch(&all)?;
    }

    // ---- report ----
    let mut wb = Workbook::new();
    let ws = wb.add_worksheet();
    for (c, h) in ["PORTAL", "BRAND", "SKU", "STATUS", "MAIN (_1) FROM", "OTHER IMAGES FROM", "IMAGE COUNT", "SOURCE FILES"].iter().enumerate() {
        ws.write_string(0, c as u16, *h)?;
    }
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for (i, (r, _)) in results.iter().enumerate() {
        let i = (i + 1) as u32;
        ws.write_string(i, 0, &r.portal)?; ws.write_string(i, 1, &r.brand)?; ws.write_string(i, 2, &r.sku)?;
        ws.write_string(i, 3, &r.status)?; ws.write_string(i, 4, &r.main_src)?; ws.write_string(i, 5, &r.oth_src)?;
        ws.write_number(i, 6, r.count as f64)?; ws.write_string(i, 7, &r.files)?;
        *counts.entry(r.status.clone()).or_default() += 1;
    }
    wb.save(base.join(if dry { "report_portalwise_dry.xlsx" } else { "report_portalwise.xlsx" }))?;
    for (k, v) in &counts { println!("{k}: {v}"); }

    // ---- AJIO size check ----
    if !dry {
        let mut wb = Workbook::new();
        let ws = wb.add_worksheet();
        for (c, h) in ["BRAND", "SKU", "_1 SIZE", "_2 SIZE"].iter().enumerate() { ws.write_string(0, c as u16, *h)?; }
        let mut n = 0u32;
        for (portal, brand, sku) in &jobs {
            if portal != "AJIO" { continue; }
            let d: PathBuf = out_root.join(portal).join(brand).join(sku);
            if let (Ok(a), Ok(b)) = (image::image_dimensions(d.join(format!("{sku}_1.jpg"))), image::image_dimensions(d.join(format!("{sku}_2.jpg")))) {
                if a != b {
                    n += 1;
                    ws.write_string(n, 0, brand)?; ws.write_string(n, 1, sku)?;
                    ws.write_string(n, 2, format!("{}x{}", a.0, a.1))?; ws.write_string(n, 3, format!("{}x{}", b.0, b.1))?;
                }
            }
        }
        wb.save(base.join("AJIO_check_list.xlsx"))?;
        println!("{n} AJIO SKUs to check -> AJIO_check_list.xlsx");
    }
    Ok(())
}
