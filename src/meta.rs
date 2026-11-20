//! Metadata writer. Same tags as the Python version, but exiftool is started
//! once per 50 images (via `-execute`) instead of once per image.

use anyhow::{bail, Result};
use chrono::{Duration, Local, NaiveDateTime};
use md5::{Digest, Md5};
use std::path::{Path, PathBuf};
use std::process::Command;

/// false = koi metadata nahi likhega (options 1 & 2 me tagging band).
pub const PIPELINE_ENABLED: bool = true;

const COMPANY: &str = "Bindal Exports Ltd";
const CREATOR: &str = "Bindal Exports Studio";
const SHOT_INTERVAL_SECS: i64 = 20;
const DAYS_BACK: u128 = 20;
const STAR_RATING: u8 = 5;
const TZ: &str = "+05:30";
const DEFAULT_CAMERA: &str = "phaseone";
const LAT: f64 = 12.9716;
const LON: f64 = 77.5946;
const ALT: u32 = 920;
const CITY: &str = "Bengaluru";
const STATE: &str = "Karnataka";
const COUNTRY: &str = "India";
const COUNTRY_CODE: &str = "IND";
const SUBLOC: &str = "Studio";

const STUDIO_COMMON: &[(&str, &str)] = &[
    ("ExposureProgram", "Manual"), ("ExposureMode", "Manual"), ("ExposureCompensation", "0"),
    ("MeteringMode", "Multi-segment"), ("Flash", "Fired"), ("WhiteBalance", "Manual"),
    ("LightSource", "Flash"), ("SceneCaptureType", "Standard"), ("Contrast", "Normal"),
    ("Saturation", "Normal"), ("Sharpness", "Normal"), ("GainControl", "None"),
    ("CustomRendered", "Normal"), ("DigitalZoomRatio", "1"), ("SubjectDistance", "3.5"),
    ("SubjectDistanceRange", "Distant"), ("SensingMethod", "One-chip color area"),
    ("FileSource", "Digital Camera"), ("SceneType", "Directly photographed"),
    ("ColorSpace", "sRGB"), ("ExifVersion", "0232"), ("XResolution", "300"),
    ("YResolution", "300"), ("ResolutionUnit", "inches"), ("YCbCrPositioning", "Co-sited"),
    ("FocalPlaneResolutionUnit", "cm"),
];

/// order: Make, Model, LensMake, LensModel, LensInfo, FocalLength, FocalLengthIn35mmFormat,
///        FNumber, ApertureValue, MaxApertureValue, ExposureTime, ShutterSpeedValue, ISO
const PRESET_TAGS: [&str; 13] = [
    "Make", "Model", "LensMake", "LensModel", "LensInfo", "FocalLength", "FocalLengthIn35mmFormat",
    "FNumber", "ApertureValue", "MaxApertureValue", "ExposureTime", "ShutterSpeedValue", "ISO",
];

fn preset(key: &str) -> (&'static str, [&'static str; 13]) {
    match key {
        "hasselblad" => ("Phocus 4.0", ["Hasselblad", "X2D 100C", "Hasselblad", "XCD 2,5/90V", "90 90 2.5 2.5", "90.0 mm", "71", "11.0", "11.0", "2.5", "1/125", "1/125", "64"]),
        "fujifilm" => ("Capture One 16.4 Windows", ["FUJIFILM", "GFX100 II", "FUJIFILM", "GF110mmF2 R LM WR", "110 110 2 2", "110.0 mm", "87", "11.0", "11.0", "2.0", "1/125", "1/125", "80"]),
        "sony" => ("Capture One 16.4 Windows", ["SONY", "ILCE-1M2", "Sony", "FE 85mm F1.4 GM II", "85 85 1.4 1.4", "85.0 mm", "85", "8.0", "8.0", "1.4", "1/200", "1/200", "100"]),
        "canon" => ("Digital Photo Professional 4.19", ["Canon", "Canon EOS R5 Mark II", "Canon", "RF85mm F1.2 L USM", "85 85 1.2 1.2", "85.0 mm", "85", "8.0", "8.0", "1.2", "1/200", "1/200", "100"]),
        "nikon" => ("NX Studio 1.7", ["NIKON CORPORATION", "NIKON Z 9", "Nikon", "NIKKOR Z 85mm f/1.2 S", "85 85 1.2 1.2", "85.0 mm", "85", "8.0", "8.0", "1.2", "1/200", "1/200", "64"]),
        _ => ("Capture One 16.4 Windows", ["Phase One", "IQ4 150MP", "Schneider Kreuznach", "Schneider Kreuznach 80mm LS f/2.8", "80 80 2.8 2.8", "80.0 mm", "51", "11.0", "11.0", "2.8", "1/125", "1/125", "50"]),
    }
}

#[derive(Clone)]
pub struct TagJob {
    pub path: PathBuf,
    pub sku: String,
    pub brand: String,
    pub portal: String,
    pub index: u32,
}

/// Python str.title()
fn title(s: &str) -> String {
    let mut out = String::new();
    let mut prev_alpha = false;
    for c in s.chars() {
        if c.is_alphabetic() {
            if prev_alpha { out.extend(c.to_lowercase()); } else { out.extend(c.to_uppercase()); }
            prev_alpha = true;
        } else {
            out.push(c);
            prev_alpha = false;
        }
    }
    out
}

/// Shoot date 1..20 days back, fixed per SKU; _N images are 20 s apart.
pub fn shot_time(sku: &str, index: u32) -> NaiveDateTime {
    let digest = Md5::digest(sku.to_uppercase().as_bytes());
    let h = u128::from_be_bytes(digest.into());
    let day = Local::now().naive_local().date().and_hms_opt(10, 0, 0).unwrap()
        - Duration::days(1 + (h % DAYS_BACK) as i64);
    let start = day + Duration::seconds(((h >> 8) % (6 * 3600)) as i64);
    start + Duration::seconds(index.saturating_sub(1) as i64 * SHOT_INTERVAL_SECS)
}

pub fn find_exiftool() -> Option<PathBuf> {
    let local = std::env::var("LOCALAPPDATA").unwrap_or_default();
    let mut cands: Vec<PathBuf> = Vec::new();
    if let Some(dir) = std::env::current_exe().ok().and_then(|p| p.parent().map(|x| x.to_path_buf())) {
        cands.push(dir.join("exiftool.exe"));
        cands.push(dir.join("exiftool(-k).exe"));
    }
    if let Ok(dir) = std::env::current_dir() {
        cands.push(dir.join("exiftool.exe"));
        cands.push(dir.join("exiftool(-k).exe"));
    }
    cands.push(PathBuf::from("exiftool"));
    cands.push(PathBuf::from("exiftool.exe"));
    cands.push(PathBuf::from(format!(r"{}\Programs\ExifTool\ExifTool.exe", local)));
    cands.push(PathBuf::from(r"C:\Program Files\ExifTool\ExifTool.exe"));
    cands.push(PathBuf::from(r"C:\Windows\exiftool.exe"));
    cands.into_iter().find(|p| {
        Command::new(p).arg("-ver").output().map(|o| o.status.success()).unwrap_or(false)
    })
}

fn args_for(j: &TagJob, w: u32, h: u32) -> Vec<String> {
    let (software, vals) = preset(DEFAULT_CAMERA);
    let pv = |name: &str| vals[PRESET_TAGS.iter().position(|t| *t == name).unwrap()];

    let brand_name = title(j.brand.trim());
    let category = title(j.sku.split('-').next().unwrap_or(""));
    let product = format!("{brand_name} {category} {}", j.sku).trim().to_string();
    let description = product.clone();
    let alt_text = format!("{product} - image {}", j.index);
    let mut keywords: Vec<String> = Vec::new();
    for k in [brand_name.as_str(), category.as_str(), j.sku.as_str(), "Women Western Wear"] {
        if !k.is_empty() && !keywords.iter().any(|x| x.eq_ignore_ascii_case(k)) {
            keywords.push(k.to_string());
        }
    }
    let shot = shot_time(&j.sku, j.index);
    let ts = shot.format("%Y:%m:%d %H:%M:%S").to_string();
    let copyright = format!("(c) {} {}. All rights reserved.", shot.format("%Y"), COMPANY);
    let focal = pv("FocalLength").split_whitespace().next().unwrap_or("").to_string();

    let mut a: Vec<String> = vec![
        "-all=".into(), "-tagsFromFile".into(), "@".into(), "-EXIF:Orientation".into(),
        "-ICC_Profile".into(), "-charset".into(), "iptc=UTF8".into(), "-codedcharacterset=utf8".into(),
    ];
    for (t, v) in STUDIO_COMMON { a.push(format!("-EXIF:{t}={v}")); }
    for t in PRESET_TAGS { a.push(format!("-EXIF:{t}={}", pv(t))); }

    let mut push = |s: String| a.push(s);
    push(format!("-EXIF:Software={software}"));
    push(format!("-EXIF:ExifImageWidth={w}")); push(format!("-EXIF:ExifImageHeight={h}"));
    push(format!("-EXIF:DateTimeOriginal={ts}")); push(format!("-EXIF:CreateDate={ts}")); push(format!("-EXIF:ModifyDate={ts}"));
    push(format!("-EXIF:OffsetTime={TZ}")); push(format!("-EXIF:OffsetTimeOriginal={TZ}")); push(format!("-EXIF:OffsetTimeDigitized={TZ}"));
    push("-EXIF:SubSecTimeOriginal=00".into());
    push(format!("-EXIF:ImageUniqueID={}", uuid::Uuid::new_v4().simple()));
    push(format!("-XMP-tiff:Make={}", pv("Make"))); push(format!("-XMP-tiff:Model={}", pv("Model")));
    push(format!("-XMP-aux:Lens={}", pv("LensModel")));
    push(format!("-XMP-exifEX:LensMake={}", pv("LensMake"))); push(format!("-XMP-exifEX:LensModel={}", pv("LensModel")));
    push(format!("-XMP-exif:FNumber={}", pv("FNumber"))); push(format!("-XMP-exif:ExposureTime={}", pv("ExposureTime")));
    push(format!("-XMP-exif:FocalLength={focal}"));
    push(format!("-XMP-exifEX:PhotographicSensitivity={}", pv("ISO")));
    push(format!("-XMP-xmp:CreatorTool={software}")); push(format!("-XMP-xmp:CreateDate={ts}{TZ}"));
    push(format!("-XMP-xmp:Rating={STAR_RATING}"));
    push(format!("-XMP-photoshop:DateCreated={ts}{TZ}")); push("-XMP-photoshop:ColorMode=RGB".into());
    // location
    push(format!("-GPS:GPSLatitude={}", LAT.abs())); push("-GPS:GPSLatitudeRef=N".into());
    push(format!("-GPS:GPSLongitude={}", LON.abs())); push("-GPS:GPSLongitudeRef=E".into());
    push(format!("-GPS:GPSAltitude={ALT}")); push("-GPS:GPSAltitudeRef=Above Sea Level".into());
    push("-GPS:GPSMapDatum=WGS-84".into()); push("-GPS:GPSVersionID=2.3.0.0".into());
    push(format!("-XMP-exif:GPSLatitude={LAT}")); push(format!("-XMP-exif:GPSLongitude={LON}"));
    push(format!("-IPTC:City={CITY}")); push(format!("-IPTC:Province-State={STATE}"));
    push(format!("-IPTC:Country-PrimaryLocationName={COUNTRY}")); push(format!("-IPTC:Country-PrimaryLocationCode={COUNTRY_CODE}"));
    push(format!("-IPTC:Sub-location={SUBLOC}"));
    push(format!("-IPTC:DateCreated={}", shot.format("%Y:%m:%d"))); push(format!("-IPTC:TimeCreated={}+05:30", shot.format("%H:%M:%S")));
    push(format!("-XMP-photoshop:City={CITY}")); push(format!("-XMP-photoshop:State={STATE}")); push(format!("-XMP-photoshop:Country={COUNTRY}"));
    push(format!("-XMP-iptcCore:CountryCode={COUNTRY_CODE}")); push(format!("-XMP-iptcCore:Location={SUBLOC}"));
    // product
    let cut = |s: &str, n: usize| s.chars().take(n).collect::<String>();
    push(format!("-EXIF:Artist={CREATOR}")); push(format!("-EXIF:Copyright={copyright}"));
    push(format!("-EXIF:ImageDescription={alt_text}")); push(format!("-EXIF:XPTitle={product}"));
    push(format!("-EXIF:XPSubject={category}")); push(format!("-EXIF:XPComment={description}"));
    push(format!("-EXIF:XPKeywords={}", keywords.join(";")));
    push(format!("-IPTC:ObjectName={}", cut(&product, 64))); push(format!("-IPTC:Headline={}", cut(&product, 256)));
    push(format!("-IPTC:Caption-Abstract={}", cut(&description, 2000))); push(format!("-IPTC:By-line={CREATOR}"));
    push(format!("-IPTC:CopyrightNotice={copyright}")); push(format!("-IPTC:Source={COMPANY}"));
    push(format!("-IPTC:Credit={COMPANY}")); push(format!("-IPTC:OriginalTransmissionReference={}", j.sku));
    push(format!("-XMP-dc:Title={product}")); push(format!("-XMP-dc:Description={description}"));
    push(format!("-XMP-dc:Creator={CREATOR}")); push(format!("-XMP-dc:Rights={copyright}"));
    push(format!("-XMP-dc:Identifier={}", j.sku));
    push(format!("-XMP-photoshop:Headline={product}")); push(format!("-XMP-photoshop:Credit={COMPANY}"));
    push(format!("-XMP-photoshop:Source={COMPANY}"));
    push(format!("-XMP-iptcCore:AltTextAccessibility={alt_text}")); push("-XMP-xmpRights:Marked=True".into());
    push("-XMP-x:XMPToolkit=".into());
    for kw in &keywords {
        push(format!("-IPTC:Keywords+={}", cut(kw, 64)));
        push(format!("-XMP-dc:Subject+={kw}"));
    }
    a
}

/// Ek JPG ko in-place tag karta hai - bilkul Python version jaisa (ek file = ek exiftool call).
fn tag_one(exif: &Path, j: &TagJob, n: usize) -> Result<()> {
    let (w, h) = image::image_dimensions(&j.path)?;
    let mut lines: Vec<String> = ["-charset", "filename=utf8", "-overwrite_original", "-q", "-m"]
        .iter().map(|s| s.to_string()).collect();
    lines.extend(args_for(j, w, h));
    lines.push(j.path.to_string_lossy().to_string());
    let argfile = std::env::temp_dir().join(format!("exif_args_{}_{}.txt", std::process::id(), n));
    std::fs::write(&argfile, lines.join("\n") + "\n")?;
    let out = Command::new(exif).arg("-@").arg(&argfile).output();
    let _ = std::fs::remove_file(&argfile);
    let out = out?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        let so = String::from_utf8_lossy(&out.stdout);
        bail!("exiftool: {} {}", err.trim(), so.trim());
    }
    Ok(())
}

/// Quiet version (options 1 & 2 use this).
pub fn tag_batch(jobs: &[TagJob]) -> Result<()> { tag_jobs(jobs, false) }

/// Tag many JPGs in place (4 exiftool processes parallel).
pub fn tag_jobs(jobs: &[TagJob], verbose: bool) -> Result<()> {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    if jobs.is_empty() { return Ok(()); }
    let Some(exif) = find_exiftool() else {
        bail!("exiftool nahi mila. exiftool.exe (aur uska exiftool_files folder) ko image_tool.exe ke bagal me rakho ya PATH me daalo.");
    };
    if verbose { println!("exiftool: {}", exif.display()); }
    let ok = AtomicUsize::new(0);
    let failed = AtomicUsize::new(0);
    let counter = AtomicUsize::new(0);
    let total = jobs.len();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
    pool.install(|| {
        jobs.par_iter().for_each(|j| {
            let n = counter.fetch_add(1, Ordering::Relaxed);
            match tag_one(&exif, j, n) {
                Ok(()) => {
                    let d = ok.fetch_add(1, Ordering::Relaxed) + 1;
                    if verbose { println!("tagged {}/{}  {}", d, total, j.path.display()); }
                }
                Err(e) => {
                    failed.fetch_add(1, Ordering::Relaxed);
                    eprintln!("SKIP {}: {e}", j.path.display());
                }
            }
        });
    });
    if verbose || failed.load(Ordering::Relaxed) > 0 {
        println!("Metadata: {} tagged, {} failed", ok.load(Ordering::Relaxed), failed.load(Ordering::Relaxed));
    }
    Ok(())
}

/// (sku, brand, portal, index) from BASE/PORTAL/BRAND/SKU/SKU_N.jpg
pub fn names(base: &Path, dir: &Path, file: &str) -> (String, String, String, u32) {
    let rel = dir.strip_prefix(base).unwrap_or(dir);
    let mut parts: Vec<String> = rel.components().map(|c| c.as_os_str().to_string_lossy().to_string()).collect();
    while parts.len() < 3 { parts.insert(0, String::new()); }
    let n = parts.len();
    let stem = Path::new(file).file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let idx = stem.rsplit_once('_').and_then(|(_, d)| d.parse::<u32>().ok()).unwrap_or(1);
    (parts[n - 1].clone(), parts[n - 2].clone(), parts[n - 3].clone(), idx)
}

pub fn run_metadata(folder: &Path) -> Result<()> {
    println!("Folder: {}", folder.display());
    if !folder.is_dir() {
        bail!("Folder nahi mila: {}\nPehle option 2 chalao, ya `image_tool 3 \"D:\\path\\folder\"` se folder do.", folder.display());
    }
    let mut jobs = Vec::new();
    for e in walkdir::WalkDir::new(folder).sort_by_file_name().into_iter().flatten() {
        let p = e.path();
        let ext = p.extension().map(|x| x.to_string_lossy().to_lowercase()).unwrap_or_default();
        if e.file_type().is_file() && (ext == "jpg" || ext == "jpeg") {
            let (sku, brand, portal, index) = names(folder, p.parent().unwrap(), &e.file_name().to_string_lossy());
            jobs.push(TagJob { path: p.to_path_buf(), sku, brand, portal, index });
        }
    }
    if jobs.is_empty() {
        bail!("{} me koi .jpg/.jpeg image nahi mili.", folder.display());
    }
    println!("{} images mili, tagging shuru...", jobs.len());
    tag_jobs(&jobs, true)?;
    println!("Done: {}", folder.display());
    Ok(())
}
