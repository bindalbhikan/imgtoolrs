//! Bindal Exports - listing image tool (Rust port of main.py)
//!
//!   image_tool            (menu)
//!   image_tool 1 [--dry]
//!   image_tool 2 [--fast | --enhance=40] [--maxmb=2]
//!   image_tool 3 [folder]
//!
//! Folder layout (exe ke bagal me): input.xlsx, PATH.csv, models/, exiftool.exe (optional if in PATH)

mod crop;
mod extract;
mod meta;

use anyhow::Result;
use std::io::{self, Write};
use std::path::PathBuf;

fn ask(prompt: &str) -> String {
    print!("{prompt}");
    io::stdout().flush().ok();
    let mut s = String::new();
    io::stdin().read_line(&mut s).ok();
    s.trim().to_string()
}

fn main() -> Result<()> {
    // BASE = folder where the exe lives; in `cargo run` fall back to the current folder
    let base: PathBuf = std::env::current_exe().ok().and_then(|p| p.parent().map(|x| x.to_path_buf()))
        .filter(|p| p.join("input.xlsx").exists() || p.join("models").exists())
        .unwrap_or_else(|| std::env::current_dir().unwrap());
    let portalwise = base.join("OUTPUT_PORTALWISE");
    let cropped = base.join("OUTPUT_CROPPED");

    let all: Vec<String> = std::env::args().skip(1).collect();
    let args: Vec<&String> = all.iter().filter(|a| !a.starts_with("--")).collect();
    let mut choice = args.first().map(|s| s.to_string()).unwrap_or_default();
    if choice.is_empty() {
        println!("\n  1. Extract images        (input.xlsx -> OUTPUT_PORTALWISE)");
        println!("  2. Auto-crop + enhance   (OUTPUT_PORTALWISE -> OUTPUT_CROPPED)");
        println!("  3. Metadata only         (re-tag images in OUTPUT_CROPPED)\n");
        choice = ask("Choose 1 / 2 / 3: ");
    }
    match choice.as_str() {
        "1" => extract::run_extract(&base, &portalwise, all.iter().any(|a| a == "--dry"))?,
        "2" => {
            let (mut strength, mut max_mb) = (crop::ENHANCE, crop::MAX_MB);
            for a in &all {
                if let Some(v) = a.strip_prefix("--maxmb=") { max_mb = v.parse().unwrap_or(max_mb); }
                if a == "--fast" { strength = 0; }
                if let Some(v) = a.strip_prefix("--enhance=") { strength = v.parse().unwrap_or(strength); }
            }
            if args.is_empty() {
                let s = ask(&format!("Enhance strength 0-100 (0 = crop only, fast; Enter = {}): ", crop::ENHANCE));
                strength = s.parse().unwrap_or(crop::ENHANCE);
                let m = ask(&format!("Max image size in MB (e.g. 2 = under 2 MB; Enter = {}): ", crop::MAX_MB));
                max_mb = m.parse().unwrap_or(crop::MAX_MB);
            }
            let max_mb = max_mb.clamp(0.5, crop::MAX_MB);
            let strength = strength.min(100);
            println!("Enhance strength: {strength}%   Max size: {max_mb} MB");
            crop::run_autocrop(&base, &portalwise, &cropped, strength, max_mb)?;
        }
        "3" => {
            let folder = args.get(1).map(|s| PathBuf::from(s.as_str())).unwrap_or(cropped);
            meta::run_metadata(&folder)?;
        }
        other => println!("Invalid choice: {other}"),
    }
    if all.is_empty() { ask("\nEnter dabao band karne ke liye..."); }
    Ok(())
}
