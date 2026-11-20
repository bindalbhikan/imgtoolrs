# image_tool (Rust port of main.py)

## Build (Windows)
1. https://rustup.rs se Rust install karo
2. `cargo build --release`
3. `target\release\image_tool.exe` ko apne folder me copy karo jahan ye hain:
   input.xlsx, PATH.csv, models\ (segformer_b2_clothes.onnx, realesr-general-x4v3.onnx), exiftool.exe (ya PATH me)
4. RUN.bat chalao ya `image_tool.exe 2 --enhance=40 --maxmb=2`

## Python se farak
- exiftool ab 50 images ke batch me ek hi baar start hota hai (pehle har image par ek process)
- path.csv matching: har file par saare SKU loop nahi, prefix lookup
- Option 1 parallel (rayon) chalta hai
- JPEG: 4:2:0 + optimized huffman, quality binary-search (4-5 encodes, pehle 15 tak)
- exiftool nahi mila to error aayega (Python wala pure-python fallback port nahi kiya)

## GPU / speed
Default build CPU par chalta hai (GitHub Actions par DirectML build fail hota tha).
GPU chahiye to local par: `cargo build --release --features directml`
`--fast` (strength 0) me upscaler load hi nahi hota.
