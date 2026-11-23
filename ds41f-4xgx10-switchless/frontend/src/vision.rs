//! Vision (docs/vision.md): the image encoder's client. The encoder (tools/vision/vision_encoder.py, the
//! checkpoint's own ViT + aligner) turns each image record of a request into its span's rows: bf16
//! [rows][5120] for `[IMAGE_START] + ([IMAGE] * w + [IMAGE_NEW_LINE]) * h + [IMAGE_END]`. The prompt's
//! placeholder ids are expanded into that many image ids, and the rows go to the engine with the SUBMIT
//! (protocol FLAG_IMAGES). Text requests never come here.
//!
//! With `--vision engine` (`VisionKind::Engine`) the encoder runs in the engine: the media worker (tools/vision/media_worker.py,
//! `--media-addr`) answers each image's padded u8 pixels and grid instead of rows, and they go to the engine in the SUBMIT
//! (protocol FLAG_PIXELS; the layout is engine/server/pixels.h's). `encode_pixels` is that path's `encode`.
//!
//! Wire format (TCP, loopback): request u32 len + JSON {"images": [record, ...]}; reply u32 len + JSON
//! {"ok", "error", "images": [{"rows", "hash", "grid", "ms", "bytes"}]}, then u64 len + the rows.
//! The request also carries "max_rows", the cap on the rows of all its images together, so that an
//! encoder that knows it stops early; the reply's JSON is checked against the cap here before any of
//! the rows are read.

use serde_json::Value;
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// `<｜deepseek_image｜>`: the placeholder the chat encoding writes, and the id of every span position.
pub const IMAGE_TOKEN_ID: u32 = 129264;
pub const IMAGE_DIM: usize = 5120;
/// The encoder's JSON reply for images is a few hundred bytes per image.
const MAX_HEADER: usize = 16 << 20;
pub const MAX_IMAGES: usize = 100;
pub const MAX_IMAGE_ROWS: usize = 1100;
/// The default cap on the rows (image tokens) of all the images of one request (--max-image-rows).
/// Memory follows rows, not the number of images: a row is 10 KiB here (bf16 x 5120) and again in the
/// engine, so 100 full-size images (994 rows each) are ~1 GB per copy. 40,000 rows is what 100 images
/// of ~380 rows, or 40 full-size ones, take.
pub const DEFAULT_MAX_IMAGE_ROWS: usize = 40_000;
/// What the engine's frame limit has room for; --max-image-rows cannot be raised past it.
pub const MAX_TOTAL_IMAGE_ROWS: usize = MAX_IMAGES * MAX_IMAGE_ROWS;

/// Where the images are encoded (--vision).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VisionKind {
    /// The rollback: the encoder container (--vision-addr) answers rows; the SUBMIT carries rows (FLAG_IMAGES).
    Container,
    /// The default: the engine encodes; the media worker (--media-addr) answers padded pixels and the SUBMIT carries them (FLAG_PIXELS).
    Engine,
}

impl VisionKind {
    pub fn parse(s: &str) -> Option<VisionKind> {
        match s {
            "container" => Some(VisionKind::Container),
            "engine" => Some(VisionKind::Engine),
            _ => None,
        }
    }
}

/// Mixed into the hash of every image the engine encodes. The engine's rows and the container's differ in the last bits, so a
/// session made under one is no candidate under the other: the hash is part of what a session must match (session.rs).
/// The container's hash is sent as the encoder made it.
pub const ENGINE_ENCODER_KIND: u64 = 0x7069_7865_6c73_0001;

/// A pixel part's limits: the engine's own (pixels.h: `kMaxRows`, `vit::kMaxPatches`, `kPatchK`) and the frame stage it reads a
/// request's frame into (--frame-stage-mib, 400 MiB by default).
pub const MAX_PIXEL_ROWS: usize = 1024;
pub const MAX_PATCHES: usize = 9216;
pub const PATCH_BYTES: usize = 588;
pub const MAX_PIXEL_PART: usize = 400 << 20;

/// An image's span rows for a grid of `n_h` x `n_w` patches: the aligner merges 3 x 3 patches; IMAGE_START and IMAGE_END and one
/// IMAGE_NEW_LINE per grid row (pixels.h `rows_of`).
pub fn rows_of_grid(n_h: u64, n_w: u64) -> u64 {
    let (lh, lw) = (n_h.div_ceil(3), n_w.div_ceil(3));
    lh * (lw + 1) + 2
}

/// A multiple of 8 at or above `n`: the padding of the pixel part's arrays and images.
pub fn pad8(n: usize) -> usize {
    (n + 7) & !7
}

/// Whether an error of `encode_pixels` is a size the request may not have (answered 413): the media worker's pixel limit or the
/// frame stage's.
pub fn is_too_large(e: &str) -> bool {
    (e.contains("exceeds limit of") && e.contains("pixels")) || e.starts_with("at most ") && e.contains("MiB of image pixels")
}

/// The default for --max-image-rows-in-flight: two requests at the default request cap.
pub const DEFAULT_MAX_IMAGE_ROWS_IN_FLIGHT: usize = 2 * DEFAULT_MAX_IMAGE_ROWS;

/// The bytes of one image row held in memory (bf16 x IMAGE_DIM).
pub const IMAGE_ROW_BYTES: usize = IMAGE_DIM * 2;

/// The most rows `n_images` images can take, before the encoder has said what they do take:
/// MAX_IMAGE_ROWS each, and no more than the request cap (over it the request is refused anyway).
pub fn rows_estimate(n_images: usize, max_rows: usize) -> usize {
    (n_images * MAX_IMAGE_ROWS).min(max_rows.min(MAX_TOTAL_IMAGE_ROWS))
}

/// The rows held by the image requests in flight, against a limit (server.rs `hold_image_rows`).
#[derive(Debug)]
pub struct RowsInFlight {
    max: usize,
    held: std::sync::Mutex<usize>,
    /// The process is over its memory limit: no request with images is taken until it is not.
    refusing: std::sync::atomic::AtomicBool,
    /// The pixel bytes held by the requests in flight (`RowsHeld::set_pixel_bytes`), counted for the image memory when `pixels`.
    pixel_bytes: std::sync::atomic::AtomicUsize,
    pixels: std::sync::atomic::AtomicBool,
}

/// One request's share of the rows in flight; dropping it gives the rows back.
#[derive(Debug)]
pub struct RowsHeld {
    budget: std::sync::Arc<RowsInFlight>,
    rows: std::sync::Mutex<usize>,
    pixel_bytes: std::sync::atomic::AtomicUsize,
}

impl RowsInFlight {
    pub fn new(max: usize) -> std::sync::Arc<RowsInFlight> {
        std::sync::Arc::new(RowsInFlight { max, held: std::sync::Mutex::new(0),
            refusing: std::sync::atomic::AtomicBool::new(false),
            pixel_bytes: std::sync::atomic::AtomicUsize::new(0),
            pixels: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// Count the image memory in pixel bytes (the engine encodes: the front-end holds pixels, not rows) rather than in rows.
    pub fn set_pixels(&self, pixels: bool) {
        self.pixels.store(pixels, std::sync::atomic::Ordering::Relaxed);
    }

    /// The bytes the image requests in flight hold, for the session table's memory cap (`--front-image-mib`): their rows at
    /// IMAGE_ROW_BYTES each, or with `set_pixels` the pixels they hold.
    pub fn image_bytes(&self) -> usize {
        if self.pixels.load(std::sync::atomic::Ordering::Relaxed) {
            self.pixel_bytes.load(std::sync::atomic::Ordering::Relaxed)
        } else {
            self.in_flight() * IMAGE_ROW_BYTES
        }
    }

    /// Refuse new image requests (true) or take them again (false), as the process is over its memory limit or not.
    pub fn set_refusing(&self, refusing: bool) {
        self.refusing.store(refusing, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn refusing(&self) -> bool {
        self.refusing.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn max(&self) -> usize {
        self.max
    }

    pub fn in_flight(&self) -> usize {
        *self.held.lock().unwrap()
    }

    /// Hold `rows` if they fit under the limit with what is held already; otherwise the rows in flight.
    pub fn reserve(self: &std::sync::Arc<Self>, rows: usize) -> Result<RowsHeld, usize> {
        let mut held = self.held.lock().unwrap();
        if *held + rows > self.max {
            return Err(*held);
        }
        *held += rows;
        Ok(RowsHeld { budget: self.clone(), rows: std::sync::Mutex::new(rows), pixel_bytes: std::sync::atomic::AtomicUsize::new(0) })
    }
}

impl RowsHeld {
    pub fn rows(&self) -> usize {
        *self.rows.lock().unwrap()
    }

    /// This request holds `bytes` of pixels (its pixel part), from now on in place of what it held.
    pub fn set_pixel_bytes(&self, bytes: usize) {
        use std::sync::atomic::Ordering::Relaxed;
        let before = self.pixel_bytes.swap(bytes, Relaxed);
        self.budget.pixel_bytes.fetch_add(bytes, Relaxed);
        self.budget.pixel_bytes.fetch_sub(before, Relaxed);
    }

    /// Hold `rows` from now on, when that is less than what is held (it never grows: growing could
    /// pass the limit).
    pub fn shrink_to(&self, rows: usize) {
        let mut mine = self.rows.lock().unwrap();
        if rows < *mine {
            *self.budget.held.lock().unwrap() -= *mine - rows;
            *mine = rows;
        }
    }
}

impl Drop for RowsHeld {
    fn drop(&mut self) {
        let mine = *self.rows.lock().unwrap();
        *self.budget.held.lock().unwrap() -= mine;
        self.budget.pixel_bytes.fetch_sub(self.pixel_bytes.load(std::sync::atomic::Ordering::Relaxed), std::sync::atomic::Ordering::Relaxed);
    }
}

#[derive(Debug, Clone)]
pub struct Encoded {
    /// rows per image, in prompt order
    pub rows: Vec<u32>,
    pub hashes: Vec<u64>,
    /// The SUBMIT's image part, as `payload` completes it: room for the span table (u32 n and 16 bytes
    /// per image), then every image's rows, bf16 little-endian, in order. The rows are read from the
    /// encoder straight into their place here, so the front-end holds them once, not once as the reply
    /// and again as the SUBMIT's part.
    pub data: Vec<u8>,
    pub ms: f64,
    /// The part carries pixels (FLAG_PIXELS), not rows (FLAG_IMAGES).
    pub pixels: bool,
}

/// A loopback encoder RPC whose answer is its JSON header: video expansion (video.rs), which returns
/// frame records and no binary data. Image rows go through `encode`, which checks the header against
/// the row cap before it reads them.
pub(crate) async fn exchange(addr: &str, body: &Value, timeout: Duration, max_data: usize) -> Result<(Value, Vec<u8>), String> {
    let body = body.to_string().into_bytes();
    if body.len() > 256 << 20 {
        return Err("media request exceeds 256 MiB".into());
    }
    let fut = async {
        let mut s = TcpStream::connect(addr).await.map_err(|e| format!("the image encoder is not reachable: {e}"))?;
        s.write_all(&(body.len() as u32).to_le_bytes()).await.map_err(|e| e.to_string())?;
        s.write_all(&body).await.map_err(|e| e.to_string())?;
        let mut n4 = [0u8; 4];
        s.read_exact(&mut n4).await.map_err(|e| format!("image encoder: {e}"))?;
        let header_len = u32::from_le_bytes(n4) as usize;
        if header_len > 128 << 20 {
            return Err("image encoder: JSON reply too large".into());
        }
        let mut h = vec![0u8; header_len];
        s.read_exact(&mut h).await.map_err(|e| format!("image encoder: {e}"))?;
        let mut n8 = [0u8; 8];
        s.read_exact(&mut n8).await.map_err(|e| format!("image encoder: {e}"))?;
        let n = u64::from_le_bytes(n8) as usize;
        if n > max_data {
            return Err("image encoder: reply too large".to_string());
        }
        let mut data = vec![0u8; n];
        s.read_exact(&mut data).await.map_err(|e| format!("image encoder: {e}"))?;
        let h: Value = serde_json::from_slice(&h).map_err(|e| format!("image encoder: {e}"))?;
        Ok::<_, String>((h, data))
    };
    tokio::time::timeout(timeout, fut).await.map_err(|_| "the image encoder timed out".to_string())?
}

/// Whether an error of `encode` is the request's fault (answered 400) rather than the encoder's (503).
pub fn is_request_error(e: &str) -> bool {
    e.starts_with("cannot read") || e.starts_with("at most ")
}

fn too_many_rows(max_rows: usize, total: usize) -> String {
    // video frames are images by now (video.rs): they count like any other
    format!("at most {max_rows} image tokens per request (--max-image-rows); the images and video frames of this request take {total}")
}

/// The rows and hash of each image from the encoder's reply header, checked against the request and the
/// cap on its rows in total. Nothing of the rows themselves has been read or allocated at this point.
fn spans(h: &Value, n_records: usize, max_rows: usize) -> Result<(Vec<u32>, Vec<u64>), String> {
    if h.get("ok").and_then(Value::as_bool) != Some(true) {
        let e = h.get("error").and_then(Value::as_str).unwrap_or("unknown error");
        // an encoder that knows the cap stops at the image that crosses it
        if h.get("code").and_then(Value::as_str) == Some("too_many_rows") && e.starts_with("at most ") {
            return Err(e.to_string());
        }
        return Err(format!("cannot read the image: {e}"));
    }
    let imgs = h.get("images").and_then(Value::as_array).ok_or("image encoder: no images in its reply")?;
    if imgs.len() != n_records {
        return Err("image encoder: wrong number of images".into());
    }
    let mut rows = Vec::new();
    let mut hashes = Vec::new();
    for i in imgs {
        let r = i.get("rows").and_then(Value::as_u64).ok_or("image encoder: rows")?;
        if !(3..=MAX_IMAGE_ROWS as u64).contains(&r) {
            return Err("image encoder: bad span".into());
        }
        rows.push(r as u32);
        hashes.push(i.get("hash").and_then(Value::as_u64).unwrap_or(0));
    }
    let total: usize = rows.iter().map(|&r| r as usize).sum();
    if total > max_rows {
        return Err(too_many_rows(max_rows, total));
    }
    Ok((rows, hashes))
}

/// The images' rows from the encoder. `max_rows` caps the rows of all the images together: over it the
/// request is refused (an error that `is_request_error`) before any row is read from the encoder.
pub async fn encode(addr: &str, records: &[Value], timeout: Duration, max_rows: usize) -> Result<Encoded, String> {
    if records.len() > MAX_IMAGES {
        return Err(format!("at most {MAX_IMAGES} images per request"));
    }
    let max_rows = max_rows.min(MAX_TOTAL_IMAGE_ROWS);
    let t0 = Instant::now();
    let fut = async {
        let mut s = TcpStream::connect(addr).await.map_err(|e| format!("the image encoder is not reachable: {e}"))?;
        // serialised from the records where they are (no copy of them first), and freed once sent:
        // inline images are megabytes each, and the wait for the encoder is the long part
        let body = EncodeRequest { images: records, max_rows }.to_bytes()?;
        s.write_all(&(body.len() as u32).to_le_bytes()).await.map_err(|e| e.to_string())?;
        s.write_all(&body).await.map_err(|e| e.to_string())?;
        drop(body);
        let mut n4 = [0u8; 4];
        s.read_exact(&mut n4).await.map_err(|e| format!("image encoder: {e}"))?;
        let header_len = u32::from_le_bytes(n4) as usize;
        if header_len > MAX_HEADER {
            return Err("image encoder: JSON reply too large".to_string());
        }
        let mut h = vec![0u8; header_len];
        s.read_exact(&mut h).await.map_err(|e| format!("image encoder: {e}"))?;
        let h: Value = serde_json::from_slice(&h).map_err(|e| format!("image encoder: {e}"))?;
        // the spans, and the cap on their rows, before the rows: a refused request reads and allocates none
        let (rows, hashes) = spans(&h, records.len(), max_rows)?;
        let total: usize = rows.iter().map(|&r| r as usize).sum();
        let mut n8 = [0u8; 8];
        s.read_exact(&mut n8).await.map_err(|e| format!("image encoder: {e}"))?;
        if u64::from_le_bytes(n8) != (total * IMAGE_DIM * 2) as u64 {
            return Err("image encoder: the rows do not match the spans".to_string());
        }
        let table = span_table_len(rows.len());
        let mut data = vec![0u8; table + total * IMAGE_DIM * 2];
        s.read_exact(&mut data[table..]).await.map_err(|e| format!("image encoder: {e}"))?;
        Ok::<_, String>((rows, hashes, data))
    };
    let (rows, hashes, data) = tokio::time::timeout(timeout, fut).await.map_err(|_| "the image encoder timed out".to_string())??;
    Ok(Encoded { rows, hashes, data, ms: t0.elapsed().as_secs_f64() * 1e3, pixels: false })
}

/// One image of the media worker's reply, from its frame's JSON `{"ok": true, "image": {rows, hash, grid, bytes, ...}}`: its span rows,
/// hash and grid (n_h, n_w patches), the rows checked against the grid and the engine's limits.
fn image_frame(h: &Value) -> Result<(u32, u64, (u16, u16), usize), String> {
    let i = h.get("image").ok_or("image encoder: no image in its frame")?;
    let r = i.get("rows").and_then(Value::as_u64).ok_or("image encoder: rows")?;
    if !(3..=MAX_IMAGE_ROWS as u64).contains(&r) {
        return Err("image encoder: bad span".into());
    }
    let g = i.get("grid").and_then(Value::as_array).ok_or("image encoder: no grid")?;
    let at = |k: usize| g.get(k).and_then(Value::as_u64).filter(|&v| (1..=u16::MAX as u64).contains(&v)).ok_or("image encoder: bad grid");
    let (n_h, n_w) = (at(0)?, at(1)?);
    if n_h * n_w > MAX_PATCHES as u64 {
        return Err(format!("cannot read the image: {} patches, over the {MAX_PATCHES} the encoder takes", n_h * n_w));
    }
    if rows_of_grid(n_h, n_w) != r {
        return Err("image encoder: the grid does not match its rows".into());
    }
    if r as usize > MAX_PIXEL_ROWS {
        return Err(format!("cannot read the image: it takes {r} rows, over the {MAX_PIXEL_ROWS} an image takes"));
    }
    Ok((r as u32, i.get("hash").and_then(Value::as_u64).unwrap_or(0), (n_h as u16, n_w as u16), PATCH_BYTES * n_h as usize * n_w as usize))
}

/// The images' padded pixels from the media worker, as the SUBMIT's pixel part (`encode_pixels_counted` with the frame stage's
/// cap and no count).
pub async fn encode_pixels(addr: &str, records: &[Value], timeout: Duration, max_rows: usize) -> Result<Encoded, String> {
    encode_pixels_counted(addr, records, timeout, max_rows, MAX_PIXEL_PART, None).await
}

/// The images' padded pixels from the media worker, as the SUBMIT's pixel part (engine/server/pixels.h): `u32 n`, n spans (the
/// table `payload` writes), n grids of `u16 n_h, u16 n_w` padded to 8 bytes, then each image's `588 x n_h x n_w` bytes padded to
/// 8. The worker answers a run of frames, each `u32 len, JSON, u64 len, bytes`: one per image in order (`{"ok": true, "image":
/// {rows, hash, grid, ms, bytes}}` and its padded u8), then the end frame `{"ok": true, "images": n}`; an error frame
/// `{"ok": false, "error", "code"}` ends the reply at any image and what came before it is dropped. Each image is read straight
/// into its place in the part, and the part's bytes are counted as the frames arrive (`held`, for `--front-image-mib`): a request
/// is refused after the frame that takes it over `max_bytes` (at most `MAX_PIXEL_PART`, the engine's frame stage), an error
/// `is_too_large`, as it is over `max_rows` (the rows of all the images together, as for `encode`). The hashes carry
/// `ENGINE_ENCODER_KIND`.
pub async fn encode_pixels_counted(addr: &str, records: &[Value], timeout: Duration, max_rows: usize, max_bytes: usize, held: Option<&RowsHeld>) -> Result<Encoded, String> {
    if records.len() > MAX_IMAGES {
        return Err(format!("at most {MAX_IMAGES} images per request"));
    }
    let max_rows = max_rows.min(MAX_TOTAL_IMAGE_ROWS);
    let max_bytes = max_bytes.min(MAX_PIXEL_PART);
    let n = records.len();
    let t0 = Instant::now();
    let fut = async {
        let mut s = TcpStream::connect(addr).await.map_err(|e| format!("the image encoder is not reachable: {e}"))?;
        let body = EncodeRequest { images: records, max_rows }.to_bytes()?;
        s.write_all(&(body.len() as u32).to_le_bytes()).await.map_err(|e| e.to_string())?;
        s.write_all(&body).await.map_err(|e| e.to_string())?;
        drop(body);
        let grid_at = span_table_len(n);
        let data_at = grid_at + pad8(4 * n);
        let mut data = vec![0u8; data_at];
        let (mut rows, mut hashes) = (Vec::with_capacity(n), Vec::with_capacity(n));
        loop {
            let mut n4 = [0u8; 4];
            s.read_exact(&mut n4).await.map_err(|e| format!("image encoder: {e}"))?;
            let header_len = u32::from_le_bytes(n4) as usize;
            if header_len > MAX_HEADER {
                return Err("image encoder: JSON reply too large".to_string());
            }
            let mut h = vec![0u8; header_len];
            s.read_exact(&mut h).await.map_err(|e| format!("image encoder: {e}"))?;
            let h: Value = serde_json::from_slice(&h).map_err(|e| format!("image encoder: {e}"))?;
            if h.get("ok").and_then(Value::as_bool) != Some(true) {
                let e = h.get("error").and_then(Value::as_str).unwrap_or("unknown error");
                // a worker that knows the cap stops at the image that crosses it
                if h.get("code").and_then(Value::as_str) == Some("too_many_rows") && e.starts_with("at most ") {
                    return Err(e.to_string());
                }
                return Err(format!("cannot read the image: {e}"));
            }
            if let Some(total) = h.get("images").and_then(Value::as_u64) {
                if total as usize != n || rows.len() != n {
                    return Err("image encoder: wrong number of images".into());
                }
                break;
            }
            if rows.len() == n {
                return Err("image encoder: wrong number of images".into());
            }
            let (r, hash, (n_h, n_w), bytes) = image_frame(&h)?;
            rows.push(r);
            hashes.push(hash ^ ENGINE_ENCODER_KIND);
            let total_rows: usize = rows.iter().map(|&r| r as usize).sum();
            if total_rows > max_rows {
                return Err(too_many_rows(max_rows, total_rows));
            }
            let part = data.len() + pad8(bytes);
            if part > max_bytes {
                return Err(format!("at most {} MiB of image pixels per request; the images of this request take {} MiB", max_bytes >> 20, part >> 20));
            }
            let mut n8 = [0u8; 8];
            s.read_exact(&mut n8).await.map_err(|e| format!("image encoder: {e}"))?;
            if u64::from_le_bytes(n8) != bytes as u64 {
                return Err("image encoder: the pixels do not match the grid".to_string());
            }
            let i = rows.len() - 1;
            data[grid_at + 4 * i..grid_at + 4 * i + 2].copy_from_slice(&n_h.to_le_bytes());
            data[grid_at + 4 * i + 2..grid_at + 4 * i + 4].copy_from_slice(&n_w.to_le_bytes());
            let at = data.len();
            data.reserve_exact(pad8(bytes));
            data.resize(at + pad8(bytes), 0);
            s.read_exact(&mut data[at..at + bytes]).await.map_err(|e| format!("image encoder: {e}"))?;
            if let Some(held) = held {
                held.set_pixel_bytes(data.len());
            }
        }
        Ok::<_, String>((rows, hashes, data))
    };
    let r = tokio::time::timeout(timeout, fut).await.map_err(|_| "the image encoder timed out".to_string())?;
    if r.is_err() {
        // what came before an error is dropped, and so is what it counted
        if let Some(held) = held {
            held.set_pixel_bytes(0);
        }
    }
    let (rows, hashes, data) = r?;
    Ok(Encoded { rows, hashes, data, ms: t0.elapsed().as_secs_f64() * 1e3, pixels: true })
}

/// The prompt's ids with each placeholder expanded into its image's span, and the spans' starts.
pub fn expand(ids: &[u32], rows: &[u32]) -> Result<(Vec<u32>, Vec<u32>), String> {
    let n = ids.iter().filter(|&&t| t == IMAGE_TOKEN_ID).count();
    if n != rows.len() {
        return Err(format!("the prompt has {n} image placeholders for {} images", rows.len()));
    }
    let extra: usize = rows.iter().map(|&r| r as usize - 1).sum();
    let mut out = Vec::with_capacity(ids.len() + extra);
    let mut starts = Vec::with_capacity(rows.len());
    let mut k = 0;
    for &t in ids {
        if t == IMAGE_TOKEN_ID {
            starts.push(out.len() as u32);
            out.extend(std::iter::repeat(IMAGE_TOKEN_ID).take(rows[k] as usize));
            k += 1;
        } else {
            out.push(t);
        }
    }
    Ok((out, starts))
}

/// The bytes of the SUBMIT's span table for n images: u32 n, then {u32 start, u32 rows, u64 hash} each.
fn span_table_len(n: usize) -> usize {
    4 + 16 * n
}

/// The prompt's text with each placeholder written once per row of its image (so that the text spells the
/// expanded ids), and where each image's run starts in the result (byte offsets).
pub fn expand_text(text: &str, rows: &[u32]) -> Result<(String, Vec<usize>), String> {
    let ph = crate::encoding::IMAGE_PLACEHOLDER;
    let n = text.matches(ph).count();
    if n != rows.len() {
        return Err(format!("the prompt has {n} image placeholders for {} images", rows.len()));
    }
    let mut out = String::with_capacity(text.len());
    let mut ats = Vec::with_capacity(n);
    let mut last = 0;
    for (k, (i, _)) in text.match_indices(ph).enumerate() {
        out.push_str(&text[last..i]);
        ats.push(out.len());
        for _ in 0..rows[k] {
            out.push_str(ph);
        }
        last = i + ph.len();
    }
    out.push_str(&text[last..]);
    Ok((out, ats))
}

/// The starts of the image spans in expanded ids: each run of `rows[k]` image ids, in order. An error when
/// the ids hold other image ids than those spans (the engine refuses such a prompt).
pub fn span_starts(ids: &[u32], rows: &[u32]) -> Result<Vec<u32>, String> {
    let mut starts = Vec::with_capacity(rows.len());
    let mut i = 0;
    while i < ids.len() {
        if ids[i] != IMAGE_TOKEN_ID {
            i += 1;
            continue;
        }
        let Some(&r) = rows.get(starts.len()) else { return Err("the prompt has more image ids than images".into()) };
        let end = i + r as usize;
        if end > ids.len() || ids[i..end].iter().any(|&t| t != IMAGE_TOKEN_ID) {
            return Err("an image's ids are cut short".into());
        }
        starts.push(i as u32);
        i = end;
    }
    if starts.len() != rows.len() {
        return Err(format!("the prompt has {} image spans for {} images", starts.len(), rows.len()));
    }
    Ok(starts)
}

/// The SUBMIT's image part (after prompt[]): u32 n, {u32 start, u32 rows, u64 hash} per image, the rows.
/// The rows are already in place in `e.data`; this writes the table in front of them and hands the
/// buffer over, without copying the rows.
pub fn payload(starts: &[u32], e: Encoded) -> Vec<u8> {
    assert_eq!(starts.len(), e.rows.len(), "one span per image");
    let mut v = e.data;
    let table = &mut v[..span_table_len(starts.len())];
    table[..4].copy_from_slice(&(starts.len() as u32).to_le_bytes());
    for i in 0..starts.len() {
        let at = 4 + 16 * i;
        table[at..at + 4].copy_from_slice(&starts[i].to_le_bytes());
        table[at + 4..at + 8].copy_from_slice(&e.rows[i].to_le_bytes());
        table[at + 8..at + 16].copy_from_slice(&e.hashes[i].to_le_bytes());
    }
    v
}

/// The encoder's request, serialised from the image records in place.
#[derive(serde::Serialize)]
struct EncodeRequest<'a> {
    images: &'a [Value],
    max_rows: usize,
}

impl EncodeRequest<'_> {
    /// The request's JSON in a buffer of exactly its size: measured first, so that megabytes of inline
    /// images are not also held in the slack of a buffer grown by doubling.
    fn to_bytes(&self) -> Result<Vec<u8>, String> {
        struct Count(usize);
        impl std::io::Write for Count {
            fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
                self.0 += b.len();
                Ok(b.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let mut n = Count(0);
        serde_json::to_writer(&mut n, self).map_err(|e| e.to_string())?;
        let mut v = Vec::with_capacity(n.0);
        serde_json::to_writer(&mut v, self).map_err(|e| e.to_string())?;
        Ok(v)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn expands_placeholders() {
        let (ids, st) = expand(&[0, 5, IMAGE_TOKEN_ID, 6, IMAGE_TOKEN_ID], &[3, 4]).unwrap();
        assert_eq!(ids.len(), 3 + 3 + 4);
        assert_eq!(st, vec![2, 6]);
        assert!(ids[2..5].iter().all(|&t| t == IMAGE_TOKEN_ID) && ids[5] == 6);
        assert!(expand(&[IMAGE_TOKEN_ID], &[3, 3]).is_err());
    }

    /// A stand-in for tools/vision/vision_encoder.py: answers one request with `reply`, then the rows
    /// of its images (zeros) when `send_rows`; otherwise it sends nothing more and keeps the connection
    /// open. Returns its address and, when it is done, the request it was sent.
    pub(crate) async fn fake_encoder(reply: Value, send_rows: bool) -> (String, tokio::task::JoinHandle<Value>) {
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = l.local_addr().unwrap().to_string();
        let h = tokio::spawn(async move {
            let (mut s, _) = l.accept().await.unwrap();
            let mut n4 = [0u8; 4];
            s.read_exact(&mut n4).await.unwrap();
            let mut body = vec![0u8; u32::from_le_bytes(n4) as usize];
            s.read_exact(&mut body).await.unwrap();
            let hdr = reply.to_string().into_bytes();
            s.write_all(&(hdr.len() as u32).to_le_bytes()).await.unwrap();
            s.write_all(&hdr).await.unwrap();
            if send_rows {
                let rows: u64 = reply["images"].as_array().map_or(0, |a| a.iter().map(|i| i["rows"].as_u64().unwrap()).sum());
                let n = rows as usize * IMAGE_DIM * 2;
                s.write_all(&(n as u64).to_le_bytes()).await.unwrap();
                let _ = s.write_all(&row_bytes(n)).await;
            } else {
                // the client must answer from the header alone: nothing else ever comes
                let _ = s.read(&mut [0u8; 1]).await;
            }
            serde_json::from_slice(&body).unwrap()
        });
        (addr, h)
    }

    /// What the stand-in encoder sends as rows: a pattern, so that a test sees where each byte lands.
    fn row_bytes(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i % 251) as u8).collect()
    }

    #[tokio::test]
    async fn rows_land_in_the_submit_part_without_a_copy() {
        let rows = [3u32, 5, 4];
        let (addr, _enc) = fake_encoder(reply_with_rows(&rows), true).await;
        let e = encode(&addr, &records(3), T, DEFAULT_MAX_IMAGE_ROWS).await.unwrap();
        let n = 12 * IMAGE_DIM * 2;
        assert_eq!(e.data.len(), 4 + 16 * 3 + n, "room for the span table, then the rows");
        let (at, hashes) = (e.data.as_ptr(), e.hashes.clone());
        let part = payload(&[7, 20, 40], e);
        assert_eq!(part.as_ptr(), at, "the same buffer: the rows are not copied");
        // the layout the engine reads (docs/protocol.md): u32 n, {u32 start, u32 rows, u64 hash} each, the rows
        let mut want = 3u32.to_le_bytes().to_vec();
        for (i, (start, r)) in [7u32, 20, 40].iter().zip(rows).enumerate() {
            want.extend_from_slice(&start.to_le_bytes());
            want.extend_from_slice(&r.to_le_bytes());
            want.extend_from_slice(&hashes[i].to_le_bytes());
        }
        want.extend_from_slice(&row_bytes(n));
        assert!(part == want, "span table and rows as before");
        // and the request the encoder gets is the same JSON as before, plus the cap
        let (addr, enc) = fake_encoder(reply_with_rows(&[3]), true).await;
        let recs = vec![json!({"url": "data:image/png;base64,QUJD", "detail": "high"})];
        encode(&addr, &recs, T, 123).await.unwrap();
        assert_eq!(enc.await.unwrap(), json!({"images": recs, "max_rows": 123}));
        let bytes = EncodeRequest { images: &recs, max_rows: 123 }.to_bytes().unwrap();
        assert_eq!(bytes, json!({"images": recs, "max_rows": 123}).to_string().into_bytes());
        assert_eq!(bytes.capacity(), bytes.len(), "sized once");
    }

    pub(crate) fn reply_with_rows(rows: &[u32]) -> Value {
        json!({"ok": true, "images": rows.iter().enumerate().map(|(i, r)| json!({"rows": r, "hash": i})).collect::<Vec<_>>()})
    }

    fn records(n: usize) -> Vec<Value> {
        vec![json!({"url": "data:image/png;base64,AAAA"}); n]
    }

    const T: Duration = Duration::from_secs(20);

    #[test]
    fn rows_in_flight() {
        assert_eq!(DEFAULT_MAX_IMAGE_ROWS_IN_FLIGHT, 80_000);
        // the estimate before the encoder answers: 1100 a picture, at most the request cap
        assert_eq!(rows_estimate(1, DEFAULT_MAX_IMAGE_ROWS), 1100);
        assert_eq!(rows_estimate(36, DEFAULT_MAX_IMAGE_ROWS), 39_600);
        assert_eq!(rows_estimate(37, DEFAULT_MAX_IMAGE_ROWS), 40_000);
        assert_eq!(rows_estimate(100, usize::MAX), MAX_TOTAL_IMAGE_ROWS);
        let b = RowsInFlight::new(10);
        let x = b.reserve(6).unwrap();
        assert_eq!(b.reserve(5).unwrap_err(), 6, "5 more do not fit: told what is in flight, and nothing is taken");
        let y = b.reserve(4).unwrap();
        assert_eq!((b.in_flight(), b.reserve(1).is_err()), (10, true));
        x.shrink_to(2);
        assert_eq!((x.rows(), b.in_flight()), (2, 6));
        x.shrink_to(9); // never grows
        assert_eq!((x.rows(), b.in_flight()), (2, 6));
        drop(y);
        assert_eq!(b.in_flight(), 2);
        drop(x);
        assert_eq!(b.in_flight(), 0);
        assert!(b.reserve(11).is_err() && b.reserve(10).is_ok());
        // from many threads at once the limit holds and everything comes back
        let b = RowsInFlight::new(1000);
        let over = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let (b, over) = (b.clone(), over.clone());
                std::thread::spawn(move || {
                    for i in 0..20_000 {
                        if let Ok(h) = b.reserve(300 + i % 7) {
                            if b.in_flight() > 1000 {
                                over.store(true, std::sync::atomic::Ordering::Relaxed);
                            }
                            h.shrink_to(100);
                        }
                    }
                })
            })
            .collect();
        threads.into_iter().for_each(|t| t.join().unwrap());
        assert!(!over.load(std::sync::atomic::Ordering::Relaxed));
        assert_eq!(b.in_flight(), 0);
    }

    #[tokio::test]
    async fn hundred_small_images_pass() {
        // 100 images of 383 rows (a 768 x 768 picture) are 38,300 rows: under the default cap
        let rows = vec![383u32; 100];
        assert!(rows.iter().sum::<u32>() as usize <= DEFAULT_MAX_IMAGE_ROWS);
        let (h, _) = spans(&reply_with_rows(&rows), 100, DEFAULT_MAX_IMAGE_ROWS).unwrap();
        assert_eq!(h, rows);
        // and end to end with the stand-in encoder (short spans, so the test moves 3 MB rather than 392)
        let (addr, enc) = fake_encoder(reply_with_rows(&[3; 100]), true).await;
        let e = encode(&addr, &records(100), T, DEFAULT_MAX_IMAGE_ROWS).await.unwrap();
        assert_eq!((e.rows.len(), e.data.len()), (100, 4 + 16 * 100 + 300 * IMAGE_DIM * 2));
        let sent = enc.await.unwrap();
        assert_eq!(sent["max_rows"], DEFAULT_MAX_IMAGE_ROWS, "the encoder is told the cap");
        assert_eq!(sent["images"].as_array().unwrap().len(), 100);
    }

    #[tokio::test]
    async fn rows_over_the_cap_are_refused_before_any_row_is_read() {
        // 41 full-size images (994 rows each) are 40,754 rows: over the default cap of 40,000
        let (addr, _enc) = fake_encoder(reply_with_rows(&[994; 41]), false).await;
        let t0 = Instant::now();
        let e = encode(&addr, &records(41), T, DEFAULT_MAX_IMAGE_ROWS).await.unwrap_err();
        assert_eq!(e, "at most 40000 image tokens per request (--max-image-rows); the images and video frames of this request take 40754");
        assert!(is_request_error(&e), "answered 400, not 503");
        assert!(t0.elapsed() < Duration::from_secs(5), "refused from the reply's header, without waiting for rows");
        // 40 of them are exactly 39,760: allowed
        assert!(spans(&reply_with_rows(&[994; 40]), 40, DEFAULT_MAX_IMAGE_ROWS).is_ok());
    }

    #[tokio::test]
    async fn the_cap_is_a_setting() {
        // exactly at the cap passes, one row over does not
        assert!(spans(&reply_with_rows(&[3, 4, 5]), 3, 12).is_ok());
        assert!(spans(&reply_with_rows(&[3, 4, 5]), 3, 11).unwrap_err().starts_with("at most 11 image tokens"));
        // raised, 100 full-size images fit; the engine's frame limit is the ceiling whatever is asked
        assert!(spans(&reply_with_rows(&[994; 100]), 100, MAX_TOTAL_IMAGE_ROWS).is_ok());
        let (addr, enc) = fake_encoder(reply_with_rows(&[3]), true).await;
        encode(&addr, &records(1), T, usize::MAX).await.unwrap();
        assert_eq!(enc.await.unwrap()["max_rows"], MAX_TOTAL_IMAGE_ROWS);
        assert!(MAX_TOTAL_IMAGE_ROWS * IMAGE_DIM * 2 + (8 << 20) <= crate::protocol::MAX_FRAME);
    }

    #[tokio::test]
    async fn errors_are_sorted_into_the_requests_and_the_encoders() {
        // an encoder that knows the cap stops early and says so: the request's fault
        let stop = json!({"ok": false, "code": "too_many_rows", "error": "at most 40000 image tokens per request; the first 41 of 100 images already take 40754"});
        let (addr, _enc) = fake_encoder(stop, false).await;
        let e = encode(&addr, &records(100), T, DEFAULT_MAX_IMAGE_ROWS).await.unwrap_err();
        assert!(e.starts_with("at most 40000 image tokens") && is_request_error(&e), "{e}");
        // an unreadable image: the request's fault, as before
        let (addr, _enc) = fake_encoder(json!({"ok": false, "error": "cannot identify image file"}), false).await;
        let e = encode(&addr, &records(1), T, DEFAULT_MAX_IMAGE_ROWS).await.unwrap_err();
        assert_eq!(e, "cannot read the image: cannot identify image file");
        assert!(is_request_error(&e));
        // more than 100 images never reaches the encoder
        let e = encode("127.0.0.1:1", &records(101), T, DEFAULT_MAX_IMAGE_ROWS).await.unwrap_err();
        assert_eq!(e, "at most 100 images per request");
        // a broken or absent encoder is not the request's fault
        let (addr, _enc) = fake_encoder(reply_with_rows(&[3, 3]), true).await;
        let e = encode(&addr, &records(1), T, DEFAULT_MAX_IMAGE_ROWS).await.unwrap_err();
        assert!(!is_request_error(&e), "{e}");
        let (addr, _enc) = fake_encoder(reply_with_rows(&[2000]), true).await;
        assert!(!is_request_error(&encode(&addr, &records(1), T, DEFAULT_MAX_IMAGE_ROWS).await.unwrap_err()));
        let gone = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap().local_addr().unwrap().to_string();
        let e = encode(&gone, &records(1), T, DEFAULT_MAX_IMAGE_ROWS).await.unwrap_err();
        assert!(e.starts_with("the image encoder is not reachable") && !is_request_error(&e), "{e}");
    }

    #[test]
    fn expands_text_and_finds_spans() {
        let ph = crate::encoding::IMAGE_PLACEHOLDER;
        let (t, at) = expand_text(&format!("a{ph}b{ph}"), &[3, 4]).unwrap();
        assert_eq!(t, format!("a{}b{}", ph.repeat(3), ph.repeat(4)));
        assert_eq!(at, vec![1, 1 + 3 * ph.len() + 1]);
        assert!(expand_text(ph, &[3, 3]).is_err());
        let ids = [0, 5, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 6, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 7];
        assert_eq!(span_starts(&ids, &[3, 4]).unwrap(), vec![2, 6]);
        assert!(span_starts(&ids, &[3, 3]).is_err());
        assert!(span_starts(&ids, &[3]).is_err());
        assert!(span_starts(&ids, &[4, 4]).is_err());
    }

    // ------------------------------------------------------------------ pixels (--vision engine)

    /// What a stand-in media worker answers for an image: its padded pixels are `fill` bytes, 588 x n_h x n_w of them.
    #[derive(Clone, Copy)]
    pub(crate) struct Img {
        pub n_h: u16,
        pub n_w: u16,
        pub fill: u8,
        pub hash: u64,
    }

    /// The frames the media worker answers for `imgs`: one per image, `({"ok": true, "image": {rows, hash, grid, ms, bytes}}, its
    /// padded pixels)`, then the end frame `({"ok": true, "images": n}, no bytes)`.
    pub(crate) fn media_frames(imgs: &[Img]) -> Vec<(Value, Vec<u8>)> {
        let mut v: Vec<(Value, Vec<u8>)> = imgs
            .iter()
            .map(|i| {
                let (lh, lw) = ((i.n_h as u64).div_ceil(3), (i.n_w as u64).div_ceil(3));
                let bytes = 588 * i.n_h as usize * i.n_w as usize;
                (json!({"ok": true, "image": {"rows": rows_of_grid(i.n_h as u64, i.n_w as u64), "hash": i.hash, "grid": [i.n_h, i.n_w, lh, lw], "ms": 1.0, "bytes": bytes}}), vec![i.fill; bytes])
            })
            .collect();
        v.push((json!({"ok": true, "images": imgs.len()}), Vec::new()));
        v
    }

    /// A stand-in for tools/vision/media_worker.py: answers one request with `frames`, each as `u32 len, JSON, u64 len, bytes`. After
    /// the frame at index `pause.0` it waits for `pause.1` to be notified (a test looks at the reader meanwhile). After the last frame
    /// it keeps the connection open until the reader goes. Returns its address and the request it was sent.
    pub(crate) async fn fake_media_frames(frames: Vec<(Value, Vec<u8>)>, pause: Option<(usize, std::sync::Arc<tokio::sync::Notify>)>) -> (String, tokio::task::JoinHandle<Value>) {
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = l.local_addr().unwrap().to_string();
        let h = tokio::spawn(async move {
            let (mut s, _) = l.accept().await.unwrap();
            let mut n4 = [0u8; 4];
            s.read_exact(&mut n4).await.unwrap();
            let mut body = vec![0u8; u32::from_le_bytes(n4) as usize];
            s.read_exact(&mut body).await.unwrap();
            for (i, (j, data)) in frames.into_iter().enumerate() {
                let hdr = j.to_string().into_bytes();
                if s.write_all(&(hdr.len() as u32).to_le_bytes()).await.is_err() || s.write_all(&hdr).await.is_err() {
                    break;
                }
                if s.write_all(&(data.len() as u64).to_le_bytes()).await.is_err() || s.write_all(&data).await.is_err() {
                    break;
                }
                if let Some((at, n)) = &pause {
                    if *at == i {
                        n.notified().await;
                    }
                }
            }
            let _ = s.read(&mut [0u8; 1]).await;
            serde_json::from_slice(&body).unwrap()
        });
        (addr, h)
    }

    /// The pixel part as engine/server/tests/pixels_test.cpp's `frame()` lays it out after the prompt, for images in `spans`
    /// order: u32 n, {start, rows, hash} each, the grids (u16 n_h, u16 n_w) padded to 8, each image's bytes padded to 8.
    fn pixel_part(starts: &[u32], imgs: &[Img], salt: u64) -> Vec<u8> {
        let mut p = (imgs.len() as u32).to_le_bytes().to_vec();
        for (s, i) in starts.iter().zip(imgs) {
            p.extend((*s).to_le_bytes());
            p.extend((rows_of_grid(i.n_h as u64, i.n_w as u64) as u32).to_le_bytes());
            p.extend((i.hash ^ salt).to_le_bytes());
        }
        let mut g = Vec::new();
        for i in imgs {
            g.extend(i.n_h.to_le_bytes());
            g.extend(i.n_w.to_le_bytes());
        }
        g.resize(pad8(g.len()), 0);
        p.extend(g);
        for i in imgs {
            let mut px = vec![i.fill; 588 * i.n_h as usize * i.n_w as usize];
            px.resize(pad8(px.len()), 0);
            p.extend(px);
        }
        p
    }

    /// The images of pixels_test's `three` (grids 3 x 3, 39 x 39, 5 x 7; starts 10, 20, 400; hash 0x1111 x i; fill i).
    pub(crate) fn three() -> Vec<Img> {
        vec![Img { n_h: 3, n_w: 3, fill: 1, hash: 0x1111 }, Img { n_h: 39, n_w: 39, fill: 2, hash: 0x2222 }, Img { n_h: 5, n_w: 7, fill: 3, hash: 0x3333 }]
    }

    #[test]
    fn the_grid_rule_is_the_engines() {
        // pixels_test: rows_of({3,3}) == 4, {39,39} == 184, {5,7} == 10, {93,93} == 31 x 32 + 2
        assert_eq!([rows_of_grid(3, 3), rows_of_grid(39, 39), rows_of_grid(5, 7), rows_of_grid(93, 93)], [4, 184, 10, 31 * 32 + 2]);
        // 588 x 35 = 20,580 pads by 4; 588 x 9 pads by 4; 588 x 2 by none (pixels_test)
        assert_eq!([pad8(588 * 35) - 588 * 35, pad8(588 * 9) - 588 * 9, pad8(588 * 2) - 588 * 2], [4, 4, 0]);
    }

    #[tokio::test]
    async fn a_pixel_part_is_the_frame_pixels_test_reads() {
        let imgs = three();
        let (addr, enc) = fake_media_frames(media_frames(&imgs), None).await;
        let recs = vec![json!({"type": "image", "url": "data:image/png;base64,AAAA"}); 3];
        let e = encode_pixels(&addr, &recs, T, DEFAULT_MAX_IMAGE_ROWS).await.unwrap();
        assert!(e.pixels, "the part says it carries pixels");
        assert_eq!(e.rows, vec![4, 184, 10]);
        let part = payload(&[10, 20, 400], e);
        assert!(part == pixel_part(&[10, 20, 400], &imgs, ENGINE_ENCODER_KIND), "the part equals pixels_test's well-formed layout, byte for byte");
        // the offsets pixels_test checks: grids at 4 + 16 n, the first image at the padded grids' end, the last ends the part
        let data_at = 4 + 16 * 3 + 16;
        assert_eq!(part[data_at], 1);
        assert_eq!(part[data_at + pad8(588 * 9)], 2);
        assert_eq!(part[data_at + pad8(588 * 9) + pad8(588 * 39 * 39)], 3);
        assert_eq!(data_at + pad8(588 * 9) + pad8(588 * 39 * 39) + pad8(588 * 35), part.len());
        // and the worker is asked what the container is asked: the records and the rows cap
        assert_eq!(enc.await.unwrap(), json!({"images": recs, "max_rows": DEFAULT_MAX_IMAGE_ROWS}));
    }

    #[tokio::test]
    async fn the_encoders_kind_is_in_the_hash() {
        let imgs = vec![Img { n_h: 3, n_w: 3, fill: 9, hash: 0xabcd }];
        let (addr, _enc) = fake_media_frames(media_frames(&imgs), None).await;
        let e = encode_pixels(&addr, &[json!({"url": "x"})], T, DEFAULT_MAX_IMAGE_ROWS).await.unwrap();
        assert_eq!(e.hashes, vec![0xabcd ^ ENGINE_ENCODER_KIND]);
        assert_ne!(e.hashes[0], 0xabcd, "the same image under the container's encoder has another hash");
        // the container's hash is sent as the encoder made it
        let (addr, _enc) = fake_encoder(reply_with_rows(&[3]), true).await;
        let e = encode(&addr, &records(1), T, DEFAULT_MAX_IMAGE_ROWS).await.unwrap();
        assert_eq!((e.hashes[0], e.pixels), (0, false));
    }

    #[tokio::test]
    async fn a_grid_that_is_not_its_rows_is_the_workers_fault() {
        let imgs = three();
        let mut frames = media_frames(&imgs);
        frames[1].0["image"]["grid"] = json!([39, 40, 13, 14]);
        let (addr, _enc) = fake_media_frames(frames, None).await;
        let e = encode_pixels(&addr, &records(3), T, DEFAULT_MAX_IMAGE_ROWS).await.unwrap_err();
        assert_eq!(e, "image encoder: the grid does not match its rows");
        assert!(!is_request_error(&e) && !is_too_large(&e), "a 503, not the request's fault");
        // pixels that are not the grid's bytes
        let mut frames = media_frames(&imgs);
        frames[2].1.truncate(10);
        let (addr, _enc) = fake_media_frames(frames, None).await;
        let e = encode_pixels(&addr, &records(3), T, DEFAULT_MAX_IMAGE_ROWS).await.unwrap_err();
        assert!(e.contains("do not match the grid") && !is_request_error(&e), "{e}");
        // an end frame that counts other than the images sent
        let mut frames = media_frames(&imgs);
        frames[3].0 = json!({"ok": true, "images": 2});
        let (addr, _enc) = fake_media_frames(frames, None).await;
        assert_eq!(encode_pixels(&addr, &records(3), T, DEFAULT_MAX_IMAGE_ROWS).await.unwrap_err(), "image encoder: wrong number of images");
    }

    #[tokio::test]
    async fn an_image_over_the_engines_rows_is_refused_before_its_pixels() {
        // 96 x 96 patches make 1,058 rows: under the span cap of 1,100 and over the 1,024 the engine takes
        let imgs = vec![Img { n_h: 96, n_w: 96, fill: 0, hash: 1 }];
        assert_eq!(rows_of_grid(96, 96), 1058);
        let mut frames = media_frames(&imgs);
        frames[0].1.clear(); // no pixels follow: the frame's JSON alone is refused
        let (addr, _enc) = fake_media_frames(frames, None).await;
        let e = encode_pixels(&addr, &records(1), T, DEFAULT_MAX_IMAGE_ROWS).await.unwrap_err();
        assert!(e.starts_with("cannot read the image: it takes 1058 rows") && is_request_error(&e) && !is_too_large(&e), "{e}");
    }

    /// An error frame after two image frames: the reply ends there, what came before is dropped (nothing returned, nothing counted),
    /// and the error is the one a worker that answered the error alone would have given.
    #[tokio::test]
    async fn an_error_frame_after_two_images_drops_both() {
        let imgs = three();
        let said = "cannot identify image file";
        let mut frames = media_frames(&imgs[..2]);
        frames.truncate(2);
        frames.push((json!({"ok": false, "error": said}), Vec::new()));
        let (addr, _enc) = fake_media_frames(frames, None).await;
        let b = RowsInFlight::new(10_000);
        b.set_pixels(true);
        let held = b.reserve(1000).unwrap();
        let e = encode_pixels_counted(&addr, &records(3), T, DEFAULT_MAX_IMAGE_ROWS, MAX_PIXEL_PART, Some(&held)).await.unwrap_err();
        assert_eq!(e, format!("cannot read the image: {said}"));
        assert!(is_request_error(&e) && !is_too_large(&e));
        assert_eq!(b.image_bytes(), 0, "what the two images counted is given back");
        // the worker's own cap: it stops at the image that crosses it, whatever came before
        let mut frames = media_frames(&imgs[..2]);
        frames.truncate(2);
        frames.push((json!({"ok": false, "code": "too_many_rows", "error": "at most 40000 image tokens per request; the first 3 of 3 images already take 40754"}), Vec::new()));
        let (addr, _enc) = fake_media_frames(frames, None).await;
        let e = encode_pixels(&addr, &records(3), T, DEFAULT_MAX_IMAGE_ROWS).await.unwrap_err();
        assert!(e.starts_with("at most 40000 image tokens") && is_request_error(&e), "{e}");
    }

    /// The pixels count against `--front-image-mib` as the frames arrive; a request is refused after the frame that crosses it.
    #[tokio::test]
    async fn the_image_memory_is_counted_as_frames_arrive() {
        let imgs = vec![Img { n_h: 39, n_w: 39, fill: 1, hash: 1 }; 3]; // 894,348 bytes each
        let one = pad8(588 * 39 * 39);
        let head = 4 + 16 * 3 + pad8(12);
        let gate = std::sync::Arc::new(tokio::sync::Notify::new());
        let (addr, _enc) = fake_media_frames(media_frames(&imgs), Some((1, gate.clone()))).await;
        let b = RowsInFlight::new(10_000);
        b.set_pixels(true);
        let held = std::sync::Arc::new(b.reserve(552).unwrap());
        let (h2, a2) = (held.clone(), addr.clone());
        let task = tokio::spawn(async move { encode_pixels_counted(&a2, &records(3), T, DEFAULT_MAX_IMAGE_ROWS, 400 << 20, Some(&*h2)).await });
        let t0 = Instant::now();
        while b.image_bytes() != head + 2 * one {
            assert!(t0.elapsed() < Duration::from_secs(10), "after two frames the part counts two images, saw {}", b.image_bytes());
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(!task.is_finished(), "the third frame has not been sent");
        gate.notify_one();
        let e = task.await.unwrap().unwrap();
        assert_eq!(e.data.len(), head + 3 * one);
        assert_eq!(b.image_bytes(), head + 3 * one);
        // a limit of 2 MiB: the first two images (1.7 MiB with the table) pass, the frame that crosses it is refused
        let (addr, _enc) = fake_media_frames(media_frames(&imgs), None).await;
        let e = encode_pixels_counted(&addr, &records(3), T, DEFAULT_MAX_IMAGE_ROWS, 2 << 20, None).await.unwrap_err();
        assert!(e.starts_with("at most 2 MiB of image pixels per request") && is_too_large(&e), "{e}");
        // and one of 1 MiB is crossed by the second frame, not the first
        let (addr, _enc) = fake_media_frames(media_frames(&imgs[..1]), None).await;
        assert!(encode_pixels_counted(&addr, &records(1), T, DEFAULT_MAX_IMAGE_ROWS, 1 << 20, None).await.is_ok());
        let (addr, _enc) = fake_media_frames(media_frames(&imgs[..2]), None).await;
        assert!(encode_pixels_counted(&addr, &records(2), T, DEFAULT_MAX_IMAGE_ROWS, 1 << 20, None).await.unwrap_err().contains("MiB of image pixels"));
        drop(held);
    }

    #[tokio::test]
    async fn a_part_over_the_frame_stage_is_refused_at_the_frame_that_crosses_it() {
        // 93 x 93 patches: 994 rows, 5,085,616 bytes padded; 82 fit the 400 MiB the engine stages a frame in, the 83rd does not
        let part = |n: usize| 4 + 16 * n + pad8(4 * n) + n * pad8(588 * 93 * 93);
        assert!(part(82) <= MAX_PIXEL_PART && part(83) > MAX_PIXEL_PART);
        // the same rule at a cap small enough to send: 4 of them are under 24 MiB, the 5th crosses it
        let imgs = vec![Img { n_h: 93, n_w: 93, fill: 0, hash: 1 }; 10];
        let (addr, _enc) = fake_media_frames(media_frames(&imgs), None).await;
        let e = encode_pixels_counted(&addr, &records(10), T, MAX_TOTAL_IMAGE_ROWS, 24 << 20, None).await.unwrap_err();
        assert!(e.starts_with("at most 24 MiB of image pixels per request") && is_too_large(&e), "{e}");
        // a cap above the stage is the stage
        let (addr, _enc) = fake_media_frames(media_frames(&imgs[..1]), None).await;
        assert!(encode_pixels_counted(&addr, &records(1), T, MAX_TOTAL_IMAGE_ROWS, usize::MAX, None).await.is_ok());
    }

    #[tokio::test]
    async fn the_workers_pixel_limit_is_too_large_not_a_bad_request() {
        let said = "Image size (41000000 pixels) exceeds limit of 40000000 pixels, could be decompression bomb DOS attack.";
        let (addr, _enc) = fake_media_frames(vec![(json!({"ok": false, "error": said}), Vec::new())], None).await;
        let e = encode_pixels(&addr, &records(1), T, DEFAULT_MAX_IMAGE_ROWS).await.unwrap_err();
        assert_eq!(e, format!("cannot read the image: {said}"));
        assert!(is_too_large(&e) && is_request_error(&e));
        assert!(!is_too_large("cannot read the image: cannot identify image file"));
        assert_eq!(encode_pixels("127.0.0.1:1", &records(101), T, 1).await.unwrap_err(), "at most 100 images per request");
    }

    #[test]
    fn pixels_count_for_the_image_memory() {
        let b = RowsInFlight::new(10_000);
        let held = b.reserve(1000).unwrap();
        assert_eq!(b.image_bytes(), 1000 * IMAGE_ROW_BYTES, "rows, as before");
        b.set_pixels(true);
        assert_eq!(b.image_bytes(), 0, "before the media worker has answered the request holds no pixels");
        held.set_pixel_bytes(5_000_000);
        let other = b.reserve(10).unwrap();
        other.set_pixel_bytes(40);
        assert_eq!(b.image_bytes(), 5_000_040, "the pixels, not rows x 10 KiB");
        held.set_pixel_bytes(3_000_000);
        assert_eq!(b.image_bytes(), 3_000_040, "a request holds what it holds now");
        drop(held);
        assert_eq!(b.image_bytes(), 40);
        drop(other);
        assert_eq!((b.image_bytes(), b.in_flight()), (0, 0));
    }

    #[test]
    fn the_vision_kinds() {
        assert_eq!(VisionKind::parse("container"), Some(VisionKind::Container));
        assert_eq!(VisionKind::parse("engine"), Some(VisionKind::Engine));
        assert_eq!(VisionKind::parse("Engine"), None);
    }
}
