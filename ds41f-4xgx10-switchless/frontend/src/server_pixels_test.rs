//! Images with `--vision engine`: the real router against the mock engine and a stand-in media worker, with a real TCP client. What the
//! engine saw of each request is the mock's record (`mock::Seen`); what the client got is the HTTP answer. `--vision container`, the
//! control, runs through a stand-in encoder container the same way.

use super::*;
use crate::mock::{MockConfig, Seen};
use crate::protocol::VERSION_PREFIX;
use crate::vision::{pad8, rows_of_grid, VisionKind, ENGINE_ENCODER_KIND, IMAGE_DIM, PATCH_BYTES};
use std::path::PathBuf;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// An image the stand-in workers know by its URL: `data:image/png;base64,G<n_h>x<n_w>x<tag>`.
fn image_url(n_h: u16, n_w: u16, tag: u32) -> String {
    format!("data:image/png;base64,G{n_h}x{n_w}x{tag}")
}

fn grid_of(url: &str) -> (u16, u16, u64) {
    let g = url.rsplit_once("base64,G").unwrap().1;
    let mut p = g.split('x');
    let (h, w, t): (u16, u16, u64) = (p.next().unwrap().parse().unwrap(), p.next().unwrap().parse().unwrap(), p.next().unwrap().parse().unwrap());
    (h, w, t)
}

/// A stand-in worker on a local port that answers any number of requests: `pixels` is the media worker (the padded pixels, each byte
/// its image's tag, one frame per image and an end frame), otherwise the encoder container (rows of zeros). An image with tag 413
/// is refused as the media worker refuses an image over its pixel limit (an error frame). Every request it was sent is kept.
async fn worker(pixels: bool) -> (String, Arc<Mutex<Vec<Value>>>) {
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = l.local_addr().unwrap().to_string();
    let sent = Arc::new(Mutex::new(Vec::new()));
    let sent2 = sent.clone();
    tokio::spawn(async move {
        loop {
            let Ok((mut s, _)) = l.accept().await else { return };
            let sent = sent2.clone();
            tokio::spawn(async move {
                let mut n4 = [0u8; 4];
                s.read_exact(&mut n4).await.unwrap();
                let mut body = vec![0u8; u32::from_le_bytes(n4) as usize];
                s.read_exact(&mut body).await.unwrap();
                let req: Value = serde_json::from_slice(&body).unwrap();
                sent.lock().unwrap().push(req.clone());
                let imgs: Vec<(u16, u16, u64)> = req["images"].as_array().unwrap().iter().map(|r| grid_of(r["url"].as_str().unwrap())).collect();
                let mut out: Vec<u8> = Vec::new();
                let mut frame = |j: Value, data: &[u8]| {
                    let hdr = j.to_string().into_bytes();
                    out.extend((hdr.len() as u32).to_le_bytes());
                    out.extend(hdr);
                    out.extend((data.len() as u64).to_le_bytes());
                    out.extend(data);
                };
                if pixels {
                    // the media worker: a frame per image, then the end frame; an image with tag 413 ends the reply with the error an
                    // image over the worker's pixel limit gets, and what came before it is dropped by the reader
                    for &(h, w, t) in &imgs {
                        if t == 413 {
                            frame(json!({"ok": false, "error": "Image size (41000000 pixels) exceeds limit of 40000000 pixels, could be decompression bomb DOS attack."}), &[]);
                            let _ = s.write_all(&out).await;
                            return;
                        }
                        frame(json!({"ok": true, "image": {"rows": rows_of_grid(h as u64, w as u64), "hash": 0x9000 + t, "grid": [h, w, h.div_ceil(3), w.div_ceil(3)],
                                                           "ms": 1.0, "bytes": 588 * h as u64 * w as u64}}), &vec![t as u8; PATCH_BYTES * h as usize * w as usize]);
                    }
                    frame(json!({"ok": true, "images": imgs.len()}), &[]);
                    let _ = s.write_all(&out).await;
                } else {
                    // the encoder container: one header, then the rows
                    let reply = json!({"ok": true, "images": imgs.iter().map(|&(h, w, t)| json!({
                        "rows": rows_of_grid(h as u64, w as u64), "hash": 0x9000 + t, "grid": [h, w, h.div_ceil(3), w.div_ceil(3)], "bytes": 588 * h as u64 * w as u64}))
                        .collect::<Vec<_>>()});
                    let hdr = reply.to_string().into_bytes();
                    s.write_all(&(hdr.len() as u32).to_le_bytes()).await.unwrap();
                    s.write_all(&hdr).await.unwrap();
                    let rows: u64 = imgs.iter().map(|&(h, w, _)| rows_of_grid(h as u64, w as u64)).sum();
                    let data = vec![0u8; rows as usize * IMAGE_DIM * 2];
                    s.write_all(&(data.len() as u64).to_le_bytes()).await.unwrap();
                    let _ = s.write_all(&data).await;
                }
            });
        }
    });
    (addr, sent)
}

/// The synthetic tokenizer with the image placeholder at the id the front-end expects (129264).
fn tokenizer() -> Arc<crate::tokenizer::Tokenizer> {
    let added = format!(r#""added_tokens":[{{"id":{},"content":{},"special":true,"normalized":false}},"#, crate::vision::IMAGE_TOKEN_ID, serde_json::to_string(crate::encoding::IMAGE_PLACEHOLDER).unwrap());
    let json = crate::tokenizer::synthetic::json().replacen(r#""added_tokens":["#, &added, 1);
    Arc::new(crate::tokenizer::Tokenizer::from_json(&json).expect("tokenizer"))
}

struct Rig {
    addr: std::net::SocketAddr,
    seen: Arc<Mutex<Vec<Seen>>>,
    workers: Arc<Mutex<Vec<Value>>>,
    app: Arc<App>,
    path: PathBuf,
    server: tokio::task::JoinHandle<()>,
    mock: tokio::task::JoinHandle<std::io::Result<()>>,
}

impl Drop for Rig {
    fn drop(&mut self) {
        self.server.abort();
        self.mock.abort();
        let _ = std::fs::remove_file(&self.path);
    }
}

async fn rig(tag: &str, kind: VisionKind) -> Rig {
    rig_mib(tag, kind, crate::server::DEFAULT_FRONT_IMAGE_MIB).await
}

/// `rig` with `--front-image-mib` set.
async fn rig_mib(tag: &str, kind: VisionKind, front_image_mib: usize) -> Rig {
    let tok = tokenizer();
    let path = std::env::temp_dir().join(format!("dsfront-pixels-{}-{tag}.sock", std::process::id()));
    let seen = Arc::new(Mutex::new(Vec::new()));
    let cfg = MockConfig {
        socket: path.clone(),
        tok: tok.clone(),
        delay: Duration::ZERO,
        prefill_tok_per_s: 0.0,
        max_context: 1 << 20,
        version: VERSION_PREFIX,
        seen: Some(seen.clone()),
    };
    let mock = tokio::spawn(crate::mock::run(cfg));
    let engine = EngineClient::start(path.clone());
    assert!(engine.wait_connected(Duration::from_secs(10)).await.is_some(), "the front-end did not connect to the mock engine");
    let (waddr, workers) = worker(kind == VisionKind::Engine).await;
    let mut scfg = super::tests::cfg();
    scfg.vision = kind;
    scfg.front_image_mib = front_image_mib;
    match kind {
        VisionKind::Engine => scfg.media_addr = Some(waddr),
        VisionKind::Container => scfg.vision_addr = Some(waddr),
    }
    let app = App::new(scfg, tok, engine);
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = l.local_addr().unwrap();
    let routes = router(app.clone());
    let server = tokio::spawn(async move { axum::serve(l, routes).await.unwrap() });
    Rig { addr, seen, workers, app, path, server, mock }
}

/// One non-streaming chat request: the status and the JSON.
async fn chat(addr: std::net::SocketAddr, messages: Value) -> (u16, Value) {
    let body = json!({"messages": messages, "max_tokens": 300}).to_string();
    let mut s = TcpStream::connect(addr).await.unwrap();
    let head = format!("POST /v1/chat/completions HTTP/1.1\r\nhost: x\r\nconnection: close\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n", body.len());
    s.write_all(head.as_bytes()).await.unwrap();
    s.write_all(body.as_bytes()).await.unwrap();
    let mut got = Vec::new();
    tokio::time::timeout(Duration::from_secs(20), s.read_to_end(&mut got)).await.expect("no answer within 20 s").unwrap();
    let text = String::from_utf8_lossy(&got).to_string();
    let (head, body) = text.split_once("\r\n\r\n").unwrap_or((&text, ""));
    let status: u16 = head.split_whitespace().nth(1).unwrap().parse().unwrap();
    (status, serde_json::from_str(body).unwrap_or(Value::Null))
}

fn user(text: &str, images: &[String]) -> Value {
    let mut parts = vec![json!({"type": "text", "text": text})];
    parts.extend(images.iter().map(|u| json!({"type": "image_url", "image_url": {"url": u}})));
    json!({"role": "user", "content": parts})
}

fn reply_of(v: &Value) -> String {
    v["choices"][0]["message"]["content"].as_str().unwrap_or_default().to_string()
}

/// The image parts the mock saw, in order.
fn parts(r: &Rig) -> Vec<(bool, Vec<u8>)> {
    r.seen.lock().unwrap().iter().filter_map(|s| if let Seen::Images { pixels, part } = s { Some((*pixels, part.clone())) } else { None }).collect()
}

/// The spans of a part as (start, rows, hash).
fn spans(part: &[u8]) -> Vec<(u32, u32, u64)> {
    let n = u32::from_le_bytes(part[..4].try_into().unwrap()) as usize;
    part[4..4 + 16 * n]
        .chunks_exact(16)
        .map(|t| (u32::from_le_bytes(t[..4].try_into().unwrap()), u32::from_le_bytes(t[4..8].try_into().unwrap()), u64::from_le_bytes(t[8..].try_into().unwrap())))
        .collect()
}

fn submits(r: &Rig) -> Vec<u64> {
    r.seen.lock().unwrap().iter().filter_map(|s| if let Seen::Submit { session, .. } = s { Some(*session) } else { None }).collect()
}

/// `--vision engine`: the SUBMIT carries pixels in the layout pixels_test reads, from the media worker's answer, and the engine (the
/// mock) takes it. The container's rows are never asked for.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn with_vision_engine_the_submit_carries_the_media_workers_pixels() {
    let r = rig("engine", VisionKind::Engine).await;
    let imgs = [image_url(3, 3, 1), image_url(39, 39, 2), image_url(5, 7, 3)];
    let (status, a) = chat(r.addr, json!([user("what are these", &imgs)])).await;
    assert_eq!(status, 200, "{a}");
    assert!(reply_of(&a).starts_with("Mock reply to:"), "{a}");
    let p = parts(&r);
    assert_eq!(p.len(), 1);
    let (pixels, part) = &p[0];
    assert!(pixels, "FLAG_PIXELS, not rows");
    // the layout of pixels_test's `three`: n, spans, grids padded to 8, each image padded to 8
    let sp = spans(part);
    assert_eq!(sp.iter().map(|s| s.1).collect::<Vec<_>>(), vec![4, 184, 10]);
    assert_eq!(sp.iter().map(|s| s.2).collect::<Vec<_>>(), vec![(0x9000 + 1) ^ ENGINE_ENCODER_KIND, (0x9000 + 2) ^ ENGINE_ENCODER_KIND, (0x9000 + 3) ^ ENGINE_ENCODER_KIND]);
    let grids_at = 4 + 16 * 3;
    assert_eq!(&part[grids_at..grids_at + 12], &[3, 0, 3, 0, 39, 0, 39, 0, 5, 0, 7, 0]);
    assert_eq!(&part[grids_at + 12..grids_at + 16], &[0, 0, 0, 0], "the grids' array is padded to 8");
    let at = grids_at + 16;
    assert_eq!(part.len(), at + pad8(588 * 9) + pad8(588 * 39 * 39) + pad8(588 * 35));
    assert_eq!((part[at], part[at + pad8(588 * 9)], part[at + pad8(588 * 9) + pad8(588 * 39 * 39)]), (1, 2, 3), "each image's bytes, from the worker");
    assert_eq!(part[at + 588 * 9..at + pad8(588 * 9)], [0; 4], "padding");
    assert_eq!(r.workers.lock().unwrap().len(), 1, "one request to the media worker, for all three images");
}

/// The control: `--vision container` sends the rows it always sent.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn with_vision_container_the_submit_is_todays_rows() {
    let r = rig("container", VisionKind::Container).await;
    let imgs = [image_url(3, 3, 1), image_url(5, 7, 3)];
    let (status, a) = chat(r.addr, json!([user("what are these", &imgs)])).await;
    assert_eq!(status, 200, "{a}");
    let p = parts(&r);
    assert_eq!(p.len(), 1);
    let (pixels, part) = &p[0];
    assert!(!pixels, "FLAG_IMAGES");
    assert_eq!(part.len(), 4 + 16 * 2 + (4 + 10) * IMAGE_DIM * 2, "span table, then the rows");
    assert_eq!(spans(part).iter().map(|s| s.2).collect::<Vec<_>>(), vec![0x9000 + 1, 0x9000 + 3], "the encoder's hashes as they are");
}

/// A media worker that refuses an image (over its pixel limit) after two image frames: both are dropped, 413 to the client, and
/// nothing reaches the engine.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_picture_over_the_workers_pixel_limit_is_413_and_never_reaches_the_engine() {
    let r = rig("big", VisionKind::Engine).await;
    let (status, a) = chat(r.addr, json!([user("look", &[image_url(3, 3, 1), image_url(5, 7, 2), image_url(3, 3, 413)])])).await;
    assert_eq!(status, 413, "{a}");
    assert!(a["error"]["message"].as_str().unwrap().contains("exceeds limit of 40000000 pixels"), "{a}");
    assert!(submits(&r).is_empty() && parts(&r).is_empty(), "the engine saw nothing: {:?}", r.seen.lock().unwrap());
    // and the front-end serves the next request
    let (status, _) = chat(r.addr, json!([user("look", &[image_url(3, 3, 1)])])).await;
    assert_eq!(status, 200);
}

/// What `--front-image-mib` counts is the pixels held: while a request is held its pixel part counts, not its rows at 10 KiB.
#[tokio::test]
async fn the_image_memory_counts_the_pixels_under_vision_engine() {
    let r = rig("count", VisionKind::Engine).await;
    let held = r.app.image_rows.reserve(184).unwrap();
    held.set_pixel_bytes(588 * 39 * 39);
    assert_eq!(r.app.image_rows.image_bytes(), 588 * 39 * 39);
    assert!(r.app.image_rows.image_bytes() < 184 * crate::vision::IMAGE_ROW_BYTES, "far under the rows' count");
    drop(held);
    assert_eq!(r.app.image_rows.image_bytes(), 0);
    let c = rig("count-c", VisionKind::Container).await;
    let held = c.app.image_rows.reserve(184).unwrap();
    assert_eq!(c.app.image_rows.image_bytes(), 184 * crate::vision::IMAGE_ROW_BYTES, "the container's rows, as before");
    drop(held);
}

/// A kept session's turn that resends five images and adds one sends all six, and the engine (the mock) sees the six
/// spans; the first five are the first turn's, byte for byte (the same images are the same hashes, so which of them the engine
/// encodes is its own rule).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_turn_that_resends_five_images_and_adds_one_sends_all_six() {
    let r = rig("f8", VisionKind::Engine).await;
    let five: Vec<String> = (1..=5).map(|t| image_url(3 + t as u16, 3, t)).collect();
    let (s1, a) = chat(r.addr, json!([user("first five", &five)])).await;
    assert_eq!(s1, 200, "{a}");
    let mut six = five.clone();
    six.push(image_url(3, 3, 6));
    let (s2, b) = chat(
        r.addr,
        json!([user("first five", &five), {"role": "assistant", "content": reply_of(&a)}, user("and one more", &six[5..])]),
    )
    .await;
    assert_eq!(s2, 200, "{b}");
    let p = parts(&r);
    assert_eq!(p.len(), 2, "two turns, each with its image part");
    assert!(p.iter().all(|(px, _)| *px));
    let (s_a, s_b) = (spans(&p[0].1), spans(&p[1].1));
    assert_eq!((s_a.len(), s_b.len()), (5, 6), "all six images are sent");
    assert_eq!(s_a.iter().map(|s| (s.1, s.2)).collect::<Vec<_>>(), s_b[..5].iter().map(|s| (s.1, s.2)).collect::<Vec<_>>(), "the five are the first turn's images");
    assert_eq!(s_b[5].2, (0x9000 + 6) ^ ENGINE_ENCODER_KIND);
    let sub = submits(&r);
    assert_eq!(sub[0], sub[1], "the same session: the turn continues what the first made");
    // the media worker was asked for all six again: the front-end does not decide which are encoded
    let w = r.workers.lock().unwrap();
    assert_eq!(w[1]["images"].as_array().unwrap().len(), 6);
}

/// `--front-image-mib` is counted as frames arrive: a request whose pixels pass it is refused (413) after the frame that crosses it,
/// and nothing reaches the engine; under the limit the same images go through.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_request_over_front_image_mib_is_refused_after_the_frame_that_crosses_it() {
    let r = rig_mib("mib", VisionKind::Engine, 1).await; // 39 x 39 patches: 0.85 MiB each
    let (status, a) = chat(r.addr, json!([user("look", &[image_url(39, 39, 1), image_url(39, 39, 2), image_url(39, 39, 3)])])).await;
    assert_eq!(status, 413, "{a}");
    assert!(a["error"]["message"].as_str().unwrap().starts_with("at most 1 MiB of image pixels per request"), "{a}");
    assert!(submits(&r).is_empty() && parts(&r).is_empty(), "nothing reached the engine");
    assert_eq!(r.app.image_rows.image_bytes(), 0, "what the frames counted is given back");
    let (status, a) = chat(r.addr, json!([user("look", &[image_url(39, 39, 1)])])).await;
    assert_eq!(status, 200, "one image is under the limit: {a}");
}
