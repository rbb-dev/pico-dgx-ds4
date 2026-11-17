#!/usr/bin/env python3
"""Interleaved prefill with images, through the front-end's HTTP API (stdlib only): X (an image in the middle of about 47K
tokens) and Y (an image at the start of about 40K tokens) are read alone, then together while a third request decodes;
their outputs and top-2 logprobs must equal the alone runs (each prompt sees its own image only). Both phases read cold:
restart the engine between them.

  image_overlap.py URL MODEL alone OUT.json
  image_overlap.py URL MODEL together OUT.json ALONE.json      exit 0 when both are equal       (key: $API_KEY)
"""
import base64, json, os, struct, sys, threading, time, urllib.request, zlib

URL, MODEL, PHASE = sys.argv[1], sys.argv[2], sys.argv[3]
KEY = os.environ.get("API_KEY", "")


def png(w, h, seed):
    rows = b"".join(b"\x00" + bytes(v for x in range(w)
                                    for v in ((x * 7 + y * 3 + seed) % 256, (x * seed + y) % 256, (y * 5 + seed * 11) % 256))
                    for y in range(h))
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + \
        chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b"")


def post(messages, max_tokens, label, out):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "logprobs": True,
            "top_logprobs": 2, "chat_template_kwargs": {"thinking": False}}
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=1200))
    u = r.get("usage") or {}
    ch = r["choices"][0]
    lp = [(c.get("token"), c.get("logprob"), [(t.get("token"), t.get("logprob")) for t in c.get("top_logprobs") or []])
          for c in ((ch.get("logprobs") or {}).get("content") or [])]
    res = {"text": ch["message"].get("content") or "", "lp": lp, "prompt": u.get("prompt_tokens"),
           "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"), "s": time.time() - t0}
    print(f"{label}: prompt {res['prompt']} cached {res['cached']} in {res['s']:.1f} s; reply {res['text'][:50]!r}", flush=True)
    out[label] = res


def prompts():
    # X's image sits in the middle of X, so X reads it after Y (the second reader, arriving meanwhile) has read all of
    # its pieces: with one image set for the whole model, Y's images would replace X's and Y's end would clear them
    x = "Field notes on harbour cranes and tides. "
    y = "Ledger of orchard yields and rainfall. "
    def image(seed):
        return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png(256, 256, seed)).decode()}}
    X = [{"role": "user", "content": [{"type": "text", "text": x * 2600}, image(7),
                                      {"type": "text", "text": x * 2600 + "\nDescribe the image in one sentence."}]}]
    Y = [{"role": "user", "content": [image(3), {"type": "text", "text": y * 5000 + "\nDescribe the image in one sentence."}]}]
    return X, Y


def main():
    X, Y = prompts()
    out = {}
    if PHASE == "alone":
        post(X, 24, "X", out)
        post(Y, 24, "Y", out)
        json.dump(out, open(sys.argv[4], "w"))
        return
    d = threading.Thread(target=post, args=([{"role": "user", "content": "Count from 1 to 3000, one number a line."}], 3000,
                                            "decoder", out))
    d.start()
    time.sleep(3)
    tx = threading.Thread(target=post, args=(X, 24, "X", out))
    tx.start()
    time.sleep(0.5)
    ty = threading.Thread(target=post, args=(Y, 24, "Y", out))
    ty.start()
    for th in (tx, ty, d):
        th.join()
    json.dump(out, open(sys.argv[4], "w"))
    al = json.load(open(sys.argv[5]))
    ok_x = out["X"]["text"] == al["X"]["text"] and json.dumps(out["X"]["lp"]) == json.dumps(al["X"]["lp"])
    ok_y = out["Y"]["text"] == al["Y"]["text"] and json.dumps(out["Y"]["lp"]) == json.dumps(al["Y"]["lp"])
    print("X (image in its middle) together vs alone: %s; Y (image at its start) together vs alone: %s; cached X %s Y %s (0: cold)" % (
        "EQUAL" if ok_x else "DIFFER", "EQUAL" if ok_y else "DIFFER", out["X"]["cached"], out["Y"]["cached"]))
    print("image_overlap: %s" % ("PASS" if ok_x and ok_y else "FAIL"))
    sys.exit(0 if ok_x and ok_y else 1)


if __name__ == "__main__":
    main()
