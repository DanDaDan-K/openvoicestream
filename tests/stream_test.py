#!/usr/bin/env python3
"""Streaming ASR quality test — prints partials + final."""
import json, wave, io, sys
import numpy as np
import websocket

def test(path, label):
    with open(path, 'rb') as f:
        raw = f.read()
    with wave.open(io.BytesIO(raw)) as wf:
        sr = wf.getframerate()
        frames = wf.getnframes()
        audio = wf.readframes(frames)
    dur = frames / sr
    samples = np.frombuffer(audio, dtype=np.int16)
    chunk_n = int(sr * 0.25)
    chunks = [samples[i:i+chunk_n].tobytes() for i in range(0, len(samples), chunk_n)]
    ws = websocket.create_connection(
        'ws://localhost:8621/asr/stream?language=Chinese&sample_rate=16000', timeout=30)
    partials = []
    ws.settimeout(0.05)
    for c in chunks:
        ws.send_binary(c)
        while True:
            try:
                msg = ws.recv()
                data = json.loads(msg)
                if data.get('type') == 'partial':
                    txt = data.get('text', '').strip()
                    if txt:
                        partials.append(txt)
            except:
                break
    ws.settimeout(None)
    ws.send_binary(b'')
    final = ''
    while True:
        data = json.loads(ws.recv())
        if data.get('type') == 'final':
            final = data.get('text', '').strip()
            break
    ws.close()
    print(f'{label} ({dur:.1f}s) partials={len(partials)} final={final}')
    seen = set()
    for p in partials:
        if p not in seen:
            seen.add(p)
            print(f'  [{p}]')
    repeats = sum(1 for i in range(1, len(partials))
                  if partials[i] == partials[i-1] and partials[i])
    if repeats:
        print(f'  REPEATS={repeats}')
    return final

if __name__ == '__main__':
    base = '/home/harvest/bench/wavs'
    for w, l in [('S0.wav', 'S0'), ('S1.wav', 'S1'), ('S3.wav', 'S3')]:
        test(f'{base}/{w}', l)
