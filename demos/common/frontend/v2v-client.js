/* V2VStreamClient — browser client for `WS /v2v/stream` (native ES module).
 *
 * Wire contract: docs/api/v2v-stream.md (authoritative). Summary:
 *   client → server : one leading {"type":"config", ...} JSON frame, then
 *                     binary int16 LE PCM (mic) frames interleaved with JSON
 *                     control frames — {"type":"text"}, {"type":"tts_flush"},
 *                     {"type":"asr_eos"}, {"type":"abort"} (barge-in).
 *   server → client : JSON events — asr_partial / asr_endpoint / asr_final /
 *                     vad_event (speech_start|speech_end) / tts_started /
 *                     tts_sentence_done / tts_done / error — plus binary TTS
 *                     audio. The FIRST binary frame of the session carries a
 *                     4-byte LE uint32 sample-rate header; every later binary
 *                     frame is raw int16 LE PCM at that fixed rate (the header
 *                     is never re-emitted, not even across utterances).
 *
 * Playback: received PCM is scheduled gap-free on a Web Audio context (same
 * scheduling approach as slv-client.js TTSStreamPlayer). `interrupt()` sends
 * {"type":"abort"} AND silences/clears everything scheduled locally, so
 * barge-in feels instant even with client-side buffering. The server also
 * cancels TTS on its own when VAD sees speech_start — the client mirrors that
 * by letting callers stopPlayback() on the vad_event.
 *
 * Unknown JSON frame types are ignored (forwarded to onEvent), as the
 * protocol doc instructs.
 */

export class V2VStreamClient {
  /**
   * @param {object} opts
   * @param {string}  [opts.baseUrl]        http(s) origin of the SLV server (default: page origin)
   * @param {string}  [opts.path]           WS path (default "/v2v/stream")
   * @param {string}  [opts.asrLanguage]    e.g. "auto" | "zh" | "en"; omit to disable ASR
   * @param {string}  [opts.ttsLanguage]    e.g. "zh" | "en" | "auto"; omit to disable TTS
   * @param {number}  [opts.ttsSpeakerId]   speaker ID from /tts/speakers (optional)
   * @param {number}  [opts.ttsSpeed]       optional; some backends only
   * @param {number}  [opts.sampleRate]     mic PCM rate (default 16000)
   * @param {string}  [opts.vad]            "silero" | "webrtcvad" | "none" (optional)
   * @param {number}  [opts.vadSilenceMs]   silence to auto-endpoint (optional)
   * @param {boolean} [opts.multiUtterance] keep session open across utterances (default false)
   * @param {boolean} [opts.playback]       schedule received audio on Web Audio (default true)
   * @param {AudioContext} [opts.audioContext] shared context (created lazily otherwise)
   *
   * Callbacks (all optional):
   *   onAsrPartial(msg)        {"type":"asr_partial","text","is_stable"}
   *   onAsrEndpoint(msg)       {"type":"asr_endpoint"}
   *   onAsrFinal(msg)          {"type":"asr_final","text",...session_complete?}
   *   onVadEvent(event, msg)   event = "speech_start" | "speech_end"
   *   onTtsStarted(msg)        {"type":"tts_started","sentence"}
   *   onTtsSentenceDone(msg)   {"type":"tts_sentence_done","sentence"}
   *   onTtsDone(msg)           {"type":"tts_done"}
   *   onError(msg)             {"type":"error","error"}
   *   onEvent(msg)             any other / unknown JSON frame
   *   onSampleRate(rate)       4-byte header parsed (once per connection)
   *   onAudioChunk(int16)      every non-empty PCM payload, pre-scheduling
   *   onPlaybackStart()        local playback transitioned silent → audible
   *   onPlaybackEnd()          scheduled audio fully played OR stopped/cleared
   *   onClose(ev)              WS closed (ev.code 4429 = session slots full)
   */
  constructor(opts = {}) {
    this.opts = opts;
    this.ws = null;
    this.sampleRate = 0;          // TTS output rate from the one-time header
    this._pre = new Uint8Array(0);      // bytes accumulated before the header
    this._leftover = new Uint8Array(0); // odd trailing byte between chunks
    // playback state
    this.ctx = opts.audioContext || null;
    this._sources = [];
    this._playhead = 0;           // ctx.currentTime-based scheduling cursor
    this._playing = false;
    this._endTimer = null;
  }

  get _wsUrl() {
    const base = this.opts.baseUrl || window.location.origin;
    const u = new URL(this.opts.path || "/v2v/stream", base);
    u.protocol = u.protocol === "https:" ? "wss:" : "ws:";
    return u.toString();
  }

  get connected() {
    return !!this.ws && this.ws.readyState === WebSocket.OPEN;
  }

  get isPlaying() {
    return this._playing;
  }

  /** Documented config keys only (docs/api/v2v-stream.md). */
  _configFrame() {
    const o = this.opts;
    const cfg = { type: "config" };
    if (o.asrLanguage != null) cfg.asr_language = o.asrLanguage;
    if (o.ttsLanguage != null) cfg.tts_language = o.ttsLanguage;
    if (o.ttsSpeakerId != null) cfg.tts_speaker_id = o.ttsSpeakerId;
    if (o.ttsSpeed != null) cfg.tts_speed = o.ttsSpeed;
    cfg.sample_rate = o.sampleRate || 16000;
    if (o.vad != null) cfg.vad = o.vad;
    if (o.vadSilenceMs != null) cfg.vad_silence_ms = o.vadSilenceMs;
    if (o.multiUtterance != null) cfg.multi_utterance = !!o.multiUtterance;
    return cfg;
  }

  /** Open the WS and send the leading config frame. */
  connect() {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(this._wsUrl);
      ws.binaryType = "arraybuffer";
      ws.onopen = () => {
        ws.send(JSON.stringify(this._configFrame()));
        resolve(this);
      };
      ws.onerror = (e) => reject(e);
      ws.onclose = (ev) => {
        this.ws = null;
        if (this.opts.onClose) this.opts.onClose(ev);
      };
      ws.onmessage = (ev) => {
        if (typeof ev.data === "string") this._onJson(ev.data);
        else this._onBinary(ev.data);
      };
      this.ws = ws;
    });
  }

  /* ── uplink ─────────────────────────────────────────────────────── */

  /** @param {Int16Array|ArrayBuffer} pcm mic PCM at config sample_rate */
  sendPcm(pcm) {
    if (!this.connected) return;
    this.ws.send(pcm instanceof Int16Array ? pcm.buffer : pcm);
  }

  /** Optional text-direct input: incremental text chunk feeding TTS. */
  sendText(text) {
    if (!this.connected) return;
    this.ws.send(JSON.stringify({ type: "text", text }));
  }

  /** Flush the remaining TTS sentence buffer. */
  ttsFlush() {
    if (!this.connected) return;
    this.ws.send(JSON.stringify({ type: "tts_flush" }));
  }

  /** Manually finalize ASR (overrides VAD). */
  asrEos() {
    if (!this.connected) return;
    this.ws.send(JSON.stringify({ type: "asr_eos" }));
  }

  /** Barge-in: {"type":"abort"} cancels the in-flight synth + queued
   *  sentences server-side; locally we stop and clear everything scheduled
   *  so the interruption is instant despite client buffering. */
  interrupt() {
    if (this.connected) {
      this.ws.send(JSON.stringify({ type: "abort" }));
    }
    this.stopPlayback();
  }

  /* ── downlink ───────────────────────────────────────────────────── */

  _onJson(data) {
    let msg;
    try { msg = JSON.parse(data); } catch { return; }
    const cb = this.opts;
    switch (msg.type) {
      case "asr_partial": cb.onAsrPartial && cb.onAsrPartial(msg); break;
      case "asr_endpoint": cb.onAsrEndpoint && cb.onAsrEndpoint(msg); break;
      case "asr_final": cb.onAsrFinal && cb.onAsrFinal(msg); break;
      case "vad_event": cb.onVadEvent && cb.onVadEvent(msg.event, msg); break;
      case "tts_started": cb.onTtsStarted && cb.onTtsStarted(msg); break;
      case "tts_sentence_done": cb.onTtsSentenceDone && cb.onTtsSentenceDone(msg); break;
      case "tts_done": cb.onTtsDone && cb.onTtsDone(msg); break;
      case "error": cb.onError && cb.onError(msg); break;
      default: cb.onEvent && cb.onEvent(msg); // unknown frames: ignorable
    }
  }

  _onBinary(buf) {
    let bytes = new Uint8Array(buf);
    // One-time 4-byte LE uint32 sample-rate header (first binary frame of
    // the session; never re-emitted). Tolerate a pathological split across
    // frames by accumulating until 4 bytes are available.
    if (this.sampleRate === 0) {
      const merged = new Uint8Array(this._pre.length + bytes.length);
      merged.set(this._pre); merged.set(bytes, this._pre.length);
      if (merged.length < 4) { this._pre = merged; return; }
      this.sampleRate = new DataView(merged.buffer).getUint32(0, true);
      this._pre = new Uint8Array(0);
      bytes = merged.subarray(4);
      if (this.opts.onSampleRate) this.opts.onSampleRate(this.sampleRate);
    }
    if (this._leftover.length) {
      const merged = new Uint8Array(this._leftover.length + bytes.length);
      merged.set(this._leftover); merged.set(bytes, this._leftover.length);
      bytes = merged;
      this._leftover = new Uint8Array(0);
    }
    const evenLen = bytes.length - (bytes.length % 2);
    if (bytes.length % 2) this._leftover = bytes.slice(evenLen);
    if (!evenLen) return;
    const int16 = new Int16Array(
      bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + evenLen));
    if (this.opts.onAudioChunk) this.opts.onAudioChunk(int16);
    if (this.opts.playback !== false) this._schedule(int16);
  }

  /* ── gap-free Web Audio playback (TTSStreamPlayer scheduling) ───── */

  _ensureCtx() {
    if (!this.ctx) this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    if (this.ctx.state === "suspended") this.ctx.resume();
    return this.ctx;
  }

  _schedule(int16) {
    if (!int16.length || !this.sampleRate) return;
    const ctx = this._ensureCtx();
    const buf = ctx.createBuffer(1, int16.length, this.sampleRate);
    const ch = buf.getChannelData(0);
    for (let i = 0; i < int16.length; i++) ch[i] = int16[i] / 32768;
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    const startAt = Math.max(ctx.currentTime + 0.03, this._playhead);
    src.start(startAt);
    this._playhead = startAt + buf.duration;
    this._sources.push(src);
    if (!this._playing) {
      this._playing = true;
      if (this.opts.onPlaybackStart) this.opts.onPlaybackStart();
    }
    this._armEndTimer();
  }

  _armEndTimer() {
    if (this._endTimer) clearTimeout(this._endTimer);
    const ctx = this.ctx;
    const waitMs = Math.max(0, (this._playhead - ctx.currentTime) * 1000) + 80;
    this._endTimer = setTimeout(() => {
      this._endTimer = null;
      if (!this._playing) return;
      if (this.ctx && this._playhead - this.ctx.currentTime > 0.05) {
        this._armEndTimer(); // more audio got scheduled meanwhile
        return;
      }
      this._sources = [];
      this._playing = false;
      if (this.opts.onPlaybackEnd) this.opts.onPlaybackEnd();
    }, waitMs);
  }

  /** Local-only: silence + clear everything scheduled (no wire frame). */
  stopPlayback() {
    if (this._endTimer) { clearTimeout(this._endTimer); this._endTimer = null; }
    for (const s of this._sources) { try { s.stop(); } catch { /* ended */ } }
    this._sources = [];
    this._playhead = 0;
    if (this._playing) {
      this._playing = false;
      if (this.opts.onPlaybackEnd) this.opts.onPlaybackEnd();
    }
  }

  /** Close the WS (server frees the session slot + cancels in-flight work). */
  close() {
    this.stopPlayback();
    if (this.ws) {
      const ws = this.ws;
      this.ws = null;
      try { ws.close(); } catch { /* already closing */ }
    }
  }
}
