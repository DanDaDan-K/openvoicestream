# P0b/P1 Memory Optimization Spec - 2026-04-28

## Section A: P0b - TRT engine IStreamReaderV2 streaming load

Goal: remove the extra host-RAM copy created by reading every `.engine` into `std::vector<char>` before `IRuntime::deserializeCudaEngine`. The observed hot path is `benchmark/cpp/tts_trt_engine.cpp:51` `LoadEngineFile`, which opens the file, allocates `std::vector<char> data(size)`, and reads the entire blob at `benchmark/cpp/tts_trt_engine.cpp:59`. That vector is then passed to TensorRT at `benchmark/cpp/tts_trt_engine.cpp:94`/`96` for `TRTTalkerEngine`, `benchmark/cpp/tts_trt_engine.cpp:275`/`278` for separate prefill, `benchmark/cpp/tts_trt_engine.cpp:1096`/`1098` for `TRTCPEngine`, `benchmark/cpp/tts_trt_engine.cpp:1265`/`1267` for `TRTCPKVEngine`, `benchmark/cpp/tts_trt_engine.cpp:2411`/`2413` for `TRTVocoderEngine`, and `benchmark/cpp/tts_trt_engine.cpp:2547`/`2549` for `TRTASRPrefillEngine`.

Modified files:

- `benchmark/cpp/tts_trt_engine.cpp:51`, `LoadEngineFile`: replace or bypass full-buffer read with a streaming engine-reader helper.
- `benchmark/cpp/tts_trt_engine.cpp:85`, `TRTTalkerEngine::TRTTalkerEngine`: deserialize via reader helper.
- `benchmark/cpp/tts_trt_engine.cpp:272`, `TRTTalkerEngine::LoadPrefillEngine`: deserialize via reader helper.
- `benchmark/cpp/tts_trt_engine.cpp:1093`, `TRTCPEngine::TRTCPEngine`: deserialize via reader helper.
- `benchmark/cpp/tts_trt_engine.cpp:1260`, `TRTCPKVEngine::TRTCPKVEngine`: deserialize via reader helper.
- `benchmark/cpp/tts_trt_engine.cpp:2408`, `TRTVocoderEngine::TRTVocoderEngine`: deserialize via reader helper.
- `benchmark/cpp/tts_trt_engine.cpp:2541`, `TRTASRPrefillEngine::TRTASRPrefillEngine`: deserialize via reader helper.
- `benchmark/cpp/tts_trt_engine.h:180`, `TRTTalkerEngine` runtime members and other engine classes need no public API changes; helper can remain file-local in the `.cpp`.

IStreamReaderV2 skeleton [unverified because TensorRT headers are not present in this workspace; `rg` under `/usr`, `/usr/local`, and `/usr/src` found no `IStreamReaderV2`]:

```cpp
// Verify exact signatures on Jetson:
// rg -n "class IStreamReaderV2|deserializeCudaEngine" /usr/include /usr/src /usr/local
class FileStreamReader final : public nvinfer1::IStreamReaderV2 {
 public:
  explicit FileStreamReader(const std::string& path, size_t chunk_size = 1 << 20);
  ~FileStreamReader() noexcept override;

  // Exact signatures must match NvInferRuntime.h on TRT 10.3 [unverified].
  void* read(size_t size, cudaStream_t stream) noexcept override;
  bool seek(int64_t offset, nvinfer1::IStreamReaderV2::SeekPosition where) noexcept override;

 private:
  int fd_ = -1;
  std::vector<char> bounce_;
  int64_t size_ = 0;
  int64_t pos_ = 0;
  bool failed_ = false;
};
```

Error handling: constructor opens and `fstat`s the engine path, throws `std::runtime_error` on open/stat failure; `read` returns `nullptr` and latches `failed_` on short read; `seek` validates bounds and returns `false` on invalid offset. The call site should check `engine_` after `deserializeCudaEngine(reader)` and preserve current abort/return behavior. Use a bounded bounce buffer, not one buffer sized to the engine.

pybind11 changes: API is unchanged. `benchmark/cpp/tts_binding.cpp:56` exposes `TRTDecoder(engine_path, ...)`, `benchmark/cpp/tts_binding.cpp:410` exposes `ASRPipeline(model_dir, engine_path, device_id)`, and neither should gain new Python arguments.

Build entry point: the actual C++ module build script is `benchmark/cpp/build.sh:1`. It creates `benchmark/cpp/build_cmake`, runs CMake with `-DORT_ROOT=/home/recomputer/ort-from-container` at `benchmark/cpp/build.sh:6`, and builds `qwen3_speech_engine` at `benchmark/cpp/build.sh:9`. Rebuild on Jetson with `cd benchmark/cpp && ./build.sh`; deploy per the script's printed copy path at `benchmark/cpp/build.sh:12`.

Verification:

- Pre/post host peak: start the service with only P0b changed, capture the process PID, and sample `/proc/$pid/status` for `VmPeak`, `VmHWM`, and `VmRSS` through startup. Keep the highest `VmPeak` before first ASR/TTS request.
- Integration: run `docker stats --no-stream` every second during container startup and compare peak memory against the current loader.
- Regression: run ASR/TTS end-to-end benchmarks using the same models and prompts; startup peak should drop, and steady-state ASR/TTS latency should not degrade beyond noise because deserialization is startup-only.

Risk analysis: `IStreamReaderV2` is the right target for TRT streaming deserialization [inferred], but TRT 10.3 stability must be verified against the installed headers and a small engine before broad rollout. `mmap` is simpler and may reduce explicit heap allocation, but it can still map the whole engine and may not avoid TensorRT's internal staging; use it only as fallback [inferred]. Concurrent multi-engine loading should be serialized during startup until the reader is proven reentrant, because the current code loads multiple engines into the same process and iGPU shared memory pressure is the failure mode [inferred].

## Section B: P1 - ASR encoder ORT CUDA EP to TRT

Goal: remove the ORT CUDA EP allocator/session overhead for the ASR encoder. The active Python path loads the encoder at `app/backends/qwen3_asr.py:825` with `ort.InferenceSession`, warms it at `app/backends/qwen3_asr.py:883`, and executes it at `app/backends/qwen3_asr.py:976` and streaming context path `app/backends/qwen3_asr.py:410`. A C++ ORT encoder also exists in `benchmark/cpp/asr_pipeline.cpp:43`/`45` and runs at `benchmark/cpp/asr_pipeline.cpp:164`.

Export script design: add `benchmark/export_asr_encoder_trt.py` rather than modifying the general ONNX exporters. Existing exporters already define the encoder graph: `benchmark/export_qwen3_asr.py:272` and `benchmark/export_qwen3_asr_unified.py:358` export `encoder.onnx` with input `mel` and output `audio_features`; dynamic axes are `mel` time dim 2 and output dim 1 at `benchmark/export_qwen3_asr_unified.py:379`. The new script should either call/reuse that export then invoke `trtexec`, or only build the engine from an existing `encoder.onnx`.

Observed chunk shapes: streaming uses `CHUNK_SIZE_SEC = 0.4` at `app/backends/qwen3_asr.py:67`, `LEFT_CONTEXT_SEC = 1.0` at `app/backends/qwen3_asr.py:68`, and mel frames use `hop_length=160` in `app/utils/whisper_mel.py:19`. `compute_whisper_log_mel` returns `[1, 128, T]` at `app/utils/whisper_mel.py:61`. Therefore TRT profiles should cover 1.4s streaming windows and full offline segments [inferred]: min `mel:1x128x100`, opt `mel:1x128x200`, max `mel:1x128x3000` for the 30s cap from `app/backends/qwen3_asr.py:1124`. If build time or memory is too high, use two profiles: streaming max 200 and offline max 3000 [inferred]. Precision should start with FP16, because current ORT path prefers `encoder_fp16.onnx` before `encoder.onnx` at `app/backends/qwen3_asr.py:822`; BF16 can be tested if FP16 accuracy fails [inferred].

C++ loader changes: add a `TRTASREncoder` class in `benchmark/cpp/tts_trt_engine.h`/`.cpp`, parallel to `TRTASRPrefillEngine` at `benchmark/cpp/tts_trt_engine.h:652`. Constructor takes `(engine_path, max_mel_frames, max_out_frames, hidden_dim=1024)`. Bind input `mel` as `[1,128,T]`, output `audio_features` as `[1,Tp,1024]`, set dynamic input shape before enqueue, copy output back to `std::vector<float>`, and return `{features, audio_len}` compatible with `ASRPipeline::EncoderOutput` at `benchmark/cpp/asr_pipeline.h:58`.

Python-side changes: preserve `self._encoder.run(None, {"mel": mel})[0]` call style by wrapping the pybind object in a tiny adapter:

```python
class _TRTEncoderAdapter:
    def __init__(self, engine): self.engine = engine
    def run(self, output_names, feeds):
        return [self.engine.run(feeds["mel"])]
```

At `app/backends/qwen3_asr.py:821`, if `ASR_ENCODER_BACKEND=trt_native`, import `qwen3_speech_engine`, create `TRTASREncoder`, and assign `self._encoder = _TRTEncoderAdapter(...)`; otherwise retain current ORT behavior. This keeps warm-up and transcription call sites unchanged.

Verification:

- Diff: feed the same mel tensors to ORT and TRT, compare `max_abs` and `mean_abs`; initial thresholds `max_abs <= 2e-2`, `mean_abs <= 2e-3` for FP16 [inferred], then validate transcript parity.
- Latency: N=10 per-chunk measurements on streaming-size mel and 30s mel; report ORT vs TRT ms.
- Memory: record startup `VmPeak`, steady `VmRSS`, and `docker stats` before/after; target is a 200-400 MB steady-state reduction [inferred from task statement, not observed in code].

Risk analysis: the encoder wrapper uses padding, reshape, convs, attention, GELU, and dynamic slicing at `benchmark/export_qwen3_asr_unified.py:226`-`257`; TensorRT may reject some shape/slice patterns [inferred]. Dynamic profile dispatch can add overhead for small streaming chunks, so benchmark both single wide profile and split profiles.

## Implementation order recommendation

Implement A before B. A is localized to engine file loading, keeps Python/pybind APIs unchanged, and addresses the startup peak that can prevent the process from reaching steady state. B depends on a new engine artifact and new C++/Python surface, so it has broader correctness risk. They can be developed in parallel only if one engineer owns `tts_trt_engine.cpp` loader changes and another owns `TRTASREncoder`/export, because both touch `benchmark/cpp/tts_trt_engine.*`. Estimated risk: P0b medium due to unverified TRT 10.3 stream-reader API; P1 medium-high due to encoder graph TRT compatibility and accuracy validation.
