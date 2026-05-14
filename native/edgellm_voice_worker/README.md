# Voice Workers (jetson-voice build entry)

Worker C++ source lives in the sibling `qwen3-edgellm-jetson` repo at
`native/edgellm_voice_worker/`. This directory holds only the
CMakeLists.txt that drives the jetson-voice build, referencing the
canonical sources via `WORKER_SRC_DIR` (default
`../../../qwen3-edgellm-jetson/native/edgellm_voice_worker`).

To edit worker code (qwen3_asr_worker, qwen3_tts_worker, mel_extractor,
kissfft), modify files in the qwen3-edgellm-jetson repo. Reproducer
(`reproduce_qwen3_highperf.sh`) clones both repos to sibling paths, so
the relative reference always resolves.
