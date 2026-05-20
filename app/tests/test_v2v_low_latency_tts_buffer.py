from app.core.v2v import LowLatencyTTSBuffer, SentenceBuffer


def test_low_latency_tts_buffer_emits_cjk_clause_before_full_sentence():
    buf = LowLatencyTTSBuffer(language="zh")

    assert list(buf.add("从前有个小狐狸，")) == ["从前有个小狐狸，"]
    assert list(buf.add("它特别喜欢冒险。")) == ["它特别喜欢冒险。"]
    assert buf.is_empty()


def test_low_latency_tts_buffer_emits_bounded_cjk_run_without_punctuation():
    buf = LowLatencyTTSBuffer(language="zh", target_chars=10, max_chars=14)

    chunks = list(buf.add("这是一个没有标点但应该尽快开口的中文回复"))

    assert chunks[0] == "这是一个没有标点但应"
    assert "".join(chunks) == "这是一个没有标点但应该尽快开口的中文回复"
    assert list(buf.flush()) == []


def test_low_latency_tts_buffer_keeps_short_soft_break_until_useful():
    buf = LowLatencyTTSBuffer(language="zh", min_chars=8, target_chars=12, max_chars=16)

    assert list(buf.add("你好，")) == []
    assert list(buf.add("我现在可以")) == ["你好，我现在可以"]
    assert list(buf.flush()) == []


def test_low_latency_tts_buffer_flushes_remainder():
    buf = LowLatencyTTSBuffer(language="zh")

    assert list(buf.add("再给我讲个")) == []
    assert list(buf.flush()) == ["再给我讲个"]
    assert buf.is_empty()


def test_sentence_buffer_still_waits_for_pysbd_or_flush():
    buf = SentenceBuffer(language="zh")

    assert list(buf.add("从前有个小狐狸，")) == []
    assert list(buf.flush()) == ["从前有个小狐狸，"]
