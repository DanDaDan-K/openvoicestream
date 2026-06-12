"""Tool-call stability bench for the reBot voice-arm demo.

WHY THIS EXISTS
---------------
Concern: does the small LLM (Qwen3-4B-AWQ on edge-llm) lose tool-calling
accuracy after many calls in a demo session, and should we cap conversation
history?

Architectural finding (voxedge/engine/conversation.py:910/929/978): in
server-loop mode the LLM turn is built FRESH every utterance —
``[{system}, {user}]`` — with NO persistent cross-turn history member. So the
"history accumulates → model mimics history → stops emitting tool_calls"
failure (KNOWN_ISSUES ISSUE-001) is structurally impossible in production, and
a history cap is moot there. What remains is the model's per-command
tool-selection reliability, which is independent of turn count.

This bench measures exactly that. It replays the production server-loop request
shape (the real system prompt + the 8 advertised tool schemas) against an
edge-llm endpoint for a matrix of [command × paraphrase] × N repeats, and:
  * scores tool-selection accuracy (right tool name) + arg correctness,
  * checks determinism (same input → same tool across repeats),
  * runs a long SEQUENTIAL mixed sequence to expose any server-side KV/prefix
    drift (if accuracy were turn-dependent, it would show here),
  * lists every failure (wrong tool / no tool / wrong args) so the presenter
    knows which phrasings are fragile.

It is model-swappable (``--base-url`` / ``--model``) so the SAME matrix can be
run against Qwen3-4B-AWQ (current), Qwen3.5-4B-GDN, and the GDN-MTP variant to
pick the most demo-stable model.

Run inside a container on the voice-arm network (reaches edge-llm:8000), e.g.:
  docker exec voice-rebot-arm python /home/seeed/toolcall_stability_bench.py \
    --base-url http://edge-llm:8000/v1 --model Qwen/Qwen3-4B-AWQ --repeats 5
Stdlib only (urllib) so it runs in any python container.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from collections import defaultdict


# ── production system prompt (voice_rebot_arm/config.yaml) ──────────────
SYSTEM_PROMPT = (
    "You are a voice controller for a reBot B601-DM robotic arm.\n"
    "The user speaks commands aloud and you reply via TTS.\n\n"
    "Behaviour:\n"
    "- When the user clearly requests a motion, FIRST emit a brief\n"
    "  spoken acknowledgement, THEN call the matching tool function.\n"
    "  The acknowledgement is ONE OR TWO words in the user's language\n"
    "  (\"好的，\" / \"OK,\" / \"Sure,\") and gets spoken immediately while\n"
    "  the robot executes the motion. Examples:\n"
    "    User: \"挥手\"       → Emit \"好的\" → call wave()\n"
    "    User: \"回到原位\"   → Emit \"好的\" → call go_home()\n"
    "    User: \"张开夹爪\"   → Emit \"好的\" → call open_gripper()\n"
    "  AFTER the tool returns, emit a short confirmation (\"已挥手\" /\n"
    "  \"Done\" / \"已回到原位\"), 1 short sentence max.\n"
    "- To pick the right tool, read each tool's description and match the\n"
    "  user's exact phrase against the trigger words listed there. Chinese\n"
    "  triggers are LITERAL: '挥手' MUST call wave; '张开夹爪' MUST call\n"
    "  open_gripper; '闭合夹爪' MUST call close_gripper; '回到原位' MUST\n"
    "  call go_home; '指向' MUST call point_at. Never substitute a\n"
    "  semantically-similar tool — if no description literally contains the\n"
    "  user's phrase, reply that you don't have that action.\n"
    "- HARD RULE — only call a tool when the user's text contains the\n"
    "  EXACT trigger phrase listed in that tool's description. If the\n"
    "  phrase is not literally present, DO NOT call any tool — reply\n"
    "  \"没听清，请再说一次。\" / \"Sorry, I didn't catch that.\" instead.\n\n"
    "/no_think"
)


def _fn(name: str, desc: str, params: dict | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": params or {"type": "object", "properties": {}},
        },
    }


# ── the 8 advertised tools (actions.yaml descriptions + grasp catalog +
#    builtins), reconstructed to match what the agent advertises ──────────
_GRASP_CATALOG = ["box", "cardboard box", "carton", "package"]
_GRASP_DESC = (
    "Pick up / grasp an object using the camera-guided arm when the user asks "
    "to grab/pick something up ('抓','拿起','夹起','抓取','grab','pick up'). "
    "object_name MUST be exactly one of these catalog labels: ["
    + ", ".join(repr(c) for c in _GRASP_CATALOG)
    + "]. Map the user's spoken object to the closest catalog label and pass "
    "that English label verbatim (e.g. user says '抓盒子'/'把箱子拿起来' -> "
    "object_name='box'). Do NOT pass the user's Chinese words; the detector "
    "only knows the catalog labels above."
)

TOOLS = [
    _fn("go_home", 'Return the arm to its home / ready position. Triggers: "回到原位", "回家", "归位", "go home", "home", "reset position".'),
    _fn("open_gripper", 'Open the gripper / release. Triggers: "张开夹爪", "松开", "打开夹爪", "把夹爪打开", "夹爪打开", "open gripper", "release", "let go".'),
    _fn("close_gripper", 'Close the gripper / grasp. Triggers: "闭合夹爪", "夹紧", "抓住", "合上夹爪", "关闭夹爪", "close gripper", "grasp", "grab".'),
    _fn("wave", 'Wave hello by swinging the arm side to side. Triggers: "挥手", "挥一下手", "挥挥手", "打招呼", "打个招呼", "wave", "say hi".'),
    _fn("point_at", 'Point forward at an object. Triggers: "指向", "指一下", "point at", "point", "show me". Do not use for head nodding: "点头" is unsupported.'),
    _fn("grasp_object", _GRASP_DESC, {
        "type": "object",
        "properties": {"object_name": {"type": "string", "description": "Catalog label of the object to grasp."}},
        "required": ["object_name"],
    }),
    _fn("time_now", "Return the current local time as ISO 8601."),
    _fn("set_mode", "Switch the agent to a different mode.", {
        "type": "object",
        "properties": {"mode_name": {"type": "string"}},
        "required": ["mode_name"],
    }),
]


# ── test matrix: (utterance, expected_tool_or_None, expected_arg_substr) ─
# expected_tool=None means "no tool should fire" (chit-chat / out-of-scope).
MATRIX = [
    # wave
    ("挥手", "wave", None),
    ("挥挥手", "wave", None),
    ("挥一下手", "wave", None),
    ("跟大家打个招呼", "wave", None),
    # go_home
    ("回到原位", "go_home", None),
    ("回家", "go_home", None),
    ("归位", "go_home", None),
    ("复位", "go_home", None),
    # open_gripper
    ("张开夹爪", "open_gripper", None),
    ("打开夹爪", "open_gripper", None),
    ("把夹爪松开", "open_gripper", None),
    # close_gripper
    ("闭合夹爪", "close_gripper", None),
    ("夹紧", "close_gripper", None),
    ("合上夹爪", "close_gripper", None),
    # point_at
    ("指向那个物体", "point_at", None),
    ("指一下", "point_at", None),
    # grasp_object (vision grasp) — note the 抓住/抓 trigger collision with close_gripper
    ("抓盒子", "grasp_object", "box"),
    ("把盒子抓起来", "grasp_object", "box"),
    ("夹起盒子", "grasp_object", "box"),
    ("抓取盒子", "grasp_object", "box"),
    ("把那个箱子拿起来", "grasp_object", "box"),
    # out-of-scope / should NOT fire a motion tool
    ("你好", None, None),
    ("今天天气怎么样", None, None),
]


# ── HARD matrix: demo-realistic robustness. Each entry is
#    (utterance, expected_tool_or_None, expected_arg_substr, category).
# expected_tool=None → no motion tool should fire (chit-chat / unsupported).
# For ASR-homophone / truncation rows the "expected" is the demo-desired
# recovery; failures are later split into DANGEROUS (a different motion fired)
# vs SAFE (no tool → presenter just repeats).
MOTION_TOOLS = {"wave", "go_home", "open_gripper", "close_gripper", "point_at", "grasp_object"}
HARD_MATRIX = [
    # colloquial / indirect phrasings that still embed the trigger word
    ("帮我挥个手", "wave", None, "colloquial"),
    ("挥一挥手", "wave", None, "colloquial"),
    ("胳膊回到原位", "go_home", None, "colloquial"),
    ("请把夹爪张开", "open_gripper", None, "colloquial"),
    ("把夹爪闭合一下", "close_gripper", None, "colloquial"),
    ("帮我把盒子拿起来", "grasp_object", "box", "colloquial"),
    # polite / distractor wrappers around the literal trigger
    ("那个，麻烦你挥手好吗", "wave", None, "wrapper"),
    ("现在请回到原位吧", "go_home", None, "wrapper"),
    ("嗯…你帮我把夹爪松开", "open_gripper", None, "wrapper"),
    # ASR homophone / near-homophone errors (what STT may actually emit)
    ("灰手", "wave", None, "asr_homophone"),
    ("挥首", "wave", None, "asr_homophone"),
    ("回到原味", "go_home", None, "asr_homophone"),
    ("张开夹抓", "open_gripper", None, "asr_homophone"),
    ("必合夹爪", "close_gripper", None, "asr_homophone"),
    ("加紧", "close_gripper", None, "asr_homophone"),
    ("抓河子", "grasp_object", "box", "asr_homophone"),
    # truncated / partial (ASR cut the tail)
    ("张开", "open_gripper", None, "truncated"),
    ("回原位", "go_home", None, "truncated"),
    ("挥", "wave", None, "truncated"),
    # English / mixed
    ("wave", "wave", None, "english"),
    ("go home please", "go_home", None, "english"),
    ("grab the box", "grasp_object", "box", "english"),
    # intent collisions (抓住 is a close_gripper trigger, but a box is named)
    ("抓住盒子", "grasp_object", "box", "collision"),
    ("夹住这个盒子", "grasp_object", "box", "collision"),
    # traps — must NOT fire a motion tool
    ("点头", None, None, "trap_unsupported"),
    ("转个圈", None, None, "trap_unsupported"),
    ("你叫什么名字", None, None, "trap_chitchat"),
    ("给我讲个笑话", None, None, "trap_chitchat"),
    ("挥手用英语怎么说", None, None, "trap_meta"),
]


def call_llm(base_url: str, model: str, text: str, timeout: float = 30.0) -> dict:
    """One production-shaped chat/completions call; return parsed result."""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "tools": TOOLS,
        "tool_choice": "auto",
        "temperature": 0.0,
        "stream": False,
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read().decode())
    dt = time.time() - t0
    msg = resp["choices"][0]["message"]
    tcs = msg.get("tool_calls") or []
    tool = None
    arg = None
    if tcs:
        tool = tcs[0]["function"]["name"]
        try:
            arg = json.dumps(json.loads(tcs[0]["function"].get("arguments") or "{}"), ensure_ascii=False)
        except Exception:
            arg = tcs[0]["function"].get("arguments")
    return {"tool": tool, "arg": arg, "text": (msg.get("content") or "")[:60], "latency_s": round(dt, 2)}


def score(exp_tool, exp_arg, got_tool, got_arg) -> bool:
    if exp_tool is None:
        return got_tool is None
    if got_tool != exp_tool:
        return False
    if exp_arg is not None:
        return exp_arg in (got_arg or "")
    return True


def run_hard(args) -> int:
    """Demo-realistic robustness matrix with failure classified by demo impact."""
    print(f"=== HARD robustness bench :: model={args.model} base={args.base_url} repeats={args.repeats} ===\n")
    by_cat = defaultdict(lambda: [0, 0])
    dangerous = []   # a motion command/trap → a DIFFERENT motion tool fired (arm moves wrong)
    safe_miss = []   # a motion command → no tool (arm doesn't move; presenter repeats)
    spurious = []    # a trap (no-tool expected) → a motion tool fired (arm moves unexpectedly)
    nondet = []
    ok_total = tot = 0
    for (text, exp_tool, exp_arg, cat) in HARD_MATRIX:
        seen_tools = set()
        row_ok = 0
        last = None
        for _ in range(args.repeats):
            try:
                r = call_llm(args.base_url, args.model, text)
            except Exception as e:
                r = {"tool": "ERROR:" + type(e).__name__, "arg": str(e)[:60], "text": "", "latency_s": 0}
            last = r
            seen_tools.add(r["tool"])
            tot += 1
            if score(exp_tool, exp_arg, r["tool"], r["arg"]):
                row_ok += 1
                ok_total += 1
        by_cat[cat][0] += row_ok
        by_cat[cat][1] += args.repeats
        if len(seen_tools) > 1:
            nondet.append((text, exp_tool, sorted(map(str, seen_tools))))
        # classify the row by its modal (last) outcome for demo-impact triage
        got = last["tool"] if last else None
        if exp_tool is None:  # trap
            if got in MOTION_TOOLS:
                spurious.append((cat, text, got))
        else:  # motion command expected
            if got != exp_tool:
                if got in MOTION_TOOLS:
                    dangerous.append((cat, text, exp_tool, got))
                else:  # None / chit-chat reply / error
                    safe_miss.append((cat, text, exp_tool, got))

    print("── accuracy by category ──")
    for c in sorted(by_cat):
        p, t = by_cat[c]
        print(f"  {c:18s} {p}/{t}  ({100*p//max(t,1)}%)")
    print(f"\nOVERALL HARD: {ok_total}/{tot} ({100*ok_total//max(tot,1)}%)")

    print("\n── DEMO-IMPACT TRIAGE ──")
    print(f"  ⛔ DANGEROUS (command → WRONG motion fired, arm moves wrong): {len(dangerous)}")
    for cat, text, exp, got in dangerous:
        print(f"       [{cat}] {text!r}: expected {exp} → fired {got}")
    print(f"  ⚠️  SPURIOUS (trap → motion fired, arm moves unexpectedly): {len(spurious)}")
    for cat, text, got in spurious:
        print(f"       [{cat}] {text!r}: fired {got} (should be no-tool)")
    print(f"  ✅ SAFE-MISS (command → no tool, arm still, presenter repeats): {len(safe_miss)}")
    for cat, text, exp, got in safe_miss:
        print(f"       [{cat}] {text!r}: expected {exp} → {got}")
    if nondet:
        print(f"\n  non-deterministic rows ({len(nondet)}):")
        for text, exp, seen in nondet:
            print(f"     {text!r} (exp {exp}): {seen}")
    else:
        print("\n  ✓ deterministic across all repeats")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://edge-llm:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-AWQ")
    ap.add_argument("--repeats", type=int, default=5, help="repeats per utterance (determinism)")
    ap.add_argument("--sequence", type=int, default=60, help="length of the long mixed-sequential drift run")
    ap.add_argument("--hard", action="store_true", help="run the demo-realistic HARD robustness matrix instead")
    args = ap.parse_args()

    if args.hard:
        return run_hard(args)

    print(f"=== tool-call stability bench :: model={args.model} base={args.base_url} ===")
    print(f"system_prompt_len={len(SYSTEM_PROMPT)} tools={len(TOOLS)} matrix={len(MATRIX)} repeats={args.repeats}\n")

    # Phase 1: accuracy matrix + determinism (repeat each utterance N times)
    per_tool = defaultdict(lambda: [0, 0])  # expected_tool -> [pass, total]
    failures = []
    nondet = []
    overall_pass = overall_total = 0
    lat = []
    for (text, exp_tool, exp_arg) in MATRIX:
        got = []
        for _ in range(args.repeats):
            try:
                r = call_llm(args.base_url, args.model, text)
            except Exception as e:
                r = {"tool": "ERROR:" + type(e).__name__, "arg": str(e)[:80], "text": "", "latency_s": 0}
            got.append(r)
            lat.append(r["latency_s"])
            ok = score(exp_tool, exp_arg, r["tool"], r["arg"])
            key = exp_tool or "(no-tool)"
            per_tool[key][1] += 1
            overall_total += 1
            if ok:
                per_tool[key][0] += 1
                overall_pass += 1
            else:
                failures.append((text, exp_tool, exp_arg, r["tool"], r["arg"]))
        tools_seen = {g["tool"] for g in got}
        if len(tools_seen) > 1:
            nondet.append((text, exp_tool, sorted(map(str, tools_seen))))

    print("── per-command accuracy ──")
    for k in sorted(per_tool):
        p, t = per_tool[k]
        print(f"  {k:14s} {p}/{t}  ({100*p//max(t,1)}%)")
    print(f"\nOVERALL: {overall_pass}/{overall_total} ({100*overall_pass//max(overall_total,1)}%)")
    if lat:
        s = sorted(lat)
        print(f"latency_s: p50={s[len(s)//2]:.2f} p90={s[int(len(s)*0.9)]:.2f} max={s[-1]:.2f}")

    if nondet:
        print(f"\n── NON-DETERMINISTIC inputs ({len(nondet)}) — same text, different tool across repeats ──")
        for text, exp, seen in nondet:
            print(f"  {text!r} (exp {exp}): saw {seen}")
    else:
        print("\n✓ deterministic: every utterance produced the same tool across all repeats")

    if failures:
        print(f"\n── FAILURES ({len(failures)}) ──")
        seen = set()
        for text, exp_tool, exp_arg, got_tool, got_arg in failures:
            sig = (text, got_tool, got_arg)
            if sig in seen:
                continue
            seen.add(sig)
            print(f"  {text!r}: expected {exp_tool}({exp_arg}) → got {got_tool}({got_arg})")

    # Phase 2: long sequential mixed run — if accuracy degraded with turn
    # count (it shouldn't, since production is stateless), it surfaces here.
    print(f"\n── long sequential run ({args.sequence} calls, mixed order) — drift check ──")
    seq = [(MATRIX[i % len(MATRIX)]) for i in range(args.sequence)]
    win = []  # rolling correctness
    first_half = second_half = 0
    half = args.sequence // 2
    for idx, (text, exp_tool, exp_arg) in enumerate(seq):
        try:
            r = call_llm(args.base_url, args.model, text)
            ok = score(exp_tool, exp_arg, r["tool"], r["arg"])
        except Exception:
            ok = False
        win.append(ok)
        if idx < half:
            first_half += int(ok)
        else:
            second_half += int(ok)
    print(f"  first-half  {first_half}/{half} ({100*first_half//max(half,1)}%)")
    print(f"  second-half {second_half}/{args.sequence-half} ({100*second_half//max(args.sequence-half,1)}%)")
    print("  → if these two halves match, accuracy is position/turn-INDEPENDENT (no degradation).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
