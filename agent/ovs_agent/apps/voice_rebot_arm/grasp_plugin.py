"""GraspPlugin — registers the ``grasp_object`` LLM tool (Phase B).

Wires the torch-free vision-grasp pipeline (:func:`grasp_service.run_grasp_once`)
into the voice agent as a single ``grasp_object(object_name)`` tool with
``response_mode="parallel"``: the tool body dispatches the grasp onto a worker
thread and returns ``{"started": True, "target": ...}`` within ~200ms so the
LLM's spoken acknowledgement overlaps the multi-second physical grasp (same
fast-dispatch pattern as ``ArmPlugin.dispatch_action``).

Cancellation: a ``threading.Event`` is set on stop-intent ("停") or sleep
(``on_user_stop_intent`` / ``on_sleep``); the grasp service polls it before
every arm motion, safe-parks the gripper (open), and aborts. Ordinary user
speech does NOT cancel an in-flight motion — an early design cancelled on
``on_user_speech_start``, which meant saying ANYTHING mid-grasp silently
dropped the held object and the new command then bounced off the
already_running guard ("第一遍没反应"). Only an explicit stop word or sleep
interrupts the arm now; a new motion command while one runs is refused with
the busy motion's name so the LLM can tell the user to wait. A short
completion tone (success/failure pitch) tells the user when the arm is done
and ready for the next command.

Heavy deps (onnxruntime via the segmenter, camera SDK) are imported lazily on
first grasp so the agent boots on hosts without them.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ovs_agent.plugin import Plugin

logger = logging.getLogger(__name__)


class GraspPlugin(Plugin):
    name = "grasp"

    def __init__(self, app: Any, config: Optional[dict] = None) -> None:
        super().__init__(app)
        self.cfg = dict(config or {})
        self._arm_plugin: Any = None  # resolved at start()
        self._segmenter: Any = None
        self._camera: Any = None
        self._hand_eye: Optional[np.ndarray] = None
        self._cancel_event = threading.Event()
        self._grasp_task: Optional[asyncio.Task] = None
        self._registered = False
        # Last successful grasp result (grasp_pose / pregrasp_pose /
        # open_distance_m). put_down replays it so the object is placed back
        # at the camera-visible spot it was picked from. Cleared on a
        # successful put_down.
        self._last_grasp: Optional[dict] = None

    # ── lifecycle ──────────────────────────────────────────────────
    def setup(self) -> bool:
        # Register the tool eagerly so it is advertised on the first wake,
        # exactly like ArmPlugin registers its action tools in setup().
        if self.cfg.get("enabled", True) is False:
            logger.info("GraspPlugin: disabled by config; not registering grasp_object")
            return True
        self._register_tool()
        return True

    async def start(self) -> None:
        await super().start()
        # Find the ArmPlugin so we can reach the underlying RebotArm. The
        # actuator owns the CAN bus; we drive it directly for grasp moves.
        for plugin in getattr(self.app, "plugins", []) or []:
            if plugin.__class__.__name__ == "ArmPlugin":
                self._arm_plugin = plugin
                break
        if self._arm_plugin is None:
            logger.warning("GraspPlugin: no ArmPlugin found; grasp_object will error")
        else:
            # Cross-plugin motion mutex: let ArmPlugin's static actions see
            # our in-flight grasp/search/put_down (and refuse to interleave).
            reg = getattr(self._arm_plugin, "register_motion_source", None)
            if callable(reg):
                reg(self._busy_motion_name)

    def _busy_motion_name(self) -> Optional[str]:
        """Name of our in-flight motion (e.g. 'grasp-box'), or None."""
        task = self._grasp_task
        if task is not None and not task.done():
            try:
                return task.get_name()
            except Exception:  # pragma: no cover — defensive
                return "grasp"
        return None

    async def stop(self) -> None:
        await super().stop()
        # Signal the in-flight grasp to abort. The grasp body runs in a worker
        # thread (asyncio.to_thread) which CANNOT be force-cancelled — the
        # pipeline polls _cancel_event before each motion and safe-parks. So
        # set the event FIRST, then wait (bounded) for the worker to wind down
        # rather than just abandoning the thread mid-bus-op.
        self._cancel_event.set()
        if self._grasp_task is not None and not self._grasp_task.done():
            try:
                # Bounded wait for the worker to observe the cancel and return.
                await asyncio.wait_for(asyncio.shield(self._grasp_task), timeout=6.0)
            except asyncio.TimeoutError:
                logger.warning("GraspPlugin: grasp worker did not stop within 6s")
                self._grasp_task.cancel()
                try:
                    await self._grasp_task
                except (asyncio.CancelledError, Exception):
                    pass
            except (asyncio.CancelledError, Exception):
                pass
        self._grasp_task = None
        # Release the camera (plugin owns it; the grasp pipeline only borrows
        # the already-opened handle and must never close the shared camera).
        cam = self._camera
        if cam is not None:
            close_fn = getattr(cam, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    logger.exception("GraspPlugin: camera close failed")
            self._camera = None

    # ── cancellation hooks (barge-in / stop / sleep) ────────────────
    async def on_user_stop_intent(self, data: str) -> None:
        self._cancel_event.set()
        logger.info("GraspPlugin: stop-intent → cancel grasp")

    async def on_user_speech_start(self) -> None:
        # Speech-start barge-in cancel is OFF by default: in a noisy demo hall
        # ambient noise / TTS echo opens the mic and would abort an in-flight
        # grasp mid-motion ("arm twitches then stops"). A grasp is a short,
        # committed motion; let it finish. The user can still abort via an
        # explicit stop-intent (on_user_stop_intent) or by going to sleep. Set
        # grasp.cancel_on_speech: true to restore the old barge-in behaviour.
        if not self.cfg.get("cancel_on_speech", False):
            return
        if self._grasp_task is not None and not self._grasp_task.done():
            self._cancel_event.set()
            logger.info("GraspPlugin: barge-in → cancel grasp")

    async def on_sleep(self, data) -> None:  # noqa: ANN001
        self._cancel_event.set()

    # ── tool registration ───────────────────────────────────────────
    def _register_tool(self) -> None:
        if self._registered:
            return
        registry = getattr(self.app, "tool_registry", None)
        if registry is None:
            logger.warning("GraspPlugin: app has no tool_registry; cannot register")
            return

        plugin = self

        # The vision model only recognises a fixed catalog of class labels (the
        # configured ``yolo_classes``). The detector filters its results by the
        # ``object_name`` we pass, so the LLM MUST fill object_name with one of
        # these EXACT labels — not the user's spoken word. The LLM maps the
        # user's intent ("抓盒子"/"把那个箱子拿起来") to the closest catalog label.
        catalog = list(self.cfg.get("yolo_classes", []))
        catalog_str = ", ".join(repr(c) for c in catalog) or "'box'"
        grasp_desc = (
            "Pick up / grasp an object using the camera-guided arm when the "
            "user asks to grab/pick something up ('抓','拿起','夹起','抓取',"
            "'grab','pick up'). "
            f"object_name MUST be exactly one of these catalog labels: [{catalog_str}]. "
            "Map the user's spoken object to the closest catalog label and pass "
            "that English label verbatim (e.g. user says '抓盒子'/'把箱子拿起来' "
            "-> object_name='box'). Do NOT pass the user's Chinese words; the "
            "detector only knows the catalog labels above."
        )

        @registry.tool(
            name="grasp_object",
            description=grasp_desc,
            timeout_s=2.0,
            preamble_text="好的。",
            response_mode="parallel",
        )
        async def grasp_object(object_name: str) -> dict:  # noqa: ANN001
            return await plugin._dispatch_grasp(object_name)

        # search_object — sweep the arm-mounted camera across observation poses
        # to FIND an object that may be outside the current view, then point at
        # it WITHOUT grasping. Separate from grasp_object so a demo can show
        # "find" and "grasp" as distinct steps.
        search_desc = (
            "Search for / locate an object by sweeping the camera around when "
            "the user asks to FIND or LOOK FOR something but not (yet) pick it "
            "up ('找一下','找找','搜索','看看有没有','find','look for','search "
            "for'). The arm scans several viewpoints, stops when it sees the "
            "object and points at it WITHOUT grasping. "
            f"object_name MUST be exactly one of these catalog labels: [{catalog_str}]. "
            "Map the user's spoken object to the closest catalog label "
            "(e.g. '找一下盒子' -> object_name='box')."
        )

        @registry.tool(
            name="search_object",
            description=search_desc,
            timeout_s=2.0,
            preamble_text="好的。",
            response_mode="parallel",
        )
        async def search_object(object_name: str) -> dict:  # noqa: ANN001
            return await plugin._dispatch_search(object_name)

        # put_down — place the held object back where grasp_object picked it
        # up (a spot the camera detected it at, so the NEXT grasp can find it
        # again), then return home. Lives here rather than actions.yaml so it
        # can replay the recorded grasp/pregrasp poses and release width.
        put_down_desc = (
            "Put down / place / release the object currently held by the "
            "gripper. The arm sets it back down at the spot it was picked up "
            "from (so the camera can find it again), opens the gripper, and "
            "returns home. Triggers: '放下', '放下来', '放回去', '放到桌上', "
            "'把它放下', 'put it down', 'put down', 'place it', 'set it down', "
            "'drop it', 'release it'."
        )

        @registry.tool(
            name="put_down",
            description=put_down_desc,
            timeout_s=2.0,
            preamble_text="好的。",
            response_mode="parallel",
        )
        async def put_down() -> dict:
            return await plugin._dispatch_put_down()

        self._registered = True
        # Tool list changed → invalidate the warmed prefix cache (mirrors
        # ArmPlugin._reregister_tools).
        session = getattr(self.app, "session", None)
        if session is not None and getattr(session, "cache_warmed", False):
            session.cache_warmed = False
        logger.info("GraspPlugin: registered grasp_object tool")

    # ── dispatch (fast-return, worker-thread grasp) ─────────────────
    async def _dispatch_grasp(self, object_name: str) -> dict:
        target = (object_name or "").strip()
        if not target:
            return {"started": False, "error": "empty object_name"}
        # Safety net: the detector filters by class label, which only knows the
        # configured catalog (English). If the LLM passed the user's word
        # instead (e.g. '盒子'), remap to a catalog label so detections aren't
        # silently filtered out. Exact or substring match against the catalog is
        # honoured; otherwise fall back to the first catalog class (the model
        # only knows box-like classes here, so any is the box).
        catalog = list(self.cfg.get("yolo_classes", []))
        if catalog:
            tl = target.lower()
            # Resolve to an EXACT catalog label (the detector filters by exact
            # class name). Prefer exact match, then substring either way, else
            # the first catalog class. Guarantees the filter can match.
            resolved = (
                next((c for c in catalog if c.lower() == tl), None)
                or next((c for c in catalog if c.lower() in tl or tl in c.lower()), None)
                or catalog[0]
            )
            if resolved != target:
                logger.info(
                    "GraspPlugin: object_name %r → catalog label %r", target, resolved
                )
                target = resolved
        if self._arm_plugin is None or getattr(self._arm_plugin, "arm", None) is None:
            return {"started": False, "target": target, "error": "arm not available"}

        actuator = self._arm_plugin.arm
        arm = getattr(actuator, "robot", None)
        if arm is None:
            return {"started": False, "target": target, "error": "arm not connected"}

        # SAFETY: the grasp pipeline drives the arm directly. Refuse to start
        # if torque is off — the torque gate is the single source of truth for
        # "may we move?", and a parallel grasp must honour it just like
        # execute_sequence does.
        if not getattr(actuator, "torque_enabled", False):
            return {"started": False, "target": target, "error": "torque disabled"}

        # SAFETY: refuse re-entry / interleave. A second grasp while one is in
        # flight would overwrite _cancel_event / _grasp_task and let two grasp
        # workers race the same arm/bus; a static action in flight would
        # interleave waypoints with ours.
        busy_err = self._refuse_if_motion_busy()
        if busy_err is not None:
            return {"started": False, "target": target, **busy_err}

        # Fresh cancel token for this grasp.
        self._cancel_event = threading.Event()
        cancel_event = self._cancel_event

        try:
            self._ensure_perception()
        except Exception as exc:
            logger.exception("GraspPlugin: perception init failed")
            return {"started": False, "target": target, "error": str(exc)}

        async def _runner() -> dict:
            from .grasp_service import run_grasp_once

            params = self._grasp_params()
            return await asyncio.to_thread(
                run_grasp_once,
                target,
                arm=arm,
                actuator=actuator,
                segmenter=self._segmenter,
                camera=self._camera,
                T_hand_eye=self._hand_eye,
                cancel_event=cancel_event,
                **params,
            )

        self._grasp_task = asyncio.create_task(_runner(), name=f"grasp-{target}")
        # Fire-and-forget: surface completion in the log; the LLM round 2 /
        # spoken ack overlaps the physical motion (parallel mode).
        self._grasp_task.add_done_callback(self._on_grasp_done)
        return {"started": True, "target": target}

    async def _dispatch_search(self, object_name: str) -> dict:
        """Fast-return dispatch for search_object (sweep + locate, no grasp)."""
        target = (object_name or "").strip()
        catalog = list(self.cfg.get("yolo_classes", []))
        if catalog and target:
            tl = target.lower()
            target = (
                next((c for c in catalog if c.lower() == tl), None)
                or next((c for c in catalog if c.lower() in tl or tl in c.lower()), None)
                or catalog[0]
            )
        elif not target and catalog:
            target = catalog[0]
        if not target:
            return {"started": False, "error": "empty object_name"}
        if self._arm_plugin is None or getattr(self._arm_plugin, "arm", None) is None:
            return {"started": False, "target": target, "error": "arm not available"}
        actuator = self._arm_plugin.arm
        arm = getattr(actuator, "robot", None)
        if arm is None:
            return {"started": False, "target": target, "error": "arm not connected"}
        if not getattr(actuator, "torque_enabled", False):
            return {"started": False, "target": target, "error": "torque disabled"}
        # Single arm-motion slot shared with grasp + ArmPlugin actions.
        busy_err = self._refuse_if_motion_busy()
        if busy_err is not None:
            return {"started": False, "target": target, **busy_err}
        self._cancel_event = threading.Event()
        cancel_event = self._cancel_event
        try:
            self._ensure_perception()
        except Exception as exc:
            logger.exception("GraspPlugin: perception init failed (search)")
            return {"started": False, "target": target, "error": str(exc)}

        async def _runner() -> dict:
            from .grasp_service import run_search_once

            return await asyncio.to_thread(
                run_search_once,
                target,
                arm=arm,
                actuator=actuator,
                segmenter=self._segmenter,
                camera=self._camera,
                T_hand_eye=self._hand_eye,
                cancel_event=cancel_event,
                **self._search_params(),
            )

        self._grasp_task = asyncio.create_task(_runner(), name=f"search-{target}")
        self._grasp_task.add_done_callback(self._on_grasp_done)
        return {"started": True, "target": target}

    def _search_params(self) -> dict:
        params: dict[str, Any] = {}
        sp = self.cfg.get("scan_poses")
        if sp:
            # each pose is a list/tuple [x,y,z,roll,pitch,yaw]
            params["scan_poses"] = [tuple(float(v) for v in p) for p in sp]
        for k in ("conf", "move_duration", "warm_up_frames", "frames", "indicate"):
            if k in self.cfg:
                v = self.cfg[k]
                if k == "indicate":
                    params[k] = v if isinstance(v, bool) else str(v).strip().lower() in {"1", "true", "yes"}
                elif k in ("warm_up_frames", "frames"):
                    try:
                        params[k] = int(float(str(v).strip()))
                    except (TypeError, ValueError):
                        pass
                else:
                    try:
                        params[k] = float(str(v).strip())
                    except (TypeError, ValueError):
                        pass
        return params

    async def _dispatch_put_down(self) -> dict:
        """Fast-return dispatch for put_down (place back where picked up)."""
        if self._arm_plugin is None or getattr(self._arm_plugin, "arm", None) is None:
            return {"started": False, "error": "arm not available"}
        actuator = self._arm_plugin.arm
        arm = getattr(actuator, "robot", None)
        if arm is None:
            return {"started": False, "error": "arm not connected"}
        if not getattr(actuator, "torque_enabled", False):
            return {"started": False, "error": "torque disabled"}
        # Single arm-motion slot shared with grasp/search + ArmPlugin actions.
        busy_err = self._refuse_if_motion_busy()
        if busy_err is not None:
            return {"started": False, **busy_err}
        last = self._last_grasp or {}
        # Admission: a recorded grasp is sufficient — place-back is harmless
        # even if the jaw somehow let go in the meantime, and the physical
        # holding check can be momentarily False mid-transition. Refuse only
        # when BOTH there is no recorded grasp AND the gripper is physically
        # not clamping anything (encoder gap + grip torque on the real arm).
        if not last.get("grasp_pose"):
            try:
                holding_attr = getattr(arm, "gripper_is_holding", None)
                held = holding_attr() if callable(holding_attr) else holding_attr
            except Exception:
                held = None
            if held is False:
                return {"started": False, "error": "nothing held"}
        kwargs: dict[str, Any] = {
            "grasp_pose": last.get("grasp_pose"),
            "pregrasp_pose": last.get("pregrasp_pose"),
        }
        try:
            kwargs["open_distance_m"] = float(last.get("open_distance_m", 0.09))
        except (TypeError, ValueError):
            kwargs["open_distance_m"] = 0.09
        pp = self.cfg.get("place_pose")
        if pp:
            try:
                kwargs["place_pose"] = tuple(float(v) for v in pp)
            except (TypeError, ValueError):
                logger.warning("GraspPlugin: ignoring malformed place_pose")
        md = self.cfg.get("move_duration")
        if md is not None:
            try:
                kwargs["move_duration"] = float(str(md).strip())
            except (TypeError, ValueError):
                pass

        self._cancel_event = threading.Event()
        cancel_event = self._cancel_event

        async def _runner() -> dict:
            from .grasp_service import run_put_down_once

            res = await asyncio.to_thread(
                run_put_down_once,
                arm=arm,
                actuator=actuator,
                cancel_event=cancel_event,
                **kwargs,
            )
            if res.get("success"):
                # Placed back — the recorded pose is consumed; the next grasp
                # re-detects from camera.
                self._last_grasp = None
            return res

        self._grasp_task = asyncio.create_task(_runner(), name="put_down")
        self._grasp_task.add_done_callback(self._on_grasp_done)
        return {"started": True, "used_recorded_pose": bool(last.get("grasp_pose"))}

    def _refuse_if_motion_busy(self) -> Optional[dict]:
        """Shared single-motion-slot guard for grasp/search/put_down, plus the
        cross-plugin check against ArmPlugin's static actions. Returns an
        error payload naming the in-flight motion (so the LLM can tell the
        user what the arm is still doing), or None when the arm is free."""
        mine = self._busy_motion_name()
        if mine:
            return {"error": "already_running", "current": mine}
        chk = getattr(self._arm_plugin, "busy_action", None)
        if callable(chk):
            try:
                busy = chk()
            except Exception:  # pragma: no cover — defensive
                busy = None
            # busy_action also consults our registered source, but our slot
            # was checked above and is idle — anything it reports now is a
            # static action sequence.
            if busy:
                return {"error": "arm_busy", "current": str(busy)}
        return None

    def _on_grasp_done(self, task: asyncio.Task) -> None:
        try:
            res = task.result()
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("GraspPlugin: grasp task crashed")
            self._play_done_tone(False)
            return
        # Remember a successful grasp so put_down can place the object back at
        # the (camera-visible, IK-validated) pickup spot. Search / put_down
        # results have no grasp_pose, so they never overwrite this.
        if res and res.get("success") and res.get("grasp_pose"):
            self._last_grasp = dict(res)
        logger.info("GraspPlugin: grasp result: %s", res)
        # Audible completion feedback: the parallel tool result returned long
        # ago ("好的"), so without this the user cannot tell when the arm is
        # done and safe to command again. Skip on cancel — the user stopped it
        # and already knows.
        if res and not res.get("cancelled"):
            self._play_done_tone(bool(res.get("success") or res.get("found")))

    def _play_done_tone(self, ok: bool) -> None:
        """Play a short local tone when a motion finishes (success/failure
        pitch), mirroring app_base's wake tone. Config (all optional):
        ``grasp.done_tone: {hz_ok, hz_fail, ms}``; set ms: 0 to disable."""
        try:
            audio = getattr(self.app, "audio", None)
            notify = getattr(audio, "play_notification", None)
            if not callable(notify):
                return
            cfg = dict(self.cfg.get("done_tone") or {})
            hz = float(cfg.get("hz_ok", 1175)) if ok else float(cfg.get("hz_fail", 330))
            ms = float(cfg.get("ms", 180))
            if hz <= 0 or ms <= 0:
                return
            import math
            import struct
            import time as _time

            sr = int(getattr(audio, "output_sr", None) or 16000)
            n = int(sr * ms / 1000)
            amp = 16000
            fade = min(n // 4, int(sr * 0.01))
            pcm = bytearray(n * 2)
            for i in range(n):
                s = amp * math.sin(2 * math.pi * hz * i / sr)
                if i < fade:
                    s *= i / fade
                elif i >= n - fade:
                    s *= (n - 1 - i) / fade
                struct.pack_into("<h", pcm, i * 2, max(-32768, min(32767, int(s))))
            notify(bytes(pcm))
            # Suppress the mic while the tone plays so it isn't captured as a
            # command (same pattern as the wake tone).
            sup = getattr(self.app, "_local_output_mic_suppress_until", None)
            if sup is not None:
                self.app._local_output_mic_suppress_until = max(
                    float(sup), _time.monotonic() + (ms + 200.0) / 1000.0
                )
            logger.info("GraspPlugin: done tone (%s) %dHz %dms",
                        "ok" if ok else "fail", int(hz), int(ms))
        except Exception:  # pragma: no cover — defensive
            logger.debug("GraspPlugin: done tone failed", exc_info=True)

    # ── perception / calibration init (lazy, device-only) ───────────
    def _ensure_perception(self) -> None:
        if self._segmenter is None:
            from .perception.yolo_onnx import YoloOnnxSegmenter

            model_path = self.cfg.get("yolo_model_path")
            if not model_path:
                raise RuntimeError("grasp config missing yolo_model_path")
            names = list(self.cfg.get("yolo_classes", []))
            providers = self.cfg.get("onnx_providers")
            kwargs: dict[str, Any] = {}
            if providers:
                kwargs["providers"] = tuple(providers)
            self._segmenter = YoloOnnxSegmenter(model_path, names, **kwargs)
        if self._camera is None:
            from .perception.camera import make_camera

            cam_cfg = {"camera": dict(self.cfg.get("camera", {}))}
            calib_dir = self.cfg.get("calib_dir")
            self._camera = make_camera(cam_cfg, calib_dir=calib_dir)
            self._camera.open()
        if self._hand_eye is None:
            self._hand_eye = self._load_hand_eye()

    def _load_hand_eye(self) -> Optional[np.ndarray]:
        path = self.cfg.get("hand_eye_path")
        if not path:
            logger.warning("GraspPlugin: no hand_eye_path; grasp transform will fail")
            return None
        p = Path(path)
        if not p.exists():
            logger.warning("GraspPlugin: hand_eye_path %s not found", path)
            return None
        try:
            if p.suffix == ".npz":
                data = np.load(str(p))
                key = "T_hand_eye" if "T_hand_eye" in data else data.files[0]
                return np.asarray(data[key], dtype=np.float64)
            return np.asarray(np.load(str(p)), dtype=np.float64)
        except Exception:
            logger.exception("GraspPlugin: failed to load hand-eye from %s", path)
            return None

    def _grasp_params(self) -> dict:
        # Numeric grasp params often arrive as "${VAR:-default}" → an env-
        # substituted STRING (e.g. conf "0.15"), which would break the numpy
        # `c < conf` gate in predict and float maths downstream. Coerce floats
        # here so run_grasp_once always receives real numbers. Empty string →
        # treat as unset (drop the key, fall back to the function default).
        float_keys = {
            "conf", "iou", "depth_quantile", "pregrasp_offset_m",
            "insertion_depth_m", "lift_height_m", "grasp_force",
            "open_distance_m", "move_duration",
        }
        int_keys = {"warm_up_frames"}
        bool_keys = {"release_after"}
        out: dict[str, Any] = {}
        # Auto-search: pass the configured scan_poses so grasp_object sweeps to
        # find the object when it is not in the immediate view (same poses
        # search_object uses).
        sp = self.cfg.get("scan_poses")
        if sp:
            try:
                out["scan_poses"] = [tuple(float(v) for v in p) for p in sp]
            except (TypeError, ValueError):
                logger.warning("GraspPlugin: ignoring malformed scan_poses")
        for k in float_keys | int_keys | bool_keys:
            if k not in self.cfg:
                continue
            v = self.cfg[k]
            try:
                if k in bool_keys:
                    out[k] = v if isinstance(v, bool) else str(v).strip().lower() in {"1", "true", "yes"}
                elif k in int_keys:
                    s = str(v).strip()
                    if s:
                        out[k] = int(float(s))
                else:  # float
                    s = str(v).strip()
                    if s:
                        out[k] = float(s)
            except (TypeError, ValueError):
                logger.warning("GraspPlugin: ignoring non-numeric %s=%r", k, v)
        return out


__all__ = ["GraspPlugin"]
