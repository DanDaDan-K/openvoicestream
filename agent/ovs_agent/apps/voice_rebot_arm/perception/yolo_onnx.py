"""Torch-free YOLOE-seg segmenter on onnxruntime + numpy/cv2.

Replaces the ultralytics ``YOLO.predict`` path used by ``reBot-DevArm-Grasp``
with a pure onnxruntime + numpy/cv2 implementation that produces a
``YoloResult`` whose ``names`` / ``boxes`` / ``masks`` / ``orig_shape`` fields
mirror the subset of the ultralytics ``Results`` API consumed downstream
(``ordinary_grasp`` + ``yolo_utils``): ``result.names`` (dict),
``result.boxes[i].xyxy/.cls/.conf``, ``result.masks.data[i]`` (numpy HxW),
``result.orig_shape`` (h, w). No torch, no ``.cpu()``, no ultralytics.

ONNX export contract (validated, see _scratch/yolo-onnx-probe):
  * The fixed-vocabulary YOLOE-seg ONNX bakes both the open-vocab class set
    AND NMS into the graph, so we do NOT run NMS ourselves.
  * ``output0``: ``[1, 300, 4 + 1 + 1 + 32]`` rows of
    ``[x1, y1, x2, y2, conf, cls_id, *32 mask_coeffs]`` in 640-letterbox xyxy.
  * ``output1``: ``[1, 32, 160, 160]`` mask prototypes.

Mask assembly replicates ultralytics ``process_mask(upsample=True)``:
  coeffs @ protos -> 160x160 logits, crop by box*ratio in proto space,
  bilinear upsample to 640, threshold > 0 (== sigmoid > 0.5), THEN strip the
  letterbox padding and resize the *content region* to the original image
  (NEAREST). The padding strip is load-bearing: ``ultralytics`` ``masks.data``
  is already letterbox-cropped to the scaled content (e.g. 640x480, not
  640x640); skipping the strip systematically under-estimates mask area by
  ~25%.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import cv2
import numpy as np

# Default ONNX Runtime provider preference: TensorRT (Jetson) → CUDA → CPU.
# onnxruntime silently drops unavailable providers, so this list is safe on a
# CPU-only Mac (falls back to CPUExecutionProvider).
DEFAULT_PROVIDERS: tuple[str, ...] = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
)


# ── ultralytics-compatible result containers (numpy-only) ────────────────────
class _Box:
    """One detection's box, mirroring ``ultralytics.engine.results.Boxes[i]``.

    ``xyxy`` is a ``(1, 4)`` array so downstream ``box.xyxy[0]`` indexing works;
    ``cls`` / ``conf`` are length-1 arrays so ``int(box.cls[0])`` /
    ``float(box.conf[0])`` work unchanged. No ``.cpu()`` needed (already numpy).
    """

    __slots__ = ("xyxy", "cls", "conf")

    def __init__(self, xyxy: np.ndarray, cls_id: int, conf: float) -> None:
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(1, 4)
        self.cls = np.asarray([cls_id], dtype=np.float32)
        self.conf = np.asarray([conf], dtype=np.float32)


class _Boxes:
    """Indexable / len-able collection of :class:`_Box`."""

    __slots__ = ("_items",)

    def __init__(self, items: Sequence[_Box]) -> None:
        self._items = list(items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> _Box:
        return self._items[index]


class _Masks:
    """Holds ``data``: an ``(N, H, W)`` numpy mask stack (orig image space).

    Mirrors ``ultralytics`` ``result.masks.data`` which downstream code reads
    as ``result.masks.data[i]`` (per-instance HxW float/uint mask). Already
    numpy → ``tensor_to_numpy`` short-circuits, no ``.cpu()`` invoked.
    """

    __slots__ = ("data",)

    def __init__(self, data: np.ndarray) -> None:
        self.data = data


@dataclass
class YoloResult:
    """Minimal stand-in for ``ultralytics.engine.results.Results``.

    Exposes exactly the attributes the vendored grasp pipeline reads:
    ``names`` (id→label dict), ``boxes`` (:class:`_Boxes`), ``masks``
    (:class:`_Masks` or ``None``), ``orig_shape`` ``(h, w)``. ``obb`` is always
    ``None`` (YOLOE-seg has no oriented boxes) so the grasp code falls through
    to its mask/box path.
    """

    names: dict[int, str]
    boxes: _Boxes
    masks: Optional[_Masks]
    orig_shape: tuple[int, int]
    obb: None = None


def _letterbox(
    image_bgr: np.ndarray,
    new_shape: int,
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, float, float]:
    """Resize+pad to a square ``new_shape`` keeping aspect ratio.

    Returns ``(padded, ratio, dw, dh)`` where ``ratio`` is the uniform scale
    and ``dw`` / ``dh`` are the *total* horizontal / vertical padding (split
    evenly across both sides). Matches the ultralytics LetterBox transform
    used at export time.
    """
    h, w = image_bgr.shape[:2]
    ratio = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * ratio)), int(round(h * ratio)))
    dw = (new_shape - new_unpad[0]) / 2.0
    dh = (new_shape - new_unpad[1]) / 2.0
    resized = image_bgr
    if (w, h) != new_unpad:
        resized = cv2.resize(image_bgr, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return padded, ratio, dw, dh


class YoloOnnxSegmenter:
    """ONNX-Runtime YOLOE-seg segmenter with numpy/cv2 seg post-process.

    Args:
        model_path: path to the exported ``*-seg.onnx`` (vocab + NMS baked in).
        names: ordered class labels; index ``i`` is class id ``i``. MUST match
            the vocabulary order used at export time.
        input_size: ``(h, w)`` network input. The export is square so only the
            first element is used as the letterbox target.
        providers: onnxruntime execution providers, most-preferred first.
            Defaults to TensorRT → CUDA → CPU; unavailable providers are
            silently dropped by onnxruntime.

    The :class:`onnxruntime.InferenceSession` is created lazily on the first
    :meth:`predict` so constructing a segmenter is cheap and SDK-free (useful
    for tests that mock inference).
    """

    def __init__(
        self,
        model_path: str,
        names: list[str],
        input_size: tuple[int, int] = (640, 640),
        providers: Sequence[str] = DEFAULT_PROVIDERS,
    ) -> None:
        self.model_path = str(model_path)
        self.names: dict[int, str] = {i: str(n) for i, n in enumerate(names)}
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.providers = tuple(providers)
        self._session: Any = None
        self._input_name: Optional[str] = None

    # ── session lifecycle ────────────────────────────────────────────────
    def _ensure_session(self) -> None:
        if self._session is not None:
            return
        # Deferred import: onnxruntime is an optional/device dep. Keep it out
        # of module import so the package loads on hosts without the wheel.
        import onnxruntime as ort  # noqa: PLC0415

        self._session = ort.InferenceSession(
            self.model_path, providers=list(self.providers)
        )
        self._input_name = self._session.get_inputs()[0].name

    # ── inference ─────────────────────────────────────────────────────────
    def predict(
        self,
        image_bgr: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.45,  # noqa: ARG002 — NMS is baked into the graph
    ) -> list[YoloResult]:
        """Run inference on one BGR frame; return ``[YoloResult]`` (list for
        ultralytics API parity — always length 1).

        ``iou`` is accepted for signature compatibility but unused: the export
        bakes NMS in. We only apply the ``conf`` confidence gate on rows.
        """
        self._ensure_session()
        img0 = np.asarray(image_bgr)
        h0, w0 = img0.shape[:2]
        net = self.input_size[0]

        padded, ratio, dw, dh = _letterbox(img0, net)
        # BGR→RGB, HWC→CHW, scale, add batch dim.
        blob = np.ascontiguousarray(
            padded[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        )
        outs = self._session.run(None, {self._input_name: blob})
        return [self._postprocess(outs, conf, (h0, w0), ratio, dw, dh, net)]

    # ── numpy/cv2 seg post-process (validated against ultralytics) ─────────
    def _postprocess(
        self,
        outs: list[np.ndarray],
        conf: float,
        orig_shape: tuple[int, int],
        ratio: float,
        dw: float,
        dh: float,
        net: int,
    ) -> YoloResult:
        det = np.asarray(outs[0])[0]      # [300, 38]
        proto = np.asarray(outs[1])[0]    # [32, 160, 160]
        mh, mw = proto.shape[1:]
        proto_flat = proto.reshape(proto.shape[0], -1)
        h0, w0 = orig_shape

        # Letterbox padding edges in 640 space (stripped from each mask).
        top = int(round(dh - 0.1))
        left = int(round(dw - 0.1))
        bot = net - int(round(dh + 0.1))
        rgt = net - int(round(dw + 0.1))

        boxes: list[_Box] = []
        masks: list[np.ndarray] = []
        for row in det:
            c = float(row[4])
            # Keep detections AT the threshold (conf is an inclusive floor);
            # only drop strictly-below-threshold rows.
            if c < conf:
                continue
            box640 = row[:4].astype(np.float32)
            cls_id = int(round(float(row[5])))
            coeffs = row[6:]

            # box in original-image space (un-letterbox).
            box_orig = box640.copy()
            box_orig[[0, 2]] -= dw
            box_orig[[1, 3]] -= dh
            box_orig /= ratio
            box_orig[[0, 2]] = box_orig[[0, 2]].clip(0, w0)
            box_orig[[1, 3]] = box_orig[[1, 3]].clip(0, h0)

            # mask logits in proto (160) space, cropped to the box.
            m = (coeffs @ proto_flat).reshape(mh, mw)
            rx, ry = mw / net, mh / net
            cx1 = int(round(max(box640[0] * rx, 0)))
            cy1 = int(round(max(box640[1] * ry, 0)))
            cx2 = int(round(box640[2] * rx))
            cy2 = int(round(box640[3] * ry))
            cropped = np.zeros_like(m)
            cropped[cy1:cy2, cx1:cx2] = m[cy1:cy2, cx1:cx2]

            # upsample to 640, threshold (>0 == sigmoid>0.5), strip padding,
            # resize content to original (NEAREST) — matches ultralytics
            # masks.data consumer path.
            m640 = cv2.resize(cropped, (net, net), interpolation=cv2.INTER_LINEAR)
            bin640 = (m640 > 0.0).astype(np.float32)
            content = bin640[top:bot, left:rgt]
            m_orig = cv2.resize(content, (w0, h0), interpolation=cv2.INTER_NEAREST)

            boxes.append(_Box(box_orig, cls_id, c))
            masks.append((m_orig > 0.5).astype(np.float32))

        mask_obj: Optional[_Masks] = None
        if masks:
            mask_obj = _Masks(np.stack(masks, axis=0))
        return YoloResult(
            names=self.names,
            boxes=_Boxes(boxes),
            masks=mask_obj,
            orig_shape=(h0, w0),
        )


__all__ = ["YoloOnnxSegmenter", "YoloResult", "DEFAULT_PROVIDERS"]
