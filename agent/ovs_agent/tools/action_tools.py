"""arm_tools — register one ovs-agent @tool per action in actions.yaml.

Each action becomes a no-argument tool whose description is the
``description:`` field from actions.yaml. The LLM picks an action by
function-calling; the tool dispatches into ArmPlugin.dispatch_action,
which returns as soon as the serial bus has accepted the first frame
(~200ms) — the remaining frames continue on a worker thread while the
server-loop template response is spoken.

Response mode model (framework-level, see ovs-agent registry):
  * ``template`` (DEFAULT for arm actions): tool body returns fast,
    runner SKIPS LLM round 2 and synthesises a fixed ``completion_text``
    instead. This keeps the LLM responsible for semantic tool selection
    while preventing duplicate acknowledgements after the tool result.
  * ``parallel``: tool body returns fast after dispatch, LLM round 2 runs
    in parallel with the physical motion.
  * ``await``: tool body blocks until the motion completes, then LLM
    round 2 runs. Legacy behaviour — only useful if the LLM should
    reason over the post-motion observation cache.

actions.yaml schema (additions on top of the existing description):
  sequences:
    <name>:
      description: "..."
      response_mode: template | parallel | await   # optional
      completion_text: "好的。"                   # optional, used by template

Why closures-with-default-args: ``for entry in actions: def _tool(): ...``
captures ``entry`` by reference, so EVERY registered tool would dispatch
the last action. Wrapping in ``_make(name=...)`` binds the name at
closure-creation time, which is the canonical Python idiom for this.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Default response mode for arm actions when actions.yaml is silent. We keep
# the LLM in the loop for choosing the tool, then use a fixed template response
# after the fast dispatch result so voxedge does not run a second LLM round that
# can repeat "好的。".
_DEFAULT_ARM_RESPONSE_MODE = "template"
_DEFAULT_ARM_COMPLETION_TEXT = "好的。"


def register_arm_tools(
    registry,
    arm_plugin: Any,
    actions: list[dict],
    timeout_s: float = 15.0,
    disabled_actions: set[str] | None = None,
) -> int:
    """Register one tool per action; return count registered.

    Each entry in ``actions`` is the dict from
    ``ActionsManager.list_with_descriptions()`` — at minimum
    ``{"name", "description"}``, optionally with ``"response_mode"``
    and ``"completion_text"`` overrides parsed from actions.yaml.
    """
    count = 0
    disabled = set(disabled_actions or set())
    for entry in actions:
        action_name = entry.get("name")
        if not isinstance(action_name, str) or not action_name:
            continue
        if action_name in disabled:
            logger.info("skipping disabled arm tool name=%r", action_name)
            continue
        description = (entry.get("description") or "").strip() or (
            f"Execute the pre-recorded arm motion named {action_name!r}."
        )
        # Per-action response mode override, else "template" (arm default).
        rmode = entry.get("response_mode") or _DEFAULT_ARM_RESPONSE_MODE
        # completion_text only meaningful in template mode, but harmless to
        # thread through for parallel too (registry stores it; the runner just
        # ignores it when not in template mode).
        ctext = entry.get("completion_text") or (
            _DEFAULT_ARM_COMPLETION_TEXT if rmode == "template" else ""
        )

        def _make(
            name: str = action_name,
            desc: str = description,
            mode: str = rmode,
            comp: str = ctext,
        ) -> None:
            if mode == "await":
                async def _tool() -> dict:
                    # Block-until-done semantics: the LLM will see the
                    # final {"success": bool, "action": name} dict and
                    # compose its reply with full knowledge of outcome.
                    return await arm_plugin.execute_action(name)
            else:
                # parallel + template both want fast return.
                async def _tool() -> dict:  # type: ignore[no-redef]
                    # Returns ~200ms after the serial bus accepts frame
                    # #1. The rest of the motion continues on a worker
                    # thread inside ArmPlugin; LLM round 2 / template
                    # reply overlap that worker.
                    return await arm_plugin.dispatch_action(name)

            _tool.__name__ = name
            _tool.__doc__ = desc
            # Do not use ``preamble_text="好的。"`` here. The server-loop
            # engine also speaks after successful tool dispatch; preamble +
            # LLM/template response is how one voice command produced two
            # separate "好的。" TTS sentences.
            registry.tool(
                name=name,
                description=desc,
                timeout_s=timeout_s,
                preamble_text="",
                completion_text=comp,
                response_mode=mode,
            )(_tool)

        _make()
        count += 1
        logger.info(
            "registered arm tool name=%r response_mode=%r completion_text=%r",
            action_name, rmode, ctext,
        )
    logger.info("registered %d arm tools", count)
    return count


__all__ = ["register_arm_tools"]
