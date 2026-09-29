"""
Enable/disable reporting shared by every Levy hook.

Nothing in here is a hook; the hooks live one-per-file alongside it (currently
only ``feature_gateway.py``).

License: TDG-Attribution-NonCommercial-NoDistrib
"""

# --------------------------------------------------------------------------- #
# Enable / disable reporting
#
# Every hook is built by a `build_*` function that returns None when its config
# block is absent or `enable: false`. Those builders call `announce` on the way
# out, so a run prints one greppable line per hook and the question "is this
# hook on?" is answered by `grep '\[LevyHook\]'` over the log rather than by
# reading the model code.
# --------------------------------------------------------------------------- #

_ACTIVE = []


def announce(hook_name, instance, where, detail=''):
    """
    Record and print that ``hook_name`` is live. Pass-through for ``instance``.

    Returns ``instance`` so a builder can end with ``return announce(...)``.
    ``None`` in means ``None`` out and nothing printed, which is the disabled
    path.
    """
    if instance is None:
        return None
    _ACTIVE.append({'hook': hook_name, 'where': where, 'detail': detail})
    print(f"[LevyHook] {hook_name:<14} ENABLED   {where}"
          + (f"   {detail}" if detail else ""))
    return instance


def note_disabled(hook_name, where, cfg):
    """
    Print a line when a config block exists but is switched off.

    A block that is present and disabled is worth one line -- it is the case
    where someone expects the hook to be running. A block that is absent
    entirely says nothing.
    """
    if cfg:
        print(f"[LevyHook] {hook_name:<14} disabled  {where}   "
              f"(block present, enable: false)")


def active_hooks():
    """Every hook announced in this process, in construction order."""
    return list(_ACTIVE)


def summary():
    """One-line-per-hook description of what is live. Empty string if nothing."""
    if not _ACTIVE:
        return "[LevyHook] no hooks enabled"
    return '\n'.join(
        f"[LevyHook] {h['hook']:<14} {h['where']}   {h['detail']}".rstrip()
        for h in _ACTIVE)


def reset_registry():
    """Forget what has been announced. For tests that build several models."""
    _ACTIVE.clear()
