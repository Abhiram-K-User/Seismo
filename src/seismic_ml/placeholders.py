"""Convention for phase-gated placeholders.

Every interface that is intentionally not implemented yet raises
:class:`NotYetImplementedError`, which records the component name and the
development phase in which it is planned (see PROJECT_CONTEXT.md §14).

To list everything that remains to be implemented::

    grep -rn "NotYetImplementedError(" src/

When a phase implements a component, remove its ``raise`` and replace the
corresponding placeholder test with real functional tests.
"""

from __future__ import annotations


class NotYetImplementedError(NotImplementedError):
    """Raised by interfaces whose implementation belongs to a later phase.

    Args:
        component: Qualified name of the unimplemented component,
            e.g. ``"GAT.forward"``.
        phase: Development phase in which the component is planned.
    """

    def __init__(self, component: str, phase: int) -> None:
        self.component = component
        self.phase = phase
        super().__init__(
            f"{component} is not implemented yet (planned for Phase {phase}). "
            "See PROJECT_CONTEXT.md before implementing it."
        )
