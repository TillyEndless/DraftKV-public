"""Fatal unwind policy for an isolated CLI worker; safe to import without CUDA."""

from contextlib import contextmanager
from contextvars import ContextVar

from .acceptance import AcceptanceStop


_isolated_cli_worker = ContextVar("draftkv_isolated_cli_worker", default=False)
_fatal_resource_roots = ContextVar("draftkv_fatal_resource_roots", default=None)


@contextmanager
def isolated_cli_worker():
    """Limit abandoned transactions to a process whose CLI will exit hard."""
    token = _isolated_cli_worker.set(True)
    roots_token = _fatal_resource_roots.set([])
    try:
        yield
    finally:
        _fatal_resource_roots.reset(roots_token)
        _isolated_cli_worker.reset(token)


def bypass_failed_transaction_rollback(error):
    """Allow the original fatal error to reach synchronous CLI reporting.

    Ordinary callers still fence asynchronous work before dropping references.
    Bounded acceptance deliberately rolls back a partial forward, even in CLI.
    """
    return (_isolated_cli_worker.get()
            and not isinstance(error, AcceptanceStop)
            and not (isinstance(error, SystemExit) and error.code in (None, 0)))


def retain_fatal_failure(error, *resources):
    """Root abandoned resources and their traceback until the CLI's hard exit.

    A failing transfer may already have popped an item from its pending queue.
    Retaining the original traceback also keeps those in-flight frame locals.
    Normal callers and bounded acceptance do not abandon or retain anything.
    """
    if not bypass_failed_transaction_rollback(error):
        return False
    _fatal_resource_roots.get().extend((error, *resources))
    return True
