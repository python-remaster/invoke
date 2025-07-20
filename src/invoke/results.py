from __future__ import annotations

from types import TracebackType
from typing import Optional

# Import some platform-specific things at top level so they can be mocked for
# tests.
try:
    import pty
except ImportError:
    pty = None  # type: ignore[assignment]

from .util import default_encoding


class Result:
    """
    A container for information about the result of a command execution.

    All params are exposed as attributes of the same name and type.

    :param str stdout:
        The subprocess' standard output.

    :param str stderr:
        Same as ``stdout`` but containing standard error (unless the process
        was invoked via a pty, in which case it will be empty; see
        `.Runner.run`.)

    :param str encoding:
        The string encoding used by the local shell environment.

    :param str command:
        The command which was executed.

    :param str shell:
        The shell binary used for execution.

    :param dict env:
        The shell environment used for execution. (Default is the empty dict,
        ``{}``, not ``None`` as displayed in the signature.)

    :param int exited:
        An integer representing the subprocess' exit/return code.

        .. note::
            This may be ``None`` in situations where the subprocess did not run
            to completion, such as when auto-responding failed or a timeout was
            reached.

    :param bool pty:
        A boolean describing whether the subprocess was invoked with a pty or
        not; see `.Runner.run`.

    :param tuple hide:
        A tuple of stream names (none, one or both of ``('stdout', 'stderr')``)
        which were hidden from the user when the generating command executed;
        this is a normalized value derived from the ``hide`` parameter of
        `.Runner.run`.

        For example, ``run('command', hide='stdout')`` will yield a `Result`
        where ``result.hide == ('stdout',)``; ``hide=True`` or ``hide='both'``
        results in ``result.hide == ('stdout', 'stderr')``; and ``hide=False``
        (the default) generates ``result.hide == ()`` (the empty tuple.)

    .. note::
        `Result` objects' truth evaluation is equivalent to their `.ok`
        attribute's value. Therefore, quick-and-dirty expressions like the
        following are possible::

            if run("some shell command"):
                do_something()
            else:
                handle_problem()

        However, remember `Zen of Python #2
        <http://zen-of-python.info/explicit-is-better-than-implicit.html#2>`_.

    .. versionadded:: 1.0
    """

    # TODO: inherit from namedtuple instead? heh (or: use attrs from pypi)
    def __init__(
        self,
        stdout: str = "",
        stderr: str = "",
        encoding: Optional[str] = None,
        command: str = "",
        shell: str = "",
        env: Optional[dict] = None,
        exited: int = 0,
        pty: bool = False,
        hide: tuple[str, ...] = tuple(),
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        if encoding is None:
            encoding = default_encoding()
        self.encoding = encoding
        self.command = command
        self.shell = shell
        self.env = {} if env is None else env
        self.exited = exited
        self.pty = pty
        self.hide = hide

    @property
    def return_code(self) -> int:
        """
        An alias for ``.exited``.

        .. versionadded:: 1.0
        """
        return self.exited

    def __bool__(self) -> bool:
        return self.ok

    def __int__(self) -> int:
        return self.exited

    def __str__(self) -> str:
        if self.exited is not None:
            desc = f"Command exited with status {self.exited}."
        else:
            desc = "Command was not fully executed due to watcher error."
        ret = [desc]
        for x in ("stdout", "stderr"):
            val = getattr(self, x)
            ret.append(
                f"=== {x} ===\n{val.rstrip()}\n" if val else f"(no {x})"
            )
        return "\n".join(ret)

    def __repr__(self) -> str:
        # TODO: more? e.g. len of stdout/err? (how to represent cleanly in a
        # 'x=y' format like this? e.g. '4b' is ambiguous as to what it
        # represents
        return f"<Result cmd={self.command!r} exited={self.exited}>"

    @property
    def ok(self) -> bool:
        """
        A boolean equivalent to ``exited == 0``.

        .. versionadded:: 1.0
        """
        return bool(self.exited == 0)

    @property
    def failed(self) -> bool:
        """
        The inverse of ``ok``.

        I.e., ``True`` if the program exited with a nonzero return code, and
        ``False`` otherwise.

        .. versionadded:: 1.0
        """
        return not self.ok

    def tail(self, stream: str, count: int = 10) -> str:
        """
        Return the last ``count`` lines of ``stream``, plus leading whitespace.

        :param str stream:
            Name of some captured stream attribute, eg ``"stdout"``.
        :param int count:
            Number of lines to preserve.

        .. versionadded:: 1.3
        """
        # TODO: preserve alternate line endings? Mehhhh
        # NOTE: no trailing \n preservation; easier for below display if
        # normalized
        return "\n\n" + "\n".join(getattr(self, stream).splitlines()[-count:])


class Promise(Result):
    """
    A promise of some future `Result`, yielded from asynchronous execution.

    This class' primary API member is `join`; instances may also be used as
    context managers, which will automatically call `join` when the block
    exits. In such cases, the context manager yields ``self``.

    `Promise` also exposes copies of many `Result` attributes, specifically
    those that derive from `~Runner.run` kwargs and not the result of command
    execution. For example, ``command`` is replicated here, but ``stdout`` is
    not.

    .. versionadded:: 1.4
    """

    def __init__(self, runner: "Runner") -> None:
        """
        Create a new promise.

        :param runner:
            An in-flight `Runner` instance making this promise.

            Must already have started the subprocess and spun up IO threads.
        """
        self.runner = runner
        # Basically just want exactly this (recently refactored) kwargs dict.
        # TODO: consider proxying vs copying, but prob wait for refactor
        for key, value in self.runner.result_kwargs.items():
            setattr(self, key, value)

    def join(self) -> Result:
        """
        Block until associated subprocess exits, returning/raising the result.

        This acts identically to the end of a synchronously executed ``run``,
        namely that:

        - various background threads (such as IO workers) are themselves
          joined;
        - if the subprocess exited normally, a `Result` is returned;
        - in any other case (unforeseen exceptions, IO sub-thread
          `.ThreadException`, `.Failure`, `.WatcherError`) the relevant
          exception is raised here.

        See `~Runner.run` docs, or those of the relevant classes, for further
        details.
        """
        try:
            return self.runner._finish()
        finally:
            self.runner.stop()

    def __enter__(self) -> "Promise":
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.join()
