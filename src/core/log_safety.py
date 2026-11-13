"""Rendering a command for a log line so it cannot carry a credential.

Pod stdout is scraped into Loki, which RETAINS it. Anything a logger prints is
therefore published to a durable store that outlives the token, so the question
for an exec log line is not "is this string tidy" but "can this string ever
contain a secret".

The previous rendering was ``" ".join(command[:3])``. ``ContainerManager
.execute_command`` wraps a STRING command as ``["sh", "-c", command]``, so
element 2 of that slice is the ENTIRE shell script — including ``git_push``'s
``printf '%s' <token> > <file>``. The guard that was in place,
``_sanitize_shell_arg``, quotes the token for the SHELL; it was never a
redaction for the LOGGER. A guard aimed at the wrong consumer.

⭐ The design rule here is ALLOWLIST, not denylist. A denylist of known token
shapes (``ghp_``/``ghs_``/``github_pat_`` …) loses to every shape it has not met
— a GitLab PAT, a bearer JWT, a basic-auth password, a secret minted by
``resolve_git_token`` from a provider nobody has added a regex for. So
``summarize_command`` does not try to scrub the secret out of the command; it
emits a projection that keeps only words in COMMAND POSITION — the program that
runs, and a few sub-command/flag/path words after it — and drops every VALUE.
So ``cd /workspace && git push -u origin topic`` still reads as ``cd ; git push
-u origin`` — an operator can see what ran — while ``umask 077 && printf '%s'
'ghs_…' > /tmp/.git_cred_ab12`` collapses to ``umask ; printf ? /tmp/…``.

⚠️ THE ACTUAL GUARANTEE, STATED NARROWLY. An earlier draft of this docstring
said a secret is "STRUCTURALLY ABSENT". That is TRUE for value positions and
FALSE for command positions, and the difference is load-bearing:

  * ✅ A secret in a VALUE POSITION cannot appear. It is the argument to
    ``printf``, the userinfo in a URL, the payload of ``--token=``: every such
    word is shape-checked and rendered ``?``. This holds for shapes no denylist
    knows, which is the property ``redact_secrets`` alone cannot give.
  * ⛔ A secret that lands in a COMMAND POSITION *and* happens to match one of
    the accepted word shapes WILL be printed, unless ``redact_secrets`` knows
    it. Two demonstrated cases:
        ``summarize_command("git push; zzq-Wv7-novel-shape-0000")``
            → ``git push ; zzq-Wv7-novel-shape-0000`` — after ``;`` that word
              is a PROGRAM, and ``_PROGRAM_WORD`` admits it.
        ``summarize_command("sshpass -p hunterhunterhunter ssh")``
            → ``sshpass -p hunterhunterhunter ssh`` — an all-lowercase
              password is indistinguishable from a sub-command word.
    So the correct claim is: *a value-position secret cannot appear; a
    command-position token unknown to the denylist can.* Do not restate this as
    "cannot contain a credential" — the callers' comments depend on which half
    they are relying on.

``redact_secrets`` is the SECOND layer and is honestly best-effort: it only
knows the shapes listed in ``_SECRET_PATTERNS``. It exists to catch exactly the
command-position case above, and to clean free-form text (exception strings)
where no structural projection is possible. It is not, and must not become, the
thing the exec log lines rely on.

⛔ ReDoS IS A LIVE CONCERN IN THIS MODULE, not a theoretical one. ``git_push``
feeds raw exec **stderr** into ``_sanitize_error_message`` → ``redact_secrets``,
capped only by ``exec_max_output_bytes`` (default 1 MiB), on an async handler
that blocks the event loop for the duration. A first cut of the URL rule used
``[a-zA-Z][a-zA-Z0-9+.-]*://``; the unbounded ``*`` backtracks at EVERY start
offset of a long alphanumeric run — no ``://`` required anywhere in the input —
which measured 0.71s/2.86s/12.10s at 32/64/128 KiB (×4 per doubling, i.e.
quadratic) and ~12 minutes at the 1 MiB cap. Every quantifier here is now
BOUNDED, and ``redact_secrets`` additionally truncates its input. If you add a
pattern, add a timing case to ``TestRedactSecretsIsLinear`` with it.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Sequence

__all__ = ["redact_secrets", "summarize_command"]

_REDACTED = "[REDACTED]"

#: Known credential shapes. Best-effort, second layer only — see the module
#: docstring. The GitHub quantifiers are deliberately ``{20,255}`` rather than
#: the exactly-36 the API-response sanitiser used to carry: a ``ghs_``
#: installation token is not guaranteed to be 36 characters, and an off-by-one
#: length is not a reason to publish it.
_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{20,255}"), _REDACTED),
    (re.compile(r"github_pat_[A-Za-z0-9_]{22,}"), _REDACTED),
    (re.compile(r"glpat-[A-Za-z0-9_-]{16,}"), _REDACTED),
    (re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"), _REDACTED),
    # Credentials embedded in a URL's userinfo, on any scheme and any host.
    #
    # ⛔ BOTH quantifiers are bounded, and both bounds are load-bearing:
    #
    #   * ``{0,15}`` on the scheme tail. With ``*`` here, a run of N letters
    #     backtracks ~N times at each of N start offsets looking for ``://``
    #     that is not there — quadratic, measured 12.10s on 128 KiB of "A".
    #     Bounded, each start offset costs at most 16 probes, so the whole scan
    #     is linear. 16 characters covers every scheme this estate can produce
    #     (``https`` 5, ``git+ssh`` 7, the longest IANA-registered one is 14).
    #     A longer scheme is not silently mangled — it simply is not matched,
    #     which is the safe direction for an over-eager redactor and is why
    #     bounding beats the ``https?|git|ssh|ftp`` alternation the review also
    #     offered: this still generalises, it just stops backtracking.
    #   * ``{1,256}`` on the userinfo. Otherwise ``a://`` repeated N times is
    #     itself quadratic: every ``://`` starts a ``[^/\s@]+`` run that scans
    #     to end-of-input before failing to find ``@``. 256 is well past any
    #     real ``user:password``.
    (
        re.compile(r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]{0,15}://)[^/\s@]{1,256}@"),
        rf"\g<scheme>{_REDACTED}@",
    ),
)

#: Defence in depth behind the bounded quantifiers above. Linear is not the same
#: as free, and the carrier here is attacker-influenced: ``git_push`` hands raw
#: remote stderr to ``_sanitize_error_message``, bounded only by
#: ``exec_max_output_bytes`` (1 MiB default). Truncating is also strictly SAFER
#: than scrubbing — bytes past the cap are not published at all, rather than
#: published if no pattern happens to match them. The marker keeps that visible
#: instead of silently shortening an operator's error message.
_MAX_REDACT_CHARS = 16 * 1024

#: A word may be reported as a PROGRAM. Conservative: no ``:``, no ``@``, no
#: ``=``, no quote characters, bounded length. A quoted or URL-shaped word fails
#: this and is reported as ``?``.
_PROGRAM_WORD = re.compile(r"\A[A-Za-z0-9_][A-Za-z0-9_.+/-]{0,31}\Z")

#: A word may be reported as a SUB-COMMAND or FLAG after the program. Lowercase
#: letters and hyphens only (plus a leading ``-``/``--``): no digits, no
#: underscore, no dot, no separator. Every credential shape this service handles
#: — ``ghs_…``, ``github_pat_…``, a base64/JWT blob, a URL, a hex id — carries at
#: least one character outside that class, so a secret cannot be mistaken for a
#: sub-command word.
_CONTINUATION_WORD = re.compile(r"\A--?[a-z][a-z-]{0,20}\Z|\A[a-z][a-z-]{0,20}\Z")

#: ⭐ A word may ALSO be reported if it is PATH-SHAPED. Without this the
#: projection was safe but close to useless at the sites that matter:
#: ``ruff format -- a.py b.py`` lost the FILES, ``python -m pytest tests/x.py``
#: lost the TEST PATH. Paths are what an operator actually needs to reproduce
#: an exec, and a path is not a value in the credential sense.
#:
#: ⚠️ This WIDENS what can be printed, so it is fenced by three independent
#: barriers, each of which alone rejects the realistic carrier — the base64 blob
#: from ``container_monitoring._create_file``, which decodes to a live
#: live github.com URL carrying ``x-access-token:ghs_…`` as its userinfo:
#:   1. it must contain ``/`` (or be a plain ``name.ext``). Every credential
#:      shape this service handles — ``ghs_``/``ghp_``/``github_pat_``/``glpat-``,
#:      a JWT, a hex key — contains no ``/`` at all;
#:   2. total length ≤ 64. The measured base64-of-a-token-URL is 104 chars;
#:   3. no unbroken run of ≥ 20 alphanumerics. Path segments are words; a
#:      base64/hex blob is one long run. This is what stops a SHORT blob that
#:      cleared (1) and (2).
#: A word must pass ALL of them, so a secret has to be simultaneously
#: slash-bearing, short, and word-shaped to survive. The residual is stated in
#: the module docstring; it is not claimed to be zero.
_PATH_SEGMENT_WORD = re.compile(r"\A[A-Za-z0-9_.-]{0,32}\Z")
_FILENAME_WORD = re.compile(r"\A[A-Za-z0-9_-]{1,32}\.[A-Za-z0-9]{1,8}\Z")
_LONG_ALNUM_RUN = re.compile(r"[A-Za-z0-9]{20,}")
_MAX_PATH_WORD_CHARS = 64

#: ``--`` carries no data whatever — it is the end-of-options separator. It is
#: emitted and does NOT consume the word budget, because spending one of three
#: slots on it is what made ``ruff format -- a.py b.py`` render as ``ruff
#: format`` with the files already out of reach.
_OPTION_TERMINATOR = "--"

#: Rendered in place of a word that failed every shape check. ⛔ NOT omitted
#: silently: ``git -c credential.helper= push`` must not render as ``git -c
#: push``, which reads as a command that was never run. A visible ``?`` keeps
#: the log line honest about the gap.
_OPAQUE = "?"

#: Shell punctuation that begins a new command position. Applied textually,
#: without honouring quoting — that errs toward MORE segments, i.e. toward
#: reading more words as programs, each of which is then shape-checked.
_COMMAND_BOUNDARY = re.compile(r"\|\||&&|\$\(|[;|&\n()`{}]")

#: Shell wrappers whose ``-c`` argument is a script, not a value. Unwrapped
#: rather than printed as ``sh -c`` and stopped: every command this service
#: builds arrives at the provisioner in exactly that shape, so refusing to look
#: inside would make the fix indistinguishable from logging nothing, and the
#: next operator who needs to debug an exec would revert it.
_SHELL_PROGRAMS = frozenset({"sh", "bash", "dash", "ash", "zsh"})

#: Per segment, and overall. A log line is not a transcript.
#:
#: ``_MAX_CONTINUATION_WORDS`` bounds what is EMITTED; ``_MAX_SCANNED_WORDS``
#: bounds what is EXAMINED. They are separate because the old code stopped dead
#: at the first word it could not render, which is how ``cd /w && git -c
#: credential.helper= push`` lost the verb ``push`` — the single most important
#: word in the line — to an opaque config argument two positions earlier.
#: Scanning past a rejected word cannot print that word (it is still ``?``); it
#: only lets a LATER word be considered, under the same shape checks.
_MAX_CONTINUATION_WORDS = 4
_MAX_SCANNED_WORDS = 12
_MAX_SEGMENTS = 8


def redact_secrets(text: str) -> str:
    """Replace known credential shapes in free-form text.

    Best-effort by construction: it can only remove shapes it has been taught.
    Use it on text that has no structure to project (an exception message), or
    as a second layer behind :func:`summarize_command` — never as the only
    thing standing between a token and a retained log store.

    Input longer than :data:`_MAX_REDACT_CHARS` is truncated BEFORE matching,
    with a marker, so neither the regex cost nor the published volume scales
    with a remote's stderr. See the ReDoS note in the module docstring.
    """
    if not text:
        return text
    if len(text) > _MAX_REDACT_CHARS:
        omitted = len(text) - _MAX_REDACT_CHARS
        text = f"{text[:_MAX_REDACT_CHARS]}… [{omitted} more characters omitted]"
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _words(segment: str) -> list[str]:
    """Split one segment into words, quote-aware where the quoting is valid.

    ``shlex.split`` raises on an unbalanced quote, which a textual segmentation
    of a real shell script produces routinely (the ``credential.helper '!f() {
    … }; f'`` line splits mid-quote). Falling back to whitespace splitting keeps
    the quote characters attached to the word, where the shape checks reject
    them — the failure mode is "reported as ``?``", never "reported raw".
    """
    try:
        return shlex.split(segment)
    except ValueError:
        return segment.split()


def _is_path_word(word: str) -> bool:
    """Is ``word`` safe to print as a PATH rather than dropped as a value?

    Three independent barriers — length, a required ``/`` (or a plain
    ``name.ext``), and no long unbroken alphanumeric run. See the comment on
    :data:`_PATH_SEGMENT_WORD` for why each one is there and what it stops.
    The length check is first so the run scan never sees an unbounded word.
    """
    if not word or len(word) > _MAX_PATH_WORD_CHARS:
        return False
    if _LONG_ALNUM_RUN.search(word):
        return False
    if "/" not in word:
        return bool(_FILENAME_WORD.match(word))
    return all(_PATH_SEGMENT_WORD.match(segment) for segment in word.split("/"))


def _render_words(words: Sequence[str]) -> str | None:
    """One command position: the program, then a few sub-command/flag/path words.

    Words that fail every shape check collapse to a single ``?`` rather than
    ending the scan, so a verb that sits behind an opaque argument survives.
    """
    if not words:
        return None
    program = words[0] if _PROGRAM_WORD.match(words[0]) else _OPAQUE
    kept = [program]
    emitted = 0
    for word in words[1 : 1 + _MAX_SCANNED_WORDS]:
        if emitted >= _MAX_CONTINUATION_WORDS:
            break
        if word == _OPTION_TERMINATOR:
            kept.append(_OPTION_TERMINATOR)
            continue
        if _CONTINUATION_WORD.match(word) or _is_path_word(word):
            kept.append(word)
            emitted += 1
            continue
        # Opaque. Collapse a run of them so a long value list does not become a
        # row of question marks.
        if kept[-1] != _OPAQUE:
            kept.append(_OPAQUE)
    return " ".join(kept)


def _render_script(script: str) -> list[str]:
    """Project a shell string to one rendering per command position."""
    segments = _COMMAND_BOUNDARY.split(script)
    return [rendered for rendered in (_render_words(_words(s)) for s in segments) if rendered]


def _basename(word: str) -> str:
    return word.rsplit("/", 1)[-1]


def summarize_command(command: str | Sequence[str] | None) -> str:
    """Render ``command`` as programs and sub-commands only, never as values.

    Accepts either shape the exec surface uses: a shell string (what
    ``execute_command`` is handed) or an argv list (what it forwards, and what
    ``exec_in_pod`` always receives). Both are projected the same way, because a
    list argv is not inherently safe either — ``["git", "push",
    "https://<token>@github.com/o/r"]`` puts a credential in element 2, which
    the old ``command[:3]`` slice printed in full.

    The result always carries a size so an operator can tell a short command
    from a long one, and so "the summary is empty" stays distinguishable from
    "nothing was logged at all".
    """
    if command is None:
        return "<none>"

    if isinstance(command, str):
        rendered = _render_script(command)
        size = f"len={len(command)}"
    else:
        argv = [str(word) for word in command]
        size = f"argc={len(argv)}"
        if len(argv) >= 3 and _basename(argv[0]) in _SHELL_PROGRAMS and argv[1] == "-c":
            rendered = _render_script(argv[2])
        else:
            rendered = [r for r in (_render_words(argv),) if r]

    if not rendered:
        return f"<empty> ({size})"

    truncated = len(rendered) > _MAX_SEGMENTS
    body = " ; ".join(rendered[:_MAX_SEGMENTS])
    if truncated:
        body += " ; ..."
    # Second layer. See the module docstring: this is belt-and-braces over an
    # already-structural projection, not the projection itself.
    return f"{redact_secrets(body)} ({size})"
