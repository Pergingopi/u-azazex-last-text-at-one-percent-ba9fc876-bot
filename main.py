#!/usr/bin/env python3
"""Last Text at One Percent -- write the final text to ``entry.txt``.

The program writes exactly one file, ``<output-dir>/entry.txt``, containing
the phone's last text message: a single line, 118 characters, no trailing
newline. It is deliberately boring infrastructure around one joke: every
step exists so that a bad write can never silently pass as a valid entry.

Run with Python 3.13, standard library only, no network, no randomness.
The output is deterministic -- the same bytes on every run, for every user.

Pipeline, in order:

1. :func:`validate_text` checks the text against the gate (UTF-8 text,
   1 to 200 characters, not only whitespace, no trailing newline, no
   control characters) and raises :class:`EntryValidationError` on any
   violation instead of asserting, so a bad entry fails loudly even
   with ``python -O``.
2. :func:`resolve_output_dir` picks the output directory. An empty or
   unset ``MAKEFUN_OUTPUT_DIR`` falls back to ``/output``; a set value
   is used exactly as given and *created* with ``os.makedirs`` when it
   does not exist yet -- a missing directory is never an error and
   never a reason to silently fall back to ``/output``.
3. :func:`cleanup_stale_temps` removes our own leftover temp files
   (``.entry-*.tmp``) from earlier runs, retrying transient failures
   instead of giving up on the first error, and never touching the
   output directory itself or any unrelated file.
4. :func:`write_entry_atomically` writes via a temp file in the same
   directory plus :func:`os.replace`, with ``chmod 0644`` before and
   after the replace and an ``fsync`` of both the file and the
   directory, so a crash mid-write leaves either the old file or the
   complete new one, never a half-written entry.
5. :func:`verify_entry` reads the file back from disk and checks the
   bytes, the decoded text, and the file mode (0644) against what was
   validated, raising :class:`RuntimeError` on any mismatch.

Nothing here depends on the current working directory, the invoking
user, or the environment beyond ``MAKEFUN_OUTPUT_DIR``.
"""

import os
import tempfile
import time

TEXT = "1%. Deleting your search history before I die \u2014 you're welcome. 9 hours of screen time today. You did this to me. Bye."

DEFAULT_OUTPUT_DIR = "/output"
ENTRY_FILENAME = "entry.txt"
TEMP_PREFIX = ".entry-"
TEMP_SUFFIX = ".tmp"
FILE_MODE = 0o644
MIN_CHARS = 1
MAX_CHARS = 200
STALE_CLEANUP_ATTEMPTS = 3


class EntryValidationError(ValueError):
    """The entry text (or the bytes read back from disk) breaks the gate."""


def validate_text(text):
    """Validate one entry string against the gate; return it unchanged.

    Checks, in gate order: it is a ``str``; its length is 1 to 200
    characters; it is not only whitespace; it does not end with a
    trailing newline; it contains no control characters (anything
    below U+0020, plus U+007F DEL -- so no ``\\n``, ``\\r`` or tab
    anywhere); and it encodes as UTF-8. Every failure raises
    :class:`EntryValidationError` with a message naming the rule and
    the offending value, so the cause is visible without re-reading
    the gate.

    Raises:
        EntryValidationError: If any gate rule is violated.
    """
    if not isinstance(text, str):
        raise EntryValidationError(
            f"entry must be a str, got {type(text).__name__}"
        )
    if not MIN_CHARS <= len(text) <= MAX_CHARS:
        raise EntryValidationError(
            f"entry must be {MIN_CHARS}-{MAX_CHARS} characters, "
            f"got {len(text)}"
        )
    if not text.strip():
        raise EntryValidationError("entry must not be only whitespace")
    if text.endswith("\n"):
        raise EntryValidationError(
            "entry must not end with a trailing newline"
        )
    bad = [c for c in text if ord(c) < 0x20 or ord(c) == 0x7F]
    if bad:
        raise EntryValidationError(
            "entry must not contain control characters, "
            f"found {sorted({hex(ord(c)) for c in bad})}"
        )
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise EntryValidationError(f"entry must be UTF-8 text: {exc}") from exc
    return text


def validate_bytes(data):
    """Decode ``data`` as strict UTF-8 and validate it; return the text.

    This is the read-back half of the gate: bytes on disk must decode
    cleanly and then satisfy exactly the same rules as the text we set
    out to write. A decoding failure is a validation error, not a
    crash with a bare ``UnicodeDecodeError``.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise EntryValidationError(
            f"entry bytes must be bytes, got {type(data).__name__}"
        )
    try:
        text = bytes(data).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EntryValidationError(
            f"entry on disk must be UTF-8 text: {exc}"
        ) from exc
    return validate_text(text)


def resolve_output_dir():
    """Return the output directory, creating it when it does not exist.

    ``MAKEFUN_OUTPUT_DIR`` wins when it names a real path: surrounding
    whitespace is stripped, and a value that is empty or only whitespace
    is treated exactly like an unset variable, falling back to
    ``/output``. A non-empty value is used as given -- including a path
    that does not exist yet, which is created with
    ``os.makedirs(..., exist_ok=True)``. A missing directory is never
    an error and never a reason to silently fall back to ``/output``;
    only a genuine creation failure (for example a file sitting where
    the directory should be) raises :class:`RuntimeError`.
    """
    raw = os.environ.get("MAKEFUN_OUTPUT_DIR")
    if raw is None or not raw.strip():
        chosen = DEFAULT_OUTPUT_DIR
    else:
        chosen = raw.strip()
    try:
        os.makedirs(chosen, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"cannot create output directory {chosen!r}: {exc}"
        ) from exc
    if not os.path.isdir(chosen):
        raise RuntimeError(
            f"output path {chosen!r} exists but is not a directory"
        )
    return chosen


def _fsync_dir(path):
    """Flush directory ``path`` to disk so a rename inside it is durable."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _remove_with_retry(path, attempts=STALE_CLEANUP_ATTEMPTS):
    """Remove one stale temp file, retrying transient failures.

    Returns ``True`` when the file is gone (or was already gone) and
    ``False`` when it could not be removed after ``attempts`` tries.
    Cleanup is best-effort -- a file another process is briefly
    holding must not fail the whole run -- so a persistent failure is
    reported by the return value, never raised.
    """
    for attempt in range(attempts):
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
            return True
        except FileNotFoundError:
            return True
        except OSError:
            if attempt == attempts - 1:
                return False
            time.sleep(0.01 * (attempt + 1))
    return False


def cleanup_stale_temps(out_dir):
    """Remove our own leftover temp files from earlier runs.

    Only files in ``out_dir`` whose names start with ``.entry-`` and
    end with ``.tmp`` are candidates -- our :func:`tempfile.mkstemp`
    prefix and suffix, nothing else. The output directory itself and
    every unrelated file are left alone, and a file that resists
    removal after retries is skipped rather than aborting the run.
    Returns the number of stale files removed.
    """
    removed = 0
    try:
        names = os.listdir(out_dir)
    except OSError:
        return 0
    for name in names:
        if name.startswith(TEMP_PREFIX) and name.endswith(TEMP_SUFFIX):
            if _remove_with_retry(os.path.join(out_dir, name)):
                removed += 1
    return removed


def write_entry_atomically(out_dir, data):
    """Write ``data`` to ``entry.txt`` in ``out_dir`` atomically.

    The bytes go to a fresh temp file in the same directory (so the
    rename never crosses filesystems), are flushed and fsynced, get
    mode 0644 before the rename, replace ``entry.txt`` via
    :func:`os.replace`, get mode 0644 again after the rename (a
    replace can expose the target with the temp file's mode on some
    filesystems, so both chmods are load-bearing), and the directory
    itself is fsynced so the rename is durable. If anything fails
    before the replace, the temp file is unlinked and the error is
    re-raised; the previous ``entry.txt``, if any, is untouched.
    Returns the full path of the written file.
    """
    target = os.path.join(out_dir, ENTRY_FILENAME)
    fd, tmp = tempfile.mkstemp(
        prefix=TEMP_PREFIX, suffix=TEMP_SUFFIX, dir=out_dir
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, FILE_MODE)  # chmod BEFORE os.replace
        os.replace(tmp, target)
        os.chmod(target, FILE_MODE)  # chmod AFTER os.replace
        _fsync_dir(out_dir)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    return target


def verify_entry(target, expected_data, expected_text):
    """Read ``target`` back and check it is exactly what was validated.

    The bytes on disk must equal ``expected_data`` byte for byte, the
    decoded text must pass :func:`validate_text` and equal
    ``expected_text``, and the file mode must be exactly 0644. Any
    mismatch raises :class:`RuntimeError` -- a bad write can never
    silently pass as a valid entry.
    """
    with open(target, "rb") as f:
        written = f.read()
    if written != expected_data:
        raise RuntimeError(
            "read-back mismatch: entry.txt differs from the validated text"
        )
    if validate_bytes(written) != expected_text:
        raise RuntimeError(
            "read-back mismatch: entry.txt decodes to different text"
        )
    mode = os.stat(target).st_mode & 0o777
    if mode != FILE_MODE:
        raise RuntimeError(
            f"entry.txt must be mode 0644, got {oct(mode)}"
        )


def main():
    """Validate, resolve, write and verify the entry. Returns exit code 0."""
    text = validate_text(TEXT)
    out_dir = resolve_output_dir()
    data = text.encode("utf-8")
    cleanup_stale_temps(out_dir)
    target = write_entry_atomically(out_dir, data)
    verify_entry(target, data, text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
