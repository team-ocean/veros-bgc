"""Stable-descriptor and cooperative-lock primitives for protocol artifacts."""

import errno
import fcntl
import hashlib
import os
from pathlib import Path
import stat


class ArtifactTrustError(RuntimeError):
    pass


def _below_root(path, root):
    return path != root and path.is_relative_to(root)


def _hash_descriptor(descriptor):
    digest = hashlib.sha256()
    offset = 0
    while True:
        chunk = os.pread(descriptor, 1024 * 1024, offset)
        if not chunk:
            break
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


class StableArtifact:
    def __init__(self, path, root, label, expected_hash=None):
        self.path = Path(path)
        self.root = Path(root)
        self.label = label
        self.expected_hash = expected_hash
        self.descriptor = None
        self.identity = None
        self.sha256 = None

    def __enter__(self):
        if not self.path.is_absolute():
            raise ArtifactTrustError("{} path is not absolute".format(self.label))
        absolute = Path(os.path.abspath(str(self.path)))
        if absolute != self.path or not _below_root(self.path, self.root):
            raise ArtifactTrustError("{} path is outside the canonical output root".format(self.label))
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            self.descriptor = os.open(str(self.path), flags)
        except OSError as error:
            if error.errno in (errno.ELOOP, errno.ENOENT):
                raise ArtifactTrustError("{} is missing or is a symlink".format(self.label)) from error
            raise ArtifactTrustError("could not open {}: {}".format(self.label, error)) from error
        try:
            info = os.fstat(self.descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise ArtifactTrustError("{} is not a regular file".format(self.label))
            resolved = Path(os.readlink("/proc/self/fd/{}".format(self.descriptor))).resolve()
            if resolved != absolute or not _below_root(resolved, self.root):
                raise ArtifactTrustError("{} resolved outside the canonical output root".format(self.label))
            self.identity = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
            self.sha256 = _hash_descriptor(self.descriptor)
            if self.expected_hash is not None and self.sha256 != self.expected_hash:
                raise ArtifactTrustError("{} SHA-256 does not match committed evidence".format(self.label))
            return self
        except BaseException:
            os.close(self.descriptor)
            self.descriptor = None
            raise

    @property
    def proc_path(self):
        return Path("/proc/self/fd/{}".format(self.descriptor))

    def verify(self):
        current = os.fstat(self.descriptor)
        identity = (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
        if identity != self.identity or _hash_descriptor(self.descriptor) != self.sha256:
            raise ArtifactTrustError("{} changed during stable use".format(self.label))
        try:
            pathname = os.lstat(str(self.path))
        except OSError as error:
            raise ArtifactTrustError("{} changed during stable use".format(self.label)) from error
        path_identity = (pathname.st_dev, pathname.st_ino, pathname.st_size, pathname.st_mtime_ns)
        if stat.S_ISLNK(pathname.st_mode) or path_identity != self.identity:
            raise ArtifactTrustError("{} changed during stable use or became a symlink".format(self.label))

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if self.descriptor is not None and exc_type is None:
                self.verify()
        finally:
            if self.descriptor is not None:
                os.close(self.descriptor)
                self.descriptor = None


def stable_artifact(path, root, label, expected_hash=None):
    return StableArtifact(path, root, label, expected_hash=expected_hash)


class RunLock:
    def __init__(self, root):
        self.root = Path(root)
        self.descriptor = None

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.root / ".protocol.lock"
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        try:
            self.descriptor = os.open(str(lock_path), flags, 0o600)
            fcntl.flock(self.descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if self.descriptor is not None:
                os.close(self.descriptor)
                self.descriptor = None
            if error.errno in (errno.EACCES, errno.EAGAIN):
                raise ArtifactTrustError("protocol run is already locked") from error
            raise ArtifactTrustError("could not acquire protocol run lock: {}".format(error)) from error
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.descriptor is not None:
            fcntl.flock(self.descriptor, fcntl.LOCK_UN)
            os.close(self.descriptor)
            self.descriptor = None


def run_lock(root):
    return RunLock(root)
