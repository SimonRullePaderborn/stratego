"""Checks that a run folder holds exactly the files of the commit named in it.  RUN ON THE CLUSTER ONLY.

    python3 -I nca_drones/cluster/check_manifest.py ~/wf/runs/<card>

The cluster runs no git (PROGRAMME.md, section 4.3). The Mac sends the files of a commit with
`git archive <commit> nca_drones` and the list of that commit's files with `git ls-tree -r -z <commit> nca_drones`,
stored as MANIFEST in the run folder. Each entry of the list is "<mode> <type> <hash>\\t<path>", ended by a NUL
byte. Git's hash of a file is the SHA-1 of b"blob <size in bytes>\\0" followed by the file's bytes; for a symbolic
link the bytes are the link's target. This script recomputes that hash for every listed file, with the standard
library only, and names every file that differs, is missing, or is present without being listed. Run it right
after the files arrive, before anything writes into the folder. It changes nothing.
"""
import hashlib
import os
import sys


def git_blob_hash(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def read_bytes(full, mode):
    if mode == "120000":
        return os.fsencode(os.readlink(full)) if os.path.islink(full) else None
    if os.path.islink(full) or not os.path.isfile(full):
        return None
    with open(full, "rb") as f:
        return f.read()


def main():
    root = os.path.abspath(os.path.expanduser(sys.argv[1]))
    with open(os.path.join(root, "MANIFEST"), "rb") as f:
        entries = [e for e in f.read().split(b"\0") if e]
    listed, bad = set(), []
    for e in entries:
        meta, path = e.split(b"\t", 1)
        mode, kind, want = meta.decode().split()
        if kind != "blob":
            continue
        rel = os.fsdecode(path)
        listed.add(rel)
        data = read_bytes(os.path.join(root, rel), mode)
        if data is None or git_blob_hash(data) != want:
            bad.append(rel)
    present = set()
    for d, _, files in os.walk(os.path.join(root, "nca_drones")):
        for name in files:
            present.add(os.path.relpath(os.path.join(d, name), root))
    extra = sorted(present - listed)
    print(f"{len(listed)} files listed: {len(listed) - len(bad)} match, {len(bad)} differ or are missing; "
          f"{len(extra)} present but not listed")
    for rel in bad:
        print("differs or missing:", rel)
    for rel in extra:
        print("present but not listed:", rel)
    sys.exit(0 if not bad and not extra else 1)


if __name__ == "__main__":
    main()
