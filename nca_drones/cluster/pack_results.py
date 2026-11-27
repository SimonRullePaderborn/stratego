"""Writes result files as a `git fast-import` stream on standard output.  RUN ON THE CLUSTER ONLY.

On the Mac, from the repository root:

    ssh <host> 'cd ~/wf/runs/<card> && python3 -I nca_drones/cluster/pack_results.py . nca_drones/results/p2/<files>' \
        | git fast-import --force --quiet
    git checkout refs/import/results -- nca_drones/results/p2

The cluster runs no git (PROGRAMME.md, section 4.3).  This script only writes the format that `git fast-import`
reads on the Mac: one blob per file, byte for byte, and one carrier commit on the ref refs/import/results.  The
Mac checks the files out of that commit into its working copy and commits them as usual; the carrier commit is
never part of the branch.  Paths are given relative to the run folder and are kept as they are.  A file that holds
the home path, the account name or the host name is refused, so that none reaches a commit.  Standard library only.
"""
import getpass
import os
import socket
import sys
import time


def main():
    root = os.path.abspath(os.path.expanduser(sys.argv[1]))
    paths = sys.argv[2:]
    if not paths:
        sys.exit("no files named")
    needles = [x.encode() for x in (os.path.expanduser("~"), getpass.getuser(), socket.gethostname()) if x]
    blobs = []
    for rel in paths:
        with open(os.path.join(root, rel), "rb") as f:
            data = f.read()
        if any(n in data for n in needles):
            sys.exit(f"refused: {rel} holds the home path, the account name or the host name")
        blobs.append((rel, data))
    out = sys.stdout.buffer
    for i, (rel, data) in enumerate(blobs, 1):
        out.write(b"blob\nmark :%d\ndata %d\n" % (i, len(data)))
        out.write(data)
        out.write(b"\n")
    message = b"results carried from the cluster; not part of the branch\n"
    out.write(b"commit refs/import/results\ncommitter results <results@invalid> %d +0000\ndata %d\n" % (int(time.time()), len(message)))
    out.write(message)
    for i, (rel, _) in enumerate(blobs, 1):
        out.write(b"M 100644 :%d %s\n" % (i, rel.encode()))
    out.write(b"\n")
    sys.stderr.write(f"packed {len(blobs)} files, {sum(len(d) for _, d in blobs)} bytes\n")


if __name__ == "__main__":
    main()
