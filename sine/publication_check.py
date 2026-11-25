"""The publication check: no generated or shipped file of the MIMO Sine suite may hold the word the set does not use for
the structure under test, a local path, or a listed name.

    python sine/publication_check.py <file or folder> [...] [--workers 6]

Reads every file under the paths given. Text files are read line by line; a binary file (the .f8 and .c16 arrays) is
searched for path bytes only: the home folder and the slash prefixes as bytes; a drive prefix (`C:\`) only when 8 or
more printable characters follow it, because three bytes turn up by chance in a megabyte of floating-point numbers (69
arrays of the 380 problems held one). Three findings, each giving the place, never the word:
  word    the whole word the set avoids for the structure under test ('system' is used), singular or plural
  path    a local path: the home folder of the account running the check, its last part as a word, or any of the
          absolute prefixes listed in PATH_PREFIXES, or a path from the home folder written with '~/'
  name    a run of one to three words whose hash is a listed name: the same hashes and normalization as the runner
          repository's name gate (tools/check_tree.py, NAMES), so the list holds no name in the clear

This file itself must spell out the path prefixes it looks for, so in it only the name finding is read.

Exit status 0 when nothing is found, 1 otherwise. The last line is `PUBLICATION CHECK: <files> files, <findings>
findings`.
"""
import argparse
import hashlib
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

# the runner repository's tools/check_tree.py NAMES (copied; phrase_hash below is its normalization)
NAMES = frozenset({
    '4a654a97330bd592', 'd6c6d1247e12746b', 'e07681543d5c76d6', 'e2a3af9c1d2aa535',
    'bd3b238df5c6bcde', '2d52485274940e77', 'bef51514796755a0', 'd9a88fab7e0b2a48',
    'd603337bb0e81eb6', 'b9dc4892374c9d9f', '846be5fc541c5039', '49e172a78edef3db',
    '8b53b6c84bab7387', '31175d1308492e88', '9c9af8df9f2079ba', '63f87d6c2ab47659',
    '70cabe495f52902c', '78de28dddab5e164', '80cdad1dcf688de4', '1d7cc59f0ff10ab7',
    'ced8f61cd041bb50', '5e4f06885c0ac383', '436f604f3bb414c1', 'e78bdebae031095b',
    '05f6ecf01ac4ebec', 'cb5c36fe59056a6a', '799dda30b3121bbf', 'b426df21b43a09fa',
    '16f45b99abfda636', 'a4074992974e30ed', '4db54b27d524b4a1', '1916eb4a702ef0dd',
    '5bd79a8938028958', '3e4d26effdc5f34d', 'fce9f2383131cd07',
})
_WORD = re.compile(r'[a-z0-9]+')
_ALPHA = re.compile(r'[a-z]')
AVOIDED = re.compile(r'\bplants?\b', re.IGNORECASE)
PATH_PREFIXES = ('/Users/', '/home/', '/private/', '/Volumes/', '/tmp/', 'C:\\', 'c:\\')
_TILDE = re.compile(r'(?<![A-Za-z0-9])~/[A-Za-z]')
BINARY = ('.f8', '.c16', '.f4', '.i4', '.npy', '.bin')
_DRIVE_BYTES = re.compile(rb'[Cc]:\\[\x20-\x7e]{8}')


def phrase_hash(text):
    return hashlib.sha256(' '.join(_WORD.findall(text.lower())).encode('utf-8')).hexdigest()[:16]


def _home():
    h = os.path.expanduser('~').rstrip('/')
    return h, os.path.basename(h).lower()


def check_text(rel, text, names=NAMES, self_file=False):
    home, user = _home()
    found = []
    lines = [rel] + text.splitlines()
    for n, line in enumerate(lines):
        where = 'its path' if n == 0 else f'line {n}'
        if n > 0 and not self_file:
            if AVOIDED.search(line):
                found.append(f'{rel}: {where}: the avoided word')
            if home in line or any(p in line for p in PATH_PREFIXES) or _TILDE.search(line):
                found.append(f'{rel}: {where}: a local path')
        words = _WORD.findall(line.lower())
        if user and user in words:
            found.append(f'{rel}: {where}: the account name')
        runs = set()
        for size in (1, 2, 3):
            for i in range(len(words) - size + 1):
                w = words[i:i + size]
                if any(_ALPHA.search(x) for x in w):           # a name is not digits alone
                    runs.add(' '.join(w))
        if any(hashlib.sha256(r.encode('utf-8')).hexdigest()[:16] in names for r in runs):
            found.append(f'{rel}: {where}: a listed name')
    return found


def check_file(args):
    path, rel = args
    home, user = _home()
    data = open(path, 'rb').read()
    if path.endswith(BINARY):
        hits = [p for p in (home.encode(),) + tuple(x.encode() for x in PATH_PREFIXES if ':' not in x) if p in data]
        hits += _DRIVE_BYTES.findall(data)
        return [f'{rel}: binary: a local path'] if hits else []
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError:
        return [f'{rel}: not UTF-8 text and not a known array file']
    return check_text(rel, text, self_file=os.path.realpath(path) == os.path.realpath(__file__))


def files_under(paths):
    out = []
    for p in paths:
        p = os.path.abspath(p)
        if os.path.isfile(p):
            out.append((p, os.path.basename(p)))
            continue
        for d, dirs, fs in os.walk(p):
            dirs[:] = sorted(x for x in dirs if x not in ('__pycache__', '.git'))
            for f in sorted(fs):
                if f.endswith('.pyc') or f == '.DS_Store':
                    continue
                full = os.path.join(d, f)
                out.append((full, os.path.relpath(full, os.path.dirname(p))))
    return out


def run(paths, workers=1):
    fl = files_under(paths)
    found = []
    if workers > 1 and len(fl) > 50:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for r in ex.map(check_file, fl, chunksize=64):
                found.extend(r)
    else:
        for a in fl:
            found.extend(check_file(a))
    return len(fl), found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+')
    ap.add_argument('--workers', type=int, default=1)
    a = ap.parse_args()
    n, found = run(a.paths, min(a.workers, 6))
    for f in found:
        print(f)
    print(f'PUBLICATION CHECK: {n} files, {len(found)} findings')
    sys.exit(1 if found else 0)


if __name__ == '__main__':
    main()
