"""Check a case file against the runner's case schema (msnb-case-1.schema.json, copied from the runner's draft at its
commit 1c0a4a24, beside FORMAT_RUN.md). A small validator for the parts of JSON Schema that file uses (type, const,
enum, required, properties, additionalProperties, items, minItems, maxItems, minimum, maximum, exclusiveMinimum,
minLength, pattern, not, oneOf, $ref), so the suite needs no package beyond NumPy.

    python sine/case_schema.py <case.json> [...]      # prints each problem found; exit 1 if any
"""
import json
import os
import re
import sys

SCHEMA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'msnb-case-1.schema.json')
_TYPES = {'object': dict, 'array': list, 'string': str, 'boolean': bool, 'null': type(None)}


def _is(v, t):
    if t == 'number':
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if t == 'integer':
        return isinstance(v, int) and not isinstance(v, bool)
    return isinstance(v, _TYPES[t])


def errors(v, s, root, where='case'):
    if '$ref' in s:
        s = root['$defs'][s['$ref'].split('/')[-1]]
    out = []
    if 'type' in s:
        ts = s['type'] if isinstance(s['type'], list) else [s['type']]
        if not any(_is(v, t) for t in ts):
            return [f'{where}: type {type(v).__name__}, wants {ts}']
    if 'const' in s and v != s['const']:
        out.append(f'{where}: not {s["const"]!r}')
    if 'enum' in s and v not in s['enum']:
        out.append(f'{where}: {v!r} not one of {s["enum"]}')
    if 'not' in s and not errors(v, s['not'], root, where):
        out.append(f'{where}: matches what is forbidden')
    if 'oneOf' in s:
        n = sum(not errors(v, x, root, where) for x in s['oneOf'])
        if n != 1:
            out.append(f'{where}: matches {n} of the oneOf forms, wants exactly 1')
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if 'minimum' in s and v < s['minimum']:
            out.append(f'{where}: {v} under {s["minimum"]}')
        if 'maximum' in s and v > s['maximum']:
            out.append(f'{where}: {v} over {s["maximum"]}')
        if 'exclusiveMinimum' in s and v <= s['exclusiveMinimum']:
            out.append(f'{where}: {v} not over {s["exclusiveMinimum"]}')
    if isinstance(v, str):
        if 'minLength' in s and len(v) < s['minLength']:
            out.append(f'{where}: too short')
        if 'pattern' in s and not re.search(s['pattern'], v):
            out.append(f'{where}: does not match {s["pattern"]}')
    if isinstance(v, list):
        if 'minItems' in s and len(v) < s['minItems']:
            out.append(f'{where}: {len(v)} items, wants at least {s["minItems"]}')
        if 'maxItems' in s and len(v) > s['maxItems']:
            out.append(f'{where}: {len(v)} items, wants at most {s["maxItems"]}')
        if 'items' in s:
            for i, x in enumerate(v):
                out += errors(x, s['items'], root, f'{where}[{i}]')
    if isinstance(v, dict):
        for k in s.get('required', []):
            if k not in v:
                out.append(f'{where}: {k} missing')
        props = s.get('properties', {})
        for k, x in v.items():
            if k in props:
                out += errors(x, props[k], root, f'{where}.{k}')
            elif s.get('additionalProperties') is False:
                out.append(f'{where}: unknown key {k}')
    return out


def check(case, schema=None):
    root = schema or json.load(open(SCHEMA_FILE))
    return errors(case, root, root)


def main():
    bad = 0
    root = json.load(open(SCHEMA_FILE))
    for p in sys.argv[1:]:
        e = check(json.load(open(p)), root)
        for x in e:
            print(f'{os.path.basename(p)}: {x}')
        bad += bool(e)
    print(f'CASE SCHEMA: {len(sys.argv) - 1} cases, {bad} with problems')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
