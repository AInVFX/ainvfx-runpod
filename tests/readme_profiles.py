#!/usr/bin/env python3
"""Rewrite the Models and Container disk columns of README.md's Profiles table from models.json.

After any change to models.json, run `python3 tests/readme_profiles.py` and commit README.md with it: the numbers are
the ones `up` computes (pod.profile_spec), so test_repo's check of the table passes again. `--check` only reports."""
import os
import sys

from test_repo import MODELS, ROOT, gb, keys_of, pod   # the same helpers as the test, in the same folder


def cells_for(label):
    """The Models and Container disk cells of one row: `image+train` -> ("118 GB, then up to 70 GB per run", "320 GB")."""
    profiles = pod.parse_profiles(label)
    keys = keys_of(profiles)
    start = round(gb(keys))
    models = "{} GB".format(start) if start else "none at start"
    if "train" in profiles:
        per_run = round(max(gb(set(fam.values()) - set(keys)) for fam in MODELS["training"].values()))
        models += (", then up to {} GB per run" if start else ", up to {} GB per run").format(per_run)
    return models, "{} GB".format(pod.profile_spec(profiles, MODELS)["disk"])


def main():
    path = os.path.join(ROOT, "README.md")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    head, sep, rest = text.partition("\n## Profiles\n")
    if not sep:
        sys.exit("README.md has no Profiles section")
    lines, changed, in_table = rest.split("\n"), [], False
    for i, line in enumerate(lines):
        if line.startswith("## "):
            break
        if not line.startswith("| `"):
            if in_table and not line.startswith("|"):
                break
            in_table = in_table or line.startswith("|")
            continue
        in_table = True
        cells = line.strip().strip("|").split("|")
        new = cells[:-2] + [" {} ".format(c) for c in cells_for(cells[0].strip().strip("`"))]
        row = "|" + "|".join(new) + "|"
        if row != line:
            changed.append((line, row))
            lines[i] = row
    for old, row in changed:
        print("- " + old + "\n+ " + row)
    if "--check" in sys.argv:
        sys.exit(1 if changed else 0)
    if changed:
        with open(path, "w", encoding="utf-8") as f:
            f.write(head + sep + "\n".join(lines))
    print("README.md: {} row(s) updated".format(len(changed)) if changed else "README.md: the Profiles table matches models.json")


if __name__ == "__main__":
    main()
