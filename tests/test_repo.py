#!/usr/bin/env python3
"""Tests that the repository's files agree with each other. Standard library only, Python 3.8 or newer, no network.

    python -m unittest discover -s tests      # from the repository root

A value written in two files drifts when only one of them changes: a default in settings.env and in bootstrap.sh, a
size in models.json and in README.md, a model file in models.json and in the self-test. Each test names one such pair,
so a change that forgets the other file fails here, before any pod is created.
"""
import json
import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import pod  # noqa: E402  (the module under test; importing it has no side effect)


def read(name):
    with open(os.path.join(ROOT, name), encoding="utf-8") as f:
        return f.read()


MODELS = json.loads(read("models.json"))
BOOTSTRAP = read("bootstrap.sh")
README = read("README.md")
SETTINGS = dict(re.findall(r"^(AINVFX_\w+)=(.*)$", read("settings.env"), re.M))   # every key, empty ones included


def table(heading):
    """The rows of the first table under a README heading, as lists of cells (backticks kept), header row left out."""
    rows = []
    for line in README.split("\n## " + heading + "\n", 1)[1].splitlines():
        if line.startswith("|"):
            rows.append([c.strip() for c in line.strip().strip("|").split("|")])
        elif rows:
            break
    return rows[2:]


def keys_of(profiles):
    """The files a set of profiles downloads at start, each once, @ includes followed as bootstrap.sh follows them."""
    out = []

    def walk(name, seen=()):
        for item in MODELS["profiles"][name]:
            if item.startswith("@"):
                if item[1:] not in seen:
                    walk(item[1:], seen + (name,))
            elif item not in out:
                out.append(item)
    for p in profiles:
        walk(p)
    return out


def gb(keys):
    return sum(MODELS["files"][k]["gb"] for k in keys)


class Repo(unittest.TestCase):
    def test_the_models_list_is_whole(self):
        """Every entry names a file, every file is used, and every file gives its repo, path, folder and size."""
        files = MODELS["files"]
        for name, items in MODELS["profiles"].items():
            for item in items:
                self.assertIn(item[1:] if item.startswith("@") else item,
                              MODELS["profiles"] if item.startswith("@") else files, "profile " + name)
        for family, roles in MODELS["training"].items():
            for role, key in roles.items():
                self.assertIn(key, files, "training family {} ({})".format(family, role))
        used = set(keys_of(MODELS["profiles"])) | {k for roles in MODELS["training"].values() for k in roles.values()}
        self.assertEqual(sorted(set(files) - used), [], "files that no profile and no training family uses")
        for key, f in files.items():
            self.assertTrue(f.get("repo") and f.get("path") and f.get("dir"), key)
            self.assertGreater(f.get("gb") or 0, 0, key)
            self.assertIsInstance(f.get("gated", False), bool, key)

    def test_pod_py_knows_every_profile(self):
        self.assertEqual(sorted(pod.PROFILES), sorted(MODELS["profiles"]))
        self.assertEqual(sorted(pod.PROFILE_ORDER), sorted(MODELS["profiles"]))

    def test_the_readme_profile_table_matches_models_json(self):
        """README.md's Profiles table gives the sizes and the disk that `up` computes from models.json."""
        rows = table("Profiles")
        self.assertTrue(rows)
        for cells in rows:
            label = cells[0].strip("`")
            profiles = pod.parse_profiles(label)
            spec, keys = pod.profile_spec(profiles, MODELS), keys_of(profiles)
            start = round(gb(keys))
            self.assertIn("{} GB".format(start) if start else "none at start", cells[-2], label)
            if "train" in profiles:
                per_run = round(max(gb(set(fam.values()) - set(keys)) for fam in MODELS["training"].values()))
                self.assertIn("up to {} GB per run".format(per_run), cells[-2], label)
            self.assertEqual(cells[-1], "{} GB".format(spec["disk"]), label)
            self.assertIn("5090" if "5090" in spec["gpu"] else "PRO 6000", cells[2], label)

    def test_settings_env_and_bootstrap_sh_have_the_same_defaults(self):
        """bootstrap.sh's own defaults serve a pod deployed without the variables: they are settings.env's values."""
        self.assertEqual(sorted(set(re.findall(r"\$\{(AINVFX_\w+)", BOOTSTRAP))), sorted(SETTINGS),
                         "the AINVFX_ variables that bootstrap.sh reads are those of settings.env")
        for key, default in re.findall(r"\$\{(AINVFX_\w+):-([^}$]*)\}", BOOTSTRAP):
            if SETTINGS[key]:
                self.assertEqual(default, SETTINGS[key], key)

    def test_the_readme_lists_every_setting_with_its_default(self):
        rows = {cells[0].strip("`"): cells[1] for cells in table("Your own settings")}
        self.assertEqual(sorted(rows), sorted(SETTINGS))
        for key, value in SETTINGS.items():
            if value:
                self.assertEqual(rows[key], "`{}`".format(value), key)

    def test_the_self_test_files_are_in_the_image_profile(self):
        """The test image loads three files by name: a file renamed in models.json would skip it on every pod."""
        names = re.search(r'^SELFTEST_FILES="([^"]+)"', BOOTSTRAP, re.M).group(1).split()
        image = {"{}/{}".format(MODELS["files"][k]["dir"], os.path.basename(MODELS["files"][k]["path"]))
                 for k in keys_of(["image"])}
        for name in names:
            self.assertIn(name, image)


if __name__ == "__main__":
    unittest.main()
