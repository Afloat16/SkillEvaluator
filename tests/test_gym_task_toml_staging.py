# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gym staging edits TOML fields without changing authored examples or other tables."""

from __future__ import annotations

import math
import tomllib
from pathlib import Path

import pytest

from skillevaluator.tier3.nemo_gym.runner import stage_for_gym


def _stage(tmp_path: Path, text: str, **kwargs) -> tuple[str, dict]:
    task = tmp_path / "case"
    task.mkdir()
    task_file = task / "task.toml"
    task_file.write_text(text, encoding="utf-8")
    stage_for_gym(tmp_path, **kwargs)
    staged = task_file.read_text(encoding="utf-8")
    return staged, tomllib.loads(staged)


@pytest.mark.parametrize(
    "environment",
    [
        '[environment]\nskills_dir = "/skills"\ncpus = 2\n',
        '["environment"]\n  "skills_dir" = "/skills"\ncpus = 2\n',
        "environment.skills_dir = '/skills'\nenvironment.cpus = 2\n",
        'environment = { skills_dir = "/skills", cpus = 2 }\n',
        'environment = { cpus = 2, skills_dir = "/skills" }\n',
        'environment = { skills_dir = "/skills" }\n',
        'environment.skills_dir = "/skills"\n',
        '[environment]\nskills_dir = """/workspace/\nskills"""\ncpus = 2\n',
    ],
)
def test_drop_hook_preserves_examples_and_other_skills_dir_fields(tmp_path: Path, environment: str) -> None:
    example = 'instruction = """Example:\nskills_dir = "/example"\n[environment]\ncpus = 999\n"""\n'
    text = example + environment + '\n[metadata]\nskills_dir = "metadata-value"\n'
    expected = tomllib.loads(text)
    del expected["environment"]["skills_dir"]

    staged, parsed = _stage(tmp_path, text)

    assert parsed == expected
    assert example in staged
    assert 'skills_dir = "metadata-value"' in staged


@pytest.mark.parametrize(
    "step",
    [
        "[steps.agent]\ntimeout_sec = 300\n[steps.verifier]\ntimeout_sec = 40\n",
        "agent = { timeout_sec = 300 }\nverifier.timeout_sec = 40\n",
    ],
)
def test_timeout_scaling_preserves_array_of_steps(tmp_path: Path, step: str) -> None:
    text = '[[steps]]\nname = "one"\n' + step + '[[steps]]\nname = "two"\n' + step
    expected = tomllib.loads(text)
    for item in expected["steps"]:
        item["agent"]["timeout_sec"] = 600.0
        item["verifier"]["timeout_sec"] = 80.0
    expected["verifier"] = {"timeout_sec": 1200.0}

    _, parsed = _stage(tmp_path, text, timeout_multiplier=2.0)

    assert parsed == expected


@pytest.mark.parametrize("comment", ['# The example uses """ strings\n', "# Literal strings use '''\n"])
def test_timeout_scaling_ignores_quotes_in_comments(tmp_path: Path, comment: str) -> None:
    text = comment + "[agent]\ntimeout_sec = 300\n[verifier]\ntimeout_sec = 40\n[environment]\ncpus = 2\n"
    staged, parsed = _stage(tmp_path, text, timeout_multiplier=2.0)

    assert parsed == {
        "agent": {"timeout_sec": 600.0},
        "verifier": {"timeout_sec": 80.0},
        "environment": {"cpus": 2, "build_timeout_sec": 1200.0},
    }
    assert staged.startswith(comment)


@pytest.mark.parametrize(
    "text",
    [
        '["agent"]\n"timeout_sec" = +300\n["environment"]\n"build_timeout_sec" = 0x100\n',
        "agent.timeout_sec = 300\nenvironment.build_timeout_sec = 256\n",
        "agent = { timeout_sec = 300 }\nenvironment = { build_timeout_sec = 256 }\n",
    ],
)
def test_timeout_scaling_uses_decoded_table_and_key_paths(tmp_path: Path, text: str) -> None:
    _, parsed = _stage(tmp_path, text, timeout_multiplier=2.0)

    assert parsed == {
        "agent": {"timeout_sec": 600.0},
        "environment": {"build_timeout_sec": 512.0},
        "verifier": {"timeout_sec": 1200.0},
    }


@pytest.mark.parametrize(
    "environment",
    [
        '["environment"]\n"memory" = "4G"\n"cpus" = 2\n',
        'environment.memory = "4G"\nenvironment.cpus = 2\n',
        'environment = { memory = "4G", cpus = 2 }\n',
    ],
)
def test_resource_overrides_preserve_multiline_examples(tmp_path: Path, environment: str) -> None:
    example = 'instruction = """Example:\n[environment]\nmemory = "example"\ncpus = 999\n"""\n'
    text = example + environment + '[metadata]\nmemory = "metadata-value"\n'
    expected = tomllib.loads(text)
    del expected["environment"]["memory"]
    expected["environment"].update(cpus=4, memory_mb=8192)

    staged, parsed = _stage(tmp_path, text, resource_overrides={"cpus": 4, "memory_mb": 8192})

    assert parsed == expected
    assert example in staged


@pytest.mark.parametrize(
    "verifier",
    [
        '["verifier"."env"]\nKEEP = "task"\n',
        '["verifier"."env"]\nKEEP = "task"\n[verifier]\ntimeout_sec = 12\n',
        'verifier.env.KEEP = "task"\n',
        'verifier = { timeout_sec = 12, env = { KEEP = "task" } }\n',
        '[verifier]\ntimeout_sec = 12\nenv = { KEEP = "task" }\n',
        "verifier = { timeout_sec = 12 }\n",
        "verifier = {}\n",
    ],
)
def test_judge_env_is_added_to_the_actual_verifier_table(tmp_path: Path, verifier: str) -> None:
    example = 'instruction = """Example:\n[verifier.env]\nKEEP = "example"\n"""\n'
    text = example + verifier
    expected = tomllib.loads(text)
    expected_env = expected["verifier"].setdefault("env", {})
    expected_env.setdefault("KEEP", "${KEEP}")
    expected_env["JUDGE.MODEL"] = "${JUDGE_MODEL}"

    staged, parsed = _stage(tmp_path, text, verifier_env={"KEEP": "${KEEP}", "JUDGE.MODEL": "${JUDGE_MODEL}"})

    assert parsed == expected
    assert example in staged


@pytest.mark.parametrize("timeout", ["true", '"300"', "1e309", "9" * 400])
def test_invalid_timeouts_are_rejected_before_task_is_written(tmp_path: Path, timeout: str) -> None:
    task = tmp_path / "case"
    task.mkdir()
    task_file = task / "task.toml"
    text = f"[agent]\ntimeout_sec = {timeout}\n"
    task_file.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError):
        stage_for_gym(tmp_path, timeout_multiplier=2.0)

    assert task_file.read_text(encoding="utf-8") == text


def test_removing_hook_preserves_unrelated_nan_metadata(tmp_path: Path) -> None:
    metadata = "[metadata]\nscores = [nan, { missing = nan, value = 2 }]\n"
    text = '[environment]\nskills_dir = "/skills"\n' + metadata

    staged, parsed = _stage(tmp_path, text)

    assert parsed["environment"] == {}
    assert math.isnan(parsed["metadata"]["scores"][0])
    assert math.isnan(parsed["metadata"]["scores"][1]["missing"])
    assert parsed["metadata"]["scores"][1]["value"] == 2
    assert metadata in staged
