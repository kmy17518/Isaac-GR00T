# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""B1K prompt plumbing: train-side language-key selection, serve-side prompt resolution, and
the observation the policy wrapper builds for the model.

CPU-only: the GR00T policy is replaced by a stub exposing ``language_key`` (and, for the
control-mode tests, a ``get_action`` that echoes a per-env marker).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS, ROBOT_OBS_CONFIGS
from gr00t.data.b1k_prompts import DEFAULT_PROMPT_SOURCE, language_key
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig
from gr00t.eval.eval_b1k_wrapper import (
    TASK_ID_OBS_KEY,
    B1KPolicyWrapper,
    _slice_batch,
    load_modality_config,
)
import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
R1PRO_PY = REPO_ROOT / "examples" / "b1k" / "r1pro.py"
R1PRO_JSON = REPO_ROOT / "examples" / "b1k" / "r1pro.json"
TAG = EmbodimentTag.NEW_EMBODIMENT

CAMERAS = (
    "robot_r1::robot_r1:zed_link:Camera:0::rgb",
    "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb",
    "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb",
)


class StubPolicy:
    """Stands in for Gr00tPolicy: only the checkpoint's language key matters here."""

    def __init__(self, language_key: str):
        self.language_key = language_key


# R1Pro action groups (examples/b1k/r1pro.json), 23 dims in total.
ACTION_DIMS = {
    "base": 3,
    "torso": 4,
    "left_arm": 7,
    "left_gripper": 1,
    "right_arm": 7,
    "right_gripper": 1,
}


class StubActionPolicy(StubPolicy):
    """``get_action`` echoes each env's marker (``proprio[..., 0]``) into every action value, so a
    test can tell which envs a (sub-)batch contained and whether outputs are routed back correctly."""

    def __init__(self, language_key: str, horizon: int = 4):
        super().__init__(language_key)
        self.horizon = horizon
        self.calls: list[list[float]] = []

    def get_action(self, batch, options=None):
        markers = batch["state"]["base_qvel"][:, 0, 0]  # (B,)
        batch_size = markers.shape[0]
        # The sub-batch must be sliced consistently across modalities.
        assert batch["video"]["head"].shape[:2] == (batch_size, 1)
        assert len(batch["language"][self.language_key]) == batch_size
        self.calls.append(markers.tolist())
        actions = {
            key: np.broadcast_to(markers[:, None, None], (batch_size, self.horizon, dim)).astype(
                np.float32
            )
            for key, dim in ACTION_DIMS.items()
        }
        return actions, {}


@pytest.fixture
def r1pro_registered():
    """Register the R1Pro modality config for the test (and unregister if we did it)."""
    registered_here = TAG.value not in MODALITY_CONFIGS
    if registered_here:
        load_modality_config(str(R1PRO_PY))
    yield MODALITY_CONFIGS[TAG.value]
    if registered_here:
        MODALITY_CONFIGS.pop(TAG.value, None)
        ROBOT_OBS_CONFIGS.pop(TAG.value, None)
        sys.modules.pop("r1pro", None)


@pytest.fixture
def modality_json():
    return json.loads(R1PRO_JSON.read_text())


def _obs(task_id: int | None = 0, batch: int | None = None) -> dict:
    """Evaluator-style observation: single env (``batch=None``) or ``batch`` envs stacked on a
    leading axis. Each env's ``proprio[0]`` carries a distinct marker (10 + env index)."""
    if batch is None:
        proprio = np.zeros(61, np.float32)
        proprio[0] = 10.0
        obs = {"robot_r1::proprio": proprio}
        for camera in CAMERAS:
            obs[camera] = np.zeros((480, 480, 4), np.uint8)
    else:
        proprio = np.zeros((batch, 61), np.float32)
        proprio[:, 0] = 10.0 + np.arange(batch)
        obs = {"robot_r1::proprio": proprio}
        for camera in CAMERAS:
            obs[camera] = np.zeros((batch, 480, 480, 4), np.uint8)
    if task_id is not None:
        # The evaluator sends a length-1 int64 tensor; msgpack delivers it as an array.
        obs[TASK_ID_OBS_KEY] = np.array([task_id], dtype=np.int64)
    return obs


class TestModalityConfigDefault:
    def test_r1pro_default_follows_default_prompt_source(self, r1pro_registered):
        assert r1pro_registered["language"].modality_keys == [language_key(DEFAULT_PROMPT_SOURCE)]


class TestTrainSideSelection:
    def _configs(self):
        return {
            "new_embodiment": {
                "language": ModalityConfig(
                    delta_indices=[0], modality_keys=["annotation.human.task_name"]
                )
            }
        }

    def test_select_prompt_source_rewrites_language_key(self):
        from scripts.b1k.train_b1k import select_prompt_source

        configs = self._configs()
        key = select_prompt_source(configs, "new_embodiment", "task_description")
        assert key == "annotation.human.task_description"
        assert configs["new_embodiment"]["language"].modality_keys == [key]
        assert configs["new_embodiment"]["language"].delta_indices == [0]
        with pytest.raises(ValueError, match="Unknown B1K prompt source"):
            select_prompt_source(configs, "new_embodiment", "coarse_action")
        with pytest.raises(ValueError, match="No modality config registered"):
            select_prompt_source(configs, "other_embodiment", "task_name")

    def test_resolve_language_key_defaults_to_modality_config(self):
        from scripts.b1k.train_b1k import resolve_language_key

        configs = self._configs()
        # No --prompt-source: keep whatever the modality config declares, untouched.
        assert resolve_language_key(configs, "new_embodiment", None) == "annotation.human.task_name"
        assert configs["new_embodiment"]["language"].modality_keys == ["annotation.human.task_name"]
        # --prompt-source overrides it.
        key = resolve_language_key(configs, "new_embodiment", "task_description")
        assert key == "annotation.human.task_description"
        assert configs["new_embodiment"]["language"].modality_keys == [key]


class TestServeSideResolution:
    def _config(self, **overrides):
        from scripts.b1k.serve_b1k import ServerConfig

        return ServerConfig(model_path="unused", modality_config_path=str(R1PRO_PY), **overrides)

    def test_prompt_kind_follows_checkpoint_language_key(self):
        from scripts.b1k.serve_b1k import resolve_prompts

        text, table = resolve_prompts(self._config(), "annotation.human.task_description")
        assert text is None and table[0].startswith("Turn on the radio")
        text, table = resolve_prompts(self._config(), "annotation.human.task_name")
        assert text is None and table[0] == "turning_on_radio" and len(table) == 100

    def test_task_name_fixes_prompt(self):
        from scripts.b1k.serve_b1k import resolve_prompts

        text, table = resolve_prompts(
            self._config(task_name="turning_on_radio"), "annotation.human.task_description"
        )
        assert table is None and text.startswith("Turn on the radio")

    def test_prompt_source_override_for_legacy_checkpoints(self):
        """Checkpoints trained before the loader fix carry the task_description key but saw task
        names; --prompt-source task_name keeps serving consistent with them."""
        from scripts.b1k.serve_b1k import resolve_prompts

        text, _ = resolve_prompts(
            self._config(task_name="turning_on_radio", prompt_source="task_name"),
            "annotation.human.task_description",
        )
        assert text == "turning_on_radio"

    def test_text_prompt_bypasses_everything(self):
        from scripts.b1k.serve_b1k import resolve_prompts

        assert resolve_prompts(
            self._config(text_prompt="do the thing", tasks_file="/nonexistent"),
            "annotation.human.coarse_action",
        ) == ("do the thing", None)

    def test_unknown_language_key_requires_explicit_choice(self):
        from scripts.b1k.serve_b1k import resolve_prompts

        with pytest.raises(ValueError, match="pass --prompt-source or --text-prompt"):
            resolve_prompts(self._config(), "annotation.human.coarse_action")


class TestDeployModalityValidation:
    def test_tasks_table_field_is_validated(self, tmp_path):
        from scripts.b1k.deploy_modality import _validate_dataset

        template = json.loads(R1PRO_JSON.read_text())
        info = {
            "features": {
                "observation.state": {"dtype": "float32", "shape": [61]},
                "action": {"dtype": "float32", "shape": [23]},
                "task_index": {"dtype": "int64", "shape": [1]},
                **{meta["original_key"]: {"dtype": "video"} for meta in template["video"].values()},
            }
        }
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()

        # No sidecar at all -> both prompt kinds are reported.
        errors = _validate_dataset(info, template, meta_dir=meta_dir)
        assert len(errors) == 2 and all("missing tasks table" in e for e in errors)

        # Sidecar without the natural-language field -> only task_description fails.
        (meta_dir / "tasks.jsonl").write_text(
            json.dumps({"task_index": 0, "task_name": "turning_on_radio"}) + "\n"
        )
        errors = _validate_dataset(info, template, meta_dir=meta_dir)
        assert errors == [
            "annotation 'human.task_description' -> meta/tasks.jsonl:1 (task_index 0) "
            "has no 'task' field"
        ]

        # Complete sidecar -> clean.
        (meta_dir / "tasks.jsonl").write_text(
            json.dumps({"task_index": 0, "task_name": "turning_on_radio", "task": "Turn it on."})
            + "\n"
        )
        assert _validate_dataset(info, template, meta_dir=meta_dir) == []


class TestPolicyWrapperLanguage:
    def test_prompt_resolved_from_task_id_under_checkpoint_key(
        self, r1pro_registered, modality_json
    ):
        wrapper = B1KPolicyWrapper(
            policy=StubPolicy("annotation.human.task_description"),
            embodiment_tag=TAG,
            modality_config=modality_json,
            task_prompts={0: "Turn on the radio.", 1: "Pick up the trash."},
        )
        processed, batch_size = wrapper.process_input(_obs(task_id=1))
        assert batch_size == 1
        # (B, T=1) list of lists, keyed by the checkpoint's language key -- not a hard-coded one.
        assert processed["language"] == {
            "annotation.human.task_description": [["Pick up the trash."]]
        }
        assert set(processed["video"]) == {"head", "left_wrist", "right_wrist"}
        assert processed["video"]["head"].shape == (1, 1, 224, 224, 3)

    def test_fixed_text_prompt(self, r1pro_registered, modality_json):
        wrapper = B1KPolicyWrapper(
            policy=StubPolicy("annotation.human.coarse_action"),
            embodiment_tag=TAG,
            modality_config=modality_json,
            text_prompt="Turn on the radio.",
        )
        processed, _ = wrapper.process_input(_obs(task_id=None))
        assert processed["language"] == {"annotation.human.coarse_action": [["Turn on the radio."]]}

    def test_language_key_falls_back_to_registered_config(self, r1pro_registered, modality_json):
        wrapper = B1KPolicyWrapper(
            policy=object(),  # no language_key attribute
            embodiment_tag=TAG,
            modality_config=modality_json,
            text_prompt="x",
        )
        assert wrapper.language_key == r1pro_registered["language"].modality_keys[0]

    def test_errors(self, r1pro_registered, modality_json):
        with pytest.raises(ValueError, match="needs a prompt"):
            B1KPolicyWrapper(StubPolicy("k"), TAG, modality_json)

        wrapper = B1KPolicyWrapper(
            StubPolicy("annotation.human.task_description"),
            TAG,
            modality_json,
            task_prompts={0: "a"},
        )
        with pytest.raises(KeyError, match="no 'task_id'"):
            wrapper.process_input(_obs(task_id=None))
        with pytest.raises(KeyError, match="task_id 7 is not in the prompt table"):
            wrapper.process_input(_obs(task_id=7))


class TestBatchedObservations:
    """N-env evaluation: a leading batch axis must stay the batch."""

    def test_batched_obs_keeps_batch_dim(self, r1pro_registered, modality_json):
        wrapper = B1KPolicyWrapper(
            StubPolicy("annotation.human.task_description"),
            TAG,
            modality_json,
            task_prompts={0: "Turn on the radio."},
        )
        processed, batch_size = wrapper.process_input(_obs(task_id=0, batch=3))
        assert batch_size == 3
        assert processed["video"]["head"].shape == (3, 1, 224, 224, 3)
        assert processed["state"]["torso"].shape == (3, 1, 4)
        # one task_id for the connection -> broadcast to every env in the batch
        assert processed["language"] == {
            "annotation.human.task_description": [["Turn on the radio."]] * 3
        }

    def test_per_env_task_ids(self, r1pro_registered, modality_json):
        wrapper = B1KPolicyWrapper(
            StubPolicy("annotation.human.task_description"),
            TAG,
            modality_json,
            task_prompts={0: "a", 1: "b"},
        )
        obs = _obs(batch=2)
        obs[TASK_ID_OBS_KEY] = np.array([1, 0], dtype=np.int64)
        processed, _ = wrapper.process_input(obs)
        assert processed["language"] == {"annotation.human.task_description": [["b"], ["a"]]}

    def test_slice_batch_recurses_into_modalities(self, r1pro_registered, modality_json):
        wrapper = B1KPolicyWrapper(
            StubPolicy("annotation.human.task_description"),
            TAG,
            modality_json,
            task_prompts={0: "a"},
        )
        processed, _ = wrapper.process_input(_obs(task_id=0, batch=3))
        sub = _slice_batch(processed, np.array([2, 0]))
        assert sub["video"]["head"].shape == (2, 1, 224, 224, 3)
        assert sub["state"]["base_qvel"][:, 0, 0].tolist() == [12.0, 10.0]
        assert sub["language"]["annotation.human.task_description"] == [["a"], ["a"]]


class TestRecedingControlModes:
    """The receding modes sub-batch the envs that need a new plan (``_slice_batch``)."""

    def _wrapper(self, control_mode, modality_json, horizon=4):
        policy = StubActionPolicy("annotation.human.task_description", horizon=horizon)
        wrapper = B1KPolicyWrapper(
            policy, TAG, modality_json, task_prompts={0: "a"}, control_mode=control_mode
        )
        return wrapper, policy

    def test_receeding_horizon_replans_and_routes_sub_batches(
        self, r1pro_registered, modality_json
    ):
        wrapper, policy = self._wrapper("receeding_horizon", modality_json, horizon=4)
        obs = _obs(task_id=0, batch=3)

        action = wrapper.act(obs)
        assert policy.calls == [[10.0, 11.0, 12.0]]  # first step: every env needs a plan
        assert action.shape == (3, 23)
        assert action[:, 0].tolist() == [10.0, 11.0, 12.0]  # env b gets env b's plan

        for _ in range(3):  # remaining steps of the 4-step plan come from the buffer
            wrapper.act(obs)
        assert len(policy.calls) == 1
        wrapper.act(obs)  # plans exhausted -> replan all
        assert policy.calls[-1] == [10.0, 11.0, 12.0]

        # Exhaust only env 1 -> only env 1 is re-planned, with its own obs slice.
        wrapper.sequence_indices[1, 0] = wrapper.sequence_lengths[1, 0]
        action = wrapper.act(obs)
        assert policy.calls[-1] == [11.0]
        assert action[:, 0].tolist() == [10.0, 11.0, 12.0]

    def test_receeding_temporal_sub_batch_replan(self, r1pro_registered, modality_json):
        # Plans (12 steps) outlast the replan interval (3), as with the real 16-step model and K=10.
        wrapper, policy = self._wrapper("receeding_temporal", modality_json, horizon=12)
        wrapper.replan_interval = 3
        obs = _obs(task_id=0, batch=2)

        action = wrapper.act(obs)
        assert action.shape == (2, 23)
        assert action[:, 0].tolist() == pytest.approx([10.0, 11.0])
        assert all(call == [10.0, 11.0] for call in policy.calls)

        # Desynchronize env 1 so it hits the replan boundary (step 3) one step before env 0.
        wrapper.step_counter[1] = 2
        wrapper.act(obs)  # counters -> [2, 3]: nobody replans
        n_calls = len(policy.calls)
        action = wrapper.act(obs)  # env 1 at 3 % 3 == 0 -> only env 1 is re-planned
        assert policy.calls[n_calls:] == [[11.0]]
        assert action[:, 0].tolist() == pytest.approx([10.0, 11.0])

    def test_per_connection_copies_have_independent_buffers(self, r1pro_registered, modality_json):
        """WebsocketPolicyServer hands each connection ``copy.copy(wrapper)`` + ``reset()`` so env
        slots don't share receding-mode buffers while still sharing the model."""
        wrapper, policy = self._wrapper("receeding_horizon", modality_json, horizon=4)
        slot_a, slot_b = copy.copy(wrapper), copy.copy(wrapper)
        slot_a.reset()
        slot_b.reset()
        assert slot_a.policy is slot_b.policy is policy

        slot_a.act(_obs(task_id=0, batch=3))  # 3 envs on connection A
        slot_b.act(_obs(task_id=0, batch=2))  # 2 envs on connection B
        assert slot_a.action_buffer.shape[0] == 3
        assert slot_b.action_buffer.shape[0] == 2
        assert slot_a.action_buffer is not slot_b.action_buffer
        assert wrapper.action_buffer is None  # the prototype itself stays untouched

    @pytest.mark.parametrize("control_mode", ["receeding_horizon", "receeding_temporal"])
    def test_single_env_obs_is_a_batch_of_one(self, r1pro_registered, modality_json, control_mode):
        # 16-step plans like the real R1Pro model, longer than the default replan interval (10).
        wrapper, _ = self._wrapper(control_mode, modality_json, horizon=16)
        for _ in range(12):
            action = wrapper.act(_obs(task_id=0))
            assert action.shape == (1, 23)
            assert float(action[0, 0]) == pytest.approx(10.0)
