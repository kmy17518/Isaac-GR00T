import logging
from pathlib import Path

import cv2
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS, ROBOT_OBS_CONFIGS
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.policy import BasePolicy
import numpy as np
import torch


logger = logging.getLogger(__name__)

# Observation key under which the OmniGibson evaluator sends the challenge task index
# (``omnigibson.eval.evaluator.Evaluator._preprocess_obs``); matches the dataset's task_index.
TASK_ID_OBS_KEY = "task_id"


def resize_with_pad(image: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    """
    Resize image to target size while maintaining aspect ratio and padding with zeros.
    
    Args:
        image: Input image of shape (B, H, W, C) or (H, W, C)
        target_height: Target height
        target_width: Target width
    
    Returns:
        Resized and padded image of shape (B, target_height, target_width, C) or (target_height, target_width, C)
    """
    has_batch = image.ndim == 4
    if not has_batch:
        image = image[np.newaxis, ...]
    
    batch_size, h, w, c = image.shape
    
    # Calculate scaling factor to fit within target size while maintaining aspect ratio
    scale = min(target_height / h, target_width / w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # Resize all images in batch
    resized_images = np.zeros((batch_size, target_height, target_width, c), dtype=image.dtype)
    for i in range(batch_size):
        resized = cv2.resize(image[i], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Calculate padding
        pad_top = (target_height - new_h) // 2
        pad_left = (target_width - new_w) // 2
        
        # Place resized image in center of padded canvas
        resized_images[i, pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    
    if not has_batch:
        return resized_images[0]
    return resized_images


# Make sure the user provided modality config is registered.
def load_modality_config(modality_config_path: str):
    import importlib
    import sys

    path = Path(modality_config_path)
    if path.exists() and path.suffix == ".py":
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
        print(f"Loaded modality config: {path}")
    else:
        raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")
    

def _slice_batch(batch, indices):
    """
    Slice a (possibly nested) model-input batch along the leading batch axis by ``indices``.

    process_input() returns a nested structure:
        {"video": {cam: (B,T,H,W,C)}, "state": {key: (B,T,D)},
         "language": {"...": [[p_0], ..., [p_{B-1}]]}}
    Sub-batching (selecting the env slots that need inference) must recurse into the nested dicts and
    slice arrays / the language list by ``indices``. A plain top-level ``v[indices]`` fails because the
    top-level values are dicts, not arrays (``dict[ndarray]`` -> "unhashable type: numpy.ndarray").
    """
    if isinstance(batch, dict):
        return {k: _slice_batch(v, indices) for k, v in batch.items()}
    if isinstance(batch, list):
        # language: one entry per batch element (outer dim == batch) -> select the requested envs.
        return [batch[int(i)] for i in indices]
    return batch[indices]  # numpy array, sliced along the leading (batch) axis


class B1KPolicyWrapper:
    """Adapts a GR00T policy to the OmniGibson evaluator's observation/action interface.

    Language prompt: either a fixed ``text_prompt`` for every request, or a
    ``task_prompts`` table (challenge ``task_id`` -> prompt) resolved per request
    from the ``task_id`` the evaluator includes in each observation (see
    ``gr00t.data.b1k_prompts`` and ``scripts/b1k/serve_b1k.py``). The prompt is
    fed under the policy's own language key, so it lands where training put it.
    """

    def __init__(
        self,
        policy: BasePolicy,
        embodiment_tag: EmbodimentTag,
        modality_config: dict,
        text_prompt: str | None = None,
        task_prompts: dict[int, str] | None = None,
        language_key: str | None = None,
        control_mode: str = "temporal_ensemble",
        obs_size: tuple[int, int] = (224, 224),
        action_horizon: int = 10,
    ) -> None:
        # Load robot config from registry
        self.robot = MODALITY_CONFIGS[embodiment_tag.value]
        self.robot_obs = ROBOT_OBS_CONFIGS[embodiment_tag.value] # Serving-only fields
        self.modality_config = modality_config
        self.policy = policy
        if text_prompt is None and not task_prompts:
            raise ValueError(
                "B1KPolicyWrapper needs a prompt: pass text_prompt (fixed for all requests) "
                "or task_prompts (task_id -> prompt, resolved from the evaluator's task_id obs)."
            )
        self.text_prompt = text_prompt
        self.task_prompts = dict(task_prompts) if task_prompts else {}
        self._logged_task_ids: set[int] = set()
        # Feed the prompt under the key the checkpoint was trained with.
        if language_key is None:
            language_key = getattr(policy, "language_key", None)
        if language_key is None:
            language_key = self.robot["language"].modality_keys[0]
        self.language_key = language_key
        self.control_mode = control_mode
        self.action_horizon = action_horizon
        self.obs_size = obs_size
        self.replan_interval = action_horizon  # K: replan every 10 steps
        self.max_len = 50  # how long the policy sequences are
        self.temporal_ensemble_max = 5  # max number of sequences to ensemble

        # Extract gripper indices from robot config (end-effectors to preserve)
        self.gripper_indices = []
        for i, action_key in enumerate(self.robot["action"].modality_keys):
            if self.robot["action"].action_configs[i].is_gripper:
                self.gripper_indices.extend(list(range(
                    self.modality_config["action"][action_key]["start"], self.modality_config["action"][action_key]["end"]
                )))

        # Vectorized action buffers (initialized on first call)
        self.batch_size = None
        self.action_buffer = None  # Shape: (batch, max_sequences, max_horizon, action_dim)
        self.sequence_indices = None  # Shape: (batch, max_sequences) - current position in each sequence
        self.sequence_lengths = None  # Shape: (batch, max_sequences) - total length of each sequence
        self.num_active_sequences = None  # Shape: (batch,) - number of active sequences per batch element
        self.step_counter = None  # Shape: (batch,)

    def reset(self):
        self.batch_size = None
        self.action_buffer = None
        self.sequence_indices = None
        self.sequence_lengths = None
        self.num_active_sequences = None
        self.step_counter = None

    def _ensure_batch_initialized(self, batch_size: int, action_dim: int = None):
        """Ensure buffers are initialized for the given batch size."""
        if self.batch_size != batch_size or self.action_buffer is None:
            self.batch_size = batch_size

            # For receeding_horizon: simple buffer
            # We'll set action_dim when we first see actions
            if action_dim is not None:
                self.action_buffer = np.zeros(
                    (batch_size, self.temporal_ensemble_max, self.max_len, action_dim), dtype=np.float32
                )
                self.sequence_indices = np.zeros((batch_size, self.temporal_ensemble_max), dtype=np.int32)
                self.sequence_lengths = np.zeros((batch_size, self.temporal_ensemble_max), dtype=np.int32)
                self.num_active_sequences = np.zeros(batch_size, dtype=np.int32)

            self.step_counter = np.zeros(batch_size, dtype=np.int32)

    def resolve_prompts(self, obs: dict, batch_size: int) -> list[str]:
        """Text prompt for each batch element: the fixed prompt, or the table entry for the
        challenge ``task_id`` the evaluator sends with the observation."""
        if self.text_prompt is not None:
            return [self.text_prompt] * batch_size

        task_ids = obs.get(TASK_ID_OBS_KEY)
        if task_ids is None:
            raise KeyError(
                f"Observation has no {TASK_ID_OBS_KEY!r} to resolve the prompt from; serve with a "
                "fixed prompt (--task-name / --text-prompt) or send task_id with each observation."
            )
        task_ids = np.asarray(task_ids).reshape(-1)
        if task_ids.size == 1:
            task_ids = np.repeat(task_ids, batch_size)
        elif task_ids.size != batch_size:
            raise ValueError(
                f"{TASK_ID_OBS_KEY!r} has {task_ids.size} entries for a batch of {batch_size}"
            )

        prompts = []
        for task_id in task_ids.tolist():
            task_id = int(task_id)
            if task_id not in self.task_prompts:
                raise KeyError(
                    f"task_id {task_id} is not in the prompt table (known: "
                    f"{sorted(self.task_prompts)}); check --tasks-file matches the evaluator."
                )
            if task_id not in self._logged_task_ids:
                self._logged_task_ids.add(task_id)
                logger.info(f"task_id {task_id} -> prompt {self.task_prompts[task_id]!r}")
            prompts.append(self.task_prompts[task_id])
        return prompts

    def process_input(self, obs: dict) -> tuple[dict, int]:
        """
        Process the input dictionary to match the expected input format for the model.
        Returns the processed input dictionary and batch size.
        """
        # Normalize proprio to (B, T=1, D). Insert the time axis at position 1 so a batched input
        # (B, D) keeps B as the batch (N robots) rather than being misread as T time-steps of 1 robot.
        prop_state = obs[f"{self.robot_obs['name']}::proprio"]
        if prop_state.ndim == 1:  # (D,) single env
            prop_state = prop_state[None, None, :]
        elif prop_state.ndim == 2:  # (B, D) batched: one obs per env
            prop_state = prop_state[:, None, :]
        batch_size = prop_state.shape[0]
        # Process camera images from robot config
        video = {}
        for camera_key in sorted(self.robot_obs["observation"].keys()):
            camera_obs = obs[self.robot_obs["observation"][camera_key]][..., :3]
            camera_obs = resize_with_pad(camera_obs, *self.obs_size)  # (H,W,C) or (B,H,W,C)
            # Normalize to (B, T=1, H, W, C) with the time axis at position 1 (same reasoning as proprio).
            if camera_obs.ndim == 3:  # (H, W, C) single env
                camera_obs = camera_obs[None, None, ...]
            elif camera_obs.ndim == 4:  # (B, H, W, C) batched
                camera_obs = camera_obs[:, None, ...]
            video[camera_key] = camera_obs  # Shape: (B, T, H, W, C)
        # Process state observations from robot config
        state = {}
        for state_key in sorted(self.modality_config["state"].keys()):
            start, end = self.modality_config["state"][state_key]["start"], self.modality_config["state"][state_key]["end"]
            state[state_key] = prop_state[..., start:end]
        # Language is list[list[str]] of shape (B, T=1), keyed by the checkpoint's language key.
        prompts = self.resolve_prompts(obs, batch_size)
        processed_input = {
            "video": video,
            "state": state,
            "language": {self.language_key: [[prompt] for prompt in prompts]},
        }
        return processed_input, batch_size

    def act_receeding_horizon(self, input_obs):
        """
        Receeding horizon: execute actions from current plan until exhausted, then replan.
        Uses vectorized buffers for efficient batch processing.
        """
        input_batch, batch_size = self.process_input(input_obs)
        if self.sequence_indices is None:
            needs_inference = np.ones(batch_size, dtype=bool)
        else:
            needs_inference = self.sequence_indices[:, 0] >= self.sequence_lengths[:, 0]

        if needs_inference.any():
            indices_needing_inference = np.where(needs_inference)[0]
            # Create sub-batch for elements that need inference (recurses into video/state/language).
            sub_batch = _slice_batch(input_batch, indices_needing_inference)
            target_action, _ = self.policy.get_action(sub_batch)  # (sub_batch_size, T, action_dim)
            target_action = np.concatenate(
                [target_action[key] for key in self.robot["action"].modality_keys], axis=-1
            )

            # Initialize buffers on first inference
            if self.action_buffer is None:
                action_dim = target_action.shape[2]
                self._ensure_batch_initialized(batch_size, action_dim)

            # Store actions in buffer
            seq_len = min(target_action.shape[1], self.max_len)
            self.action_buffer[indices_needing_inference, 0, :seq_len] = target_action[:, :seq_len]
            self.sequence_lengths[indices_needing_inference, 0] = seq_len
            self.sequence_indices[indices_needing_inference, 0] = 0

        # Extract next action for each batch element (fully vectorized!)
        batch_range = np.arange(batch_size)
        current_indices = self.sequence_indices[batch_range, 0]
        final_actions = self.action_buffer[batch_range, 0, current_indices]

        # Increment indices (vectorized)
        self.sequence_indices[:, 0] += 1

        return torch.from_numpy(final_actions)

    def act_receeding_temporal(self, input_obs):
        """
        Receeding temporal: infer every K steps and smooth actions across recent sequences.
        Uses vectorized buffers for efficient batch processing.
        """
        input_batch, batch_size = self.process_input(input_obs)

        # Initialize buffers on first call
        if self.action_buffer is None:
            # Need to infer once to get action_dim
            target_action, _ = self.policy.get_action(input_batch)
            target_action = np.concatenate([target_action[key] for key in self.robot["action"].modality_keys], axis=-1)
            action_dim = target_action.shape[2]
            self._ensure_batch_initialized(batch_size, action_dim)

            # Store first inference for all batch elements
            seq_len = min(target_action.shape[1], self.max_len)
            self.action_buffer[:, 0, :seq_len] = target_action[:, :seq_len]
            self.sequence_lengths[:, 0] = seq_len
            self.sequence_indices[:, 0] = 0
            self.num_active_sequences[:] = 1

        # Step 1: Run policy for elements that need replanning
        needs_replan = (self.step_counter % self.replan_interval) == 0
        if needs_replan.any():
            indices_needing_replan = np.where(needs_replan)[0]

            # Run inference only on sub-batch (recurses into video/state/language).
            sub_batch = _slice_batch(input_batch, indices_needing_replan)
            target_action, _ = self.policy.get_action(sub_batch)  # (sub_batch_size, T, action_dim)
            target_action = np.concatenate(
                [target_action[key] for key in self.robot["action"].modality_keys], axis=-1
            )

            # Add new sequences (vectorized where possible)
            seq_len = min(target_action.shape[1], self.max_len)

            # Handle elements that need shifting
            needs_shift = self.num_active_sequences[indices_needing_replan] >= self.temporal_ensemble_max
            if needs_shift.any():
                shift_indices = indices_needing_replan[needs_shift]
                # Vectorized shift for all elements that need it
                self.action_buffer[shift_indices, :-1] = self.action_buffer[shift_indices, 1:]
                self.sequence_indices[shift_indices, :-1] = self.sequence_indices[shift_indices, 1:]
                self.sequence_lengths[shift_indices, :-1] = self.sequence_lengths[shift_indices, 1:]

            # Calculate insert indices vectorized
            insert_indices = np.where(
                needs_shift, self.temporal_ensemble_max - 1, self.num_active_sequences[indices_needing_replan]
            )

            # Increment active sequences for elements that don't need shifting
            self.num_active_sequences[indices_needing_replan[~needs_shift]] += 1

            # Store new sequences (fully vectorized using advanced indexing)
            self.action_buffer[indices_needing_replan, insert_indices, :seq_len] = target_action[:, :seq_len]
            self.sequence_indices[indices_needing_replan, insert_indices] = 0
            self.sequence_lengths[indices_needing_replan, insert_indices] = seq_len

        # Step 2: Extract and ensemble current actions
        action_dim = self.action_buffer.shape[3]
        max_seq = self.temporal_ensemble_max
        seq_range = np.arange(max_seq)
        batch_range = np.arange(batch_size)
        # Create mask for active sequences (B, max_seq)
        active_mask = seq_range[None, :] < self.num_active_sequences[:, None]
        current_indices = self.sequence_indices[:, :max_seq]  # (B, max_seq)
        batch_idx = batch_range[:, None, None]
        seq_idx = seq_range[None, :, None]
        pos_idx = current_indices[:, :, None]
        actions_current = self.action_buffer[batch_idx, seq_idx, pos_idx].squeeze(-2)  # (B, max_seq, action_dim)

        k = 0.005
        exp_weights = np.exp(k * seq_range)[None, :]  # (1, max_seq)
        masked_weights = exp_weights * active_mask  # (B, max_seq)
        normalized_weights = masked_weights / masked_weights.sum(axis=1, keepdims=True)  # (B, max_seq)
        final_actions = (actions_current * normalized_weights[:, :, None]).sum(axis=1)  # (B, action_dim)

        # Preserve grippers from most recent rollout
        # Get the last active sequence index for each batch element
        last_seq_idx = self.num_active_sequences - 1
        for gripper_idx in self.gripper_indices:
            final_actions[:, gripper_idx] = actions_current[batch_range, last_seq_idx, gripper_idx]

        # Increment sequence indices
        self.sequence_indices[:, :max_seq] += 1

        # Check which sequences are still active (B, max_seq)
        still_active = self.sequence_indices[:, :max_seq] < self.sequence_lengths[:, :max_seq]
        still_active = still_active & active_mask  # Only consider currently active sequences
        # Drop exhausted sequences (compaction per batch element)
        for b in range(batch_size):
            active_in_slot = still_active[b, :]
            if not active_in_slot[: self.num_active_sequences[b]].all():
                # Some sequences exhausted, compact
                active_count = active_in_slot.sum()
                active_indices = np.where(active_in_slot)[0]
                self.action_buffer[b, :active_count] = self.action_buffer[b, active_indices]
                self.sequence_indices[b, :active_count] = self.sequence_indices[b, active_indices]
                self.sequence_lengths[b, :active_count] = self.sequence_lengths[b, active_indices]
                self.num_active_sequences[b] = active_count

        self.step_counter += 1
        return torch.from_numpy(final_actions)

    def act_temporal_ensemble(self, input_obs):
        """
        Temporal ensemble: infer at every step and smooth via exponentially weighted average of all recent action sequences.
        """
        batched = input_obs[f"{self.robot_obs['name']}::proprio"].ndim != 1
        input_batch, batch_size = self.process_input(input_obs)
        target_action, _ = self.policy.get_action(input_batch) # (B, T, action_dim)
        target_action = np.concatenate([target_action[key] for key in self.robot["action"].modality_keys], axis=-1)
        action_dim = target_action.shape[2]

        # Initialize buffers on first call
        if self.action_buffer is None:
            self._ensure_batch_initialized(batch_size, action_dim)
            self.num_active_sequences[:] = 0

        final_actions = np.empty((batch_size, action_dim))
        # Vectorized shift for all elements that need it, increment active counts otherwise
        needs_shift = self.num_active_sequences >= self.temporal_ensemble_max
        if needs_shift.any():
            shift_indices = np.where(needs_shift)[0]
            self.action_buffer[shift_indices, :-1] = self.action_buffer[shift_indices, 1:]
            self.sequence_indices[shift_indices, :-1] = self.sequence_indices[shift_indices, 1:]
            self.sequence_lengths[shift_indices, :-1] = self.sequence_lengths[shift_indices, 1:]
        self.num_active_sequences[~needs_shift] += 1

        # Insert new sequences to action buffer
        insert_indices = np.where(needs_shift, self.temporal_ensemble_max - 1, self.num_active_sequences - 1)
        seq_len = target_action.shape[1]
        batch_range = np.arange(batch_size)
        self.action_buffer[batch_range, insert_indices, :seq_len] = target_action
        self.sequence_indices[batch_range, insert_indices] = 0
        self.sequence_lengths[batch_range, insert_indices] = seq_len

        # Extract current actions
        seq_range = np.arange(self.temporal_ensemble_max)
        active_mask = seq_range[None, :] < self.num_active_sequences[:, None]
        current_indices = self.sequence_indices[:, : self.temporal_ensemble_max]  # (B, max_seq)
        batch_idx = batch_range[:, None, None]
        seq_idx = seq_range[None, :, None]
        pos_idx = current_indices[:, :, None]
        actions_current = self.action_buffer[batch_idx, seq_idx, pos_idx].squeeze(-2)  # (B, max_seq, action_dim)

        k = 0.005
        exp_weights = np.exp(k * seq_range)[None, :]  # (1, max_seq)
        masked_weights = exp_weights * active_mask  # (B, max_seq)
        normalized_weights = masked_weights / masked_weights.sum(axis=1, keepdims=True)  # (B, max_seq)
        final_actions = (actions_current * normalized_weights[:, :, None]).sum(axis=1)  # (B, action_dim)

        # Preserve grippers from most recent rollout
        for gripper_idx in self.gripper_indices:
            final_actions[:, gripper_idx] = target_action[:, 0, gripper_idx]

        # Increment all sequence indices
        self.sequence_indices[:, : self.temporal_ensemble_max] += 1

        if not batched:
            final_actions = final_actions[0]
        return torch.from_numpy(final_actions)

    def act(self, input_obs):
        """Dispatch to the appropriate action method based on control mode."""
        if self.control_mode == "receeding_temporal":
            return self.act_receeding_temporal(input_obs)
        elif self.control_mode == "receeding_horizon":
            return self.act_receeding_horizon(input_obs)
        elif self.control_mode == "temporal_ensemble":
            return self.act_temporal_ensemble(input_obs)
        else:
            raise ValueError(f"Unknown control mode: {self.control_mode}")
