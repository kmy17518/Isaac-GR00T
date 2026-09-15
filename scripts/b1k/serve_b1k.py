from dataclasses import dataclass
import json
import os

from gr00t.data.b1k_prompts import (
    DEFAULT_TASKS_FILE,
    PromptSource,
    find_b1k_task,
    load_b1k_tasks,
    prompt_source_from_language_key,
)
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.eval_b1k_wrapper import B1KPolicyWrapper, load_modality_config
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.websocket_b1k_server import WebsocketPolicyServer
import tyro


DEFAULT_MODEL_SERVER_PORT = 8000


@dataclass
class ServerConfig:
    """Configuration for running the Groot N1.5 inference server."""

    # Gr00t policy configs
    model_path: str
    """Path to the model checkpoint directory"""

    modality_config_path: str
    """Path to the modality configuration python file"""

    embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
    """Embodiment tag"""

    device: str = "cuda"
    """Device to run the model on"""

    control_mode: str = "temporal_ensemble"
    """Control mode during inference."""

    # Language prompt configs (see gr00t.data.b1k_prompts)
    task_name: str | None = None
    """BEHAVIOR task to serve, e.g. ``turning_on_radio``; fixes the prompt for every request.
    Default: resolve the prompt per request from the ``task_id`` the evaluator sends."""

    tasks_file: str = str(DEFAULT_TASKS_FILE)
    """``tasks.jsonl`` with the challenge task names and descriptions (repo copy of the
    dataset's ``meta/tasks.jsonl``; a dataset's own file works too)."""

    prompt_source: PromptSource | None = None
    """Kind of text to prompt with: ``task_description`` (natural language) or ``task_name``
    (snake_case). Default: the kind the checkpoint was trained on, read from its language key."""

    text_prompt: str | None = None
    """Explicit prompt for every request; bypasses --task-name / task_id / --tasks-file."""

    # Server configs
    host: str = "127.0.0.1"
    """Host address for the server"""

    port: int = DEFAULT_MODEL_SERVER_PORT
    """Port number for the server"""

    strict: bool = True
    """Whether to enforce strict input and output validation"""


def resolve_prompts(
    config: ServerConfig, language_key: str
) -> tuple[str | None, dict[int, str] | None]:
    """Turn the CLI prompt options into ``(text_prompt, task_prompts)`` for B1KPolicyWrapper.

    ``language_key`` is the checkpoint's language modality key; it decides which kind of
    text (description vs. name) to feed unless ``--prompt-source`` overrides it.
    """
    if config.text_prompt is not None:
        return config.text_prompt, None

    prompt_source = config.prompt_source
    if prompt_source is None:
        try:
            prompt_source = prompt_source_from_language_key(language_key)
        except ValueError as e:
            raise ValueError(
                f"{e}. Cannot infer which kind of text this checkpoint was trained on; "
                "pass --prompt-source or --text-prompt explicitly."
            ) from e
    print(f"  Prompt source: {prompt_source} (checkpoint language key: {language_key})")

    tasks = load_b1k_tasks(config.tasks_file)
    if config.task_name is not None:
        text_prompt = find_b1k_task(tasks, config.task_name).prompt(prompt_source)
        print(f"  Prompt (fixed, task {config.task_name}): {text_prompt!r}")
        return text_prompt, None

    task_prompts = {task_index: task.prompt(prompt_source) for task_index, task in tasks.items()}
    print(f"  Prompt: resolved per request from the evaluator's task_id ({len(task_prompts)} tasks)")
    return None, task_prompts


def main(config: ServerConfig):
    print("Starting GR00T inference server...")
    print(f"  Embodiment tag: {config.embodiment_tag}")
    print(f"  Model path: {config.model_path}")
    print(f"  Modality config path: {config.modality_config_path}")
    print(f"  Device: {config.device}")
    print(f"  Host: {config.host}")
    print(f"  Port: {config.port}")

    # check if the model path exists
    if config.model_path.startswith("/") and not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model path {config.model_path} does not exist")

    # load modality config if provided
    assert os.path.exists(config.modality_config_path) and config.modality_config_path.endswith(".py"), (
        f"Modality config path {config.modality_config_path} does not exist or is not a Python file"
    )
    load_modality_config(config.modality_config_path)
    modality_json = config.modality_config_path.replace(".py", ".json")
    assert os.path.exists(modality_json), (f"Modality config JSON file {modality_json} does not exist. ")
    with open(modality_json, "r") as f:
        modality_config = json.load(f)

    # Create and start the server
    policy = Gr00tPolicy(
        embodiment_tag=config.embodiment_tag,
        model_path=config.model_path,
        device=config.device,
        strict=config.strict,
    )

    # Prompt with the same kind of text the checkpoint was trained on.
    text_prompt, task_prompts = resolve_prompts(config, policy.language_key)

    # Wrap with B1K policy wrapper
    policy = B1KPolicyWrapper(
        policy=policy,
        embodiment_tag=config.embodiment_tag,
        modality_config=modality_config,
        text_prompt=text_prompt,
        task_prompts=task_prompts,
        control_mode=config.control_mode,
    )

    server = WebsocketPolicyServer(
        policy=policy,
        host=config.host,
        port=config.port,
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")


if __name__ == "__main__":
    config = tyro.cli(ServerConfig)
    main(config)
