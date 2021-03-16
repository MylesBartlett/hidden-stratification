import os
from pathlib import Path

import torch
import wandb

from stratification.harness import GEORGEHarness
from stratification.utils.parse_args import get_config
from stratification.utils.utils import init_cuda, set_seed


def main():
    config = get_config()
    use_cuda = config["use_cuda"] and torch.cuda.is_available()
    set_seed(config["seed"], use_cuda)  # set seeds for reproducibility
    init_cuda(config["deterministic"], config["allow_multigpu"])

    # Initialize wandb with online-logging as the default
    local_dir = Path(".", "local_logging")
    local_dir.mkdir(exist_ok=True)
    if config.get("log_offline", False):
        os.environ["WANDB_MODE"] = "dryrun"
    cluster_model_name = config["cluster_config"]["model"]
    if cluster_model_name == "topograd":
        if config["cluster_config"]["method_kwargs"].get("iters", -1) == 0:
            cluster_model_name = "tomato"

    wandb.init(
        entity="predictive-analytics-lab",
        project="hidden-stratification",
        dir=str(local_dir),
        config=config,
        reinit=True,
        group=config.get("group", f"{config['dataset']}/{cluster_model_name}"),
    )
    torch.multiprocessing.set_sharing_strategy("file_system")
    harness = GEORGEHarness(config, use_cuda=use_cuda)
    harness.save_full_config(config)

    first_mode = "erm" if (config["mode"] == "george") else config["mode"]
    dataloaders = harness.get_dataloaders(config, mode=first_mode)
    num_classes = dataloaders["train"].dataset.get_num_classes("superclass")
    model = harness.get_nn_model(config, num_classes=num_classes, mode=first_mode)

    activ_done = config["activations_dir"] != "NONE"
    rep_done = config["representation_dir"] != "NONE"
    activ_done = (
        activ_done or rep_done
    )  # don't need to get activations if we already have reduced ones

    # Train a model with ERM
    if activ_done and not (
        config["classification_config"]["eval_only"]
        or config["classification_config"]["save_act_only"]
    ):
        erm_dir = config["activations_dir"]
    else:
        if (
            config["classification_config"]["eval_only"]
            or config["classification_config"]["save_act_only"]
        ):
            erm_dir = config["activations_dir"]
            model_path = os.path.join(
                erm_dir, f'{config["classification_config"]["eval_mode"]}_model.pt'
            )
            print(f"Loading model from {model_path}...")
            model.load_state_dict(torch.load(model_path)["state_dict"])
        erm_dir = harness.classify(
            config["classification_config"], model, dataloaders, mode=first_mode
        )

    if not config["classification_config"]["bit_pretrained"] and not rep_done:
        model.load_state_dict(torch.load(os.path.join(erm_dir, "best_model.pt"))["state_dict"])

    set_seed(config["seed"], use_cuda)
    # Dimensionality-reduce the model activations
    if rep_done:
        reduction_dir = config["representation_dir"]
    else:
        reduction_model = harness.get_reduction_model(config, nn_model=model)
        reduction_dir = harness.reduce(
            config["reduction_config"],
            reduction_model,
            inputs_path=os.path.join(erm_dir, "outputs.pt"),
        )

    cluster_model = harness.get_cluster_model(config)
    harness.cluster(
        config["cluster_config"],
        cluster_model,
        inputs_path=os.path.join(reduction_dir, "outputs.pt"),
    )


if __name__ == "__main__":
    main()
