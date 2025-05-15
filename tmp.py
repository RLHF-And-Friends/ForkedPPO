import wandb
from tqdm import tqdm

api = wandb.Api()
runs = api.runs("karantonis-/FedRL_BeamRiderNoFrameskip-v4")

for run in tqdm(runs):
    print(run.name)

    run.config["mode"] = ""
    if run.config["use_fedavg"]:
        run.config["mode"] = "FedAvg"
    elif run.config["use_comm_penalty"]:
        run.config["mode"] = "PR"
    else:
        run.config["mode"] = "baseline"

    if run.config["objective_mode"] == 3:
        run.config["mode"] += "-PPO"
    elif run.config["objective_mode"] == 4:
        run.config["mode"] += "-MDPO"

    if run.config["use_fedavg"]:
        if "fedavg_average_weights_mode" not in run.config:
            run.config["fedavg_average_weights_mode"] = "classic-average"

        if run.config["fedavg_average_weights_mode"] == "weighted-average":
            run.config["mode"] += "-WeightedAvg"
        elif run.config["fedavg_average_weights_mode"] in ["classic-average", "classic-avg"]:
            run.config["mode"] += "-ClassicAvg"
    elif run.config["use_comm_penalty"]:
        if "fedrl_average_policies_mode" not in run.config:
            run.config["fedrl_average_policies_mode"] = "weighted-average"

        if run.config["fedrl_average_policies_mode"] == "weighted-average":
            run.config["mode"] += "-WeightedAvg"
        elif run.config["fedrl_average_policies_mode"] == "classic-average":
            run.config["mode"] += "-ClassicAvg"

    run.update()
