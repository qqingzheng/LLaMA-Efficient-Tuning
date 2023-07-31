from llmtuner.tuner import get_train_args, run_pt, run_sft, run_rm, run_ppo
import wandb
import time
import copy
from dataclasses import asdict

def main():
    model_args, data_args, training_args, finetuning_args, general_args = get_train_args()
    
    # wandb
    if training_args.local_rank in [-1, 0]:
        wandb_config = copy.deepcopy(asdict(training_args))
        wandb_config.update(asdict(model_args))
        wandb_config.update(asdict(data_args))
        wandb_config.update(asdict(finetuning_args))
        wandb.init(
            project="GeneralModel",
            name= general_args.stage + "_" +finetuning_args.finetuning_type + "_" +time.strftime("%Y-%m-%d-%H-%M-%S"),
            config=wandb_config
        )
    if general_args.stage == "pt":
        run_pt(model_args, data_args, training_args, finetuning_args)
    elif general_args.stage == "sft":
        run_sft(model_args, data_args, training_args, finetuning_args)
    elif general_args.stage == "rm":
        run_rm(model_args, data_args, training_args, finetuning_args)
    elif general_args.stage == "ppo":
        run_ppo(model_args, data_args, training_args, finetuning_args)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
