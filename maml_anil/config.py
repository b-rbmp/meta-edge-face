import argparse
from dataclasses import dataclass
from typing import Union


@dataclass
class MAMLTrainingConfig:
    ways: int
    shots: int
    meta_learning_rate: float
    fast_learning_rate: float
    adaptation_steps: int
    meta_batch_size: int
    max_batch_size: int | None
    iterations: int
    use_cuda: int
    seed: int
    number_train_tasks: int
    number_valid_tasks: int
    number_test_tasks: int
    patience: int
    debug_mode: bool
    use_wandb: bool

    network: str = "edgeface_xs_gamma_06"
    fp16: bool = False
    embedding_size: int = 512
    loss_s: float = 64.0
    loss_m1: float = 1.0
    loss_m2: float = 0.0
    loss_m3: float = 0.4
    resume_from_checkpoint: bool = False

    run_str: str = "train"



def get_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser for MAML-based few-shot learning training.
    """
    parser = argparse.ArgumentParser(description='MAML-based few-shot learning')
    parser.add_argument('--ways', type=int, default=5, help='Number of classes (ways) per task')
    parser.add_argument('--shots', type=int, default=5, help='Number of examples (shots) per class')
    parser.add_argument('--meta-learning-rate', type=float, default=0.001, help='Meta-learning (outer loop) learning rate')
    parser.add_argument('--fast-learning-rate', type=float, default=0.1, help='Fast (inner loop) adaptation learning rate')
    parser.add_argument('--adaptation-steps', type=int, default=5, help='Number of adaptation steps')
    parser.add_argument('--meta-batch-size', type=int, default=128, help='Number of tasks per meta-batch')
    parser.add_argument('--max-batch-size', type=Union[int, None], default=None, help='Maximum batch size to be passed through the NN - Adjust if GPU memory insufficient')
    parser.add_argument('--iterations', type=int, default=10000, help='Number of meta-training iterations')
    parser.add_argument('--use-cuda', type=int, default=1, help='Use CUDA (1) or CPU (0)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--number-train-tasks', type=int, default=-1, help='Number of tasks to sample for meta-training')
    parser.add_argument('--number-valid-tasks', type=int, default=-1, help='Number of tasks to sample for meta-validation')
    parser.add_argument('--number-test-tasks', type=int, default=-1, help='Number of tasks to sample for meta-testing')
    parser.add_argument('--patience', type=int, default=200, help='Number of iterations to wait for improvement')
    parser.add_argument("--debug_mode", action=argparse.BooleanOptionalAction, default=False, help="Enable Debug Mode")
    parser.add_argument("--use_wandb", action=argparse.BooleanOptionalAction, default=False, help="Enable WandDB logging")

    parser.add_argument("--network", type=str, default="edgeface_xs_gamma_06", help="Network architecture to use")
    parser.add_argument("--embedding_size", type=int, default=512, help="Size of the embedding layer")
    parser.add_argument("--loss_s", type=float, default=5.0, help="Scaling factor for the loss function")
    parser.add_argument("--loss_m1", type=float, default=1.0, help="Margin 1 for the loss function")
    parser.add_argument("--loss_m2", type=float, default=0.0, help="Margin 2 for the loss function")
    parser.add_argument("--loss_m3", type=float, default=0.0, help="Margin 3 for the loss function")

    parser.add_argument("--run_str", type=str, default="run_without_script", help="Run string for the logfile")

    parser.add_argument("--resume_from_checkpoint", action=argparse.BooleanOptionalAction, default=False, help="Resume training from a checkpoint")
    return parser


def parse_args() -> MAMLTrainingConfig:
    """
    Parse command-line arguments and return a configuration object.
    """
    parser = get_parser()
    args = parser.parse_args()

    return MAMLTrainingConfig(
        ways=args.ways,
        shots=args.shots,
        meta_learning_rate=args.meta_learning_rate,
        fast_learning_rate=args.fast_learning_rate,
        adaptation_steps=args.adaptation_steps,
        meta_batch_size=args.meta_batch_size,
        max_batch_size=args.max_batch_size,
        iterations=args.iterations,
        use_cuda=args.use_cuda,
        seed=args.seed,
        number_train_tasks=args.number_train_tasks,
        number_valid_tasks=args.number_valid_tasks,
        number_test_tasks=args.number_test_tasks,
        patience=args.patience,
        debug_mode=args.debug_mode,
        use_wandb=args.use_wandb,
        network=args.network,
        embedding_size=args.embedding_size,
        loss_s=args.loss_s,
        loss_m1=args.loss_m1,
        loss_m2=args.loss_m2,
        loss_m3=args.loss_m3,
        resume_from_checkpoint=args.resume_from_checkpoint,
        run_str=args.run_str
    )
