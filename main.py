import argparse
from pathlib import Path

from localization.config.presets import (
    config_5g,
    config_dichasus,
    config_passive,
    config_custom
)

def get_config_by_name(config_name):
    """Map config name to actual config object"""
    config_map = {
        "dichasus": config_dichasus,
        "5g": config_5g,
        "passive": config_passive,
        "custom": config_custom,
    }
    
    if config_name not in config_map:
        raise ValueError(
            f"Unknown config: {config_name}. "
            f"Available configs: {', '.join(config_map.keys())}"
        )
    
    return config_map[config_name]

def parse_args():

    parser = argparse.ArgumentParser(
    description=(
        "Localization Training Runner\n\n"
        "This script provides two main commands:\n\n"
        "1) train:  Run training experiments (UPGrad or PCGrad)\n"
        "2) analyze:  Analyze logs using LogAnalyzer\n\n"
        "Examples:\n"
        "  Train:\n"
        "    python script.py train --mode upgrad --config 5g\n\n"
        "  Analyze:\n"
        "    python script.py analyze --name dataset --start_version 1 --end_version 10\n"
    ),
    formatter_class=argparse.RawTextHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # -------- TRAIN BRANCH --------
    parser_train = subparsers.add_parser(
        "train",
        help="Run training experiments",
        description=(
            "Run training using different gradient strategies.\n\n"
            "Modes:\n"
            "  upgrad\n"
            "  pcgrad\n\n"
            "Example:\n"
            "  python script.py train --mode upgrad --config 5g --k 5"
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser_train.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["upgrad", "pcgrad", "NashMTL"],
        help=(
            "Select which function to run:\n"
            "  upgrad  - Run UPGrad experiments\n"
            "  pcgrad  - Run PCGrad experiments\n"
            "  NashMTL - Run NashMTL experiments\n"
        )
    )

    parser_train.add_argument(
        "--config",
        type=str,
        required=True,
        choices=["dichasus", "5g", "custom", "passive"],
        help=(
            "Select which config to use:\n"
            "  dichasus - Dichasus config\n"
            "  5g       - 5G config\n"
            "  custom   - config for custom datasets\n"
            "  passive  - Passive config"
        )
    )

    parser_train.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of iterations for PCGrad (default: 5)"
    )

    parser_train.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic training"
    )
    parser_train.set_defaults(func=run_train)   

    # -------- ANALYZE BRANCH --------
    parser_analyze = subparsers.add_parser(
    "analyze",
    help="Analyze logs",
    description=(
        "Analyze experiment logs using LogAnalyzer.\n\n"
        "Example:\n"
        "  python script.py analyze --name dataset --start_version 1 --end_version 10 --log_dir ./logs"
    ),
    formatter_class=argparse.RawTextHelpFormatter
    )

    parser_analyze.add_argument("--name", type=str, required=True, help="Dataset name")
    parser_analyze.add_argument("--start_version", type=int, required=True, help="Start version")
    parser_analyze.add_argument("--end_version", type=int, required=True, help="End version")
    parser_analyze.add_argument("--log_dir", type=str, default=None, help="Path to log directory")
    parser_analyze.set_defaults(func=run_analyze)  

    return parser.parse_args()


def run_train(args):
    try:
        config = get_config_by_name(args.config)
        if args.deterministic:
            config.deterministic = True
        print(f"Using config: {args.config}")
    except ValueError as e:
        print(f"Error: {e}")
        return

    if args.mode == "upgrad":
        from localization.training.strategies.upgrad import run_upgrad

        print("=" * 40)
        print(f"Running UPGrad experiments with {args.config}...")
        print("=" * 40)
        config.aggregator_name = "UPGrad"
        run_upgrad(config, K=args.k)

    elif args.mode == "pcgrad":
        from localization.training.strategies.pcgrad import run_pcgrad

        print("=" * 40)
        print(f"Running PCGrad experiments with {args.config}...")
        print("=" * 40)
        config.aggregator_name = "PCGrad"
        run_pcgrad(config, K=args.k)
    elif args.mode == "NashMTL":
        from localization.training.trainer import run

        print("=" * 40)
        print(f"Running NashMTL experiments with {args.config}...")
        print("=" * 40)
        config.aggregator_name = "NashMTL"
        run(config)

    print("\nAll selected experiments completed.")
    
    
def run_analyze(args):
    from localization.evaluation.reporting.lightning_logs import run_log_analysis

    run_log_analysis(
        dataset_name=args.name,
        log_dir=Path(args.log_dir) if args.log_dir else None,
        start_version=args.start_version,
        end_version=args.end_version
    )

    
def main():
    args = parse_args()
    args.func(args)   
    
if __name__ == "__main__":
    main()
