import argparse

from src.utils import utils
from .experiments import run_tests

def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='./results', help='Path to directory where to store test results.')

    args = parser.parse_args(argv)

    csv_path = utils.check_for_alzheimers_dataset()
    print("Using dataset:", csv_path)

    # run_tests(input_dir=csv_path, results_dir=args.output_dir)

if __name__ == "__main__":
    SystemExit(main())