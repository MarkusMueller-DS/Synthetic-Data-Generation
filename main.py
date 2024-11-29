import os
import argparse
import importlib.util


def parse_arguments():
    """
    Function parses the aruments form the command line
    """
    parser = argparse.ArgumentParser(
        description="Train and sample from synthetic data models."
    )
    # Add arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model to use (e.g., 'model1').",
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Name of the dataset (e.g., 'adult')"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "sample"],
        help="Action to perform: 'train' or 'sample'.",
    )

    return parser.parse_args()


def execute_function(model, mode):
    if model == "ctgan":
        module_name = "sdg-models.ctgan.ctgan"

    try:
        module = importlib.import_module(module_name)
        function = getattr(module, "main")
    except ModuleNotFoundError:
        print(f"Module {module_name} not found.")
        exit(1)
    except AttributeError:
        print(f"Function 'main' not found in module {module_name}.")
        exit(1)

    return function


if __name__ == "__main__":
    args = parse_arguments()
    print(f"model: {args.model}, dataset: {args.dataset}, mode: {args.mode}")
    # Call the module
    main_fn = execute_function(args.model, args.mode)
    # call main funciton from imported module
    # provide args
    main_fn(args)
