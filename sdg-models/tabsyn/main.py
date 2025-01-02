import sys
import torch
from utils import execute_function, get_args
from process_dataset import process_data

if __name__ == "__main__":
    args = get_args()

    dataname = args.dataname
    print(dataname)

    DATA_PATH = "../../data"
    INFO_PATH = "../../data/info"

    # process data
    if args.mode == "process":
        process_data(INFO_PATH, DATA_PATH, dataname)

    if torch.cuda.is_available():
        args.device = f"cuda:{1}"
    else:
        args.device = "cpu"
    print("cuda device:", args.device)

    if not args.save_path:
        args.save_path = f"{DATA_PATH}/synthetic/{dataname}/tabsyn.csv"
    main_fn = execute_function(args.method, args.mode)

    main_fn(args)
