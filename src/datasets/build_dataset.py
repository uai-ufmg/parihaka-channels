from argparse import ArgumentParser
import glob
import os
import csv


def is_ds_allowed(ds_name):
    ds = ds_name.split("-")[0]

    if args.include_ds is not None:
        if ds not in args.include_ds:
            return False

    if args.exclude_ds is not None:
        if ds in args.exclude_ds:
            return False

    return True


def get_data_paths(data_path, dataset_name, attr_name="seismic"):
    paths = []

    # get only seismic data, bc we just need one per horizon
    for path in glob.glob(os.path.join(data_path, f"*_{attr_name}.npy")):
        rel_path = path.split(args.base_dir)[1]
        if rel_path[0] == "/":
            rel_path = rel_path[1:]
        # get the path to both seismic and label
        rel_path = rel_path.split(f"_{attr_name}")[0]
        paths.append((dataset_name, rel_path))

    return paths


def get_data_paths_from_dirs(root, attr_name="seismic"):
    paths = []

    for path in glob.glob(os.path.join(root, "canal_*")):
        ds_name = path.split("canal_")[1]

        if is_ds_allowed(ds_name):
            ds_paths = get_data_paths(os.path.join(path, "images"), ds_name, attr_name)
            paths.extend(ds_paths)

    return paths


def write_to_file(out_file_path, data, header=("dataset_name", "relative_path")):
    with open(out_file_path, "w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(header)
        csv_writer.writerows(data)


if __name__ == "__main__":
    parser = ArgumentParser(description="Build the dataset to use.")
    parser.add_argument(
        "--base_dir",
        "-b",
        help="Directory where your datasets are stored. (Will not be stored in the relative path)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dataset_dir", "-d", help="Relative path of your dataset to the base dir.", type=str, required=True
    )
    parser.add_argument("--out_file", "-o", help="CSV file where your dataset will be stored.", type=str, required=True)
    parser.add_argument(
        "--dataset_name",
        "-n",
        help="Name your dataset will be saved as. (Required if data is not synthetic)",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--is_synth",
        "-s",
        help="If your dataset is synthetic or not. (Will loop through folders in the dataset_dir)",
        action="store_true",
    )
    parser.add_argument("--exclude_ds", "-e", help="Which of the synthetic datasets to ignore.", nargs="*")
    parser.add_argument("--include_ds", "-i", help="Which of the synthetic datasets to use.", nargs="*")
    parser.add_argument(
        "--attribute", "-a", help="Name of the attribute to look for in the dataset.", type=str, default="seismic"
    )

    args = parser.parse_args()

    assert (args.is_synth) or (args.dataset_name is not None), "'dataset_name' is required for non synthetic data"

    full_path = os.path.join(args.base_dir, args.dataset_dir)

    if args.is_synth:
        data_paths = get_data_paths_from_dirs(full_path, args.attribute)
    else:
        data_paths = get_data_paths(full_path, args.dataset_name, args.attribute)

    write_to_file(args.out_file, data_paths)
