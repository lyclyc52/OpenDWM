import argparse
import dwm.common
import json
import numpy as np
import os
import torch


def create_parser():
    parser = argparse.ArgumentParser(
        description="The script to run the diffusion model to generate data for"
        "detection evaluation.")
    parser.add_argument(
        "-c", "--config-path", type=str, required=True,
        help="The config to load the train model and dataset.")
    parser.add_argument(
        "-o", "--output-path", type=str, required=True,
        help="The path to save checkpoint files.")
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # set distributed training (if enabled), log, random number generator, and
    # load the checkpoint (if required).
    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(config["device"], local_rank)
        if config["device"] == "cuda":
            torch.cuda.set_device(local_rank)

        torch.distributed.init_process_group(backend=config["ddp_backend"])
    else:
        device = torch.device(config["device"])

    should_log = (ddp and local_rank == 0) or not ddp

    pipeline = dwm.common.create_instance_from_config(
        config["pipeline"], output_path=args.output_path, config=config,
        device=device)
    if should_log:
        print("The pipeline is loaded.")

    # load the dataset
    validation_dataset = dwm.common.create_instance_from_config(
        config["validation_dataset"])
    if ddp:
        validation_datasampler = \
            torch.utils.data.distributed.DistributedSampler(
                validation_dataset)
        validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset,
            **dwm.common.instantiate_config(config["validation_dataloader"]),
            sampler=validation_datasampler)
    else:
        validation_datasampler = None
        validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset,
            **dwm.common.instantiate_config(config["validation_dataloader"]))

    if should_log:
        print("The validation dataset is loaded with {} items.".format(
            len(validation_dataset)))

    if ddp:
        validation_datasampler.set_epoch(0)

    for batch in validation_dataloader:
        with torch.no_grad():
            pipeline_output = pipeline.inference_pipeline(batch, "pil")

        if "images" in pipeline_output:
            paths = [
                os.path.join(args.output_path, k["filename"])
                for i in batch["sample_data"]
                for j in i
                for k in j if not k["filename"].endswith(".bin")
            ]
            image_results = pipeline_output["images"]
            image_sizes = batch["image_size"].flatten(0, 2)
            for path, image, image_size in zip(paths, image_results, image_sizes):
                dir = os.path.dirname(path)
                os.makedirs(dir, exist_ok=True)
                image.resize(tuple(image_size.int().tolist()))\
                    .save(path, quality=95)

        if "raw_points" in pipeline_output:
            paths = [
                os.path.join(args.output_path, k["filename"])
                for i in batch["sample_data"]
                for j in i
                for k in j if k["filename"].endswith(".bin")
            ]
            raw_points = [
                j
                for i in pipeline_output["raw_points"]
                for j in i
            ]
            for path, points in zip(paths, raw_points):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                points = points.numpy()
                padded_points = np.concatenate([
                    points, np.zeros((points.shape[0], 2), dtype=np.float32)
                ], axis=-1)
                with open(path, "wb") as f:
                    f.write(padded_points.tobytes())
