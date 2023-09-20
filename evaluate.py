import datetime
import numpy as np
import torch
import os
import yaml

from utils.metrics import calcMetrics
from utils.loaddataset import SoundfieldDatasetLoader
from utils.modelhandler import loadtrainedmodel
from utils.util import load_config_yaml


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def denoise(im_noise: torch.tensor, config: dict) -> torch.tensor:
    """Denoise input image tensor by DNN defined by config

    Args:
        im_noise (torch.tensor): input noisy data
        config (dict): configuration dict of "model" and "weights_file"

    Returns:
        torch.tensor: deonised data
    """
    net = loadtrainedmodel(config["model"], config["weights_file"]).to(device)

    im_noise = im_noise.to(device)
    im_denoise = torch.zeros_like(im_noise)
    with torch.no_grad():
        # Denoise image by image, just to avoid memory overflow (16GB). If your memory is adequate, you can remove for loop
        for ii in range(im_noise.shape[0]):
            im_denoise[ii, ...] = net(torch.unsqueeze(im_noise[ii, ...], 0))
    im_denoise = im_denoise.cpu()

    return im_denoise


def save_results(config, im_denoise, PSNR, SSIM, RMSE, timestamp):
    """save config, denoise data, and metrics into files. A new folder with the name of the current time in "%Y%m%d_%H%M%S" format is generated under config["save_dir"] directory

    Args:
        config (_type_): configuration file
        im_denoise (_type_): denoised data
        PSNR (_type_): PSNR array save to PSNR.txt
        SSIM (_type_): SSIM array save to SSIM.txt
        RMSE (_type_): RMSE array save to RMSE.txt
    """
    save_dir = os.path.join(config["save_dir"], timestamp)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, "config.yml"), "w") as f:
        yaml.dump(config, f)
    np.savetxt(os.path.join(save_dir, "PSNR.txt"), PSNR)
    np.savetxt(os.path.join(save_dir, "SSIM.txt"), SSIM)
    np.savetxt(os.path.join(save_dir, "RMSE.txt"), RMSE)
    np.save(os.path.join(save_dir, "denoise"), im_denoise)


def main():
    print("--- running: evaluate.py ---")
    """Evaluate deep-sound-field-denoiser by test dataset"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"--- current timestamp: {timestamp}")

    # Load config file
    config_file_path = "config.yml"
    yaml_contents = load_config_yaml(config_file_path)
    config = yaml_contents["eval"]

    # Load dataset
    print("--- loading dataset: start ---")
    loader = SoundfieldDatasetLoader(config["dataset"])
    dataset = loader.load()
    im_noise, im_true = dataset[:]
    print("--- loading dataset: finished ---")

    print("--- sound-field denoising: start ---")
    # Denoising by DNN
    im_denoise = denoise(im_noise, config["network"])
    print("--- sound-field denoising: finished ---")

    # Calculate metrics
    PSNR, SSIM, RMSE = calcMetrics(im_true, im_denoise, verbose=2)

    # Save results
    if config["save_dir"]:
        save_results(config, im_denoise, PSNR, SSIM, RMSE, timestamp)
        print("--- save results: finished ---")


if __name__ == "__main__":
    main()
