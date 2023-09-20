import numpy as np
from sklearn.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


def calcMetrics(true_data, eval_data, verbose=1):
    """calculate metrics from eval and true data

    Args:
        true_data (_type_): true data in 4D tensor
        eval_data (_type_): eval data in 4D tensor
        verbose (int, optional): Setting print info (0: None, 1: Overall average metrics, 2: Metrics of all images). Defaults to 1.

    Returns:
        Tuple: Tuple of three lists (PSNR, SSIM, RMSE)
    """
    num_image = len(true_data)
    psnr, ssim, rmse = [], [], []
    for i in range(num_image):
        # Get images
        im_true = np.squeeze(true_data[i, :, :, :]).numpy().astype(np.float32)
        im_eval = np.squeeze(eval_data[i, :, :, :]).numpy().astype(np.float32)

        # PSNR
        p = peak_signal_noise_ratio(im_true, im_eval)

        # RSME
        r = mean_squared_error(im_true.flatten(), im_eval.flatten(), squared=False)

        # SSIM
        val_min = im_true.min()
        val_range = im_true.max() - val_min
        im_true_norm = (im_true - val_min) / val_range
        im_eval_norm = (im_eval - val_min) / val_range
        im_max = max(im_eval_norm.max(), im_true_norm.max())
        im_min = min(im_eval_norm.min(), im_true_norm.min())
        s = structural_similarity(
            im_true_norm, im_eval_norm, data_range=im_max - im_min, channel_axis=0
        )

        if verbose == 2:
            print(f"#{i}: PSNR = {p:.1f}, SSIM = {s:.3f}, RMSE = {r:.3f}")

        psnr.append(p)
        ssim.append(s)
        rmse.append(r)

    if verbose > 0:
        print(
            f"PSNR = {np.mean(psnr):.1f}, SSIM = {np.mean(ssim):.3f}, RMSE = {np.mean(rmse):.3f}"
        )

    return psnr, ssim, rmse
