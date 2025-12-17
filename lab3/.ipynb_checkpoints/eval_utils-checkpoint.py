import numpy as np

def mse(x, y):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    return float(np.mean((x - y) ** 2))

def psnr(x, y, data_range=1.0, eps=1e-12):
    """
    PSNR in dB. Assumes images are in [0, data_range].
    """
    m = mse(x, y)
    return float(10.0 * np.log10((data_range ** 2) / (m + eps)))

def evaluate_denoiser(model, x_noisy, x_clean, clip_pred=True):
    """
    Returns a dict with MSE and PSNR for:
      - noisy vs clean
      - denoised vs clean
    """
    pred = model.predict(x_noisy, verbose=0)
    if clip_pred:
        pred = np.clip(pred, 0.0, 1.0)

    results = {
        "mse_noisy": mse(x_noisy, x_clean),
        "psnr_noisy": psnr(x_noisy, x_clean),
        "mse_denoised": mse(pred, x_clean),
        "psnr_denoised": psnr(pred, x_clean),
    }
    return results
