import torch
import numpy as np
import lightkurve as lk
import tensorflow as tf

def pytorch_fold_and_bin(time, flux, period, n_bins, t0=0.0, device='cpu'):
    """
    Folds a time-series dataset at a single period and bins the folded data into equal-sized bins.

    Args:
        time (array): Array of time values.
        flux (array): Array of flux values corresponding to `time`.
        period (float): Period at which to fold the data.
        n_bins (int): Number of bins for the folded data.
        t0 (float): Reference time for folding.
        device (str): Device to use ('cpu' or 'cuda').

    Returns:
        np.ndarray: Binned averages of the folded light curve.
    """
    with torch.no_grad():
        # Filter valid (finite) data points
        valid_indices = np.isfinite(time) & np.isfinite(flux)
        time = time[valid_indices]
        flux = flux[valid_indices]

        # Ensure array length is compatible with binning
        points_per_bin = len(time) // n_bins
        resize_length = n_bins * points_per_bin
        time = time[:resize_length]
        flux = flux[:resize_length]

        # Convert data to PyTorch tensors
        tensor_args = dict(dtype=torch.float32, device=device)
        time_tensor = torch.as_tensor(time - t0, **tensor_args)  # Center time around t0
        flux_tensor = torch.as_tensor(flux, dtype=torch.float64, device=device)

        # Compute phase for folding
        period_tensor = torch.tensor(period, **tensor_args)
        phase = (time_tensor / period_tensor + 0.5) % 1.0

        # Sort flux values by phase
        sorted_indices = torch.argsort(phase)
        sorted_flux = flux_tensor[sorted_indices]

        # Bin the sorted flux values using convolution
        kernel = torch.ones((1, 1, 1, points_per_bin), dtype=torch.float64, device=device)
        binned_flux = torch.nn.functional.conv2d(
            sorted_flux.unsqueeze(0).unsqueeze(0),
            kernel,
            stride=(1, points_per_bin),
            padding=0
        )[0, 0]

        # Normalize by the number of points per bin
        binned_flux /= points_per_bin

        # Convert result to NumPy and return
        return binned_flux.cpu().numpy()
        
def load_and_clean(kic, eb_df):
    """
    Retrieves light curve and associated parameters of a Kepler target using the Kepler archive and MAST keplerebs record.

    Parameters:
        kic (int): Kepler Input Catalog index of the target.
        eb_df (DataFrame): DataFrame containing eclipsing binary parameters, including period and primary eclipse times.

    Returns:
        lc (LightCurve): Cleaned light curve as a LightKurve object.
        t (np.ndarray): Target time series.
        f (np.ndarray): Target flux intensity.
        ferr (np.ndarray): Target measurement uncertainty.
        p (float): Target eclipsing binary period in days.
        t0_1 (float): BKJD position of the primary eclipse in days.
    """
    # Load light curve
    search_result = lk.search_lightcurve(f'KIC {kic}', author='Kepler', exptime=1800)
    lc_collection = search_result.download_all(quality_bitmask="hard")
    lc_stitch = lc_collection.stitch()
    lc = lc_stitch.remove_nans()

    # Retrieve binary parameters
    params = eb_df.loc[eb_df['#KIC'] == kic]
    p = params['period'].item()
    t0_1 = params['bjd0'].item() - 54833

    # Clean flux values using 5-sigma clipping
    f = lc.flux.value
    clip_limit = 1 + 5 * np.std(f)
    mask = np.abs(f) < clip_limit
    t = lc.time.bkjd[mask]
    f = f[mask]
    ferr = lc.flux_err.value[mask]

    return lc, t, f, ferr, p, t0_1

def get_parameters(t, f, p, t0_1):
    """
    Retrieves orbital parameters (duration, t0) for both components of an eclipsing binary.

    Parameters:
        t (np.ndarray): Target time series.
        f (np.ndarray): Target flux intensity.
        p (float): Target eclipsing binary period in days.
        t0_1 (float): BKJD position of the primary eclipse.

    Returns:
        dur_prim (float): Duration of the primary eclipse in days.
        dur_sec (float or None): Duration of the secondary eclipse in days. None if no secondary is detected.
        t0_2 (float or None): BKJD position of the secondary eclipse. None if no secondary is detected.
    """
    num = int(p * 48)  # Number of bins for folding
    sample_fold = pytorch_fold_and_bin(t, f, p, num, t0=t0_1 + p / 4, device='cpu')
    gradient = np.gradient(np.gradient(sample_fold))

    # Extract primary and secondary eclipse indices
    idx1, idx2 = np.argpartition(gradient[:num // 2], 1)[:2]
    sec = gradient[num // 2:]
    idx3, idx4 = np.argpartition(sec, 1)[:2] + num // 2

    # Calculate durations and secondary eclipse parameters
    dur_prim = (np.abs(idx1 - idx2) + 22) / 48
    dur_sec = (np.abs(idx3 - idx4) + 22) / 48
    t0_2 = t0_1 + ((idx4 + idx3) - (idx2 + idx1)) / 96

    # Filter out grazing binaries
    if sec[idx3 - num // 2] > (-3 * np.std(sec)):
        return dur_prim, None, None

    return dur_prim, dur_sec, t0_2

def find_transits(time_norm, flux_norm, model, proc_hardware_name):
    """
    Uses an external convolutional neural network to search a detrended light curve for planetary transits.

    Parameters:
        time_norm (np.ndarray): Detrended target time series.
        flux_norm (np.ndarray): Detrended target flux intensity.
        model (tf.keras.Model): Pre-trained CNN model for transit detection.
        proc_hardware_name (str): Hardware configuration for TensorFlow (e.g., '/cpu:0' or '/gpu:0').

    Returns:
        segment_times (np.ndarray): Time indices of detected transits.
        y_pred (np.ndarray): Model confidence scores for each segment.
        sample_matrix (np.ndarray): Array of sample segments for model input.
    """
    segment_times = []
    segment_transform = []

    # Divide light curve into 240-point segments
    for i in range(len(flux_norm) // 240):
        segment = flux_norm[240 * i:240 * (i + 1)]
        segment_times.append(time_norm[240 * i])
        sigma = np.std(segment)
        segment_transform.append((1 - segment) / (2 * sigma))

    # Prepare input for the CNN
    sample_matrix = np.array(segment_transform).reshape(-1, 240, 1)

    # Predict transit probabilities
    with tf.device(proc_hardware_name):
        y_pred = model.predict(sample_matrix).flatten()

    return np.array(segment_times), y_pred, sample_matrix

def split_lc(time, flux, other_arrs=None, time_gap_delta=0.75, flux_gap_delta=0.05, min_chunk_len=30, verbose=False):
    """
    Splits light curve into segments based on data gaps.

    Parameters:
        time (np.ndarray): Target time series.
        flux (np.ndarray): Target flux intensity.
        other_arrs (list of np.ndarray, optional): Additional arrays to split. Defaults to None.
        time_gap_delta (float): Maximum allowable time gap in days.
        flux_gap_delta (float): Maximum allowable flux gap.
        min_chunk_len (int): Minimum chunk length in data points.
        verbose (bool): Whether to print debugging information.

    Returns:
        tuple: Split time, flux, and other arrays.
    """
    other_arrs = other_arrs or []

    # Eliminate NaNs
    valid_mask = np.isfinite(time) & np.isfinite(flux)
    time_clean, flux_clean = time[valid_mask], flux[valid_mask]
    other_clean = [arr[valid_mask] for arr in other_arrs]

    # Identify gap indices
    dt, df = np.abs(np.diff(time_clean)), np.abs(np.diff(flux_clean))
    split_ixs = np.where((dt > time_gap_delta) | (df > flux_gap_delta))[0] + 1

    if verbose:
        print(f"Time gaps: {np.count_nonzero(dt > time_gap_delta)}, Flux gaps: {np.count_nonzero(df > flux_gap_delta)}")

    # Split arrays
    time_chunks = np.split(time_clean, split_ixs)
    flux_chunks = np.split(flux_clean, split_ixs)
    other_chunks = [np.split(arr, split_ixs) for arr in other_clean]

    # Filter short chunks
    valid_chunks = [len(chunk) >= min_chunk_len for chunk in time_chunks]
    time_chunks = np.array(time_chunks, dtype=object)[valid_chunks]
    flux_chunks = np.array(flux_chunks, dtype=object)[valid_chunks]
    other_chunks = [[chunk for chunk, valid in zip(chunks, valid_chunks) if valid] for chunks in other_chunks]

    return time_chunks, flux_chunks, other_chunks, split_ixs

def perform_fft(data_raw):
    #Required: raw dataset array data_raw
    
    #Re-center data array around 0
    data = data_raw - np.mean(data_raw)
    
    #Perform FFT
    fft_result = np.fft.fft(data)
    # frequencies = np.fft.fftfreq(len(fft_result), 1)
    
    #Get magnitude of positive FFT portion
    magnitude = np.abs(fft_result)
    magnitude = magnitude[0:len(magnitude)//2]
    
    #Returns: complex ndarray (complex FFT) fft_result | array (frequency magnitude array) magnitude
    return fft_result, magnitude

def invert_fft(fft, fft_mag, bound):
    #Required: complex FFT array fft, FFT magnitude array fft_mag, integer cutoff (in pixels post-peak) bound
    
    #Find cutoff using numpy argmax
    bound_index = np.argmax(fft_mag) + bound
    
    #Cut FFT array 
    fft[bound_index:-bound_index] = 0
    
    #Restore to original with upper degrees trimmed off
    inverse = np.fft.ifft(fft)
    
    #Returns: complex ndarray (inverse reduced-degree FFT restoration) inverse
    return inverse

print("All functions loaded successfully.")