import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import mplfinance as mpf
from hurst import compute_Hc
from scipy.fftpack import fft, ifft, fftfreq



def import_data(file_path):
    """
    function to import the dataset
    """
    data = pd.read_csv(file_path, sep="\t")
    # Combine the DATE and TIME columns into a single datetime column
    data['datetime'] = pd.to_datetime(data['<DATE>'] + ' ' + data['<TIME>'])
    # Set datetime as the index (optional but helpful for time series)
    data.set_index('datetime', inplace=True)
    return data, data['<CLOSE>'] #Returns both the entire dataset and just close

def interval_selector(n_s, n_e):
    starting_position=96*n_s #96 corresponds to 1 day of data
    if (starting_position!=0):
        start = df.index[starting_position]   
    else:
        start=pd.to_datetime('2020.09.24 08:45:00')  #<-->if n==0 start from the beginning
    start_index=df.index.get_loc(start) #to get the numeric index
    end_index=96*n_e
    if (end_index!=0):
        end = df.index[end_index]
        print(end)
    else:
        end = pd.to_datetime('2020.10.12 08:45:00')
        end_index=df.index.get_loc(end)
    return start_index, end_index

def plot_close(data, start=None, end=None):
    """
    Function to plot the closing prices over a specified time period.

    Parameters:
    - data: pandas Series, the time series of closing prices
    - start: str, datetime, or int, start of the time period (optional)
    - end: str, datetime, or int, end of the time period (optional)
    """
    # Filter data based on the type of start and end
    if start is not None or end is not None:
        if isinstance(start, (str, pd.Timestamp)) or isinstance(end, (str, pd.Timestamp)):
            # Use .loc for datetime filtering
            data = data.loc[start:end]
        elif isinstance(start, int) or isinstance(end, int):
            # Use positional slicing
            data = data[start:end]

    # Plot the closing prices
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data["<CLOSE>"], label='Closing Price', color='blue')

    # Customize the plot
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Closing Price', fontsize=12)
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()
    



def filter_signal(sig, datetime_index, time_step=15*60, freq_factor=100):
    """
    Filters the input signal by removing high-frequency components based on the peak frequency.
    
    Parameters:
    sig (array-like): The signal to be filtered.
    datetime_index (pd.DatetimeIndex): The datetime index for the signal.
    time_step (float): The time step (in seconds) between each data point (default is 15 minutes).
    freq_factor (float): The factor used to determine the cutoff frequency for filtering (default is 10).
    
    Returns:
    filtered_sig_series (pd.Series): A Pandas Series containing the filtered signal with the datetime index.
    """
    # The corresponding frequencies
    sample_freq = fftfreq(sig.size, d=time_step)

    # Perform the FFT
    sig_fft = fft(np.array(sig))
    power = np.abs(sig_fft)

    # Plot the power spectrum
    plt.figure(figsize=(6, 5))
    plt.plot(sample_freq, np.log(power))
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power')
    plt.title("Power spectrum of the signal")

    # Find the peak frequency (only considering positive frequencies)
    pos_mask = np.where(sample_freq > 0)
    freqs = sample_freq[pos_mask]
    peak_freq = freqs[power[pos_mask].argmax()]

    # Filter high frequencies based on the peak frequency
    high_freq_fft = sig_fft.copy()
    high_freq_fft[np.abs(sample_freq) > freq_factor * peak_freq] = 0  # Apply frequency filtering
    filtered_sig = ifft(high_freq_fft)

    # Convert filtered signal to real values (the result of IFFT might be complex)
    filtered_sig = np.real(filtered_sig)

    # Create a Pandas Series with the datetime index and the filtered signal
    filtered_sig_series = pd.Series(filtered_sig, index=datetime_index)

    return filtered_sig_series


def filter_signal_by_auc(sig, datetime_index, time_step=15*60, discard_fraction=0.1,plot_spectrum=False):
    """
    Filters the input signal by removing high-frequency components based on the area under the curve (AUC).

    Parameters:
    sig (array-like): The signal to be filtered.
    datetime_index (pd.DatetimeIndex): The datetime index for the signal.
    time_step (float): The time step (in seconds) between each data point (default is 15 minutes).
    discard_fraction (float): Fraction of the total power to discard (default is 10%).

    Returns:
    filtered_sig_series (pd.Series): A Pandas Series containing the filtered signal with the datetime index.
    """
    # The corresponding frequencies
    sample_freq = fftfreq(sig.size, d=time_step)
    pos_mask = sample_freq > 0
    freqs = sample_freq[pos_mask]

    # Perform the FFT
    sig_fft = fft(np.array(sig))
    power = np.abs(sig_fft) ** 2

    # Compute cumulative power
    cumulative_power = np.cumsum(power[pos_mask])
    total_power = cumulative_power[-1]
    target_power = (1 - discard_fraction) * total_power

    # Determine the cutoff frequency
    cutoff_idx = np.searchsorted(cumulative_power, target_power)
    cutoff_freq = freqs[cutoff_idx]

    # Apply the frequency filter
    filtered_fft = sig_fft.copy()
    filtered_fft[np.abs(sample_freq) > cutoff_freq] = 0
    filtered_sig = ifft(filtered_fft)

    # Convert filtered signal to real values (IFFT output might be complex)
    filtered_sig = np.real(filtered_sig)

    # Create a Pandas Series with the datetime index and the filtered signal
    filtered_sig_series = pd.Series(filtered_sig, index=datetime_index)
    
    if(plot_spectrum==True):
        # Plot for visualization
        plt.figure(figsize=(6, 5))
        plt.plot(freqs, np.cumsum(power[pos_mask]) / total_power, label="Cumulative Power")
        plt.axvline(cutoff_freq, color='red', linestyle='--', label=f"Cutoff: {cutoff_freq:.2e} Hz")
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Normalized Cumulative Power')
        plt.title("Cumulative Power Spectrum")
        plt.legend()
        plt.show()

    return filtered_sig_series