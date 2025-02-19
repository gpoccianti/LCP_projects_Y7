import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
import yfinance as yf
import mplfinance as mpf
from hurst import compute_Hc
from scipy.fftpack import fft, ifft, fftfreq
from scipy.stats import norm
from pymannkendall import original_test, hamed_rao_modification_test



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
    """
    function to select the interval of data to consider
    """
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


def mann_kendall(chunk_size, start_index, end_index, df):
    """
    function to apply the mann kendall test to a given window
    """
    test_result=[]
    test_result_filtered=[]
    for i in range(0,int((end_index-start_index)/chunk_size)):
        
        s=start_index+i*chunk_size
        e=start_index+(i+1)*chunk_size
        
        prices=np.array(df.loc[df.index[s]:df.index[e],['HA_close']])
        prices=prices.reshape(chunk_size+1)
        time_indices=df.index[s:e+1]
        prices_filtered=filter_signal_by_auc(prices,time_indices,plot_spectrum=False)
        
        trend_test_modified_filtered = hamed_rao_modification_test(prices_filtered)
            
        if (trend_test_modified_filtered[0]=='no trend'):
            test_result_filtered.append(0)
        elif (trend_test_modified_filtered[0]=='increasing'):
            test_result_filtered.append(1)
        else:
            test_result_filtered.append(-1)
    return test_result_filtered

def heatmap(stride_values, start_index, end_index, chunk_size, df):
    """
    function to partially assess the stability of the test.
    This is done via applying the test on a rolling basis, but starting from a stride between intervals much bigger than 1,
    and then progressively decreasing it.
    """
    test_results_matrix = []
    for rolling_stride in stride_values:
        test_results = []
        z_stat=[]
        significance=[]
        for start_idx in range(start_index, end_index - chunk_size + 1, rolling_stride):
            end_idx = start_idx + chunk_size
    
            # Extract prices and time indices
            prices = np.array(df.loc[df.index[start_idx]:df.index[end_idx], ['HA_close']].copy())
            prices = prices.reshape(chunk_size + 1)
            time_indices = df.index[start_idx:end_idx + 1]
    
            # Apply FFT filtering
            prices_filtered = filter_signal_by_auc(prices, time_indices, plot_spectrum=False)
    
            # Apply Mann-Kendall Trend Test
            trend_test_modified = hamed_rao_modification_test(prices)
            significance.append(trend_test_modified[1])
            z_stat.append(trend_test_modified[3])
            
            # Save test results as numerical values for plotting
            if trend_test_modified[0] == 'no trend':
                test_results.append(0)
            elif trend_test_modified[0] == 'increasing':
                test_results.append(1)
            else:
                test_results.append(-1)
    
        # Pad test_results to match the maximum length, needed bc we'll convert test_results_matrix into an array --> uniform len needed
        max_length = max(len(tr) for tr in test_results_matrix) if test_results_matrix else len(test_results)
        while len(test_results) < max_length:
            test_results.append(np.nan)  # Use np.nan to indicate missing values
    
        test_results_matrix.append(test_results)
    
    test_results_matrix = np.array(test_results_matrix)
    
    # Plot the heatmap with custom settings
    plt.figure(figsize=(15, 6))
    custom_cmap = ListedColormap(['red', 'blue', 'green'])
    mesh = plt.pcolormesh(
        test_results_matrix, 
        cmap=custom_cmap, 
        linewidths=0.5,
        edgecolors='black',
        
    )    
    # Add a custom legend
    legend_patches = [
        mpatches.Patch(color='red', label='Decreasing Trend (-1)'),
        mpatches.Patch(color='blue', label='No Trend (0)'),
        mpatches.Patch(color='green', label='Increasing Trend (1)')
    ]
    plt.legend(handles=legend_patches, loc='upper right', title="Trend Type")
    plt.yticks(ticks=np.arange(0.5, len(stride_values)), labels=stride_values)
    plt.xlabel('# of Time Windows', fontsize=14)
    plt.ylabel('Stride Length',fontsize=14)
    plt.title('Trend Detection Heatmap vs. Stride Length',fontsize=16)
    plt.show()
    
def z_statistic(start_index, end_index, df, gif=False):
    '''
    function to apply the Mann Kendall test on a rolling basis and to properly display the result
    '''
    def compute_stat(x):
        return hamed_rao_modification_test(x)[3]
    def compute_significance(x):
        return hamed_rao_modification_test(x)[1]
    def p_value_to_z(p_value): #convert the pvalue to the corrisponding value of the statistic
        return norm.ppf(1 - p_value / 2)
    
    
    df_filtered = df['HA_close'][start:end].copy()
    df_filtered = filter_signal_by_auc(df_filtered.values, df_filtered.index)
    df_filtered = df_filtered[start_index+10:end_index-10]
    df_filtered=df_filtered.to_frame(name='HA_close') 
    
    df_filtered['stat']=df_filtered['HA_close'].rolling(window=chunk_size, min_periods=3).apply(compute_stat)
    df_filtered['significance']=df_filtered['HA_close'].rolling(window=chunk_size, min_periods=3).apply(compute_significance)
    
    significant=df_filtered['stat'][df_filtered['significance'] > 0]
    not_significant=df_filtered['stat'][df_filtered['significance'] < 1]
    z_bound=p_value_to_z(0.05) #0.05 because we choose a CI of 95%

    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={'height_ratios': [3,1]})
    fig.tight_layout()
    
    ax[0].plot(df_filtered['HA_close'])
    xcoords=range(start_index,end_index,chunk_size)
    xcoords=df.index[xcoords]
    for xc in xcoords:
        for a in ax:
            a.axvline(x=xc, color='grey', linestyle='--', linewidth=1, alpha=.5)
            a.grid(visible=False)
    ax[0].set_ylabel('price')
            
    ax[1].scatter(df_filtered['stat'].index[:],df_filtered['stat'], s=.5, label='z-statistic')
    ax[1].axhline(y=z_bound, color='black', linestyle='--', alpha=.5)
    ax[1].axhline(y=-z_bound, color='black', linestyle='--', alpha=.5, label='CI=95%')
    ax[1].axhspan(z_bound, ax[1].get_ylim()[1], facecolor='green', alpha=0.2)
    ax[1].axhspan(-z_bound, z_bound, facecolor='blue', alpha=0.2)
    ax[1].axhspan(ax[1].get_ylim()[0], -z_bound, facecolor='red', alpha=0.2)
    ax[1].set_ylabel('z-statistic')
    ax[1].legend()

    if(gif==True):
        fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={'height_ratios': [3,1]})
        fig.tight_layout()
        
        def update(frame):
            ax[0].cla()  # Clear previous plot
            ax[1].cla()  # Clear previous plot
        
            ax[0].set_ylabel('price')
            ax[1].set_ylabel('z-statistic')
           
            # Highlight rolling window in ax[0] (gray shading and vertical lines)
            start_idx = frame
            end_idx = start_idx + chunk_size
            if end_idx < len(df_filtered):
                ax[0].plot(df_filtered['HA_close'], label="HA_close")
                ax[0].axvspan(df_filtered.index[start_idx], df_filtered.index[end_idx], color='gray', alpha=0.3)
                ax[0].axvline(df_filtered.index[start_idx], color='red', linestyle='--', linewidth=1)
                ax[0].axvline(df_filtered.index[end_idx], color='red', linestyle='--', linewidth=1)
            
            # In ax[1], plot the computed stats and show window as well
            ax[1].scatter(df_filtered.index[:end_idx], df_filtered['stat'][:end_idx], s=0.5, label="z-statistic")
            if end_idx < len(df_filtered):
                ax[1].axvspan(df_filtered.index[start_idx], df_filtered.index[end_idx], color='gray', alpha=0.3)
            # Show the z-boundaries for significance in ax[1]
            ax[1].axhline(y=z_bound, color='black', linestyle='--', alpha=0.5)
            ax[1].axhline(y=-z_bound, color='black', linestyle='--', alpha=0.5, label='no trend (CI=95%)')
            ax[1].axhspan(z_bound, ax[1].get_ylim()[1], facecolor='green', alpha=0.2)
            ax[1].axhspan(-z_bound, z_bound, facecolor='blue', alpha=0.2)
            ax[1].axhspan(ax[1].get_ylim()[0], -z_bound, facecolor='red', alpha=0.2)        
            ax[1].legend()
            
            return ax
    
        # Set the number of frames based on the rolling window size and length of df_filtered['stat']
        num_frames = len(df_filtered['stat']) - chunk_size
        
        # Animation setup
        ani = FuncAnimation(fig, update, frames=range(0, num_frames), interval=200, repeat=False)
        
        fig.suptitle('Prices vs z-stat')
        ax[0].set_ylabel('price')
        ax[1].set_ylabel('z-statistic')
        
        # Save the animation as a gif
        ani.save('rolling_computation_evolution.gif', writer='Pillow', fps=9)
