import os
import pandas as pd
from datetime import datetime
from pathlib import Path
import re
import time
import logging
from ta import TechnicalIndicators  # Import the user's TechnicalIndicators class

# Custom dummy TQDM replacement that does nothing but maintains the interface
class DummyTQDM:
    """A no-operation replacement for tqdm that maintains the same interface."""
    def __init__(self, iterable=None, **kwargs):
        self.iterable = iterable if iterable is not None else []
        self.kwargs = kwargs  # Store parameters for compatibility
        
    def __enter__(self):
        return self  # Return self to allow context manager usage
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass  # No-op for cleanup
    
    def __iter__(self):
        for obj in self.iterable:
            yield obj
            
    def update(self, n=1):
        pass
        
    def close(self):
        pass
        
    def set_description(self, desc="", refresh=True):
        pass

class CryptoTAProcessor:
    """
    A class for processing cryptocurrency CSV data and calculating technical indicators
    using the TechnicalIndicators class from the ta module.
    """
    
    def __init__(self, csv_filename, ta_indicators=None, output_dir='processed_data', 
                 output_filename=None, start_date=None, price_source='close', verbose=False):
        """
        Initialize the Technical Analysis processor
        
        Parameters:
        csv_filename (str): Path to the CSV file containing crypto market data
        ta_indicators (list): List of dictionaries defining technical indicators to calculate
                             Each dict should have 'name' and 'periods' keys
                             Optionally can include 'price_source' to override the default
        output_dir (str): Directory to save the processed data
        output_filename (str): Custom filename for the output file (without directory path)
                              If None, will generate based on input filename
        start_date (str): Start date filter in 'YYYY-MM-DD' format
        price_source (str): Default source price calculation for technical indicators. 
                           Options: 'close', 'formula1', 'formula2', 'formula3', 'formula4', 
                           'conditional1', 'conditional2'
        verbose (bool): Whether to display progress bars and log messages (default: True)
        """
        self.csv_filename = csv_filename
        self.output_dir = output_dir
        self.output_filename = output_filename
        self.start_date = start_date
        self.price_source = price_source
        self.verbose = verbose
        
        # Configure logging based on verbosity
        if verbose:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            self.logger = logging.getLogger("CryptoTAProcessor")
        else:
            # Create a null logger that won't output anything
            self.logger = logging.getLogger("CryptoTAProcessor")
            self.logger.addHandler(logging.NullHandler())
            self.logger.propagate = False
        
        # Create progress bar function based on verbosity
        if verbose:
            from tqdm.notebook import tqdm as tqdm_notebook
            self.tqdm = tqdm_notebook
        else:
            self.tqdm = DummyTQDM
        
        if verbose:
            self.logger.info(f"Initializing CryptoTAProcessor for file: {csv_filename}")
            self.logger.info(f"Output directory: {output_dir}")
            if output_filename:
                self.logger.info(f"Custom output filename: {output_filename}")
            self.logger.info(f"Start date filter: {start_date if start_date else 'None (processing all data)'}")
            self.logger.info(f"Default price source for indicators: {price_source}")
        
        # Default technical indicators if none specified
        self.ta_indicators = ta_indicators or [
            {'name': 'SMA', 'periods': [20, 50, 200]},
            {'name': 'EMA', 'periods': [9, 21, 55]},
            {'name': 'RSI', 'periods': [14]},
            {'name': 'MACD', 'periods': []}
        ]
        
        # Log the indicators that will be calculated (only if verbose)
        if verbose:
            indicator_summary = []
            for ind in self.ta_indicators:
                ind_price_source = ind.get('price_source', self.price_source)
                periods_info = f"{len(ind.get('periods', []))} periods"
                indicator_summary.append(f"{ind['name']}({periods_info}, src={ind_price_source})")
            
            self.logger.info(f"Technical indicators to calculate: {', '.join(indicator_summary)}")
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(exist_ok=True)
        if verbose:
            self.logger.info(f"Ensured output directory exists: {output_dir}")
    
    def load_data(self):
        """
        Load CSV data, identify OHLCV columns, and filter by start date
        
        Returns:
        pandas.DataFrame: Preprocessed DataFrame with OHLCV columns
        """
        self.logger.info(f"Loading data from: {self.csv_filename}")
        start_time = time.time()
        
        try:
            # Read CSV file
            self.logger.info("Reading CSV file...")
            df = pd.read_csv(self.csv_filename)
            self.logger.info(f"CSV loaded with {len(df)} rows and {len(df.columns)} columns")
            
            if len(df.columns) == 0:
                raise ValueError("Empty CSV file")
                
            # Determine column for timestamp (first column)
            timestamp_col = df.columns[0]
            self.logger.info(f"Using '{timestamp_col}' as timestamp column")
            
            # Convert timestamp column to datetime if it's not already
            self.logger.info("Converting timestamp column to datetime...")
            if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                self.logger.info("Timestamp column converted to datetime")
            
            # Set timestamp as index
            df.set_index(timestamp_col, inplace=True)
            self.logger.info(f"Set '{timestamp_col}' as DataFrame index")
            
            # Filter by start date if specified
            if self.start_date:
                self.logger.info(f"Filtering data from start date: {self.start_date}")
                start_date = pd.to_datetime(self.start_date)
                original_rows = len(df)
                df = df[df.index >= start_date]
                filtered_rows = len(df)
                self.logger.info(f"Filtered data from {start_date}: {original_rows - filtered_rows} rows removed, {filtered_rows} rows remaining")
            
            # Match OHLCV columns using regex patterns (case-insensitive)
            self.logger.info("Identifying OHLCV columns...")
            ohlcv_mapping = {}
            ohlcv_patterns = {
                'open': r'open|^o$',
                'high': r'high|^h$',
                'low': r'low|^l$',
                'close': r'close|^c$',
                'volume': r'volume|^v$'
            }
            
            for std_name, pattern in ohlcv_patterns.items():
                for col in df.columns:
                    if re.search(pattern, col.lower()):
                        ohlcv_mapping[std_name] = col
                        break
            
            # Check if required columns are present
            required_cols = ['open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in ohlcv_mapping]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Extract only the OHLCV columns
            self.logger.info("Extracting and standardizing OHLCV columns...")
            ohlcv_df = pd.DataFrame(index=df.index)
            for standard_name, orig_name in ohlcv_mapping.items():
                ohlcv_df[standard_name] = df[orig_name]
                self.logger.info(f"Mapped '{orig_name}' to standardized '{standard_name}'")
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Data loading completed in {elapsed_time:.2f} seconds")
            self.logger.info(f"Final dataframe shape: {ohlcv_df.shape}")
            
            return ohlcv_df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise ValueError(f"Error loading data: {str(e)}")

    def _get_price_data(self, df, source_type='close'):
        """
        Calculate the price data based on the selected price source formula
        
        Parameters:
        df (pandas.DataFrame): DataFrame with OHLCV data
        source_type (str): Price source calculation type
        
        Returns:
        pandas.Series: Calculated price data based on the selected formula
        """
        self.logger.info(f"Calculating price data using source: {source_type}")
        
        # Check if all required columns are available
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            self.logger.warning(f"Not all required columns {required_cols} available. Falling back to close price.")
            return df['close']
        
        # Shortcuts for readability
        open_price = df['open']
        high_price = df['high']
        low_price = df['low']
        close_price = df['close']
        
        if source_type == 'close':
            # Default: just use close price
            self.logger.info("Using close price")
            return close_price
            
        elif source_type == 'formula1':
            # (open+close+3*(high+low))/8
            self.logger.info("Using formula: (open+close+3*(high+low))/8")
            return (open_price + close_price + 3 * (high_price + low_price)) / 8
            
        elif source_type == 'formula2':
            # close+high+low-2*open
            self.logger.info("Using formula: close+high+low-2*open")
            return close_price + high_price + low_price - 2 * open_price
            
        elif source_type == 'formula3':
            # (close+5*(high+low)-7*(open))/4
            self.logger.info("Using formula: (close+5*(high+low)-7*(open))/4")
            return (close_price + 5 * (high_price + low_price) - 7 * open_price) / 4
            
        elif source_type == 'formula4':
            # (open+close+5*(high+low))/12
            self.logger.info("Using formula: (open+close+5*(high+low))/12")
            return (open_price + close_price + 5 * (high_price + low_price)) / 12
            
        elif source_type == 'conditional1':
            # (close > open ? high : low)
            self.logger.info("Using conditional: (close > open ? high : low)")
            return pd.Series(
                [high_price.iloc[i] if close_price.iloc[i] > open_price.iloc[i] else low_price.iloc[i] 
                 for i in range(len(close_price))],
                index=close_price.index
            )
            
        elif source_type == 'conditional2':
            # (ohlc4 > h_open ? high : low) where ohlc4 = (open+high+low+close)/4
            self.logger.info("Using conditional: (ohlc4 > open ? high : low)")
            ohlc4 = (open_price + high_price + low_price + close_price) / 4
            return pd.Series(
                [high_price.iloc[i] if ohlc4.iloc[i] > open_price.iloc[i] else low_price.iloc[i] 
                 for i in range(len(ohlc4))],
                index=close_price.index
            )
            
        else:
            self.logger.warning(f"Unknown price source '{source_type}'. Falling back to close price.")
            return close_price

    def calculate_technical_indicators(self, df):
        """
        Calculate technical indicators for the DataFrame using the 
        TechnicalIndicators class from the ta module.
        
        Parameters:
        df (pandas.DataFrame): DataFrame with OHLCV data
        
        Returns:
        pandas.DataFrame: DataFrame with technical indicators
        """
        self.logger.info("Starting technical indicator calculations...")
        results = pd.DataFrame(index=df.index)
        
        # Cache for price data to avoid recalculating the same formula multiple times
        price_data_cache = {}
        
        # Initialize the TechnicalIndicators class from the ta module
        ti = TechnicalIndicators()
        
        # Count total indicators for progress bar
        total_indicators = sum(len(ind.get('periods', [])) if len(ind.get('periods', [])) > 0 else 1 
                              for ind in self.ta_indicators)
        
        # Create progress bar for all indicators
        pbar = self.tqdm(total=total_indicators, desc="Calculating indicators", position=0, leave=True)
        
        for indicator in self.ta_indicators:
            name = indicator['name'].upper()
            periods = indicator.get('periods', [])
            
            # Get indicator-specific price source or use default
            ind_price_source = indicator.get('price_source', self.price_source)
            
            # Get or calculate price data for this indicator
            if ind_price_source not in price_data_cache:
                price_data_cache[ind_price_source] = self._get_price_data(df, ind_price_source)
            
            price_data = price_data_cache[ind_price_source]
            
            try:
                # Handle special cases first
                if name == 'KDJ':
                    self.logger.info(f"Calculating KDJ indicator using price source: {ind_price_source}...")
                    if all(col in df.columns for col in ['high', 'low']):
                        # KDJ needs high, low and a third price (using our calculated price)
                        k, d, j = ti.kdj(df['high'], df['low'], price_data)
                        results['KDJ_K'] = k
                        results['KDJ_D'] = d
                        results['KDJ_J'] = j
                        self.logger.info("KDJ indicator calculation completed")
                    else:
                        self.logger.warning("Skipping KDJ: requires high, low data")
                    pbar.update(1)
                    continue
                # if name == 'KDJ':
                    # Implementing exactly the user's provided KDJ method with robust indexing
                    # if 'high' in df.columns and 'low' in df.columns:
                    #     try:
                    #         # Create a temporary DataFrame
                    #         temp_df = pd.DataFrame({
                    #             'high': df['high'],
                    #             'low': df['low'],
                    #             'close': price_data
                    #         })
                            
                    #         # Print index information for debugging
                    #         if self.verbose:
                    #             print(f"Index type: {type(temp_df.index)}")
                    #             print(f"First few indices: {temp_df.index[:5]}")
                                
                    #         n = 9
                            
                    #         # Step 1: Calculate RSV
                    #         rolling_low = temp_df['low'].rolling(window=n).min()
                    #         rolling_high = temp_df['high'].rolling(window=n).max()
                            
                    #         rsv = (temp_df['close'] - rolling_low) * 100 / (rolling_high - rolling_low)
                    #         rsv = rsv.fillna(0)  # Fill NaN values with 0
                            
                    #         # Step 2 & 3: Initialize K and D using EXPLICIT positional indexing
                    #         k = pd.Series(0.0, index=temp_df.index)
                    #         d = pd.Series(0.0, index=temp_df.index)
                            
                    #         # Use iloc for guaranteed positional indexing
                    #         k.iloc[0] = 50.0
                    #         d.iloc[0] = 50.0
                            
                    #         # Step 4: Calculate K and D using explicit positional indexing
                    #         for i in range(1, len(temp_df)):
                    #             k.iloc[i] = (2/3) * k.iloc[i-1] + (1/3) * rsv.iloc[i]
                    #             d.iloc[i] = (2/3) * d.iloc[i-1] + (1/3) * k.iloc[i]
                            
                    #         # Step 5: Calculate J
                    #         j = 3 * k - 2 * d
                            
                    #         # Add directly to results
                    #         results['KDJ_K'] = k
                    #         results['KDJ_D'] = d
                    #         results['KDJ_J'] = j
                            
                    #         # Print successful completion
                    #         if self.verbose:
                    #             print("KDJ calculated successfully")
                                
                    #     except Exception as e:
                    #         # Print detailed error information
                    #         print(f"KDJ calculation error: {str(e)}")
                    #         print(f"Error type: {type(e).__name__}")
                    #         import traceback
                    #         print(f"Traceback: {traceback.format_exc()}")
                    
                    # pbar.update(1)
                    # continue
                    
                    
                if name == 'OBV':
                    self.logger.info(f"Calculating OBV indicator using price source: {ind_price_source}...")
                    if 'volume' in df.columns:
                        results['OBV'] = ti.obv(price_data, df['volume'])
                        self.logger.info("OBV indicator calculation completed")
                    else:
                        self.logger.warning(f"Skipping OBV: volume data required but not found")
                    pbar.update(1)
                    continue
                    
                if name == 'WILLIAMS_R':
                    self.logger.info(f"Calculating Williams %R with periods {periods} using price source: {ind_price_source}...")
                    for period in periods:
                        if all(col in df.columns for col in ['high', 'low']):
                            # Williams %R typically needs high, low, close data
                            # Create a temporary df with high, low and our calculated price as "close"
                            temp_df = pd.DataFrame({
                                'high': df['high'],
                                'low': df['low'],
                                'close': price_data
                            })
                            results[f"{name}_{period}"] = ti.williams_r(temp_df, period)
                            self.logger.info(f"Williams %R_{period} calculation completed")
                        else:
                            # Fallback to just using the calculated price
                            results[f"{name}_{period}"] = ti.williams_r(price_data, period)
                            self.logger.info(f"Williams %R_{period} calculation completed (using calculated price only)")
                        pbar.update(1)
                    continue
                    
                if name == 'STOCH_RSI':
                    self.logger.info(f"Calculating Stochastic RSI with periods {periods} using price source: {ind_price_source}...")
                    for period in periods:
                        k, d = ti.stoch_rsi(price_data, period)
                        results[f'STOCH_RSI_K_{period}'] = k
                        results[f'STOCH_RSI_D_{period}'] = d
                        self.logger.info(f"Stochastic RSI_{period} calculation completed")
                        pbar.update(1)
                    continue
                
                if name == 'MACD':
                    self.logger.info(f"Calculating MACD indicator using price source: {ind_price_source}...")
                    macd_line, signal_line, histogram = ti.macd(price_data)
                    results['MACD_LINE'] = macd_line
                    results['MACD_SIGNAL'] = signal_line
                    results['MACD_HIST'] = histogram
                    self.logger.info("MACD indicator calculation completed")
                    pbar.update(1)
                    continue
                
                # Add these special cases after the existing special case handlers:

                if name == 'AO':
                    self.logger.info(f"Calculating Awesome Oscillator using high/low data...")
                    if all(col in df.columns for col in ['high', 'low']):
                        # AO needs high and low data
                        results['AO'] = ti.ao(df['high'], df['low'])
                        self.logger.info("Awesome Oscillator calculation completed")
                    else:
                        self.logger.warning("Skipping AO: requires high, low data")
                    pbar.update(1)
                    continue
                    
                if name == 'APO':
                    self.logger.info(f"Calculating Absolute Price Oscillator...")
                    # Check if custom periods are provided
                    if len(periods) >= 2:
                        fast_period, slow_period = periods[0], periods[1]
                        results['APO'] = ti.apo(price_data, fast_period, slow_period)
                        self.logger.info(f"APO with fast={fast_period}, slow={slow_period} calculation completed")
                    else:
                        # Use default periods
                        results['APO'] = ti.apo(price_data)
                        self.logger.info("APO with default periods calculation completed")
                    pbar.update(1)
                    continue
                    
                if name == 'TSI':
                    self.logger.info(f"Calculating True Strength Index with periods {periods}...")
                    for period in periods:
                        # Adjust this if your TSI implementation takes different parameters
                        tsi_values, signal = ti.tsi(price_data, period)
                        results[f'TSI_{period}'] = tsi_values
                        results[f'TSI_SIGNAL_{period}'] = signal
                        self.logger.info(f"TSI_{period} calculation completed")
                        pbar.update(1)
                    continue
                
                if name == 'RVSI':
                    self.logger.info(f"Calculating RVSI indicator with periods {periods} using price source: {ind_price_source}...")
                    
                    # Get additional parameters specific to RVSI from the indicator configuration
                    mode = indicator.get('mode', 'tfs')  # Default to 'tfs' mode
                    self.logger.info(f"Using RVSI mode: {mode}")
                    
                    # Create a dictionary of additional parameters to pass through to the RVSI function
                    rvsi_params = {}
                    
                    # Mode-specific parameter mapping
                    if mode == 'tfs':
                        if 'vol_len' in indicator:
                            rvsi_params['vol_len'] = indicator['vol_len']
                            self.logger.info(f"Using custom vol_len: {rvsi_params['vol_len']}")
                        # Include open price if available
                        if 'open' in df.columns:
                            rvsi_params['open'] = df['open']
                            
                    elif mode == 'kvo':
                        if 'fast_x' in indicator:
                            rvsi_params['fast_x'] = indicator['fast_x']
                            self.logger.info(f"Using custom fast_x: {rvsi_params['fast_x']}")
                        if 'slow_x' in indicator:
                            rvsi_params['slow_x'] = indicator['slow_x']
                            self.logger.info(f"Using custom slow_x: {rvsi_params['slow_x']}")
                            
                    elif mode == 'vzo':
                        if 'z_len' in indicator:
                            rvsi_params['z_len'] = indicator['z_len']
                            self.logger.info(f"Using custom z_len: {rvsi_params['z_len']}")
                            
                    elif mode == 'cvo':
                        if 'ema1_len' in indicator:
                            rvsi_params['ema1_len'] = indicator['ema1_len']
                            self.logger.info(f"Using custom ema1_len: {rvsi_params['ema1_len']}")
                        if 'ema2_len' in indicator:
                            rvsi_params['ema2_len'] = indicator['ema2_len']
                            self.logger.info(f"Using custom ema2_len: {rvsi_params['ema2_len']}")
                        if 'base' in indicator:
                            rvsi_params['base'] = indicator['base']
                            self.logger.info(f"Using custom base: {rvsi_params['base']}")
                            
                            # If base is 'cvd', we need high, low, open data
                            if rvsi_params['base'] == 'cvd':
                                if all(col in df.columns for col in ['high', 'low']):
                                    rvsi_params['high'] = df['high']
                                    rvsi_params['low'] = df['low']
                                else:
                                    self.logger.warning("CVO with base 'cvd' requires high and low data, which was not found. Results may be incorrect.")
                    
                    # Check if volume data is available
                    if 'volume' not in df.columns:
                        self.logger.warning("RVSI calculation requires volume data, which was not found. Skipping RVSI calculation.")
                        pbar.update(len(periods) if periods else 1)
                        continue
                    
                    # Calculate RVSI for each period
                    for period in periods:
                        try:
                            # Call the RVSI method with the appropriate parameters
                            results[f"RVSI_{period}"] = ti.rvsi(
                                data=price_data,
                                volume=df['volume'],
                                length=period,
                                mode=mode,
                                **rvsi_params
                            )
                            
                            self.logger.info(f"RVSI_{period} calculation completed")
                        except Exception as e:
                            self.logger.error(f"Error calculating RVSI_{period}_{mode.upper()}: {str(e)}")
                        finally:
                            pbar.update(1)
                    
                    # If no periods specified, use default period
                    if not periods:
                        try:
                            # Use default period (14)
                            default_period = 14
                            results[f"RVSI_{default_period}"] = ti.rvsi(
                                data=price_data,
                                volume=df['volume'],
                                length=default_period,
                                mode=mode,
                                **rvsi_params
                            )
                            
                            self.logger.info(f"RVSI_{default_period} calculation completed (using default period)")
                            pbar.update(1)
                        except Exception as e:
                            self.logger.error(f"Error calculating RVSI with default period: {str(e)}")
                            pbar.update(1)
                    
                    continue

                if name == 'SAVITZKY_GOLAY' or name == 'SG':
                    self.logger.info(f"Calculating Savitzky-Golay filter with window lengths {periods} using price source: {ind_price_source}...")
                    
                    # Get optional polynomial order parameter or use default of 2
                    poly_order = indicator.get('poly_order', 2)
                    self.logger.info(f"Using polynomial order: {poly_order}")
                    
                    # Get optional implementation type or use default "rolling"
                    implementation = indicator.get('implementation', 'rolling')
                    self.logger.info(f"Using SG filter implementation: {implementation}")
                    
                    for window_length in periods:
                        try:
                            # Ensure window_length is valid (odd number and > poly_order)
                            if window_length % 2 == 0:
                                actual_window = window_length + 1
                                self.logger.info(f"Adjusted window length from {window_length} to {actual_window} (must be odd)")
                            else:
                                actual_window = window_length
                                
                            if actual_window <= poly_order:
                                actual_window = poly_order + 1
                                if actual_window % 2 == 0:
                                    actual_window += 1
                                self.logger.info(f"Adjusted window length to {actual_window} (must be > poly_order)")
                                
                            # Calculate the Savitzky-Golay filter with specified implementation
                            results[f"SG_{actual_window}"] = ti.savitzky_golay(
                                price_data, 
                                window_length=actual_window, 
                                poly_order=poly_order,
                                implementation=implementation
                            )
                            self.logger.info(f"Savitzky-Golay filter with window={actual_window}, poly_order={poly_order}, implementation={implementation} completed")
                        except Exception as e:
                            self.logger.error(f"Error calculating Savitzky-Golay with window={window_length}, poly_order={poly_order}, implementation={implementation}: {str(e)}")
                        finally:
                            pbar.update(1)
                    continue
                
                # For standard indicators with periods
                self.logger.info(f"Calculating {name} with periods {periods} using price source: {ind_price_source}...")
                for period in periods:
                    try:
                        # Special handling for volume-weighted indicators
                        if name in ['VWMA', 'EVWMA']:
                            if 'volume' in df.columns:
                                method = getattr(ti, name.lower())
                                results[f"{name}_{period}"] = method(
                                    data=price_data,
                                    volume=df['volume'],
                                    period=period
                                )
                                self.logger.info(f"{name}_{period} calculation completed")
                            else:
                                self.logger.warning(f"Skipping {name}_{period}: volume data required but not found")
                        else:
                            # Call the appropriate method in the TechnicalIndicators class
                            method_name = name.lower()
                            
                            # Standard indicators
                            if hasattr(ti, method_name):
                                method = getattr(ti, method_name)
                                results[f"{name}_{period}"] = method(price_data, period)
                                self.logger.info(f"{name}_{period} calculation completed")
                            else:
                                self.logger.warning(f"Indicator method {method_name} not found in TechnicalIndicators class")
                    except Exception as e:
                        self.logger.error(f"Error calculating {name}_{period}: {str(e)}")
                    finally:
                        pbar.update(1)
                        
            except Exception as e:
                print(f"Error calculating {name}: {str(e)}")
                self.logger.error(f"Error calculating {name}: {str(e)}")
                # Update progress bar for all periods in this failed indicator
                if len(periods) > 0:
                    pbar.update(len(periods))
                else:
                    pbar.update(1)
                continue
        
        pbar.close()
        self.logger.info(f"Technical indicator calculations completed. {len(results.columns)} indicators calculated.")
        return results
    
    def add_time_features(self, df, timeframe=None):
        """
        Add time features for LSTM modeling with automatic timeframe adaptation.
        
        Features added depend on the timeframe:
        - For intraday data: hour features, session encoding, minute features (for small timeframes)
        - For daily+ data: day of week, month, week of year features
        - All timeframes: appropriate cyclic encodings
        
        Parameters:
        df (pandas.DataFrame): DataFrame with timestamp index or timestamp column
        timeframe (str, optional): Force specific timeframe ('1m', '15m', '1h', '1d')
                                If None, will be auto-detected
        
        Returns:
        pandas.DataFrame: DataFrame with added time features
        """
        import numpy as np
        
        self.logger.info("Adding adaptive time features for LSTM modeling...")
        
        # Make sure we're working with a copy
        result_df = df.copy()
        
        # Ensure we have a datetime index
        if not isinstance(result_df.index, pd.DatetimeIndex):
            if 'timestamp' in result_df.columns:
                result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
                result_df.set_index('timestamp', inplace=True)
                self.logger.info("Set timestamp column as DatetimeIndex")
            else:
                self.logger.error("DataFrame must have a timestamp index or a 'timestamp' column")
                raise ValueError("Cannot find timestamp in DataFrame")
        
        # Auto-detect timeframe if not provided
        if timeframe is None:
            timeframe = self._detect_timeframe(result_df)
            self.logger.info(f"Auto-detected timeframe: {timeframe}")
        
        # -------------------------------
        # 1. Universal time components
        # -------------------------------
        # These are useful for all timeframes
        
        result_df['day_of_week_sin'] = np.sin(2 * np.pi * result_df.index.dayofweek / 7)
        result_df['day_of_week_cos'] = np.cos(2 * np.pi * result_df.index.dayofweek / 7)
        
        result_df['month_sin'] = np.sin(2 * np.pi * (result_df.index.month - 1) / 12)
        result_df['month_cos'] = np.cos(2 * np.pi * (result_df.index.month - 1) / 12)
        
        result_df['is_weekend'] = result_df.index.dayofweek >= 5
        
        # -------------------------------
        # 2. Timeframe-specific features
        # -------------------------------
        
        # For intraday timeframes (minute and hour-based)
        if timeframe in ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h']:
            # Add hour components
            result_df['hour'] = result_df.index.hour
            result_df['hour_sin'] = np.sin(2 * np.pi * result_df.index.hour / 24)
            result_df['hour_cos'] = np.cos(2 * np.pi * result_df.index.hour / 24)
            
            # Add trading session categorical encoding
            # Add a SINGLE COLUMN with session encoded as a number
            conditions = [
                (result_df.index.hour >= 0) & (result_df.index.hour < 8),
                (result_df.index.hour >= 8) & (result_df.index.hour < 16),
                (result_df.index.hour >= 16) & (result_df.index.hour < 24)
            ]
            choices = [0, 1, 2]  # Asia=0, Europe=1, America=2
            result_df['session_encoded'] = np.select(conditions, choices, default=-1)
            
            # Text label for session
            conditions = [
                (result_df.index.hour >= 0) & (result_df.index.hour < 8),
                (result_df.index.hour >= 8) & (result_df.index.hour < 16),
                (result_df.index.hour >= 16) & (result_df.index.hour < 24)
            ]
            choices = ['asia', 'europe', 'america']
            result_df['session'] = np.select(conditions, choices, default='unknown')
            
            # Session progress (0-1 from start to end of session)
            hour_fraction = result_df.index.hour + result_df.index.minute / 60
            result_df['session_progress'] = (hour_fraction % 8) / 8
            
            # For very small timeframes, add minute components
            if timeframe in ['1m', '3m', '5m', '15m', '30m']:
                result_df['minute'] = result_df.index.minute
                
                # For 1-5 minute data, add minute cyclic encoding
                if timeframe in ['1m', '3m', '5m']:
                    result_df['minute_sin'] = np.sin(2 * np.pi * result_df.index.minute / 60)
                    result_df['minute_cos'] = np.cos(2 * np.pi * result_df.index.minute / 60)
        
        # For daily and higher timeframes
        else:  # '1d', '3d', '1w', '1M'
            # Add week of year
            result_df['week_of_year'] = result_df.index.isocalendar().week
            
            # Weekly cyclic encoding
            result_df['week_sin'] = np.sin(2 * np.pi * result_df.index.isocalendar().week / 53)
            result_df['week_cos'] = np.cos(2 * np.pi * result_df.index.isocalendar().week / 53)
            
            # For daily data, add day of year
            if timeframe == '1d':
                days_in_year = np.where(result_df.index.is_leap_year, 366, 365)
                result_df['day_of_year'] = result_df.index.dayofyear
                result_df['day_of_year_sin'] = np.sin(2 * np.pi * result_df.index.dayofyear / days_in_year)
                result_df['day_of_year_cos'] = np.cos(2 * np.pi * result_df.index.dayofyear / days_in_year)
                
                # For daily data, we can still add a notion of which session has most impact
                # based on the day's primary trading location or the closing time
                # This is an approximation since daily data doesn't have hour information
                # Here we'll add a "dominant session" column based on day of week
                # (This is just an example - you may want to customize this based on your specific use case)
                workdays = (result_df.index.dayofweek < 5)  # Monday-Friday
                result_df['dominant_session'] = np.where(workdays, 'global', 'weekend')
        
        self.logger.info(f"Added {len(result_df.columns) - len(df.columns)} time features for {timeframe} timeframe")
        return result_df

    def _detect_timeframe(self, df):
        """
        Auto-detect the timeframe of the DataFrame based on timestamp differences
        
        Parameters:
        df (pandas.DataFrame): DataFrame with DatetimeIndex
        
        Returns:
        str: Detected timeframe ('1m', '15m', '1h', '1d', etc.)
        """
        # Check if we have enough data points
        if len(df) <= 1:
            self.logger.warning("Not enough data points to detect timeframe. Defaulting to 1d")
            return '1d'
        
        # Calculate time differences in seconds
        time_diffs = df.index.to_series().diff().dropna()
        
        if time_diffs.empty:
            self.logger.warning("Cannot detect timeframe from index. Defaulting to 1d")
            return '1d'
        
        # Get the most common difference in seconds
        # For better accuracy, we'll find the mode after rounding to the nearest second
        most_common_diff = time_diffs.dt.total_seconds().round().mode().iloc[0]
        
        # Map the difference to a standard timeframe string
        if most_common_diff <= 60:
            return '1m'
        elif most_common_diff <= 180:
            return '3m'
        elif most_common_diff <= 300:
            return '5m'
        elif most_common_diff <= 900:
            return '15m'
        elif most_common_diff <= 1800:
            return '30m'
        elif most_common_diff <= 3600:
            return '1h'
        elif most_common_diff <= 7200:
            return '2h'
        elif most_common_diff <= 14400:
            return '4h'
        elif most_common_diff <= 21600:
            return '6h'
        elif most_common_diff <= 28800:
            return '8h'
        elif most_common_diff <= 43200:
            return '12h'
        elif most_common_diff <= 86400:
            return '1d'
        elif most_common_diff <= 259200:
            return '3d'
        elif most_common_diff <= 604800:
            return '1w'
        else:
            return '1M'
    
    def process_and_save(self):
        """
        Process the CSV file, calculate indicators, and save the result
        
        Returns:
        pandas.DataFrame: Combined DataFrame with original data and indicators
        """
        overall_start = time.time()
        self.logger.info(f"==== Starting processing of {self.csv_filename} ====")
        
        # Load data with progress tracking
        with self.tqdm(total=1, desc="Loading data", position=0, leave=True) as pbar:
            df = self.load_data()
            pbar.update(1)
        
        # Calculate technical indicators (has its own progress bar)
        ta_df = self.calculate_technical_indicators(df)
        
        # Combine original data with technical indicators
        self.logger.info("Combining original data with technical indicators...")
        combined_df = pd.concat([df, ta_df], axis=1)
        
            
        # Add cyclic time features for LSTM model
        self.logger.info("Adding cyclic time features")
        with self.tqdm(total=1, desc="Adding time features", position=0, leave=True) as pbar:
            combined_df = self.add_time_features(combined_df)
            pbar.update(1)
        
        # Reset index to make timestamp a column named 'timestamp'
        combined_df = combined_df.reset_index().rename(columns={combined_df.index.name: 'timestamp'})
        self.logger.info("Reset index to create 'timestamp' column")
        
        # Determine output filename
        if self.output_filename:
            # Use the custom output filename
            output_path = f"{self.output_dir}/{self.output_filename}"
        else:
            # Generate filename based on input filename
            base_filename = os.path.basename(self.csv_filename)
            name_without_ext = os.path.splitext(base_filename)[0]
            date_str = datetime.now().strftime('%Y%m%d')
            output_path = f"{self.output_dir}/{name_without_ext}_ta_{date_str}.csv"
        
        # Save combined data with progress tracking
        self.logger.info(f"Saving processed data to {output_path}...")
        with self.tqdm(total=1, desc="Saving results", position=0, leave=True) as pbar:
            combined_df.to_csv(output_path, index=False)  # No need for index column since timestamp is a column now
            pbar.update(1)
        
        overall_elapsed = time.time() - overall_start
        self.logger.info(f"==== Processing completed in {overall_elapsed:.2f} seconds ====")
        self.logger.info(f"Final dataset: {len(combined_df)} rows × {len(combined_df.columns)} columns")
        self.logger.info(f"Results saved to: {output_path}")
        
        return combined_df