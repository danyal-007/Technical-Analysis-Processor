import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from numba import cuda, jit, float64, int64, prange, njit
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import unfold
from scipy.signal import savgol_coeffs, savgol_filter
from numpy.lib.stride_tricks import sliding_window_view

# Set random seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

class TechnicalIndicators:
    """Class containing implementations of various technical indicators"""
    @staticmethod
    def ensure_numerical_stability(data):
        stable_data = data.copy()
        stable_data.fillna(0.0, inplace=True)
        stable_data.replace([np.inf, -np.inf], 0.0, inplace=True)
        return stable_data

    @staticmethod
    def sma(data, period):
        """Simple Moving Average (Optimized with PyTorch)"""
        stable_data = TechnicalIndicators.ensure_numerical_stability(data)
        
        if len(stable_data) == 0:  # Handle empty input
            return pd.Series([], index=data.index, dtype=np.float64)
        
        # Convert to PyTorch tensor with GPU support
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = 'cpu'
        data_torch = torch.tensor(stable_data.values, dtype=torch.float64, device=device)
        
        # Create sliding windows using unfold
        try:
            windows = torch.nn.functional.unfold(
                data_torch.view(1, 1, -1, 1),  # Reshape to 4D (B,C,H,W)
                kernel_size=(period, 1),
                stride=1
            ).squeeze().T  # Transpose to [num_windows, period]
        except RuntimeError:
            return pd.Series(np.nan, index=data.index, dtype=np.float64)
        
        # Vectorized mean computation
        sma_tensor = windows.mean(dim=1)
        
        # Convert back to pandas with proper alignment
        valid_values = sma_tensor.cpu().numpy()
        valid_index = data.index[period-1:]
        return pd.Series(valid_values, index=valid_index, dtype=np.float64).reindex(
            data.index, fill_value=np.nan
        )

    @staticmethod
    @njit(fastmath=True, cache=True)
    def _compute_ema_numba(data_np, period):
        alpha = 2.0 / (period + 1.0)
        n = data_np.size
        ema = np.empty_like(data_np)
        if n == 0:
            return ema
            
        ema[0] = data_np[0]
        for i in range(1, n):
            ema[i] = alpha * data_np[i] + (1.0 - alpha) * ema[i-1]
        return ema

    @staticmethod
    @torch.jit.script
    def _ema_kernel(data: torch.Tensor, alpha: float) -> torch.Tensor:
        result = torch.empty_like(data)
        result[0] = data[0]
        for i in range(1, data.size(0)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result

    @staticmethod
    def _compute_ema_torch(data_tensor, period):
        alpha = 2.0 / (period + 1.0)
        return TechnicalIndicators._ema_kernel(data_tensor, alpha)

    @staticmethod
    def ema(data, period):
        """Optimized EMA with JIT-compiled backends"""
        if not isinstance(data, pd.Series):
            data = pd.Series(data)

        stable_data = TechnicalIndicators.ensure_numerical_stability(data)
        data_np = stable_data.to_numpy(dtype=np.float64)

        try:
            # if torch.cuda.is_available():
            #     data_tensor = torch.as_tensor(data_np, device='cuda', dtype=torch.float64)
            #     result_tensor = TechnicalIndicators._compute_ema_torch(data_tensor, period)
            #     result = result_tensor.cpu().numpy()
            # else:
            # Use pre-allocated arrays for Numba
            result = TechnicalIndicators._compute_ema_numba(data_np, period)
        except Exception as e:
            result = stable_data.ewm(span=period, adjust=False).mean().to_numpy()

        return pd.Series(result, index=data.index, name='EMA')

    @staticmethod
    def wma(data, period):
        """Weighted Moving Average (GPU-accelerated with PyTorch)"""
        # Ensure input stability and convert to tensor
        stable_data = TechnicalIndicators.ensure_numerical_stability(data)
        device = 'cpu'
        data_tensor = torch.tensor(stable_data.values, 
                                dtype=torch.float64, 
                                device=device)

        # Precompute weights and their sum
        weights = torch.arange(1, period+1, 
                            dtype=torch.float64, 
                            device=data_tensor.device)
        sum_weights = torch.sum(weights)  # Precompute once

        # Create sliding windows using unfold
        if len(data_tensor) >= period:
            # Shape: [num_windows, period]
            windows = data_tensor.unfold(0, period, 1)
            # Vectorized weighted sum
            wma_values = (windows * weights).sum(dim=1) / sum_weights
        else:
            wma_values = torch.tensor([], device=data_tensor.device)

        # Convert back to pandas with proper alignment
        result = pd.Series(wma_values.cpu().numpy(), 
                        index=data.index[period-1:period-1+len(wma_values)])
        return result.reindex_like(data)

    @staticmethod
    def hma(data, period):
        """Optimized Hull Moving Average using PyTorch for GPU acceleration."""
        # Ensure numerical stability and convert to tensor
        stable_data = TechnicalIndicators.ensure_numerical_stability(data)
        data_tensor = torch.tensor(stable_data.values, dtype=torch.float64)
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = 'cpu'
        data_tensor = data_tensor.to(device)
        period_half = period // 2
        sqrt_period = int(np.sqrt(period))
        
        if len(data) < period + sqrt_period - 1:
            return pd.Series(np.nan, index=data.index, dtype=np.float64)


        # WMA(half-period)
        weights_half = torch.arange(1, period_half + 1, dtype=torch.float64, device=device)
        windows_half = unfold(data_tensor.view(1, 1, -1), (1, period_half), stride=1).squeeze(0).t()
        wma_half = (windows_half * weights_half).sum(dim=1) / weights_half.sum()

        # WMA(full-period)
        weights_full = torch.arange(1, period + 1, dtype=torch.float64, device=device)
        windows_full = unfold(data_tensor.view(1, 1, -1), (1, period), stride=1).squeeze(0).t()
        wma_full = (windows_full * weights_full).sum(dim=1) / weights_full.sum()

        # Align and compute 2*WMA_half - WMA_full
        overlap_start = period - period_half  # Start index in wma_half tensor
        combined_values = 2 * wma_half[overlap_start : overlap_start + len(wma_full)] - wma_full

        # Final WMA on the combined values
        weights_final = torch.arange(1, sqrt_period + 1, dtype=torch.float64, device=device)
        windows_final = unfold(combined_values.view(1, 1, -1), (1, sqrt_period), stride=1).squeeze(0).t()
        hma_values = (windows_final * weights_final).sum(dim=1) / weights_final.sum()

        # Map results to original index with NaNs
        start_idx = period + sqrt_period - 2  # First valid index in original data
        hma_series = pd.Series(np.nan, index=data.index, dtype=np.float64)
        if len(hma_values) > 0:
            hma_series.iloc[start_idx : start_idx + len(hma_values)] = hma_values.cpu().numpy()
        
        return hma_series
    
    @staticmethod
    def thma(data, period):
        """Triple Hull Moving Average - Enhanced version of HMA"""
        stable_data = TechnicalIndicators.ensure_numerical_stability(data)
        hma1 = TechnicalIndicators.hma(stable_data, period)
        hma2 = TechnicalIndicators.hma(hma1, period)
        hma3 = TechnicalIndicators.hma(hma2, period)
        return (hma1 + hma2 + hma3) / 3
    
    @staticmethod
    def dema(data, period):
        """Double Exponential Moving Average"""
        stable_data = TechnicalIndicators.ensure_numerical_stability(data)
        ema1 = TechnicalIndicators.ema(stable_data, period)
        ema2 = TechnicalIndicators.ema(ema1, period)
        return 2 * ema1 - ema2
    
    @staticmethod
    def tema(data, period):
        """Triple Exponential Moving Average"""
        stable_data = TechnicalIndicators.ensure_numerical_stability(data)
        ema1 = TechnicalIndicators.ema(stable_data, period)
        ema2 = TechnicalIndicators.ema(ema1, period)
        ema3 = TechnicalIndicators.ema(ema2, period)
        return 3 * ema1 - 3 * ema2 + ema3
 
    @staticmethod
    def lsma(data, period):
        """Least Squares Moving Average - Linear regression based MA (optimized with PyTorch)"""
        stable_data = TechnicalIndicators.ensure_numerical_stability(data)
        
        # Handle invalid periods and short data
        if period < 2 or len(data) < period:
            return pd.Series(np.nan, index=data.index, name=data.name if data.name else 'LSMA')
        
        # Convert to PyTorch tensor with GPU support
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = 'cpu'
        data_tensor = torch.tensor(stable_data.values, dtype=torch.float64, device=device)
        
        # Precompute regression coefficients
        x_tensor = torch.arange(period, dtype=torch.float64, device=device)
        sum_x = x_tensor.sum()
        sum_x2 = (x_tensor ** 2).sum()
        denominator = period * sum_x2 - sum_x ** 2
        
        # Create sliding windows
        windows = data_tensor.unfold(0, period, 1)
        
        # Vectorized regression calculations
        sum_y = windows.sum(dim=1)
        sum_xy = (windows * x_tensor).sum(dim=1)
        
        # Compute regression parameters
        numerator = period * sum_xy - sum_x * sum_y
        slope = numerator / denominator
        intercept = (sum_y - slope * sum_x) / period
        predicted_values = slope * (period - 1) + intercept
        
        # Convert back to pandas with proper NaN alignment
        values = predicted_values.cpu().numpy()
        result = pd.Series(values, index=data.index[period-1:], name=data.name if data.name else 'LSMA')
        
        return result.reindex(data.index, fill_value=np.nan) 
    
    @staticmethod
    def rma(data, period):
        """Optimized RSI Moving Average (Wilder's Smoothing) using existing EMA implementation with adjusted period"""
        return TechnicalIndicators.ema(data, 2 * period - 1)

    @staticmethod
    def smma(data, period):
        """Smoothed Moving Average"""
        stable_data = TechnicalIndicators.ensure_numerical_stability(data)
        smma = pd.Series(index=stable_data.index, dtype=np.float64)
        smma.iloc[period-1] = stable_data.iloc[:period].mean()
        for i in range(period, len(stable_data)):
            smma.iloc[i] = (smma.iloc[i-1] * (period-1) + stable_data.iloc[i]) / period
        return smma

    @staticmethod
    def vwma(data, volume, period):
        """
        Optimized Volume-Weighted Moving Average using PyTorch for GPU acceleration.
        """
        # Ensure numerical stability and convert to tensors
        stable_data = TechnicalIndicators.ensure_numerical_stability(data)
        stable_volume = TechnicalIndicators.ensure_numerical_stability(volume)
        
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = 'cpu'
        data_tensor = torch.tensor(stable_data.values, dtype=torch.float64, device=device)
        volume_tensor = torch.tensor(stable_volume.values, dtype=torch.float64, device=device)
        
        # Compute price * volume product
        product_tensor = data_tensor * volume_tensor
        
        # Extract sliding windows for product and volume
        if len(data_tensor) >= period:
            product_windows = product_tensor.unfold(0, period, 1)
            volume_windows = volume_tensor.unfold(0, period, 1)
            
            # Sum windowed values
            weighted_sum = product_windows.sum(dim=1)
            volume_sum = volume_windows.sum(dim=1)
            
            # Compute VWMA with division masking
            mask = volume_sum != 0
            vwma_values = torch.where(
                mask, 
                weighted_sum / volume_sum, 
                torch.full_like(weighted_sum, torch.nan)
            )
        else:
            vwma_values = torch.tensor([], dtype=torch.float64, device=device)
        
        # Build full-length result with NaN padding
        result_tensor = torch.full((len(data_tensor),), torch.nan, 
                                dtype=torch.float64, device=device)
        if len(vwma_values) > 0:
            result_tensor[period-1:] = vwma_values
        
        # Return pandas Series with original index
        return pd.Series(
            result_tensor.cpu().numpy(), 
            index=data.index, 
            name=f"VWMA_{period}"
        )
    
    @staticmethod
    def evwma(data, volume, period):
        stable_data = TechnicalIndicators.ensure_numerical_stability(data)
        stable_volume = TechnicalIndicators.ensure_numerical_stability(volume)
        
        # Convert to PyTorch tensors with float64 precision
        data_tensor = torch.tensor(stable_data.values, dtype=torch.float64)
        volume_tensor = torch.tensor(stable_volume.values, dtype=torch.float64)
        
        # Use GPU if available
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = 'cpu'
        data_tensor, volume_tensor = data_tensor.to(device), volume_tensor.to(device)
        
        # Calculate volume-weighted price
        weighted_price = data_tensor * volume_tensor
        
        # Vectorized rolling sums using sliding windows
        def rolling_sum(x):
            return torch.nn.functional.conv1d(
                x.view(1, 1, -1), 
                torch.ones(1, 1, period, dtype=torch.float64, device=device),
                padding=0
            ).squeeze()
        
        sum_weighted = rolling_sum(weighted_price.view(1, -1))
        sum_vol = rolling_sum(volume_tensor.view(1, -1))
        
        # Handle division with GPU-accelerated masking
        evwma_tensor = torch.full_like(data_tensor, torch.nan, dtype=torch.float64)
        valid_mask = sum_vol != 0
        valid_indices = torch.arange(len(sum_weighted), device=device) + period - 1
        
        evwma_tensor[valid_indices[valid_mask]] = (
            sum_weighted[valid_mask] / sum_vol[valid_mask]
        )
        
        # Convert back to pandas Series with original index alignment
        result = pd.Series(evwma_tensor.cpu().numpy(), index=data.index)
        return result.iloc[period-1:].reindex(data.index, fill_value=np.nan)

    @staticmethod
    @njit(fastmath=True, cache=True)
    def _compute_zlema_adjusted_numba(data_np, lag):
        n = data_np.size
        adjusted = np.empty_like(data_np)
        if n == 0:
            return adjusted
        for i in range(n):
            if i >= lag:
                adjusted[i] = data_np[i] + (data_np[i] - data_np[i - lag])
            else:
                adjusted[i] = data_np[i]
        return adjusted

    @staticmethod
    def zema(data, period):
        """Zero-Lag Exponential Moving Average (ZLEMA)"""
        if not isinstance(data, pd.Series):
            data = pd.Series(data)

        lag = (period - 1) // 2

        stable_data = TechnicalIndicators.ensure_numerical_stability(data)
        data_np = stable_data.to_numpy(dtype=np.float64)

        # Compute adjusted data
        try:
            adjusted_data_np = TechnicalIndicators._compute_zlema_adjusted_numba(data_np, lag)
        except Exception as e:
            # Fallback to pandas-based adjustment if Numba fails
            shifted = data.shift(lag).fillna(0).to_numpy()
            adjusted_data_np = data_np + (data_np - shifted)

        # Compute EMA on adjusted data
        try:
            result = TechnicalIndicators._compute_ema_numba(adjusted_data_np, period)
        except Exception as e:
            # Fallback to pandas EMA calculation
            result = pd.Series(adjusted_data_np).ewm(span=period, adjust=False).mean().to_numpy()

        return pd.Series(result, index=data.index, name='ZLEMA')
    
    @staticmethod
    def t3(data: pd.Series, period: int, volume_factor: float = 0.7) -> pd.Series:
        """Optimized Tillson T3 Moving Average using pre-optimized EMA"""
        stable_data = TechnicalIndicators.ensure_numerical_stability(data)
        c1 = -(volume_factor ** 3)
        c2 = 3 * (volume_factor ** 2) + 3 * (volume_factor ** 3)
        c3 = -6 * (volume_factor ** 2) - 3 * volume_factor - 3 * (volume_factor ** 3)
        c4 = 1 + 3 * volume_factor + (volume_factor ** 3) + 3 * (volume_factor ** 2)

        device = 'cpu'
        
        # Convert to tensor once (reuse existing EMA's tensor logic)
        data_tensor = torch.as_tensor(stable_data.values, dtype=torch.float64, device=device)
        
        original_index = stable_data.index

        # Compute all EMAs sequentially on GPU
        def _tensor_to_series(t: torch.Tensor) -> pd.Series:
            return pd.Series(t.cpu().numpy(), index=original_index, name=data.name)

        # Reuse pre-optimized EMA method with tensor-through-pandas wrapping
        e1 = TechnicalIndicators.ema(_tensor_to_series(data_tensor), period)
        e2 = TechnicalIndicators.ema(e1, period)
        e3 = TechnicalIndicators.ema(e2, period)
        e4 = TechnicalIndicators.ema(e3, period)
        e5 = TechnicalIndicators.ema(e4, period)
        e6 = TechnicalIndicators.ema(e5, period)

        
        
        # Convert final components to tensors for GPU math
        components = [
            c1 * torch.as_tensor(e6.values, device=device),
            c2 * torch.as_tensor(e5.values, device=device),
            c3 * torch.as_tensor(e4.values, device=device),
            c4 * torch.as_tensor(e3.values, device=device)
        ]

        # Sum components and convert back to pandas
        t3_tensor = sum(components)
        return pd.Series(t3_tensor.cpu().numpy(), index=original_index, name=data.name)
    
    @staticmethod
    def gma(data, period):
        """Geometric Moving Average"""
        stable_data = TechnicalIndicators.ensure_numerical_stability(data)
        def geometric_mean(x):
            try:
                return stats.gmean(x[x > 0])  # Only positive values
            except:
                return np.nan
        return stable_data.rolling(period).apply(geometric_mean, raw=True)
    
    @staticmethod
    def wwma(data, period):
        """Welles Wilder Moving Average"""
        # Same as RMA, included for completeness
        return TechnicalIndicators.rma(data, period)

    @staticmethod
    def cma(data, period):
        """Corrective Moving Average (Optimized)"""
        stable_data = TechnicalIndicators.ensure_numerical_stability(data)
        sma = TechnicalIndicators.sma(stable_data, period)
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = 'cpu'
        
        # Convert to tensors with GPU support
        data_tensor = torch.tensor(stable_data.values, dtype=torch.float64, device=device)
        sma_tensor = torch.tensor(sma.values, dtype=torch.float64, device=device)
        n = len(data_tensor)
        
        # Compute difference between data and SMA
        diff = data_tensor - sma_tensor
        
        # GPU-accelerated rolling mean with strict period enforcement
        def compute_correction(diff_tensor, window_size):
            if len(diff_tensor) < window_size:
                return torch.full_like(diff_tensor, torch.nan)
            
            windows = diff_tensor.unfold(0, window_size, 1)
            mask = ~torch.isnan(windows)
            valid_counts = mask.sum(dim=1)
            valid_sum = (windows * mask).sum(dim=1)
            
            # Apply min_periods=period constraint
            correction_values = torch.where(
                valid_counts >= window_size,
                valid_sum / valid_counts,
                torch.tensor(np.nan, dtype=torch.float64, device=device)
            )
            
            # Align with original index
            correction = torch.full_like(diff_tensor, torch.nan)
            correction[window_size-1:] = correction_values
            return correction

        correction = compute_correction(diff, period)
        
        # Combine SMA and correction
        cma_tensor = sma_tensor + correction
        
        # Convert to pandas Series with proper index alignment
        return pd.Series(cma_tensor.cpu().numpy(), index=data.index)
    
    @staticmethod
    def gmma(data, period):
        """Corrected Geometric Mean Moving Average with rigorous NaN/zero handling."""
        stable_data = TechnicalIndicators.ensure_numerical_stability(data)
        data_jax = jnp.array(stable_data.values, dtype=jnp.float64)
        
        # 1. Track original zeros
        zero_mask = (data_jax == 0)
        
        # 2. Prepare log data (zeros and unstable values become NaNs)
        data_log_safe = jnp.where(zero_mask, jnp.nan, data_jax)
        log_data = jnp.log(data_log_safe)
        
        # 3. Valid mask now excludes both zeros and unstable values
        valid_mask = ~jnp.isnan(log_data)
        
        # 4. Windowed computations
        kernel = jnp.ones(period, dtype=jnp.float64)
        
        # 5. Zero presence detection
        zero_counts = jax.scipy.signal.convolve(
            zero_mask.astype(jnp.float64), 
            kernel, 
            mode='valid'
        )
        has_zeros = zero_counts > 0
        
        # 6. Valid log summation
        sum_logs = jax.scipy.signal.convolve(
            jnp.where(valid_mask, log_data, 0.0),  # Replace invalid with 0 for summation
            kernel, 
            mode='valid'
        )
        valid_counts = jax.scipy.signal.convolve(
            valid_mask.astype(jnp.float64), 
            kernel, 
            mode='valid'
        )
        
        # 7. Geometric mean calculation
        log_mean = jnp.where(
            valid_counts >= period, 
            sum_logs / valid_counts, 
            jnp.nan
        )
        geometric_mean = jnp.exp(log_mean)
        
        # 8. Final value logic
        final_values = jnp.where(
            has_zeros, 
            0.0,  # Zero if any original zero in window
            geometric_mean
        )
        final_values = jnp.where(
            valid_counts < period, 
            jnp.nan, 
            final_values
        )
        
        # 9. Pad with NaNs for alignment
        padded = jnp.concatenate([
            jnp.full(period-1, jnp.nan), 
            final_values
        ])
        
        return pd.Series(padded, index=data.index, dtype=np.float64)

    @staticmethod
    def ealf(data, period, gamma=0.5):
        """
        Optimized Ehler's Adaptive Laguerre Filter using JAX for GPU acceleration.
        Maintains strict sequential processing with JAX's scan operator.
        """
        stable_data = TechnicalIndicators.ensure_numerical_stability(data)
        
        # Convert to JAX array with float64 precision
        data_jax = jnp.array(stable_data.values.astype(np.float64))
        if len(data_jax) == 0:
            return pd.Series([], index=data.index, dtype=np.float64)
        
        # Initial state setup
        initial_state = (data_jax[0], data_jax[0], data_jax[0], data_jax[0])
        inputs = data_jax[1:]  # Skip first element already in initial state

        # JIT-compiled scan operation for GPU acceleration
        @jax.jit
        def jax_scan(carry, x):
            l0_prev, l1_prev, l2_prev, l3_prev = carry
            l0_new = (1 - gamma) * x + gamma * l0_prev
            l1_new = -gamma * l0_new + l0_prev + gamma * l1_prev
            l2_new = -gamma * l1_new + l1_prev + gamma * l2_prev
            l3_new = -gamma * l2_new + l2_prev + gamma * l3_prev
            return (l0_new, l1_new, l2_new, l3_new), (l0_new, l1_new, l2_new, l3_new)

        # Execute scan with JAX
        _, (l0_scan, l1_scan, l2_scan, l3_scan) = lax.scan(
            jax_scan, initial_state, inputs
        )

        # Reconstruct full sequences with initial values
        l0 = jnp.concatenate([jnp.array([initial_state[0]]), l0_scan])
        l1 = jnp.concatenate([jnp.array([initial_state[1]]), l1_scan])
        l2 = jnp.concatenate([jnp.array([initial_state[2]]), l2_scan])
        l3 = jnp.concatenate([jnp.array([initial_state[3]]), l3_scan])

        # Compute final result and convert to pandas
        result = (l0 + 2*l1 + 2*l2 + l3) / 6
        return pd.Series(
            np.asarray(result, dtype=np.float64), 
            index=data.index, 
            dtype=np.float64
        )
    
    @staticmethod
    def elf(data, period):
        """Ehler's Laguerre Filter - Simplified version of EALF"""
        return TechnicalIndicators.ealf(data, period, gamma=0.382)  # Golden ratio conjugate
    
    @staticmethod
    def rema(data, period, lambda_param=0.5):
        """
        Range EMA - Adaptive EMA based on price range
        """
        stable_data = TechnicalIndicators.ensure_numerical_stability(data)
        
        # Calculate needed components
        alpha = 2.0 / (period + 1)
        ranges = stable_data.rolling(period).max() - stable_data.rolling(period).min()
        range_means = ranges.rolling(period).mean()
        
        # Calculate normalized ranges safely
        normalized_ranges = pd.Series(index=stable_data.index, dtype=np.float64)
        mask = range_means != 0  # Create a boolean mask for non-zero values
        normalized_ranges[mask] = ranges[mask] / range_means[mask]
        normalized_ranges[~mask] = 1.0  # Set default value where mean is zero
        
        # Calculate adaptive alpha
        adaptive_alpha = alpha * (1 + lambda_param * normalized_ranges)
        adaptive_alpha = adaptive_alpha.clip(lower=0, upper=1)  # Ensure alpha stays in [0,1]
        
        # Calculate REMA
        rema = pd.Series(index=stable_data.index, dtype=np.float64)
        rema.iloc[0] = stable_data.iloc[0]
        
        for i in range(1, len(stable_data)):
            rema.iloc[i] = (adaptive_alpha.iloc[i] * stable_data.iloc[i] + 
                        (1 - adaptive_alpha.iloc[i]) * rema.iloc[i-1])
        
        return rema

    @staticmethod
    def swma(data, period):
        """Sine-Weighted Moving Average (Optimized with PyTorch)"""
        stable_data = TechnicalIndicators.ensure_numerical_stability(data)
        if len(data) < period or period <= 0:
            return pd.Series(np.nan, index=data.index)

        # Precompute sine weights using PyTorch
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = 'cpu'
        weights = torch.sin(torch.linspace(0, torch.pi, period, 
                        dtype=torch.float64, device=device))
        sum_weights = weights.sum()
        
        # Convert data to PyTorch tensor with numerical stability
        data_tensor = torch.tensor(stable_data.values, 
                        dtype=torch.float64, device=device)
        
        # Create sliding windows using unfold
        windows = data_tensor.unfold(0, period, 1)
        if windows.shape[0] == 0:  # Handle insufficient data
            return pd.Series(np.nan, index=data.index)
        
        # Vectorized weighted sum calculation
        weighted_sums = (windows * weights).sum(dim=1) / sum_weights
        
        # Convert back to pandas with proper NaN alignment
        result = pd.Series(
            weighted_sums.cpu().numpy(),
            index=data.index[period-1:period-1 + len(weighted_sums)]
        )
        return result.reindex(data.index, fill_value=np.nan)

    @staticmethod
    def calculate_mama_fama(data, fast_limit=0.5, slow_limit=0.05):
        # Ensure numerical stability and convert to numpy array
        price = TechnicalIndicators.ensure_numerical_stability(data)
        data_np = price.values.astype(np.float64)
        n = len(data_np)
        
        # Initialize numpy arrays
        mama = np.zeros(n, dtype=np.float64)
        fama = np.zeros(n, dtype=np.float64)
        detrender = np.zeros(n, dtype=np.float64)
        I1 = np.zeros(n, dtype=np.float64)
        Q1 = np.zeros(n, dtype=np.float64)
        phase = np.zeros(n, dtype=np.float64)
        
        if n < 10:
            # Handle insufficient data case
            mama[:] = np.nan
            fama[:] = np.nan
        else:
            # Initialize first 10 values
            mama[:10] = data_np[:10]
            fama[:10] = data_np[:10]
            
            # Execute optimized loop with JIT compilation
            TechnicalIndicators._mama_fama_loop(
                data_np, fast_limit, slow_limit, mama, fama, detrender, I1, Q1, phase
            )
        
        # Convert back to pandas Series with original index
        return (
            pd.Series(mama, index=price.index, dtype=np.float64),
            pd.Series(fama, index=price.index, dtype=np.float64)
        )

    @staticmethod
    @njit
    def _mama_fama_loop(data, fast_limit, slow_limit, mama, fama, detrender, I1, Q1, phase):
        for i in range(10, len(data)):
            # Compute detrender (fixed multiplier from period_prices=0)
            detrender[i] = (
                0.0962 * data[i] 
                + 0.5769 * data[i-2] 
                - 0.5769 * data[i-4] 
                - 0.0962 * data[i-6]
            ) * 0.54  # 0.075*0 + 0.54 = 0.54
            
            # Compute Q1 component
            Q1[i] = (
                0.0962 * detrender[i] 
                + 0.5769 * detrender[i-2] 
                - 0.5769 * detrender[i-4] 
                - 0.0962 * detrender[i-6]
            ) * 0.54
            
            # Set InPhase component
            I1[i] = detrender[i-3]
            
            # Compute phase calculation intermediates
            jI = 1.57 * I1[i-1] - 0.707 * I1[i-2]
            jQ = 1.57 * Q1[i-1] - 0.707 * Q1[i-2]
            
            # Calculate phase with stability checks
            if jI != 0.0 and I1[i] != 0.0:
                phase_val = np.arctan(np.abs(jQ / jI))
            else:
                phase_val = 0.0
            
            # Adjust phase quadrant
            if I1[i] < 0:
                phase_val = np.pi - phase_val
            if jQ < 0:
                phase_val = -phase_val
            phase[i] = phase_val
            
            # Compute adaptive alpha
            delta_phase = phase[i-1] - phase[i]
            if delta_phase < 1.0:
                delta_phase = 1.0
            alpha = fast_limit / delta_phase
            alpha = max(min(alpha, fast_limit), slow_limit)
            
            # Update MAMA and FAMA
            mama[i] = alpha * data[i] + (1 - alpha) * mama[i-1]
            fama[i] = 0.5 * alpha * mama[i] + (1 - 0.5 * alpha) * fama[i-1]
    
    @staticmethod
    def mama(data, period):
        """MESA Adaptive Moving Average (MAMA)"""
        mama, _ = TechnicalIndicators.calculate_mama_fama(data)
        return mama

    @staticmethod
    def fama(data, period):
        """Following Adaptive Moving Average (FAMA)"""
        _, fama = TechnicalIndicators.calculate_mama_fama(data)
        return fama

    @staticmethod
    def hkama(data, period, fast_period=2, slow_period=30):
        """
        Hilbert-based Kaufman's Adaptive Moving Average
        A modification of KAMA using Hilbert Transform concepts
        
        Parameters:
        data (pd.Series): Price data
        period (int): Calculation period
        fast_period (int): Fast EMA period
        slow_period (int): Slow EMA period
        """
        stable_data = TechnicalIndicators.ensure_numerical_stability(data)
        
        # Calculate price change and volatility
        price_change = stable_data.diff(period).abs()
        volatility = stable_data.diff().abs().rolling(period).sum()
        
        # Calculate efficiency ratio
        er = price_change / volatility
        
        # Calculate smoothing constant
        fast_alpha = 2.0 / (fast_period + 1)
        slow_alpha = 2.0 / (slow_period + 1)
        sc = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2
        
        # Initialize HKAMA series
        hkama = pd.Series(index=stable_data.index, dtype=np.float64)
        hkama.iloc[0] = stable_data.iloc[0]
        
        # Calculate HKAMA values
        for i in range(1, len(stable_data)):
            hkama.iloc[i] = hkama.iloc[i-1] + sc.iloc[i] * (stable_data.iloc[i] - hkama.iloc[i-1])
        
        return hkama

    @staticmethod
    def _ensure_torch(data):
        """Utility to convert pandas Series to torch tensor on GPU."""
        data_np = data.values
        device = 'cpu'
        return torch.tensor(data_np, dtype=torch.float64, device=device)

    @staticmethod
    def _convert_to_pandas(tensor, index):
        """Convert torch tensor back to pandas Series with original index."""
        return pd.Series(tensor.cpu().numpy(), index=index)

    @staticmethod
    @njit
    def _compute_edma_numba(src, length):
        """Numba-accelerated computation of hexp and lexp arrays."""
        n = len(src)
        hexp = np.full(n, np.nan, dtype=np.float64)
        lexp = np.full(n, np.nan, dtype=np.float64)
        hexp[0] = lexp[0] = src[0]
        smoothness = 1.0
        h_len = max(2, int(round(length / 1.5)))  # Minimum period 2
        for i in range(1, n):
            prev_hexp = hexp[i - 1]
            if src[i] >= prev_hexp:
                hexp[i] = src[i]
            else:
                hexp[i] = prev_hexp + (src[i] - prev_hexp) * (smoothness / (h_len + 1))
            
            prev_lexp = lexp[i - 1]
            if src[i] <= prev_lexp:
                lexp[i] = hexp[i]
            else:
                lexp[i] = prev_lexp + (src[i] - prev_lexp) * (smoothness / (h_len + 1))
        return hexp, lexp

    @staticmethod
    @njit
    def _compute_exact_wma_numba(src, period):
        """Numba-accelerated WMA with min_periods=1 and custom weights per window size."""
        n = len(src)
        result = np.full(n, np.nan, dtype=np.float64)
        for i in range(n):
            m = min(period, i + 1)
            if m == 1:
                result[i] = src[i]
            else:
                window = src[i - m + 1 : i + 1]
                weights = np.arange(1, m + 1, dtype=np.float64)
                weighted_sum = np.sum(window * weights)
                denominator = np.sum(weights)
                result[i] = weighted_sum / denominator if denominator != 0 else np.nan
        return result

    @staticmethod
    def edma(data, length, i_Symmetrical=False):
        """NaN-resistant EDMA implementation with Numba and PyTorch optimizations."""
        # Validate input
        if len(data) < 1 or length < 1:
            return pd.Series(np.nan, index=data.index, dtype=np.float64)

        # Ensure numerical stability
        stable_data = TechnicalIndicators.ensure_numerical_stability(data)
        src = stable_data.to_numpy().astype(np.float64)
        n = len(src)

        # Compute hexp and lexp with Numba (stateful, no data leakage)
        hexp, lexp = TechnicalIndicators._compute_edma_numba(src, length)
        lexp_series = pd.Series(lexp, index=stable_data.index).ffill()

        # Compute WMA components with Numba
        def exact_wma(series, period):
            """Numba-optimized WMA calculation with dynamic window sizes."""
            return TechnicalIndicators._compute_exact_wma_numba(series.values, period)

        h_len = max(2, int(round(length / 1.5)))
        wma_half_len = max(1, h_len // 2)
        wma_half = pd.Series(exact_wma(lexp_series, wma_half_len), index=stable_data.index)
        wma_full = pd.Series(exact_wma(lexp_series, h_len), index=stable_data.index)
        hma_input = 2 * wma_half - wma_full
        hma_final_len = max(1, int(round(np.sqrt(h_len))))
        edma_result = pd.Series(exact_wma(hma_input, hma_final_len), index=stable_data.index)

        # Apply SWMA (if enabled) with PyTorch for GPU acceleration
        if i_Symmetrical:
            data_torch = TechnicalIndicators._ensure_torch(edma_result)
            window_size = 4
            weights = torch.tensor([1.0, 2.0, 2.0, 1.0], device=data_torch.device)

            # Use PyTorch to compute SWMA with vectorized operations
            windows = torch.nn.functional.unfold(
                data_torch.unsqueeze(0), 
                kernel_size=(window_size, 1), 
                padding=0, 
                stride=1
            ).t()
            weighted_sum = (windows * weights).sum(dim=1) / weights.sum()
            result_torch = torch.zeros(len(data), dtype=torch.float64)
            result_torch[window_size - 1 :] = weighted_sum
            result_torch[:window_size - 1] = data_torch[:window_size - 1]  # Fill first window_size - 1 with original data
            edma_result = TechnicalIndicators._convert_to_pandas(result_torch, data.index)

        # Ensure index alignment and return as pandas Series
        return edma_result.reindex(data.index, method="ffill").astype(np.float64)

    @staticmethod
    def macd(data, fast_period=12, slow_period=26, signal_period=9):
        """
        Moving Average Convergence Divergence (MACD)
        
        This indicator consists of three components:
        1. MACD Line: Difference between fast and slow EMAs
        2. Signal Line: EMA of the MACD line
        3. Histogram: MACD line minus Signal line
        
        Parameters:
        data (pd.Series): Input price data
        fast_period (int): Period for fast EMA (default: 12)
        slow_period (int): Period for slow EMA (default: 26)
        signal_period (int): Period for signal line EMA (default: 9)
        
        Returns:
        tuple: (MACD line, Signal line, Histogram)
        """
        # Ensure numerical stability
        stable_data = TechnicalIndicators.ensure_numerical_stability(data)
        
        # Calculate fast and slow EMAs
        fast_ema = TechnicalIndicators.ema(stable_data, fast_period)
        slow_ema = TechnicalIndicators.ema(stable_data, slow_period)
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate Signal line (EMA of MACD line)
        signal_line = TechnicalIndicators.ema(macd_line, signal_period)
        
        # Calculate MACD histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram

    @staticmethod
    def cci(data, period=20, constant=0.015):
        """
        Optimized Commodity Channel Index (CCI) using PyTorch for GPU acceleration.
        """
        # Handle numerical stability and convert to tensor
        stable_data = TechnicalIndicators.ensure_numerical_stability(data)
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = 'cpu'
        
        data_tensor = torch.tensor(stable_data.values, dtype=torch.float64, device=device)
        n = len(stable_data)
        
        if n < period:
            return pd.Series(np.full(n, np.nan), index=data.index)
        
        # Create sliding windows and compute MA
        windows = data_tensor.unfold(0, period, 1)
        ma = windows.mean(dim=1)
        
        # Compute mean absolute deviation within each window
        mean_deviation = (windows - ma.unsqueeze(1)).abs().mean(dim=1)
        
        # Calculate CCI values (aligned to original data indices starting at period-1)
        numerator = data_tensor[period-1:] - ma
        denominator = constant * mean_deviation
        cci_tensor = numerator / denominator  # Handles division by zero as NaN/Inf
        
        # Reconstruct full-length tensor with NaNs for initial period-1 values
        full_cci = torch.full((n,), torch.nan, dtype=torch.float64, device=device)
        full_cci[period-1:] = cci_tensor
        
        # Convert to pandas Series and preserve original index
        return pd.Series(full_cci.cpu().numpy(), index=data.index)

    @staticmethod
    def volatility_index(data, period=14, smoothing_period=3, timeframe='1H'):
        """
        Optimized Volatility Index using PyTorch for GPU acceleration.
        Maintains original behavior but removes redundant True Range calculation.
        """
        annualization_factors = {
            # Minutes
            '1m': np.sqrt(365 * 24 * 60),
            '5m': np.sqrt(365 * 24 * 12), 
            '15m': np.sqrt(365 * 24 * 4),
            '30m': np.sqrt(365 * 24 * 2),
            '60m': np.sqrt(365 * 24),
            
            # Hours
            '1H': np.sqrt(365 * 24),
            '4H': np.sqrt(365 * 6),
            
            # Days
            'D': np.sqrt(252),       # Trading days
            'W': np.sqrt(52),        # Weeks
            'M': np.sqrt(12),        # Months
            
            # Crypto (24/7)
            'C': np.sqrt(365)        # Calendar days
        }
        ann_factor = annualization_factors.get(timeframe, np.sqrt(252))
        
        # Handle empty input
        if len(data) < 2:
            return pd.Series(dtype=float, index=data.index)
        
        # Ensure numerical stability on input
        stable_data = TechnicalIndicators.ensure_numerical_stability(data)
        device = 'cpu'
        
        data_tensor = torch.tensor(stable_data.values, dtype=torch.float64, device=device)
        
        # Calculate percentage returns (vectorized)
        prev_values = data_tensor[:-1]
        curr_values = data_tensor[1:]
        returns = torch.where(
            prev_values != 0,
            (curr_values - prev_values) / prev_values,
            torch.zeros_like(prev_values)
        )
        
        # Handle insufficient data post-returns
        if len(returns) < period:
            return pd.Series(np.nan, index=data.index)
        
        # Vectorized rolling standard deviation
        window_view = returns.unfold(0, period, 1)
        std_dev = window_view.std(dim=1, correction=1)  # ddof=1
        
        # Annualize volatility
        volatility = std_dev * torch.sqrt(torch.tensor(ann_factor)) * 100
        
        # Convert to pandas Series with proper index alignment
        volatility_series = pd.Series(
            volatility.cpu().numpy(),
            index=data.index[period:]  # First valid index after rolling window
        )
        
        # Use existing optimized EMA implementation
        volatility_index = TechnicalIndicators.ema(volatility_series, smoothing_period)
        
        # Reindex to match original data's index
        return volatility_index.reindex(data.index, fill_value=np.nan)

    @staticmethod
    def rsi(data, period=14, **kwargs):
        """
        State-aware RSI implementation with chunk consistency
        Pass {'last_avg_gain': x, 'last_avg_loss': y} in kwargs for chunked processing
        """
        stable_data = TechnicalIndicators.ensure_numerical_stability(data)
        data_np = stable_data.to_numpy().astype(np.float64)
        
        delta = np.zeros_like(data_np)
        delta[1:] = np.diff(data_np)
        
        gains = np.where(delta > 0, delta, 0.0)
        losses = np.where(delta < 0, -delta, 0.0)
        
        avg_gains = np.full_like(data_np, np.nan)
        avg_losses = np.full_like(data_np, np.nan)
        
        # State initialization
        last_avg_gain = kwargs.get('last_avg_gain')
        last_avg_loss = kwargs.get('last_avg_loss')
        start_idx = 0
        
        if last_avg_gain is None or last_avg_loss is None:
            # Initial chunk - compute first average normally
            valid_start = 1
            first_avg_gain = np.mean(gains[valid_start:valid_start+period])
            first_avg_loss = np.mean(losses[valid_start:valid_start+period])
            avg_gains[period] = first_avg_gain
            avg_losses[period] = first_avg_loss
            start_idx = period + 1
        else:
            # Subsequent chunk - use provided state
            avg_gains[0] = last_avg_gain
            avg_losses[0] = last_avg_loss
            start_idx = 1

        @njit(nogil=True)
        def calculate_emas(gains, losses, avg_gains, avg_losses, period, start_idx):
            for i in range(start_idx, len(gains)):
                prev_idx = i-1 if i > 0 else 0
                avg_gains[i] = (avg_gains[prev_idx] * (period-1) + gains[i]) / period
                avg_losses[i] = (avg_losses[prev_idx] * (period-1) + losses[i]) / period
            return avg_gains, avg_losses

        avg_gains, avg_losses = calculate_emas(gains, losses, avg_gains, avg_losses, period, start_idx)

        with np.errstate(divide='ignore', invalid='ignore'):
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
        
        rsi = np.where(avg_losses == 0, 100.0, rsi)
        rsi = np.where(avg_gains == 0, 0.0, rsi)
        
        # Prepare state for next chunk
        final_gain = avg_gains[-1] if len(avg_gains) > 0 else np.nan
        final_loss = avg_losses[-1] if len(avg_losses) > 0 else np.nan
        
        result = pd.Series(rsi, index=data.index, name='RSI')
        result.attrs['state'] = {'last_avg_gain': final_gain, 'last_avg_loss': final_loss}
        
        return result
    
    @staticmethod
    def rsi_divergence(data, rsi_values, period=14):
        """
        Calculate RSI divergence signals
        
        Divergence occurs when price makes a new high/low but RSI doesn't confirm it.
        This can signal potential trend reversals.
        
        Parameters:
        data (pd.Series): Price data
        rsi_values (pd.Series): RSI values
        period (int): Look-back period for divergence
        
        Returns:
        tuple: (bullish divergence signals, bearish divergence signals)
        """
        # Initialize signal series
        bullish_div = pd.Series(0, index=data.index)
        bearish_div = pd.Series(0, index=data.index)
        
        # Look for divergences over the specified period
        for i in range(period, len(data)):
            # Get the window of data we're examining
            price_window = data.iloc[i-period:i+1]
            rsi_window = rsi_values.iloc[i-period:i+1]
            
            # Find price and RSI extremes
            price_low = price_window.min()
            price_high = price_window.max()
            rsi_low = rsi_window.min()
            rsi_high = rsi_window.max()
            
            # Check for bullish divergence (lower price low but higher RSI low)
            if (price_window.iloc[-1] == price_low and 
                rsi_window.iloc[-1] > rsi_low):
                bullish_div.iloc[i] = 1
                
            # Check for bearish divergence (higher price high but lower RSI high)
            if (price_window.iloc[-1] == price_high and 
                rsi_window.iloc[-1] < rsi_high):
                bearish_div.iloc[i] = 1
        
        return bullish_div, bearish_div 

    @staticmethod
    def kdj(high, low, close, n=9):
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = 'cpu'
        dtype = torch.float64
        
        # Convert to PyTorch tensors with proper numerical stability
        high_t = torch.as_tensor(high.values, device=device, dtype=dtype)
        low_t = torch.as_tensor(low.values, device=device, dtype=dtype)
        close_t = torch.as_tensor(close.values, device=device, dtype=dtype)

        # Rolling min/max with vectorized unfold
        def rolling_extremes(x, window):
            x = x.clone()
            if len(x) < window:
                return torch.full_like(x, torch.nan), torch.full_like(x, torch.nan)
                
            unfolded = x.unfold(0, window, 1)
            mins = unfolded.min(dim=1).values
            maxs = unfolded.max(dim=1).values
            return (
                F.pad(mins, (window-1, 0), value=torch.nan),
                F.pad(maxs, (window-1, 0), value=torch.nan)
            )

        # Compute RSV with fused kernel
        low_min, high_max = rolling_extremes(low_t, n)
        rsv = (close_t - low_min) / (high_max - low_min + 1e-12) * 100
        rsv = torch.nan_to_num(rsv, 0.0)
        
        # Vectorized SMA calculations using optimized convolution
        def vectorized_sma(x, period, init_val=50.0):
            if len(x) < period:
                return torch.full_like(x, init_val)
                
            kernel = torch.full((period,), 1/period, device=device, dtype=dtype)
            conv = F.conv1d(x[None, None], kernel[None, None], padding=period-1)[0, 0]
            valid_conv = conv[period-1:-period+1] if period > 1 else conv
            return torch.cat([
                torch.full((period-1,), init_val, device=device),
                valid_conv
            ])

        # Compute K and D with batched SMAs
        k = vectorized_sma(rsv, 3)
        d = vectorized_sma(k, 3)
        j = 3*k - 2*d

        # Convert back to pandas with proper index alignment
        return (
            pd.Series(k.cpu().numpy(), index=high.index),
            pd.Series(d.cpu().numpy(), index=high.index),
            pd.Series(j.cpu().numpy(), index=high.index)
        )

    @staticmethod
    def obv(data, volume):
        """
        On Balance Volume (OBV)
        
        OBV is a momentum indicator that uses volume to predict changes in price.
        The idea is that volume precedes price movements, so we can use volume
        changes to predict future price trends.
        
        Parameters:
        data (pd.Series): Close price data
        volume (pd.Series): Volume data
        
        Returns:
        pd.Series: OBV values
        """
        # Ensure numerical stability
        stable_data = TechnicalIndicators.ensure_numerical_stability(data)
        stable_volume = TechnicalIndicators.ensure_numerical_stability(volume)
        
        # Calculate price changes
        price_change = stable_data.diff()
        
        # Initialize OBV series
        obv = pd.Series(0.0, index=data.index)
        obv.iloc[0] = stable_volume.iloc[0]
        
        # Calculate OBV
        for i in range(1, len(data)):
            if price_change.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i-1] + stable_volume.iloc[i]
            elif price_change.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i-1] - stable_volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv

    @staticmethod
    def williams_r(data, period=14):
        # Extract input data based on type (DataFrame or Series)
        if isinstance(data, pd.DataFrame):
            high = data['high']
            low = data['low']
            close = data['close']
        else:
            stable_data = TechnicalIndicators.ensure_numerical_stability(data)
            high = stable_data
            low = stable_data
            close = stable_data
        
        # Convert to PyTorch tensors with GPU support
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = 'cpu'
        
        high_tensor = torch.tensor(high.values, dtype=torch.float64, device=device)
        low_tensor = torch.tensor(low.values, dtype=torch.float64, device=device)
        close_tensor = torch.tensor(close.values, dtype=torch.float64, device=device)

        def rolling_op(tensor, period, reducer):
            """Compute rolling window operation with NaN padding"""
            if len(tensor) < period:
                return torch.full_like(tensor, torch.nan)
            windows = tensor.unfold(0, period, 1)
            reduced = reducer(windows, dim=1).values
            return torch.cat([
                torch.full((period-1,), torch.nan, device=device, dtype=torch.float64),
                reduced
            ])

        # Calculate rolling highs and lows
        highest_high = rolling_op(high_tensor, period, torch.max)
        lowest_low = rolling_op(low_tensor, period, torch.min)

        # Core Williams %R calculation
        denominator = highest_high - lowest_low
        numerator = highest_high - close_tensor
        valid_mask = denominator != 0
        
        # Avoid division by zero using mask
        williams_r_tensor = torch.where(
            valid_mask,
            (-100 * numerator) / denominator,
            torch.tensor(torch.nan, device=device)
        )

        # Final NaN/inf cleanup
        williams_r_tensor = torch.nan_to_num(williams_r_tensor, nan=torch.nan)

        # Convert to pandas Series with original index
        williams_r_series = pd.Series(
            williams_r_tensor.cpu().numpy(),
            index=data.index,
            name=f'WilliamsR_{period}'
        )
        
        return williams_r_series

    @staticmethod
    def stoch_rsi(data, period=14, smooth_k=3, smooth_d=3):
        """
        Optimized Stochastic RSI using PyTorch for GPU acceleration and sliding window operations.
        Maintains pandas Series output with identical NaN alignment to the original implementation.
        """
        # Calculate RSI using existing optimized method
        rsi_values = TechnicalIndicators.rsi(data, period)
        
        # Handle insufficient data early
        if len(rsi_values) < period:
            empty = pd.Series(np.nan, index=data.index)
            return empty.copy(), empty.copy()
        
        # PyTorch device configuration
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = 'cpu'
        
        rsi_tensor = torch.tensor(rsi_values.values, dtype=torch.float64, device=device)
        
        # Rolling min/max using unfold
        windows = rsi_tensor.unfold(0, period, 1)
        rsi_high = windows.max(dim=1).values
        rsi_low = windows.min(dim=1).values
        
        # Calculate k_raw with division guard
        valid_rsi = rsi_tensor[period-1:]  # Align with rolling window results
        denominator = rsi_high - rsi_low
        k_raw = torch.where(
            denominator == 0, 
            torch.zeros_like(valid_rsi), 
            100 * (valid_rsi - rsi_low) / denominator
        )
        
        # Nested function for sliding window means
        def sliding_mean(tensor, window):
            if len(tensor) < window:
                return torch.full(tensor.shape, torch.nan, device=device)[:0]  # Empty but valid tensor
            return tensor.unfold(0, window, 1).mean(dim=1)
        
        # Calculate %K and %D with cascading windows
        k = sliding_mean(k_raw, smooth_k)
        d = sliding_mean(k, smooth_d) if len(k) >= smooth_d else torch.tensor([], device=device)
        
        # Convert tensors to properly aligned pandas Series
        def align_series(tensor, window_offset):
            if tensor.numel() == 0:
                return pd.Series(np.nan, index=data.index)
            start_idx = period - 1 + window_offset
            valid_indices = data.index[start_idx : start_idx + len(tensor)]
            return pd.Series(
                tensor.cpu().numpy(), 
                index=valid_indices
            ).reindex(data.index, fill_value=np.nan)
        
        return (
            align_series(k, smooth_k - 1),  # %K starts after smooth_k-1 offset from period-1
            align_series(d, smooth_k + smooth_d - 2)  # %D adds smooth_d-1 offset to %K
        )

    @staticmethod
    def ao(high, low, fast_period=5, slow_period=34):
        """
        Awesome Oscillator (AO)
        
        The Awesome Oscillator is a momentum indicator that shows the difference between
        a 5-period and 34-period simple moving average of the midpoints of price bars.
        It helps identify changes in momentum and potential trend reversals.
        
        Parameters:
        high (pd.Series): High price data
        low (pd.Series): Low price data
        fast_period (int): Fast moving average period (default: 5)
        slow_period (int): Slow moving average period (default: 34)
        
        Returns:
        pd.Series: Awesome Oscillator values
        """
        # Ensure numerical stability
        stable_high = TechnicalIndicators.ensure_numerical_stability(high)
        stable_low = TechnicalIndicators.ensure_numerical_stability(low)
        
        # Calculate midpoint prices
        midpoint = (stable_high + stable_low) / 2
        
        # Calculate simple moving averages
        fast_sma = TechnicalIndicators.sma(midpoint, fast_period)
        slow_sma = TechnicalIndicators.sma(midpoint, slow_period)
        
        # Calculate Awesome Oscillator
        ao = fast_sma - slow_sma
        
        return ao

    @staticmethod
    def _ema_jit(data, period):
        """
        Just-In-Time compiled EMA calculation using Numba.
        
        Parameters:
        data (np.ndarray): Input data
        period (int): EMA period
        
        Returns:
        np.ndarray: EMA values
        """
        alpha = 2.0 / period
        ema = np.zeros_like(data)
        for i in range(period - 1, data.shape[0]):
            if i == period - 1:
                ema[i] = data[i]
            else:
                ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        return ema
    
    @staticmethod
    def apo(data, fast_period=12, slow_period=26):
        """
        Absolute Price Oscillator (APO)
        
        The Absolute Price Oscillator measures the difference between two exponential
        moving averages. It's similar to MACD but shows the absolute difference rather
        than using a signal line.
        
        Parameters:
        data (pd.Series): Price data
        fast_period (int): Fast EMA period (default: 12)
        slow_period (int): Slow EMA period (default: 26)
        
        Returns:
        pd.Series: APO values
        """
        # Ensure numerical stability
        stable_data = TechnicalIndicators.ensure_numerical_stability(data)
        
        # Convert data to numpy array for Numba processing
        data_np = stable_data.values
        
        # JIT compile the EMA function
        ema_jit = jit(TechnicalIndicators._ema_jit)
        
        # Calculate fast and slow EMAs using JIT
        fast_ema = ema_jit(data_np, fast_period)
        slow_ema = ema_jit(data_np, slow_period)
        
        # Compute APO
        apo = np.zeros_like(data_np)
        valid_start = max(fast_period, slow_period) - 1
        for i in range(valid_start, data_np.shape[0]):
            apo[i] = fast_ema[i] - slow_ema[i]
        
        # Convert back to pandas Series with the same index
        result = pd.Series(apo, index=data.index)
        
        # Set initial values to NaN based on the maximum period
        warmup = max(fast_period, slow_period)
        result.iloc[:warmup - 1] = np.nan
        
        return result

    @staticmethod
    def mom(data, period=10):
        """
        Momentum (MOM)
        
        Momentum measures the amount that a security's price has changed over a given
        time period. It helps identify the strength or weakness of a price trend.
        
        Parameters:
        data (pd.Series): Price data
        period (int): Calculation period (default: 10)
        
        Returns:
        pd.Series: Momentum values
        """
        # Ensure numerical stability
        stable_data = TechnicalIndicators.ensure_numerical_stability(data)
        
        # Calculate momentum (current price - price n periods ago)
        momentum = stable_data - stable_data.shift(period)
        
        return momentum

    @staticmethod
    def tsi(data, long_period=25, short_period=13, signal_period=7):
        stable_data = TechnicalIndicators.ensure_numerical_stability(data)
        values = stable_data.values.astype(np.float64)
        
        # Calculate price changes with NaN handling
        pc = np.empty_like(values)
        pc[:] = np.nan
        pc[1:] = values[1:] - values[:-1]
        abs_pc = np.abs(pc)
        
        # Calculate EMAs with fixed initialization
        pc_ema1 = TechnicalIndicators._numba_ema(pc, long_period)
        abs_pc_ema1 = TechnicalIndicators._numba_ema(abs_pc, long_period)
        pc_ema2 = TechnicalIndicators._numba_ema(pc_ema1, short_period)
        abs_pc_ema2 = TechnicalIndicators._numba_ema(abs_pc_ema1, short_period)
        
        # Calculate TSI values
        tsi_values = np.full_like(pc_ema2, np.nan)
        valid_mask = abs_pc_ema2 != 0
        tsi_values[valid_mask] = 100 * (pc_ema2[valid_mask] / abs_pc_ema2[valid_mask])
        
        # Calculate signal line
        signal = TechnicalIndicators._numba_ema(tsi_values, signal_period)
        
        return (
            pd.Series(tsi_values, index=data.index),
            pd.Series(signal, index=data.index)
        )

    @staticmethod
    @njit(nogil=True)
    def _numba_ema(arr, period):
        """Sequential EMA calculation with proper NaN handling"""
        alpha = 2.0 / (period + 1.0)
        result = np.empty_like(arr)
        result[:] = np.nan
        
        # Find first valid value
        start_idx = -1
        for i in range(len(arr)):
            if not np.isnan(arr[i]):
                start_idx = i
                break
                
        if start_idx == -1:
            return result  # All NaNs
        
        result[start_idx] = arr[start_idx]
        
        # Sequential EMA calculation
        for i in range(start_idx + 1, len(arr)):
            if np.isnan(arr[i]):
                result[i] = result[i-1]
            else:
                result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
                
        return result

    @staticmethod
    def cmo(data, period=14):
        """
        Optimized Chande Momentum Oscillator (CMO) using PyTorch for GPU acceleration.

        Parameters:
        data (pd.Series): Price data
        period (int): Calculation period (default: 14)

        Returns:
        pd.Series: CMO values ranging from -100 to +100
        """
        # Ensure numerical stability and convert to PyTorch tensor
        stable_data = TechnicalIndicators.ensure_numerical_stability(data)
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = 'cpu'
        
        data_tensor = torch.tensor(stable_data.values, dtype=torch.float64, device=device)

        # Compute price change with initial NaN
        if len(data_tensor) > 0:
            price_change = torch.cat([
                torch.tensor([float('nan')], dtype=torch.float64, device=device),
                data_tensor[1:] - data_tensor[:-1]
            ])
        else:
            return pd.Series(dtype=np.float64)  # Handle empty input

        # Calculate gains and losses
        gains = torch.where(price_change > 0, price_change, torch.tensor(0.0, dtype=torch.float64, device=device))
        losses = torch.where(price_change < 0, -price_change, torch.tensor(0.0, dtype=torch.float64, device=device))

        # Rolling sum using unfold for vectorized window operations
        def rolling_sum(tensor, window):
            if len(tensor) < window:
                return torch.full_like(tensor, float('nan'))
            unfolded = tensor.unfold(0, window, 1)
            sums = torch.nansum(unfolded, dim=1)  # Handle NaNs in windows
            padded = torch.cat([
                torch.full((window - 1,), float('nan'), dtype=torch.float64, device=device),
                sums
            ])
            return padded

        sum_gains = rolling_sum(gains, period)
        sum_losses = rolling_sum(losses, period)

        # Compute CMO with division-by-zero handling
        denominator = sum_gains + sum_losses
        cmo_values = 100 * (sum_gains - sum_losses) / denominator
        cmo_values = torch.where(denominator == 0, torch.tensor(0.0, dtype=torch.float64, device=device), cmo_values)

        # Align with original index and convert to pandas
        return pd.Series(cmo_values.cpu().numpy(), index=data.index)
    
    @staticmethod
    def savitzky_golay(data, window_length=11, poly_order=2, implementation="rolling"):
        """
        Accelerated rolling SG filter using scipy's batched processing.
        No Python loops, full GPU-like speed with strict scipy usage.
        """
        
        # Apply numerical stability
        stable_data = TechnicalIndicators.ensure_numerical_stability(data)
        dtype = stable_data.dtype
        n = len(stable_data)
        
        # Validate window length
        window_length = max(window_length, poly_order + 1)
        if window_length % 2 == 0:
            window_length += 1  # Force odd window length
        
        # Direct mode - unchanged
        if implementation == "direct":
            return pd.Series(
                savgol_filter(stable_data.values, window_length, poly_order),
                index=data.index
            )
        
        # Accelerated rolling mode
        if implementation == "rolling":
            if window_length > n:
                return pd.Series(np.full_like(stable_data, np.nan), index=data.index)
            
            # Create strided window representation
            windowed_data = sliding_window_view(stable_data, window_shape=window_length).copy()
            
            # Critical optimization: Batch apply savgol_filter to all windows
            # 1D batched filter application via optimized scipy backend
            filtered = savgol_filter(windowed_data, window_length, poly_order, axis=1)
            
            # Extract last value from each filtered window (causal alignment)
            results = np.full(n, np.nan)
            results[window_length-1:] = filtered[:, -1]
            
            return pd.Series(results, index=data.index)
 
    @staticmethod
    def rvsi(data, volume, length=14, mode='tfs', **kwargs):
        # Input validation
        if not isinstance(data, pd.Series) or not isinstance(volume, pd.Series):
            raise ValueError("Input data and volume must be pandas Series")
        
        # Clean and convert data
        data_clean = TechnicalIndicators.ensure_numerical_stability(data)
        vol_clean = TechnicalIndicators.ensure_numerical_stability(volume)
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = 'cpu'
        
        # Convert to PyTorch tensors
        with torch.no_grad():
            data_t = torch.tensor(data_clean.values, dtype=torch.float64, device=device)
            vol_t = torch.tensor(vol_clean.values, dtype=torch.float64, device=device)
            
            # Calculate volume oscillator
            if mode == 'tfs':
                vol_result = TechnicalIndicators._tfs_volume(data_t, vol_t, length, kwargs)
            elif mode == 'obv':
                vol_result = TechnicalIndicators._obv(data_t, vol_t)
            elif mode == 'kvo':
                vol_result = TechnicalIndicators._kvo(data_t, vol_t, kwargs)
            elif mode == 'vzo':
                vol_result = TechnicalIndicators._vzo(data_t, vol_t, length, kwargs)
            elif mode == 'cvo':
                vol_result = TechnicalIndicators._cvo(data_t, vol_t, kwargs)
            else:
                raise ValueError(f"Invalid mode: {mode}")
            
            # Convert back to numpy
            vol_np = vol_result.cpu().numpy()
            
        # Apply smoothing stages
        vol_series = pd.Series(vol_np, index=data.index)
        sma_result = TechnicalIndicators.sma(vol_series, length)
        rsi_result = TechnicalIndicators.rsi(sma_result, length)
        rvsi_final = TechnicalIndicators.hma(rsi_result, length)
        
        return rvsi_final

    # Region: Volume Mode Implementations
    @staticmethod
    def _tfs_volume(data_t, vol_t, length, kwargs):
        open_data = kwargs.get('open', torch.roll(data_t, 1))
        if isinstance(open_data, pd.Series):
            open_data = torch.tensor(
                TechnicalIndicators.ensure_numerical_stability(open_data).values,
                dtype=torch.float64, device=data_t.device
            )
        
        # Vectorized direction calculation
        dir_t = torch.where(data_t > open_data, 1.0, 
                          torch.where(data_t < open_data, -1.0, 0.0))
        vol_accum = dir_t * vol_t
        
        # Rolling window sum
        vol_len = kwargs.get('vol_len', length)
        cum_vol = vol_accum.cumsum(0)
        windows = unfold(cum_vol[None,None,:], (1, vol_len), stride=1)
        result = windows[0, -1] / vol_len
        
        # Pad with NaNs
        return torch.cat([torch.full((vol_len-1,), torch.nan, device=data_t.device), result])

    @staticmethod
    def _obv(data_t, vol_t):
        # Vectorized OBV calculation
        changes = torch.sign(data_t[1:] - data_t[:-1])
        changes = torch.cat([torch.tensor([0.0], device=data_t.device), changes])
        obv = torch.cumsum(changes * vol_t, 0)
        obv[0] = vol_t[0]
        return obv

    @staticmethod
    def _kvo(data_t, vol_t, kwargs):
        # Klinger Volume Oscillator
        changes = data_t[1:] - data_t[:-1]
        direction = torch.where(changes > 0, 1.0, -1.0)
        direction = torch.cat([torch.tensor([0.0], device=data_t.device), direction])
        x_trend = direction * vol_t * 100
        
        # Optimized EMA
        fast_ema = TechnicalIndicators._ema_torch(x_trend, kwargs.get('fast_x', 13))
        slow_ema = TechnicalIndicators._ema_torch(x_trend, kwargs.get('slow_x', 34))
        return fast_ema - slow_ema

    @staticmethod
    def _vzo(data_t, vol_t, length, kwargs):
        # Volume Zone Oscillator
        changes = data_t[1:] - data_t[:-1]
        direction = torch.where(changes > 0, 1.0, -1.0)
        direction = torch.cat([torch.tensor([0.0], device=data_t.device), direction])
        vp = direction * vol_t
        
        # EMA calculations
        vp_ema = TechnicalIndicators._ema_torch(vp, kwargs.get('z_len', length))
        vol_ema = TechnicalIndicators._ema_torch(vol_t, kwargs.get('z_len', length))
        
        # Safe division
        with np.errstate(divide='ignore', invalid='ignore'):
            vzo = torch.where(vol_ema != 0, 100 * (vp_ema / vol_ema), 0.0)
        return vzo.nan_to_num(0.0)

    @staticmethod
    def _cvo(data_t, vol_t, kwargs):
        base = kwargs.get('base', 'pvt')
        if base == 'obv':
            cv = TechnicalIndicators._obv(data_t, vol_t)
        elif base == 'pvt':
            cv = TechnicalIndicators._pvt(data_t, vol_t)
        elif base == 'cvd':
            cv = TechnicalIndicators._cvd(
                data_t,
                torch.tensor(kwargs['high'].values, device=data_t.device),
                torch.tensor(kwargs['low'].values, device=data_t.device),
                torch.tensor(kwargs['open'].values, device=data_t.device),
                vol_t
            )
        else:
            raise ValueError(f"Invalid CVO base: {base}")
        
        # EMA calculations
        ema1 = TechnicalIndicators._ema_torch(cv, kwargs.get('ema1_len', 3))
        ema2 = TechnicalIndicators._ema_torch(cv, kwargs.get('ema2_len', 13))
        return ema1 - ema2

    # Region: Helper Functions
    @staticmethod
    def _ema_torch(data, period):
        alpha = 2.0 / (period + 1.0)
        result = torch.zeros_like(data)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result

    @staticmethod
    def _pvt(data_t, vol_t):
        pct_change = (data_t[1:] - data_t[:-1]) / data_t[:-1].clamp_min(1e-8)
        pct_change = torch.cat([torch.tensor([0.0], device=data_t.device), pct_change])
        return torch.cumsum(vol_t * pct_change, 0)

    # @staticmethod
    # def _cvd(close, high, low, open, vol):
    #     # GPU-accelerated CVD calculation
    #     cvd = torch.zeros_like(close)
    #     blocks = 128
    #     threads = 64
    #     TechnicalIndicators._cuda_cvd[blocks, threads](close, high, low, open, vol, cvd)
    #     return cvd

    @staticmethod
    def _cvd(close, high, low, open, vol):
        """
        Convert PyTorch tensors to NumPy arrays and process with Numba for CPU acceleration.
        """
        # Convert PyTorch tensors to NumPy arrays
        close_np = close.cpu().numpy()
        high_np = high.cpu().numpy()
        low_np = low.cpu().numpy()
        open_np = open.cpu().numpy()
        vol_np = vol.cpu().numpy()

        device = 'cpu'
        
        # Initialize result array
        cvd_np = np.zeros_like(close_np, dtype=np.float64)

        # Call Numba-compiled function
        TechnicalIndicators._numba_cvd(close_np, high_np, low_np, open_np, vol_np, cvd_np)

        # Convert back to PyTorch tensor
        cvd_tensor = torch.tensor(cvd_np, device=device, dtype=torch.float64)
        return cvd_tensor

    @staticmethod
    @njit(fastmath=True, cache=True)
    def _numba_cvd(close, high, low, open, vol, cvd):
        n = len(close)
        for i in range(n):
            h = high[i]
            l = low[i]
            o = open[i]
            c = close[i]
            v = vol[i]

            # Compute t_w, b_w, body
            tw = h - max(o, c)
            bw = min(o, c) - l
            body = abs(c - o)
            total = tw + bw + body

            if total == 0:
                rate_up = 0.5
                rate_dn = 0.5
            else:
                if o <= c:
                    rate_up = 0.5 * (tw + bw + 2 * body) / total
                else:
                    rate_up = 0.5 * (tw + bw) / total

                if o > c:
                    rate_dn = 0.5 * (tw + bw + 2 * body) / total
                else:
                    rate_dn = 0.5 * (tw + bw) / total

            # Calculate delta
            if c >= o:
                delta = v * rate_up
            else:
                delta = -v * rate_dn

            # Update cumulative volume delta (CVD)
            if i > 0:
                cvd[i] = cvd[i - 1] + delta
            else:
                cvd[i] = delta

    @staticmethod
    @cuda.jit
    def _cuda_cvd(close, high, low, open, vol, cvd):
        i = cuda.grid(1)
        if i < len(close):
            h = high[i]
            l = low[i]
            o = open[i]
            c = close[i]
            v = vol[i]

            tw = h - max(o, c)
            bw = min(o, c) - l
            body = abs(c - o)

            if tw + bw + body == 0:
                rate_up = 0.5
                rate_dn = 0.5
            else:
                if o <= c:
                    rate_up = 0.5 * (tw + bw + 2 * body) / (tw + bw + body)
                else:
                    rate_up = 0.5 * (tw + bw) / (tw + bw + body)

                if o > c:
                    rate_dn = 0.5 * (tw + bw + 2 * body) / (tw + bw + body)
                else:
                    rate_dn = 0.5 * (tw + bw) / (tw + bw + body)

            delta = v * rate_up if c >= o else -v * rate_dn
            if i > 0:
                cvd[i] = cvd[i-1] + delta
            else:
                cvd[i] = delta

    @staticmethod
    def _wma_torch(data, period):
        stable_data = TechnicalIndicators.ensure_numerical_stability(data)
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = 'cpu'
        
        data_t = torch.tensor(stable_data.values, dtype=torch.float64, device=device)
        
        weights = torch.arange(1, period+1, dtype=torch.float64, device=device)
        windows = unfold(data_t[None,None,:], (1, period), stride=1)
        wma = (windows * weights[:,None]).sum(dim=0) / weights.sum()
        wma_np = torch.cat([
            torch.full((period-1,), torch.nan, device=device),
            wma.squeeze()
        ]).cpu().numpy()
        
        return pd.Series(wma_np, index=data.index)
    
    