#!/usr/bin/env python3
"""
Time-Domain Visualization System
===============================

Phase 3.3: Advanced time-domain visualization for EEG and ERP data.
Provides comprehensive plotting capabilities for time series, ERPs,  
butterfly plots, channel arrays, and interactive exploration tools.

Features:
- Multi-channel time series plots with flexible layouts
- ERP waveform plots with confidence intervals and statistics
- Butterfly plots for overview visualization
- Channel array plots with topographic arrangement
- Interactive time navigation and zoom
- Publication-ready figure export
- Real-time plot updates for data exploration

Author: porfanid  
Version: 1.0
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import SpanSelector, Button
import seaborn as sns
from scipy import signal
import mne
from mne.viz import plot_topomap


class PlotType(Enum):
    """Types of time-domain plots"""
    TIMESERIES = "timeseries"
    ERP = "erp"
    BUTTERFLY = "butterfly"
    CHANNEL_ARRAY = "channel_array"
    COMPARISON = "comparison"
    DIFFERENCE_WAVE = "difference_wave"


class LayoutType(Enum):
    """Channel layout types"""
    GRID = "grid"
    TOPOGRAPHIC = "topographic"
    LINEAR = "linear"
    CUSTOM = "custom"


@dataclass
class PlotConfig:
    """Configuration for time-domain plots"""
    
    # Figure settings
    figsize: Tuple[float, float] = (12, 8)
    dpi: int = 100
    style: str = 'whitegrid'  # seaborn style
    
    # Color settings
    color_palette: str = 'husl'  # seaborn color palette
    line_width: float = 1.5
    alpha: float = 0.8
    
    # Time axis settings
    time_unit: str = 's'  # 's' for seconds, 'ms' for milliseconds
    show_zero_line: bool = True
    zero_line_style: Dict = None
    
    # Channel settings
    show_channel_names: bool = True
    channel_name_size: int = 8
    max_channels_per_plot: int = 64
    
    # Statistical overlays
    show_confidence_intervals: bool = True
    ci_alpha: float = 0.3
    show_significance: bool = True
    
    # Interactive features
    enable_zoom: bool = True
    enable_selection: bool = True
    show_cursor: bool = True
    
    # Export settings
    export_format: str = 'png'
    export_dpi: int = 300
    
    def __post_init__(self):
        """Initialize default values"""
        if self.zero_line_style is None:
            self.zero_line_style = {'color': 'black', 'linestyle': '--', 'alpha': 0.5}


@dataclass
class PlotAnnotation:
    """Annotation for time-domain plots"""
    
    time: float  # Time point
    amplitude: Optional[float] = None  # Amplitude (for specific channel)
    channel: Optional[str] = None  # Channel name
    text: str = ""  # Annotation text
    color: str = 'red'
    marker: str = 'o'
    size: int = 8


class TimeDomainVisualizer:
    """
    Advanced time-domain visualization system for EEG/ERP data.
    
    This class provides comprehensive plotting capabilities including:
    - Multi-channel time series with flexible layouts
    - ERP visualization with statistical overlays
    - Butterfly plots for data overview
    - Interactive navigation and selection tools
    - Publication-ready figure export
    """
    
    def __init__(self, config: PlotConfig = None):
        """Initialize the visualizer"""
        self.config = config or PlotConfig()
        self.figures_ = {}
        self.axes_ = {}
        self.annotations_ = {}
        
        # Set matplotlib/seaborn style
        plt.style.use('default')
        sns.set_style(self.config.style)
        
        # Interactive widgets
        self.selectors_ = {}
        self.buttons_ = {}
    
    def plot_timeseries(self, raw: mne.io.Raw, channels: List[str] = None,
                       time_range: Tuple[float, float] = None,
                       layout: LayoutType = LayoutType.GRID,
                       title: str = None) -> plt.Figure:
        """
        Plot multi-channel time series data.
        
        Args:
            raw: MNE Raw object with EEG data
            channels: List of channels to plot (None = all channels)
            time_range: Time range to plot (tmin, tmax) in seconds
            layout: Channel layout type
            title: Plot title
            
        Returns:
            Matplotlib figure object
        """
        try:
            # Select channels and time range
            if channels is None:
                channels = raw.ch_names[:self.config.max_channels_per_plot]
            
            if time_range is None:
                time_range = (raw.times[0], raw.times[-1])
            
            # Extract data
            raw_cropped = raw.copy().crop(tmin=time_range[0], tmax=time_range[1])
            raw_cropped.pick_channels(channels)
            
            data, times = raw_cropped.get_data(return_times=True)
            
            # Convert time units if needed
            if self.config.time_unit == 'ms':
                times = times * 1000
                time_label = 'Time (ms)'
            else:
                time_label = 'Time (s)'
            
            # Create figure and subplots
            n_channels = len(channels)
            fig = self._create_channel_layout(n_channels, layout, title or 'EEG Time Series')
            
            # Plot each channel
            colors = sns.color_palette(self.config.color_palette, n_channels)
            
            for i, (ch_name, ch_data) in enumerate(zip(channels, data)):
                if layout == LayoutType.GRID:
                    ax = fig.axes[i]
                elif layout == LayoutType.LINEAR:
                    ax = fig.axes[0]
                else:
                    ax = fig.axes[i] if i < len(fig.axes) else fig.axes[0]
                
                # Plot channel data
                line = ax.plot(times, ch_data, 
                             color=colors[i], 
                             linewidth=self.config.line_width,
                             alpha=self.config.alpha,
                             label=ch_name)[0]
                
                # Channel-specific formatting
                if layout == LayoutType.GRID:
                    ax.set_title(ch_name, fontsize=self.config.channel_name_size)
                    ax.set_ylabel('Amplitude (µV)')
                    
                    # Add zero line
                    if self.config.show_zero_line:
                        ax.axhline(y=0, **self.config.zero_line_style)
                
                # Store line for later reference
                self.annotations_[f'timeseries_{ch_name}'] = line
            
            # Format axes
            if layout == LayoutType.LINEAR:
                ax = fig.axes[0]
                ax.set_xlabel(time_label)
                ax.set_ylabel('Amplitude (µV)')
                ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
                
                if self.config.show_zero_line:
                    ax.axhline(y=0, **self.config.zero_line_style)
            else:
                # Set x-label on bottom plots only
                for i, ax in enumerate(fig.axes):
                    if i >= len(fig.axes) - int(np.sqrt(len(fig.axes))):
                        ax.set_xlabel(time_label)
            
            # Add interactive features
            if self.config.enable_selection:
                self._add_time_selector(fig)
            
            plt.tight_layout()
            
            # Store figure
            self.figures_['timeseries'] = fig
            
            return fig
            
        except Exception as e:
            raise RuntimeError(f"Failed to plot time series: {str(e)}")
    
    def plot_erp(self, erps: Union[mne.Evoked, List[mne.Evoked]], 
                channels: List[str] = None,
                conditions: List[str] = None,
                show_ci: bool = None,
                title: str = None) -> plt.Figure:
        """
        Plot Event-Related Potentials with statistical overlays.
        
        Args:
            erps: Single ERP or list of ERPs to plot
            channels: List of channels to plot
            conditions: List of condition names
            show_ci: Show confidence intervals (overrides config)
            title: Plot title
            
        Returns:
            Matplotlib figure object
        """
        try:
            # Ensure erps is a list
            if isinstance(erps, mne.Evoked):
                erps = [erps]
            
            if conditions is None:
                conditions = [f'Condition {i+1}' for i in range(len(erps))]
            
            if channels is None:
                channels = erps[0].ch_names[:min(9, len(erps[0].ch_names))]  # Max 9 for 3x3 grid
            
            # Create figure
            n_channels = len(channels)
            if n_channels == 1:
                fig, ax = plt.subplots(1, 1, figsize=self.config.figsize, dpi=self.config.dpi)
                axes = [ax]
            else:
                n_rows = int(np.ceil(np.sqrt(n_channels)))
                n_cols = int(np.ceil(n_channels / n_rows))
                fig, axes = plt.subplots(n_rows, n_cols, figsize=self.config.figsize, dpi=self.config.dpi)
                axes = axes.flatten() if n_channels > 1 else [axes]
            
            fig.suptitle(title or 'Event-Related Potentials', fontsize=14, fontweight='bold')
            
            # Get colors for conditions
            colors = sns.color_palette(self.config.color_palette, len(erps))
            
            # Convert time units
            times = erps[0].times
            if self.config.time_unit == 'ms':
                times = times * 1000
                time_label = 'Time (ms)'
            else:
                time_label = 'Time (s)'
            
            # Plot each channel
            for ch_idx, ch_name in enumerate(channels):
                if ch_idx >= len(axes):
                    break
                
                ax = axes[ch_idx]
                
                # Plot each condition
                for erp_idx, (erp, condition, color) in enumerate(zip(erps, conditions, colors)):
                    if ch_name not in erp.ch_names:
                        continue
                    
                    ch_data_idx = erp.ch_names.index(ch_name)
                    erp_data = erp.data[ch_data_idx, :]
                    
                    # Main ERP line
                    line = ax.plot(times, erp_data * 1e6,  # Convert to µV
                                 color=color, 
                                 linewidth=self.config.line_width,
                                 alpha=self.config.alpha,
                                 label=condition)[0]
                    
                    # Add confidence intervals if available and requested
                    show_ci_flag = show_ci if show_ci is not None else self.config.show_confidence_intervals
                    if show_ci_flag and hasattr(erp, 'metadata') and erp.metadata:
                        if 'ci_lower' in erp.metadata and 'ci_upper' in erp.metadata:
                            ci_lower = erp.metadata['ci_lower'][ch_data_idx, :] * 1e6
                            ci_upper = erp.metadata['ci_upper'][ch_data_idx, :] * 1e6
                            
                            ax.fill_between(times, ci_lower, ci_upper,
                                          color=color, alpha=self.config.ci_alpha)
                
                # Format subplot
                ax.set_title(ch_name, fontsize=self.config.channel_name_size)
                ax.set_ylabel('Amplitude (µV)')
                
                # Add zero line
                if self.config.show_zero_line:
                    ax.axhline(y=0, **self.config.zero_line_style)
                    ax.axvline(x=0, **self.config.zero_line_style)
                
                # Add legend to first subplot
                if ch_idx == 0:
                    ax.legend(loc='upper right')
            
            # Set x-labels on bottom row
            for i in range(max(0, len(axes) - n_cols), len(axes)):
                if i < len(axes):
                    axes[i].set_xlabel(time_label)
            
            # Hide empty subplots
            for i in range(len(channels), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            # Store figure
            self.figures_['erp'] = fig
            
            return fig
            
        except Exception as e:
            raise RuntimeError(f"Failed to plot ERP: {str(e)}")
    
    def plot_butterfly(self, data: Union[mne.io.Raw, mne.Evoked, List[mne.Evoked]],
                      channels: List[str] = None,
                      title: str = None,
                      highlight_channels: List[str] = None) -> plt.Figure:
        """
        Create butterfly plot showing all channels overlaid.
        
        Args:
            data: EEG data (Raw, Evoked, or list of Evoked)
            channels: List of channels to include
            title: Plot title
            highlight_channels: Channels to highlight with different style
            
        Returns:
            Matplotlib figure object
        """
        try:
            fig, ax = plt.subplots(1, 1, figsize=self.config.figsize, dpi=self.config.dpi)
            
            # Handle different data types
            if isinstance(data, mne.io.Raw):
                if channels is None:
                    channels = data.ch_names
                data_array, times = data.get_data(picks=channels, return_times=True)
                data_array *= 1e6  # Convert to µV
                
            elif isinstance(data, mne.Evoked):
                if channels is None:
                    channels = data.ch_names
                ch_indices = [data.ch_names.index(ch) for ch in channels if ch in data.ch_names]
                data_array = data.data[ch_indices, :] * 1e6
                times = data.times
                
            elif isinstance(data, list) and all(isinstance(d, mne.Evoked) for d in data):
                # Multiple ERPs - create grand average butterfly
                if channels is None:
                    channels = data[0].ch_names
                
                # Stack all ERP data
                all_data = []
                for erp in data:
                    ch_indices = [erp.ch_names.index(ch) for ch in channels if ch in erp.ch_names]
                    all_data.append(erp.data[ch_indices, :])
                
                data_array = np.mean(all_data, axis=0) * 1e6
                times = data[0].times
            
            else:
                raise ValueError("Unsupported data type for butterfly plot")
            
            # Convert time units
            if self.config.time_unit == 'ms':
                times = times * 1000
                time_label = 'Time (ms)'
            else:
                time_label = 'Time (s)'
            
            # Plot all channels with low alpha
            highlight_set = set(highlight_channels or [])
            normal_channels = [ch for ch in channels if ch not in highlight_set]
            
            # Normal channels in gray with low alpha
            for i, ch in enumerate(normal_channels):
                if ch in channels:
                    ch_idx = channels.index(ch)
                    ax.plot(times, data_array[ch_idx, :],
                           color='gray', alpha=0.3, linewidth=0.5)
            
            # Highlighted channels with distinct colors
            if highlight_channels:
                colors = sns.color_palette(self.config.color_palette, len(highlight_channels))
                for ch, color in zip(highlight_channels, colors):
                    if ch in channels:
                        ch_idx = channels.index(ch)
                        ax.plot(times, data_array[ch_idx, :],
                               color=color, alpha=self.config.alpha, 
                               linewidth=self.config.line_width * 1.5,
                               label=ch)
                
                ax.legend(loc='upper right')
            
            # Compute and plot grand average
            grand_avg = np.mean(data_array, axis=0)
            ax.plot(times, grand_avg, 
                   color='black', linewidth=self.config.line_width * 2,
                   alpha=1.0, label='Grand Average')
            
            # Format plot
            ax.set_xlabel(time_label)
            ax.set_ylabel('Amplitude (µV)')
            ax.set_title(title or f'Butterfly Plot ({len(channels)} channels)')
            
            # Add zero lines
            if self.config.show_zero_line:
                ax.axhline(y=0, **self.config.zero_line_style)
                if isinstance(data, (mne.Evoked, list)):
                    ax.axvline(x=0, **self.config.zero_line_style)
            
            # Add grand average to legend
            handles, labels = ax.get_legend_handles_labels()
            if 'Grand Average' not in labels:
                handles.append(plt.Line2D([0], [0], color='black', linewidth=2))
                labels.append('Grand Average')
                ax.legend(handles, labels, loc='upper right')
            
            plt.tight_layout()
            
            # Store figure
            self.figures_['butterfly'] = fig
            
            return fig
            
        except Exception as e:
            raise RuntimeError(f"Failed to create butterfly plot: {str(e)}")
    
    def plot_channel_array(self, data: Union[mne.io.Raw, mne.Evoked],
                          time_point: float = None,
                          layout: str = 'biosemi64',
                          title: str = None) -> plt.Figure:
        """
        Plot channel array with topographic arrangement.
        
        Args:
            data: EEG data
            time_point: Specific time point to plot (for Evoked data)
            layout: Channel layout name
            title: Plot title
            
        Returns:
            Matplotlib figure object
        """
        try:
            # This would require MNE layout information
            # For now, create a simplified grid arrangement
            
            if isinstance(data, mne.io.Raw):
                # Use current time point or middle of data
                if time_point is None:
                    time_point = data.times[len(data.times)//2]
                
                time_idx = np.argmin(np.abs(data.times - time_point))
                data_snapshot = data.get_data()[:, time_idx] * 1e6
                channels = data.ch_names
                
            elif isinstance(data, mne.Evoked):
                if time_point is None:
                    time_point = 0.0  # Peak time or stimulus onset
                
                time_idx = np.argmin(np.abs(data.times - time_point))
                data_snapshot = data.data[:, time_idx] * 1e6
                channels = data.ch_names
            
            # Create grid layout
            n_channels = len(channels)
            n_cols = int(np.ceil(np.sqrt(n_channels)))
            n_rows = int(np.ceil(n_channels / n_cols))
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=self.config.figsize, dpi=self.config.dpi)
            if n_channels == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            # Color mapping for amplitudes
            vmin, vmax = np.percentile(data_snapshot, [5, 95])
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.cm.RdBu_r
            
            # Plot each channel
            for i, (ch_name, amplitude) in enumerate(zip(channels, data_snapshot)):
                if i >= len(axes):
                    break
                
                ax = axes[i]
                
                # Create a simple representation (circle with color-coded amplitude)
                circle = plt.Circle((0.5, 0.5), 0.4, 
                                  color=cmap(norm(amplitude)), 
                                  alpha=self.config.alpha)
                ax.add_patch(circle)
                
                # Add channel name and amplitude value
                ax.text(0.5, 0.5, f'{ch_name}\n{amplitude:.1f}µV', 
                       ha='center', va='center', 
                       fontsize=self.config.channel_name_size,
                       fontweight='bold')
                
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_aspect('equal')
                ax.axis('off')
            
            # Hide empty subplots
            for i in range(len(channels), len(axes)):
                axes[i].set_visible(False)
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=axes, shrink=0.8)
            cbar.set_label('Amplitude (µV)')
            
            # Format figure
            time_str = f" at t = {time_point:.3f}s" if time_point is not None else ""
            fig.suptitle(title or f'Channel Array{time_str}', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            # Store figure
            self.figures_['channel_array'] = fig
            
            return fig
            
        except Exception as e:
            raise RuntimeError(f"Failed to create channel array plot: {str(e)}")
    
    def plot_comparison(self, erp1: mne.Evoked, erp2: mne.Evoked,
                       condition1: str = "Condition 1",
                       condition2: str = "Condition 2", 
                       channels: List[str] = None,
                       show_difference: bool = True,
                       title: str = None) -> plt.Figure:
        """
        Plot comparison between two ERPs with difference wave.
        
        Args:
            erp1: First ERP condition
            erp2: Second ERP condition
            condition1: Name of first condition
            condition2: Name of second condition
            channels: Channels to plot
            show_difference: Whether to show difference wave
            title: Plot title
            
        Returns:
            Matplotlib figure object
        """
        try:
            if channels is None:
                channels = erp1.ch_names[:6]  # Limit to 6 channels for comparison
            
            # Create figure with subplots
            if show_difference:
                fig, axes = plt.subplots(2, len(channels), figsize=(3*len(channels), 8), dpi=self.config.dpi)
                if len(channels) == 1:
                    axes = axes.reshape(-1, 1)
            else:
                fig, axes = plt.subplots(1, len(channels), figsize=(3*len(channels), 4), dpi=self.config.dpi)
                if len(channels) == 1:
                    axes = axes.reshape(1, -1)
            
            # Convert time units
            times = erp1.times
            if self.config.time_unit == 'ms':
                times = times * 1000
                time_label = 'Time (ms)'
            else:
                time_label = 'Time (s)'
            
            colors = ['blue', 'red']
            
            # Plot each channel
            for ch_idx, ch_name in enumerate(channels):
                if ch_name not in erp1.ch_names or ch_name not in erp2.ch_names:
                    continue
                
                # Get data
                ch_idx1 = erp1.ch_names.index(ch_name)
                ch_idx2 = erp2.ch_names.index(ch_name)
                data1 = erp1.data[ch_idx1, :] * 1e6
                data2 = erp2.data[ch_idx2, :] * 1e6
                
                # Top subplot: overlay comparison
                ax_top = axes[0, ch_idx] if show_difference else axes[0, ch_idx]
                
                ax_top.plot(times, data1, color=colors[0], 
                           linewidth=self.config.line_width, 
                           label=condition1, alpha=self.config.alpha)
                ax_top.plot(times, data2, color=colors[1],
                           linewidth=self.config.line_width,
                           label=condition2, alpha=self.config.alpha)
                
                ax_top.set_title(f'{ch_name} - Comparison')
                ax_top.set_ylabel('Amplitude (µV)')
                
                if self.config.show_zero_line:
                    ax_top.axhline(y=0, **self.config.zero_line_style)
                    ax_top.axvline(x=0, **self.config.zero_line_style)
                
                if ch_idx == 0:
                    ax_top.legend()
                
                # Bottom subplot: difference wave
                if show_difference:
                    ax_bottom = axes[1, ch_idx]
                    diff_data = data1 - data2
                    
                    ax_bottom.plot(times, diff_data, color='black',
                                  linewidth=self.config.line_width,
                                  alpha=self.config.alpha)
                    ax_bottom.fill_between(times, 0, diff_data, 
                                         alpha=0.3, color='gray')
                    
                    ax_bottom.set_title(f'{ch_name} - Difference')
                    ax_bottom.set_ylabel('Difference (µV)')
                    ax_bottom.set_xlabel(time_label)
                    
                    if self.config.show_zero_line:
                        ax_bottom.axhline(y=0, **self.config.zero_line_style)
                        ax_bottom.axvline(x=0, **self.config.zero_line_style)
                else:
                    ax_top.set_xlabel(time_label)
            
            # Format figure
            comparison_title = title or f'{condition1} vs {condition2}'
            fig.suptitle(comparison_title, fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            # Store figure
            self.figures_['comparison'] = fig
            
            return fig
            
        except Exception as e:
            raise RuntimeError(f"Failed to create comparison plot: {str(e)}")
    
    def _create_channel_layout(self, n_channels: int, layout: LayoutType, title: str) -> plt.Figure:
        """Create appropriate subplot layout for channels"""
        
        if layout == LayoutType.LINEAR:
            fig, ax = plt.subplots(1, 1, figsize=self.config.figsize, dpi=self.config.dpi)
            
        elif layout == LayoutType.GRID:
            n_cols = int(np.ceil(np.sqrt(n_channels)))
            n_rows = int(np.ceil(n_channels / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=self.config.figsize, dpi=self.config.dpi)
            
            # Flatten axes if multiple subplots
            if n_channels > 1:
                axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        
        else:  # TOPOGRAPHIC or CUSTOM
            # For now, fall back to grid
            n_cols = int(np.ceil(np.sqrt(n_channels)))
            n_rows = int(np.ceil(n_channels / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=self.config.figsize, dpi=self.config.dpi)
            
            if n_channels > 1:
                axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        return fig
    
    def _add_time_selector(self, fig: plt.Figure):
        """Add interactive time selection to figure"""
        
        def onselect(xmin, xmax):
            print(f"Selected time range: {xmin:.3f} - {xmax:.3f} s")
            # Here you could trigger replotting with new time range
        
        # Add span selector to first axis
        if fig.axes:
            ax = fig.axes[0]
            selector = SpanSelector(ax, onselect, 'horizontal', 
                                   useblit=True, rectprops=dict(alpha=0.3, facecolor='yellow'))
            self.selectors_[f'span_{id(fig)}'] = selector
    
    def add_annotations(self, fig_name: str, annotations: List[PlotAnnotation]):
        """
        Add annotations to existing plot.
        
        Args:
            fig_name: Name of the figure to annotate
            annotations: List of annotations to add
        """
        if fig_name not in self.figures_:
            raise ValueError(f"Figure '{fig_name}' not found")
        
        fig = self.figures_[fig_name]
        
        for annotation in annotations:
            # Find appropriate axis (first one for now)
            ax = fig.axes[0]
            
            # Add marker
            ax.scatter(annotation.time, annotation.amplitude or 0,
                      color=annotation.color, marker=annotation.marker,
                      s=annotation.size**2, zorder=10)
            
            # Add text
            if annotation.text:
                ax.annotate(annotation.text, 
                           xy=(annotation.time, annotation.amplitude or 0),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=annotation.color, alpha=0.7),
                           arrowprops=dict(arrowstyle='->', color=annotation.color))
        
        fig.canvas.draw()
    
    def export_figure(self, fig_name: str, filename: str = None, 
                     format: str = None, dpi: int = None):
        """
        Export figure to file.
        
        Args:
            fig_name: Name of figure to export
            filename: Output filename (None = auto-generate)
            format: Export format ('png', 'pdf', 'svg', etc.)
            dpi: Resolution for raster formats
        """
        if fig_name not in self.figures_:
            raise ValueError(f"Figure '{fig_name}' not found")
        
        fig = self.figures_[fig_name]
        
        # Set defaults
        if filename is None:
            filename = f"{fig_name}_plot"
        
        if format is None:
            format = self.config.export_format
        
        if dpi is None:
            dpi = self.config.export_dpi
        
        # Add extension if not present
        if not filename.endswith(f'.{format}'):
            filename = f"{filename}.{format}"
        
        # Export
        fig.savefig(filename, format=format, dpi=dpi, 
                   bbox_inches='tight', facecolor='white')
        
        print(f"Figure exported to {filename}")
    
    def clear_figures(self):
        """Clear all stored figures"""
        for fig in self.figures_.values():
            plt.close(fig)
        
        self.figures_.clear()
        self.axes_.clear()
        self.annotations_.clear()
        self.selectors_.clear()
        self.buttons_.clear()
    
    def get_figure_info(self) -> Dict[str, Dict]:
        """Get information about all stored figures"""
        info = {}
        for name, fig in self.figures_.items():
            info[name] = {
                'n_axes': len(fig.axes),
                'figsize': fig.get_size_inches(),
                'dpi': fig.dpi,
                'title': fig._suptitle.get_text() if fig._suptitle else None
            }
        return info