#!/usr/bin/env python3
"""
EEG Preprocessing Pipeline
=========================

Comprehensive preprocessing pipeline integrating:
- Advanced filtering
- Re-referencing
- Channel management
- Bad channel detection and interpolation
- Montage management

Author: porfanid
Version: 1.0
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import mne
import numpy as np

from .filters import EEGFilterProcessor, FilterConfig, FilterPresets
from .referencing import EEGReferenceProcessor, ReferenceConfig, ReferencePresets
from .channel_management import EEGChannelManager

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("WARNING")


class PreprocessingConfig:
    """Configuration for preprocessing pipeline"""
    
    def __init__(
        self,
        # Filtering
        apply_filters: bool = True,
        filter_configs: Optional[List[FilterConfig]] = None,
        
        # Channel management
        detect_bad_channels: bool = True,
        interpolate_bad_channels: bool = True,
        load_montage: bool = True,
        montage_name: str = 'standard_1020',
        
        # Re-referencing
        apply_reference: bool = True,
        reference_config: Optional[ReferenceConfig] = None,
        
        # General
        copy_data: bool = True,
        verbose: bool = False
    ):
        """
        Initialize preprocessing configuration
        
        Args:
            apply_filters: Whether to apply filtering
            filter_configs: List of filter configurations to apply
            detect_bad_channels: Whether to detect bad channels
            interpolate_bad_channels: Whether to interpolate bad channels
            load_montage: Whether to load channel montage
            montage_name: Name of montage to load
            apply_reference: Whether to apply re-referencing
            reference_config: Reference configuration
            copy_data: Whether to copy data at each step
            verbose: Whether to print verbose output
        """
        self.apply_filters = apply_filters
        self.filter_configs = filter_configs or FilterPresets.get_preprocessing_preset()
        self.detect_bad_channels = detect_bad_channels
        self.interpolate_bad_channels = interpolate_bad_channels
        self.load_montage = load_montage
        self.montage_name = montage_name
        self.apply_reference = apply_reference
        self.reference_config = reference_config or ReferencePresets.get_clinical_preset()
        self.copy_data = copy_data
        self.verbose = verbose


class PreprocessingPipeline:
    """
    Complete EEG preprocessing pipeline
    
    Integrates all preprocessing steps in a configurable pipeline:
    1. Channel management and bad channel detection
    2. Montage loading
    3. Bad channel interpolation
    4. Filtering
    5. Re-referencing
    """
    
    def __init__(self):
        """Initialize preprocessing pipeline"""
        self.filter_processor = EEGFilterProcessor()
        self.reference_processor = EEGReferenceProcessor()
        self.channel_manager = EEGChannelManager()
        
        self.pipeline_history = []
    
    def run_pipeline(
        self, 
        raw: mne.io.Raw, 
        config: PreprocessingConfig
    ) -> Tuple[mne.io.Raw, Dict]:
        """
        Run complete preprocessing pipeline
        
        Args:
            raw: Raw EEG data
            config: Preprocessing configuration
            
        Returns:
            Tuple of (preprocessed_raw, pipeline_results)
        """
        if config.copy_data:
            raw = raw.copy()
        
        pipeline_results = {
            'original_info': self._get_raw_info(raw),
            'steps_completed': [],
            'step_results': {},
            'success': False,
            'error': None
        }
        
        try:
            # Step 1: Channel analysis
            if config.verbose:
                print("Step 1: Analyzing channels...")
            
            channel_analysis = self.channel_manager.analyze_channels(raw)
            pipeline_results['step_results']['channel_analysis'] = channel_analysis
            pipeline_results['steps_completed'].append('channel_analysis')
            
            # Step 2: Montage loading
            if config.load_montage:
                if config.verbose:
                    print("Step 2: Loading montage...")
                
                try:
                    raw, montage_info = self.channel_manager.montage_manager.load_standard_montage(
                        raw, config.montage_name
                    )
                    pipeline_results['step_results']['montage'] = montage_info
                    pipeline_results['steps_completed'].append('montage')
                except Exception as e:
                    if config.verbose:
                        print(f"Montage loading failed: {e}")
                    pipeline_results['step_results']['montage'] = {'success': False, 'error': str(e)}
            
            # Step 3: Bad channel detection and interpolation
            if config.detect_bad_channels:
                if config.verbose:
                    print("Step 3: Detecting and handling bad channels...")
                
                bad_channels_info = channel_analysis['bad_channels']
                all_bad_channels = set()
                for bad_type, channels in bad_channels_info.items():
                    all_bad_channels.update(channels)
                
                if all_bad_channels and config.interpolate_bad_channels:
                    try:
                        raw, interp_info = self.channel_manager.channel_interpolator.interpolate_channels(
                            raw, list(all_bad_channels), copy=False
                        )
                        pipeline_results['step_results']['interpolation'] = interp_info
                        pipeline_results['steps_completed'].append('interpolation')
                    except Exception as e:
                        if config.verbose:
                            print(f"Channel interpolation failed: {e}")
                        pipeline_results['step_results']['interpolation'] = {'success': False, 'error': str(e)}
                else:
                    pipeline_results['step_results']['bad_channels'] = bad_channels_info
                    pipeline_results['steps_completed'].append('bad_channel_detection')
            
            # Step 4: Filtering
            if config.apply_filters:
                if config.verbose:
                    print("Step 4: Applying filters...")
                
                filter_results = []
                for i, filter_config in enumerate(config.filter_configs):
                    try:
                        raw, filter_info = self.filter_processor.apply_filter(
                            raw, filter_config, copy=False
                        )
                        filter_results.append(filter_info)
                        if config.verbose:
                            print(f"  Applied {filter_config.filter_type} filter")
                    except Exception as e:
                        filter_info = {
                            'success': False,
                            'error': str(e),
                            'type': filter_config.filter_type
                        }
                        filter_results.append(filter_info)
                        if config.verbose:
                            print(f"  Filter {filter_config.filter_type} failed: {e}")
                
                pipeline_results['step_results']['filtering'] = filter_results
                pipeline_results['steps_completed'].append('filtering')
            
            # Step 5: Re-referencing
            if config.apply_reference:
                if config.verbose:
                    print("Step 5: Applying re-referencing...")
                
                try:
                    raw, ref_info = self.reference_processor.apply_reference(
                        raw, config.reference_config
                    )
                    pipeline_results['step_results']['referencing'] = ref_info
                    pipeline_results['steps_completed'].append('referencing')
                except Exception as e:
                    if config.verbose:
                        print(f"Re-referencing failed: {e}")
                    pipeline_results['step_results']['referencing'] = {'success': False, 'error': str(e)}
            
            # Final step: Update results
            pipeline_results['final_info'] = self._get_raw_info(raw)
            pipeline_results['success'] = True
            
            # Store in history
            self.pipeline_history.append(pipeline_results.copy())
            
            if config.verbose:
                print("Preprocessing pipeline completed successfully!")
            
        except Exception as e:
            pipeline_results['error'] = str(e)
            pipeline_results['success'] = False
            if config.verbose:
                print(f"Pipeline failed: {e}")
            raise RuntimeError(f"Preprocessing pipeline failed: {str(e)}")
        
        return raw, pipeline_results
    
    def _get_raw_info(self, raw: mne.io.Raw) -> Dict:
        """Get summary information about raw data"""
        ch_types = raw.get_channel_types()
        return {
            'n_channels': len(raw.ch_names),
            'n_times': raw.n_times,
            'sfreq': raw.info['sfreq'],
            'duration': raw.times[-1],
            'channel_types': {ch_type: ch_types.count(ch_type) for ch_type in set(ch_types)},
            'has_montage': raw.info.get_montage() is not None,
            'bad_channels': raw.info['bads'].copy()
        }
    
    def create_custom_pipeline(
        self,
        raw: mne.io.Raw,
        steps: List[str],
        **kwargs
    ) -> PreprocessingConfig:
        """
        Create custom preprocessing configuration based on data analysis
        
        Args:
            raw: Raw EEG data for analysis
            steps: List of preprocessing steps to include
            **kwargs: Additional configuration parameters
            
        Returns:
            Custom preprocessing configuration
        """
        # Analyze the data
        analysis = self.channel_manager.analyze_channels(raw)
        recommendations = self.channel_manager.get_channel_recommendations(raw)
        
        # Create configuration based on analysis
        config_params = {
            'apply_filters': 'filtering' in steps,
            'detect_bad_channels': 'bad_channel_detection' in steps,
            'interpolate_bad_channels': 'interpolation' in steps,
            'load_montage': 'montage' in steps,
            'apply_reference': 'referencing' in steps,
            'copy_data': True,
            'verbose': kwargs.get('verbose', False)
        }
        
        # Customize based on data characteristics
        sfreq = raw.info['sfreq']
        
        # Select appropriate filters based on sampling rate
        if config_params['apply_filters']:
            if sfreq >= 500:
                # High sampling rate - use standard preprocessing
                config_params['filter_configs'] = FilterPresets.get_preprocessing_preset()
            elif sfreq >= 250:
                # Medium sampling rate - lighter filtering
                config_params['filter_configs'] = [
                    FilterConfig('highpass', freq_low=0.5),
                    FilterConfig('lowpass', freq_low=min(40.0, sfreq/6)),
                    FilterConfig('notch', freq_notch=50.0)
                ]
            else:
                # Low sampling rate - minimal filtering
                config_params['filter_configs'] = [
                    FilterConfig('highpass', freq_low=0.1),
                    FilterConfig('lowpass', freq_low=min(30.0, sfreq/8))
                ]
        
        # Select montage based on recommendations
        if config_params['load_montage']:
            montage_suggestions = recommendations.get('montage_suggestions', [])
            if montage_suggestions:
                best_montage = max(montage_suggestions, key=lambda x: x['match_percentage'])
                config_params['montage_name'] = best_montage['montage']
        
        # Select reference based on data characteristics
        if config_params['apply_reference']:
            n_eeg_channels = analysis['channel_types'].get('eeg', 0)
            if n_eeg_channels >= 16:
                config_params['reference_config'] = ReferencePresets.get_clinical_preset()
            else:
                config_params['reference_config'] = ReferencePresets.get_research_preset()
        
        # Override with any provided kwargs
        config_params.update(kwargs)
        
        return PreprocessingConfig(**config_params)
    
    def get_pipeline_summary(self, pipeline_results: Dict) -> str:
        """
        Get human-readable summary of pipeline results
        
        Args:
            pipeline_results: Results from run_pipeline
            
        Returns:
            Formatted summary string
        """
        summary = ["EEG Preprocessing Pipeline Summary", "=" * 40]
        
        if pipeline_results['success']:
            summary.append("✅ Pipeline completed successfully")
        else:
            summary.append("❌ Pipeline failed")
            if pipeline_results['error']:
                summary.append(f"Error: {pipeline_results['error']}")
        
        summary.append(f"\nSteps completed: {len(pipeline_results['steps_completed'])}")
        for step in pipeline_results['steps_completed']:
            summary.append(f"  ✓ {step}")
        
        # Original vs final info
        original = pipeline_results['original_info']
        final = pipeline_results.get('final_info', original)
        
        summary.extend([
            "\nData Changes:",
            f"  Channels: {original['n_channels']} → {final['n_channels']}",
            f"  Duration: {original['duration']:.1f}s (unchanged)",
            f"  Sampling rate: {original['sfreq']:.1f} Hz (unchanged)"
        ])
        
        # Step-specific results
        step_results = pipeline_results['step_results']
        
        if 'filtering' in step_results:
            filter_results = step_results['filtering']
            successful_filters = [f for f in filter_results if f['success']]
            summary.append(f"  Filters applied: {len(successful_filters)}/{len(filter_results)}")
        
        if 'interpolation' in step_results:
            interp_info = step_results['interpolation']
            if interp_info['success']:
                n_interp = interp_info['n_interpolated']
                summary.append(f"  Channels interpolated: {n_interp}")
        
        if 'referencing' in step_results:
            ref_info = step_results['referencing']
            if ref_info['success']:
                summary.append(f"  Reference: {ref_info['type']}")
        
        return "\n".join(summary)
    
    def get_pipeline_history(self) -> List[Dict]:
        """Get history of pipeline runs"""
        return self.pipeline_history.copy()
    
    def clear_history(self):
        """Clear pipeline history"""
        self.pipeline_history.clear()
        self.filter_processor.clear_history()
        self.reference_processor.clear_history()


class PreprocessingPresets:
    """Predefined preprocessing presets for common scenarios"""
    
    @staticmethod
    def get_clinical_preset() -> PreprocessingConfig:
        """Clinical EEG preprocessing preset"""
        return PreprocessingConfig(
            apply_filters=True,
            filter_configs=FilterPresets.get_preprocessing_preset(),
            detect_bad_channels=True,
            interpolate_bad_channels=True,
            load_montage=True,
            montage_name='standard_1020',
            apply_reference=True,
            reference_config=ReferencePresets.get_clinical_preset(),
            verbose=True
        )
    
    @staticmethod
    def get_research_preset() -> PreprocessingConfig:
        """Research EEG preprocessing preset"""
        return PreprocessingConfig(
            apply_filters=True,
            filter_configs=FilterPresets.get_preprocessing_preset(),
            detect_bad_channels=True,
            interpolate_bad_channels=True,
            load_montage=True,
            montage_name='standard_1020',
            apply_reference=True,
            reference_config=ReferencePresets.get_research_preset(),
            verbose=True
        )
    
    @staticmethod
    def get_erp_preset() -> PreprocessingConfig:
        """ERP analysis preprocessing preset"""
        return PreprocessingConfig(
            apply_filters=True,
            filter_configs=FilterPresets.get_erp_preset(),
            detect_bad_channels=True,
            interpolate_bad_channels=True,
            load_montage=True,
            montage_name='standard_1020',
            apply_reference=True,
            reference_config=ReferencePresets.get_erp_preset(),
            verbose=True
        )
    
    @staticmethod
    def get_sleep_preset() -> PreprocessingConfig:
        """Sleep study preprocessing preset"""
        return PreprocessingConfig(
            apply_filters=True,
            filter_configs=[
                FilterConfig('highpass', freq_low=0.3),
                FilterConfig('lowpass', freq_low=35.0),
                FilterConfig('notch', freq_notch=50.0)
            ],
            detect_bad_channels=True,
            interpolate_bad_channels=False,  # Usually don't interpolate in sleep studies
            load_montage=True,
            montage_name='standard_1020',
            apply_reference=True,
            reference_config=ReferencePresets.get_sleep_preset(),
            verbose=True
        )
    
    @staticmethod
    def get_minimal_preset() -> PreprocessingConfig:
        """Minimal preprocessing (just basic filtering)"""
        return PreprocessingConfig(
            apply_filters=True,
            filter_configs=[
                FilterConfig('highpass', freq_low=0.5),
                FilterConfig('lowpass', freq_low=40.0)
            ],
            detect_bad_channels=False,
            interpolate_bad_channels=False,
            load_montage=False,
            apply_reference=False,
            verbose=False
        )