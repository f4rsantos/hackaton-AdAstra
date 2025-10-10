#!/usr/bin/env python3
"""
Improved Confidence Configuration

This module provides optimized confidence thresholds based on analysis results.
"""

import numpy as np

class ImprovedConfidenceConfig:
    """
    Optimized confidence configuration based on analysis results
    """
    
    def __init__(self):
        """Initialize with optimized defaults"""
        
        # Based on analysis results:
        # - 0.3/0.1 gives 388 detections (too many false positives)
        # - 0.62/0.35 gives 17 detections (too restrictive) 
        # - 0.5/0.2 gives 49 detections (better balance)
        
        self.configs = {
            'permissive': {
                'threshold': 0.4,
                'min_confidence': 0.15,
                'description': 'More detections, higher false positive rate',
                'use_case': 'Initial detection, broad search'
            },
            'balanced': {
                'threshold': 0.55,
                'min_confidence': 0.25,
                'description': 'Good balance of precision and recall',
                'use_case': 'General purpose detection'
            },
            'strict': {
                'threshold': 0.65,
                'min_confidence': 0.35,
                'description': 'Fewer false positives, may miss some matches',
                'use_case': 'High precision required'
            },
            'very_strict': {
                'threshold': 0.75,
                'min_confidence': 0.45,
                'description': 'Very conservative detection',
                'use_case': 'Critical applications'
            }
        }
        
        # Smart Astra optimal defaults (updated based on analysis)
        self.smart_defaults = {
            'threshold': 0.55,      # Better than 0.62 (less restrictive)
            'min_confidence': 0.25,  # Better than 0.35 (less restrictive)
            'border_threshold': 0.6,
            'overlap_sensitivity': 0.3
        }
    
    def get_config(self, mode='balanced'):
        """Get configuration for specified mode"""
        return self.configs.get(mode, self.configs['balanced'])
    
    def get_smart_defaults(self):
        """Get Smart Astra optimized defaults"""
        return self.smart_defaults.copy()
    
    def analyze_confidence_distribution(self, confidence_values):
        """
        Analyze confidence distribution and suggest optimal thresholds
        
        Args:
            confidence_values: List of confidence values from detections
            
        Returns:
            dict: Analysis results and recommendations
        """
        if not confidence_values:
            return {
                'status': 'no_data',
                'recommendation': 'permissive'
            }
        
        confidences = np.array(confidence_values)
        
        analysis = {
            'status': 'success',
            'count': len(confidences),
            'mean': np.mean(confidences),
            'std': np.std(confidences),
            'min': np.min(confidences),
            'max': np.max(confidences),
            'median': np.median(confidences),
            'q25': np.percentile(confidences, 25),
            'q75': np.percentile(confidences, 75)
        }
        
        # Suggest configuration based on distribution
        if analysis['mean'] < 0.3:
            recommendation = 'permissive'
            reason = 'Low average confidence detected'
        elif analysis['mean'] > 0.6:
            recommendation = 'strict'
            reason = 'High average confidence detected'
        else:
            recommendation = 'balanced'
            reason = 'Moderate confidence distribution'
        
        analysis['recommendation'] = recommendation
        analysis['reason'] = reason
        analysis['suggested_config'] = self.get_config(recommendation)
        
        return analysis
    
    def suggest_thresholds(self, expected_detections=None, false_positive_tolerance='medium'):
        """
        Suggest optimal thresholds based on requirements
        
        Args:
            expected_detections: Expected number of detections (None for unknown)
            false_positive_tolerance: 'low', 'medium', 'high'
            
        Returns:
            dict: Suggested configuration
        """
        tolerance_map = {
            'low': 'strict',
            'medium': 'balanced', 
            'high': 'permissive'
        }
        
        base_config = tolerance_map.get(false_positive_tolerance, 'balanced')
        config = self.get_config(base_config).copy()
        
        # Adjust based on expected detections
        if expected_detections is not None:
            if expected_detections > 50:
                # Many expected detections - be more permissive
                config['threshold'] *= 0.9
                config['min_confidence'] *= 0.85
            elif expected_detections < 10:
                # Few expected detections - be more strict
                config['threshold'] *= 1.1
                config['min_confidence'] *= 1.15
        
        # Clamp values to reasonable ranges
        config['threshold'] = np.clip(config['threshold'], 0.1, 0.95)
        config['min_confidence'] = np.clip(config['min_confidence'], 0.05, 0.8)
        
        return config

def test_improved_confidence():
    """Test the improved confidence configuration"""
    
    config = ImprovedConfidenceConfig()
    
    # Test with different confidence distributions
    test_cases = [
        {
            'name': 'Low confidence scenario',
            'confidences': [0.1, 0.15, 0.2, 0.12, 0.18, 0.22, 0.08, 0.25]
        },
        {
            'name': 'High confidence scenario', 
            'confidences': [0.7, 0.8, 0.75, 0.65, 0.9, 0.72, 0.68, 0.85]
        },
        {
            'name': 'Mixed confidence scenario',
            'confidences': [0.3, 0.45, 0.6, 0.2, 0.55, 0.4, 0.35, 0.5, 0.25, 0.65]
        }
    ]

if __name__ == "__main__":
    test_improved_confidence()