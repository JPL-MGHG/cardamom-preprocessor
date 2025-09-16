"""
Data Validation and Quality Control for CARDAMOM Preprocessing

This module provides comprehensive validation functions for ensuring data quality
in CARDAMOM preprocessing workflows. It includes spatial coverage validation,
temporal continuity checks, physical range validation, and QA report generation.
"""

import numpy as np
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path


def validate_spatial_coverage(data: np.ndarray,
                            expected_grid: Dict[str, Any],
                            min_coverage_fraction: float = 0.95) -> Dict[str, Any]:
    """
    Ensure data covers expected spatial domain with sufficient coverage.

    Args:
        data: Input data array (2D or 3D)
        expected_grid: Dictionary with expected grid information
        min_coverage_fraction: Minimum fraction of valid data points required

    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'test_name': 'spatial_coverage',
        'status': 'pass',
        'expected_shape': expected_grid.get('shape'),
        'actual_shape': data.shape[:2],
        'total_grid_points': data.size if data.ndim == 2 else data.shape[0] * data.shape[1],
        'valid_data_points': 0,
        'missing_data_points': 0,
        'coverage_fraction': 0.0,
        'messages': []
    }

    # Calculate spatial coverage
    if data.ndim == 2:
        valid_mask = ~np.isnan(data)
        num_valid = np.sum(valid_mask)
        total_points = data.size
    elif data.ndim == 3:
        # For 3D data, check coverage across all time steps
        valid_mask = ~np.isnan(data)
        num_valid = np.sum(np.any(valid_mask, axis=2))  # Points valid in any time step
        total_points = data.shape[0] * data.shape[1]
    else:
        raise ValueError(f"Unsupported data dimensions: {data.ndim}")

    validation_results['valid_data_points'] = int(num_valid)
    validation_results['missing_data_points'] = int(total_points - num_valid)
    validation_results['coverage_fraction'] = float(num_valid / total_points)

    # Check against expected shape
    if validation_results['expected_shape'] is not None:
        if validation_results['actual_shape'] != tuple(validation_results['expected_shape']):
            validation_results['status'] = 'fail'
            validation_results['messages'].append(
                f"Data shape {validation_results['actual_shape']} does not match "
                f"expected shape {validation_results['expected_shape']}"
            )

    # Check coverage fraction
    if validation_results['coverage_fraction'] < min_coverage_fraction:
        validation_results['status'] = 'fail'
        validation_results['messages'].append(
            f"Spatial coverage {validation_results['coverage_fraction']:.3f} below "
            f"minimum required {min_coverage_fraction:.3f}"
        )

    if validation_results['status'] == 'pass':
        validation_results['messages'].append(
            f"Spatial coverage validation passed with {validation_results['coverage_fraction']:.3f} coverage"
        )

    return validation_results


def check_temporal_continuity(data: np.ndarray,
                            time_coordinates: np.ndarray,
                            expected_time_range: Optional[Tuple[float, float]] = None,
                            max_gap_tolerance: float = 2.0) -> Dict[str, Any]:
    """
    Verify temporal coverage and identify gaps.

    Args:
        data: Input data array with time dimension
        time_coordinates: Array of time coordinates
        expected_time_range: Expected (start, end) time range
        max_gap_tolerance: Maximum acceptable gap in time units

    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'test_name': 'temporal_continuity',
        'status': 'pass',
        'time_range': [float(time_coordinates.min()), float(time_coordinates.max())],
        'expected_time_range': expected_time_range,
        'num_time_steps': len(time_coordinates),
        'time_gaps_detected': [],
        'largest_gap': 0.0,
        'messages': []
    }

    # Check against expected time range
    if expected_time_range is not None:
        expected_start, expected_end = expected_time_range
        actual_start, actual_end = validation_results['time_range']

        if actual_start > expected_start + 0.1 or actual_end < expected_end - 0.1:
            validation_results['status'] = 'warning'
            validation_results['messages'].append(
                f"Time range [{actual_start:.1f}, {actual_end:.1f}] does not fully cover "
                f"expected range [{expected_start:.1f}, {expected_end:.1f}]"
            )

    # Check for temporal gaps
    if len(time_coordinates) > 1:
        time_diffs = np.diff(time_coordinates)
        median_time_step = np.median(time_diffs)

        # Identify gaps larger than expected
        gap_indices = np.where(time_diffs > median_time_step * max_gap_tolerance)[0]

        if len(gap_indices) > 0:
            validation_results['status'] = 'warning'
            for gap_idx in gap_indices:
                gap_size = time_diffs[gap_idx]
                gap_location = time_coordinates[gap_idx]
                validation_results['time_gaps_detected'].append({
                    'location': float(gap_location),
                    'gap_size': float(gap_size),
                    'expected_size': float(median_time_step)
                })

            validation_results['largest_gap'] = float(np.max(time_diffs[gap_indices]))
            validation_results['messages'].append(
                f"Detected {len(gap_indices)} temporal gaps, largest: {validation_results['largest_gap']:.2f}"
            )

    if validation_results['status'] == 'pass':
        validation_results['messages'].append(
            f"Temporal continuity check passed for {validation_results['num_time_steps']} time steps"
        )

    return validation_results


def validate_physical_ranges(data: np.ndarray,
                           variable_type: str,
                           custom_ranges: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, Any]:
    """
    Check if data values are within physically reasonable ranges.

    Args:
        data: Input data array
        variable_type: Type of variable (e.g., 'temperature', 'precipitation')
        custom_ranges: Optional custom range definitions

    Returns:
        Dictionary with validation results
    """
    # Default physical ranges for common variables
    default_ranges = {
        'temperature': (-100.0, 60.0),  # °C
        'temperature_kelvin': (173.0, 333.0),  # K
        'precipitation': (0.0, 500.0),  # mm/day
        'radiation': (0.0, 400.0),  # W/m²
        'vpd': (0.0, 100.0),  # hPa
        'co2': (300.0, 500.0),  # ppm
        'wind_speed': (0.0, 100.0),  # m/s
        'pressure': (800.0, 1100.0),  # hPa
        'humidity': (0.0, 100.0),  # %
        'carbon_flux': (-50.0, 50.0)  # gC/m²/day
    }

    # Use custom ranges if provided
    physical_ranges = custom_ranges or default_ranges

    validation_results = {
        'test_name': 'physical_ranges',
        'variable_type': variable_type,
        'status': 'pass',
        'data_range': [float(np.nanmin(data)), float(np.nanmax(data))],
        'expected_range': physical_ranges.get(variable_type),
        'num_below_min': 0,
        'num_above_max': 0,
        'total_points': data.size,
        'messages': []
    }

    if variable_type not in physical_ranges:
        validation_results['status'] = 'warning'
        validation_results['messages'].append(
            f"No physical range defined for variable type '{variable_type}'"
        )
        return validation_results

    min_value, max_value = physical_ranges[variable_type]
    validation_results['expected_range'] = [min_value, max_value]

    # Check for values outside physical ranges
    below_min_mask = data < min_value
    above_max_mask = data > max_value

    num_below_min = np.sum(below_min_mask & ~np.isnan(data))
    num_above_max = np.sum(above_max_mask & ~np.isnan(data))

    validation_results['num_below_min'] = int(num_below_min)
    validation_results['num_above_max'] = int(num_above_max)

    if num_below_min > 0:
        validation_results['status'] = 'fail'
        validation_results['messages'].append(
            f"{num_below_min} values below physical minimum {min_value} for {variable_type}"
        )

    if num_above_max > 0:
        validation_results['status'] = 'fail'
        validation_results['messages'].append(
            f"{num_above_max} values above physical maximum {max_value} for {variable_type}"
        )

    if validation_results['status'] == 'pass':
        validation_results['messages'].append(
            f"Physical range validation passed for {variable_type}"
        )

    return validation_results


def validate_data_consistency(data_dict: Dict[str, np.ndarray],
                            coordinate_consistency: bool = True) -> Dict[str, Any]:
    """
    Validate consistency across multiple data variables.

    Args:
        data_dict: Dictionary of data arrays with variable names as keys
        coordinate_consistency: Whether to check spatial/temporal coordinate consistency

    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'test_name': 'data_consistency',
        'status': 'pass',
        'num_variables': len(data_dict),
        'consistent_shapes': True,
        'shapes': {},
        'messages': []
    }

    if len(data_dict) == 0:
        validation_results['status'] = 'warning'
        validation_results['messages'].append("No data provided for consistency check")
        return validation_results

    # Check shape consistency
    shapes = {var_name: data.shape for var_name, data in data_dict.items()}
    validation_results['shapes'] = shapes

    reference_shape = list(shapes.values())[0]
    inconsistent_shapes = [name for name, shape in shapes.items() if shape != reference_shape]

    if inconsistent_shapes:
        validation_results['status'] = 'fail'
        validation_results['consistent_shapes'] = False
        validation_results['messages'].append(
            f"Inconsistent shapes detected: {inconsistent_shapes} do not match reference shape {reference_shape}"
        )

    if validation_results['status'] == 'pass':
        validation_results['messages'].append(
            f"Data consistency check passed for {validation_results['num_variables']} variables"
        )

    return validation_results


def generate_qa_report(processed_data: Dict[str, Any],
                      validation_results: List[Dict[str, Any]],
                      output_path: str) -> None:
    """
    Generate comprehensive quality assurance report for processed data.

    Args:
        processed_data: Dictionary containing processed data and metadata
        validation_results: List of validation result dictionaries
        output_path: Path for output QA report file
    """
    qa_report = {
        'qa_report_metadata': {
            'creation_date': datetime.now().isoformat(),
            'cardamom_version': 'v1.0',
            'report_type': 'data_quality_assessment'
        },
        'data_summary': {
            'num_variables': len(processed_data.get('variables', {})),
            'spatial_dimensions': processed_data.get('spatial_dims'),
            'temporal_dimensions': processed_data.get('temporal_dims'),
            'processing_date': processed_data.get('processing_date')
        },
        'validation_summary': {
            'total_tests': len(validation_results),
            'tests_passed': sum(1 for result in validation_results if result['status'] == 'pass'),
            'tests_failed': sum(1 for result in validation_results if result['status'] == 'fail'),
            'tests_warning': sum(1 for result in validation_results if result['status'] == 'warning')
        },
        'detailed_results': validation_results,
        'recommendations': []
    }

    # Generate recommendations based on validation results
    failed_tests = [result for result in validation_results if result['status'] == 'fail']
    warning_tests = [result for result in validation_results if result['status'] == 'warning']

    if failed_tests:
        qa_report['recommendations'].append(
            "CRITICAL: Failed validation tests detected. Review data quality before proceeding."
        )

    if warning_tests:
        qa_report['recommendations'].append(
            "WARNING: Some validation tests raised warnings. Consider reviewing data quality."
        )

    if not failed_tests and not warning_tests:
        qa_report['recommendations'].append(
            "All validation tests passed. Data quality is acceptable for CARDAMOM processing."
        )

    # Write report to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(qa_report, f, indent=2, default=str)

    print(f"QA report generated: {output_path}")


class QualityAssurance:
    """
    Comprehensive quality assurance system for CARDAMOM data processing.

    This class provides a unified interface for running all quality control
    checks and generating comprehensive QA reports.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize QA system with configuration.

        Args:
            config: Quality control configuration dictionary
        """
        self.config = config
        self.enable_validation = config.get('enable_validation', True)
        self.physical_range_checks = config.get('physical_range_checks', True)
        self.spatial_continuity_checks = config.get('spatial_continuity_checks', True)
        self.temporal_continuity_checks = config.get('temporal_continuity_checks', True)
        self.missing_data_tolerance = config.get('missing_data_tolerance', 0.05)

    def run_full_qa_suite(self,
                         data_dict: Dict[str, np.ndarray],
                         metadata: Dict[str, Any],
                         output_dir: str) -> Dict[str, Any]:
        """
        Run complete quality assurance suite on processed data.

        Args:
            data_dict: Dictionary of processed data arrays
            metadata: Metadata about the processing
            output_dir: Directory for QA output files

        Returns:
            Dictionary with overall QA results
        """
        if not self.enable_validation:
            return {'status': 'skipped', 'message': 'QA validation disabled'}

        validation_results = []
        overall_status = 'pass'

        # Run individual validation tests
        for var_name, data_array in data_dict.items():
            var_metadata = metadata.get('variables', {}).get(var_name, {})

            # Spatial coverage validation
            if self.spatial_continuity_checks:
                spatial_result = validate_spatial_coverage(
                    data_array,
                    var_metadata.get('grid_info', {}),
                    1.0 - self.missing_data_tolerance
                )
                spatial_result['variable'] = var_name
                validation_results.append(spatial_result)

            # Physical range validation
            if self.physical_range_checks:
                variable_type = var_metadata.get('type', var_name)
                range_result = validate_physical_ranges(data_array, variable_type)
                range_result['variable'] = var_name
                validation_results.append(range_result)

            # Temporal continuity (for 3D data)
            if self.temporal_continuity_checks and data_array.ndim == 3:
                time_coords = metadata.get('time_coordinates')
                if time_coords is not None:
                    temporal_result = check_temporal_continuity(
                        data_array,
                        time_coords,
                        var_metadata.get('expected_time_range')
                    )
                    temporal_result['variable'] = var_name
                    validation_results.append(temporal_result)

        # Data consistency check across variables
        consistency_result = validate_data_consistency(data_dict)
        validation_results.append(consistency_result)

        # Determine overall status
        if any(result['status'] == 'fail' for result in validation_results):
            overall_status = 'fail'
        elif any(result['status'] == 'warning' for result in validation_results):
            overall_status = 'warning'

        # Generate QA report
        qa_output_path = Path(output_dir) / 'qa_report.json'
        generate_qa_report(metadata, validation_results, str(qa_output_path))

        return {
            'status': overall_status,
            'num_tests': len(validation_results),
            'validation_results': validation_results,
            'qa_report_path': str(qa_output_path)
        }