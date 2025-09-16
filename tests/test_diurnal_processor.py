"""
Test Suite for Diurnal Processing Modules

Basic functionality tests for Phase 4 diurnal processing implementation.
Focus on module imports, basic execution, and integration testing
without extensive MATLAB validation.

Test Coverage:
- Module imports and initialization
- Basic processing workflow execution
- Configuration and data flow
- Output file creation
"""

import unittest
import numpy as np
import tempfile
import shutil
import os
from unittest.mock import patch, MagicMock

# Import diurnal processing modules
try:
    from src.diurnal_processor import DiurnalProcessor, DiurnalFluxData
    from src.cms_flux_loader import CMSFluxLoader
    from src.met_driver_loader import ERA5DiurnalLoader
    from src.diurnal_calculator import DiurnalCalculator
    from src.gfed_diurnal_loader import GFEDDiurnalLoader
    from src.diurnal_output_writers import DiurnalFluxWriter
except ImportError as e:
    print(f"Warning: Could not import diurnal modules: {e}")


class TestDiurnalProcessor(unittest.TestCase):
    """Test DiurnalProcessor main class functionality."""

    def setUp(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.processor = DiurnalProcessor()

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_diurnal_processor_initialization(self):
        """Test DiurnalProcessor initialization."""
        processor = DiurnalProcessor()

        # Check basic attributes
        self.assertIsNotNone(processor.aux_data)
        self.assertIsNotNone(processor.coordinate_grid)
        self.assertEqual(len(processor.years_range), 2)
        self.assertEqual(len(processor.region_bounds), 4)

    def test_auxiliary_data_setup(self):
        """Test auxiliary data structure setup."""
        aux_data = self.processor._setup_auxiliary_data()

        # Check required keys
        required_keys = ['destination_path', 'lon_range', 'lat_range', 'grid_resolution']
        for key in required_keys:
            self.assertIn(key, aux_data)

        # Check CONUS coordinate ranges
        self.assertEqual(len(aux_data['lon_range']), 2)
        self.assertEqual(len(aux_data['lat_range']), 2)
        self.assertEqual(aux_data['grid_resolution'], 0.5)

    @patch('src.diurnal_processor.CMSFluxLoader')
    def test_process_diurnal_fluxes_basic(self, mock_cms_loader):
        """Test basic diurnal flux processing workflow."""
        # Mock CMS flux data
        mock_fluxes = {
            'GPP': np.random.rand(120, 160, 240),  # CONUS grid, 20 years monthly
            'REC': np.random.rand(120, 160, 240),
            'FIR': np.random.rand(120, 160, 240),
            'NEE': np.random.rand(120, 160, 240),
            'NBE': np.random.rand(120, 160, 240),
            'GPPunc': np.random.rand(120, 160, 240) * 0.1,
            'RECunc': np.random.rand(120, 160, 240) * 0.1,
            'FIRunc': np.random.rand(120, 160, 240) * 0.1,
            'NEEunc': np.random.rand(120, 160, 240) * 0.1,
            'NBEunc': np.random.rand(120, 160, 240) * 0.1
        }

        mock_cms_loader.return_value.load_monthly_fluxes.return_value = mock_fluxes

        # Mock other dependencies
        with patch('src.diurnal_processor.ERA5DiurnalLoader'), \
             patch('src.diurnal_processor.GFEDDiurnalLoader'), \
             patch('src.diurnal_processor.DiurnalCalculator'), \
             patch('src.diurnal_processor.DiurnalFluxWriter'):

            # Test basic processing call
            try:
                result = self.processor.process_diurnal_fluxes(
                    experiment_number=1,
                    years=[2020],
                    months=[1]
                )

                # Check result structure
                self.assertIsInstance(result, DiurnalFluxData)
                self.assertIsNotNone(result.monthly_fluxes)
                self.assertIsNotNone(result.processing_metadata)

            except Exception as e:
                self.fail(f"Basic processing workflow failed: {e}")


class TestCMSFluxLoader(unittest.TestCase):
    """Test CMS flux loader functionality."""

    def setUp(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.loader = CMSFluxLoader(data_dir=self.temp_dir)

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cms_loader_initialization(self):
        """Test CMS loader initialization."""
        loader = CMSFluxLoader()

        # Check configuration
        self.assertIn(1, loader.experiment_files)
        self.assertIn(2, loader.experiment_files)
        self.assertIn('mean', loader.experiment_files[1])
        self.assertIn('std', loader.experiment_files[1])

    def test_coordinate_grid_creation(self):
        """Test coordinate grid creation."""
        x_coords, y_coords = self.loader._get_coordinate_grids()

        # Check coordinate shapes and ranges
        self.assertEqual(x_coords.shape, y_coords.shape)
        self.assertTrue(np.all(x_coords >= -125))
        self.assertTrue(np.all(x_coords <= -65))
        self.assertTrue(np.all(y_coords >= 24))
        self.assertTrue(np.all(y_coords <= 61))

    def test_flux_data_validation(self):
        """Test flux data validation."""
        # Create mock flux data
        mock_fluxes = {
            'GPP': np.random.rand(50, 60, 12) * 20,    # Realistic GPP range
            'REC': np.random.rand(50, 60, 12) * 15,    # Realistic respiration range
            'FIR': np.random.rand(50, 60, 12) * 5,     # Realistic fire range
            'NEE': np.random.rand(50, 60, 12) * 10 - 5,  # NEE can be negative
            'NBE': np.random.rand(50, 60, 12) * 10 - 3   # NBE can be negative
        }

        # Test validation
        try:
            validation_result = self.loader.validate_flux_data(mock_fluxes)
            self.assertTrue(validation_result)
        except Exception as e:
            self.fail(f"Flux data validation failed: {e}")


class TestERA5DiurnalLoader(unittest.TestCase):
    """Test ERA5 diurnal field loader."""

    def setUp(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.loader = ERA5DiurnalLoader(data_dir=self.temp_dir)

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_era5_loader_initialization(self):
        """Test ERA5 loader initialization."""
        loader = ERA5DiurnalLoader()

        # Check configuration
        self.assertIsNotNone(loader.file_pattern)
        self.assertIn('SST', loader.variable_mapping)
        self.assertIn('SSRD', loader.variable_mapping)

    def test_meteorological_data_validation(self):
        """Test meteorological data validation."""
        # Create mock meteorological data
        mock_skt = np.random.rand(120, 160, 744) * 20 + 270  # Temperature in K
        mock_ssrd = np.random.rand(120, 160, 744) * 1000     # Solar radiation in J/m²

        # Test validation
        try:
            self.loader._validate_met_data(mock_ssrd, mock_skt)
        except Exception as e:
            self.fail(f"Meteorological data validation failed: {e}")

    def test_cumulative_radiation_conversion(self):
        """Test cumulative to instantaneous radiation conversion."""
        # Create mock cumulative radiation data
        mock_cumulative = np.cumsum(np.random.rand(10, 10, 24) * 1000, axis=2)

        # Convert to instantaneous
        instantaneous = self.loader.convert_cumulative_to_instantaneous_radiation(mock_cumulative)

        # Check properties
        self.assertEqual(instantaneous.shape, mock_cumulative.shape)
        self.assertTrue(np.all(instantaneous >= 0))  # Should be non-negative


class TestDiurnalCalculator(unittest.TestCase):
    """Test diurnal calculation engine."""

    def setUp(self):
        """Setup test environment."""
        self.calculator = DiurnalCalculator()

    def test_calculator_initialization(self):
        """Test calculator initialization."""
        calculator = DiurnalCalculator()

        # Check attributes
        self.assertIsNotNone(calculator.q10_factor)
        self.assertIsNotNone(calculator.unit_conversion_factor)

    def test_gpp_diurnal_calculation(self):
        """Test GPP diurnal pattern calculation."""
        # Create mock data
        gpp_monthly = np.random.rand(50, 60) * 20  # Monthly GPP
        ssrd = np.random.rand(50, 60, 24) * 1000   # Hourly solar radiation

        # Calculate diurnal pattern
        gpp_diurnal = self.calculator._calculate_gpp_diurnal(gpp_monthly, ssrd)

        # Check result properties
        self.assertEqual(gpp_diurnal.shape, ssrd.shape)
        self.assertTrue(np.all(gpp_diurnal >= 0))  # GPP should be non-negative

    def test_respiration_diurnal_calculation(self):
        """Test respiration diurnal pattern calculation."""
        # Create mock data
        rec_monthly = np.random.rand(50, 60) * 15  # Monthly respiration
        skt = np.random.rand(50, 60, 24) * 20 + 280  # Hourly temperature

        # Calculate diurnal pattern
        rec_diurnal = self.calculator._calculate_rec_diurnal(rec_monthly, skt)

        # Check result properties
        self.assertEqual(rec_diurnal.shape, skt.shape)
        self.assertTrue(np.all(rec_diurnal >= 0))  # Respiration should be non-negative

    def test_3hourly_to_hourly_conversion(self):
        """Test 3-hourly to hourly data conversion."""
        # Create mock 3-hourly data
        data_3h = np.random.rand(10, 10, 8)  # 8 × 3-hour periods

        # Convert to hourly
        data_1h = self.calculator._triplicate_3hourly_to_hourly(data_3h)

        # Check result properties
        self.assertEqual(data_1h.shape, (10, 10, 24))  # Should be 24 hours


class TestGFEDDiurnalLoader(unittest.TestCase):
    """Test GFED diurnal pattern loader."""

    def setUp(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.loader = GFEDDiurnalLoader(gfed_data_dir=self.temp_dir)

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_gfed_loader_initialization(self):
        """Test GFED loader initialization."""
        loader = GFEDDiurnalLoader()

        # Check configuration
        self.assertIsNotNone(loader.region_bounds)
        self.assertIsNotNone(loader.emission_factors)
        self.assertEqual(len(loader.region_bounds), 4)

    def test_emission_factors_setup(self):
        """Test emission factors configuration."""
        factors = self.loader._setup_emission_factors()

        # Check required species
        required_species = ['CO2', 'CO', 'CH4', 'C']
        for species in required_species:
            self.assertIn(species, factors)

        # Check vegetation types
        required_veg_types = ['SAVA', 'BORF', 'TEMF', 'DEFO', 'PEAT', 'AGRI']
        for veg_type in required_veg_types:
            self.assertIn(veg_type, factors['CO2'])

    def test_resolution_aggregation(self):
        """Test 0.25° to 0.5° resolution aggregation."""
        # Create mock 0.25° data
        data_025 = np.random.rand(720, 1440, 8)  # Global 0.25° grid

        # Aggregate to 0.5°
        data_05 = self.loader._aggregate_025_to_05_degree(data_025)

        # Check result properties
        self.assertEqual(data_05.shape, (360, 720, 8))  # 0.5° global grid

    def test_days_in_month_calculation(self):
        """Test days in month calculation including leap years."""
        # Test regular year
        self.assertEqual(self.loader._get_days_in_month(2021, 2), 28)

        # Test leap year
        self.assertEqual(self.loader._get_days_in_month(2020, 2), 29)

        # Test other months
        self.assertEqual(self.loader._get_days_in_month(2021, 1), 31)
        self.assertEqual(self.loader._get_days_in_month(2021, 4), 30)


class TestDiurnalOutputWriters(unittest.TestCase):
    """Test diurnal output writers."""

    def setUp(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.writer = DiurnalFluxWriter(self.temp_dir)

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_writer_initialization(self):
        """Test output writer initialization."""
        writer = DiurnalFluxWriter(self.temp_dir)

        # Check attributes
        self.assertEqual(writer.output_base_dir, self.temp_dir)
        self.assertIsNotNone(writer.netcdf_writer)

    def test_directory_structure_creation(self):
        """Test output directory structure creation."""
        # Test monthly directory
        monthly_dir = self.writer._setup_monthly_output_directory('GPP', 1)
        self.assertTrue(os.path.exists(monthly_dir))
        self.assertIn('MONTHLY_GPP', monthly_dir)

        # Test diurnal directory
        diurnal_dir = self.writer._setup_diurnal_output_directory('REC', 2)
        self.assertTrue(os.path.exists(diurnal_dir))
        self.assertIn('DIURNAL_REC', diurnal_dir)

    def test_coordinate_generation(self):
        """Test coordinate array generation."""
        # Test longitude coordinates
        lons = self.writer._get_longitude_coordinates(160)
        self.assertEqual(len(lons), 160)
        self.assertTrue(np.all(lons >= -125))
        self.assertTrue(np.all(lons <= -65))

        # Test latitude coordinates
        lats = self.writer._get_latitude_coordinates(120)
        self.assertEqual(len(lats), 120)
        self.assertTrue(np.all(lats >= 24))
        self.assertTrue(np.all(lats <= 61))

    def test_uncertainty_factor_calculation(self):
        """Test uncertainty factor calculation."""
        # Create mock data
        flux_data = np.random.rand(10, 10) * 20
        uncertainty_data = np.random.rand(10, 10) * 2

        # Calculate uncertainty factors
        factors = self.writer._calculate_uncertainty_factor(flux_data, uncertainty_data)

        # Check properties
        self.assertEqual(factors.shape, flux_data.shape)
        self.assertTrue(np.all(factors >= 1.0))
        self.assertTrue(np.all(factors <= 10.0))


class TestDiurnalIntegration(unittest.TestCase):
    """Integration tests for diurnal processing workflow."""

    def setUp(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_module_imports(self):
        """Test that all diurnal modules can be imported."""
        try:
            from src import diurnal_processor
            from src import cms_flux_loader
            from src import met_driver_loader
            from src import diurnal_calculator
            from src import gfed_diurnal_loader
            from src import diurnal_output_writers
        except ImportError as e:
            self.fail(f"Failed to import diurnal modules: {e}")

    @patch('src.diurnal_processor.CMSFluxLoader')
    @patch('src.diurnal_processor.ERA5DiurnalLoader')
    @patch('src.diurnal_processor.GFEDDiurnalLoader')
    @patch('src.diurnal_processor.DiurnalCalculator')
    @patch('src.diurnal_processor.DiurnalFluxWriter')
    def test_end_to_end_workflow(self, mock_writer, mock_calculator,
                                mock_gfed, mock_era5, mock_cms):
        """Test end-to-end diurnal processing workflow."""
        # Setup mocks
        mock_cms.return_value.load_monthly_fluxes.return_value = {
            'GPP': np.random.rand(120, 160, 240),
            'REC': np.random.rand(120, 160, 240),
            'FIR': np.random.rand(120, 160, 240),
            'NEE': np.random.rand(120, 160, 240),
            'NBE': np.random.rand(120, 160, 240),
            'GPPunc': np.random.rand(120, 160, 240) * 0.1,
            'RECunc': np.random.rand(120, 160, 240) * 0.1,
            'FIRunc': np.random.rand(120, 160, 240) * 0.1,
            'NEEunc': np.random.rand(120, 160, 240) * 0.1,
            'NBEunc': np.random.rand(120, 160, 240) * 0.1
        }

        mock_era5.return_value.load_diurnal_fields.return_value = (
            np.random.rand(120, 160, 24) * 1000,  # SSRD
            np.random.rand(120, 160, 24) * 20 + 280  # SKT
        )

        mock_gfed.return_value.load_diurnal_fields.return_value = np.random.rand(120, 160, 8)

        mock_calculator.return_value.calculate_diurnal_fluxes.return_value = {
            'GPP': np.random.rand(120, 160, 24),
            'REC': np.random.rand(120, 160, 24),
            'FIR': np.random.rand(120, 160, 24),
            'NEE': np.random.rand(120, 160, 24),
            'NBE': np.random.rand(120, 160, 24)
        }

        # Test full workflow
        try:
            processor = DiurnalProcessor()
            result = processor.process_diurnal_fluxes(
                experiment_number=1,
                years=[2020],
                months=[1],
                output_dir=self.temp_dir
            )

            # Check result
            self.assertIsInstance(result, DiurnalFluxData)

        except Exception as e:
            self.fail(f"End-to-end workflow failed: {e}")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)