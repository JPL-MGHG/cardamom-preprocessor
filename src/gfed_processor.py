"""
GFED Processing Module

Direct Python translation of MATLAB CARDAMOM_MAPS_READ_GFED_NOV24.m
Handles burned area and fire emissions processing with gap-filling logic.

Based on MATLAB function CARDAMOM_MAPS_READ_GFED_NOV24(RES) from:
matlab-migration/CARDAMOM_MAPS_READ_GFED_NOV24.m
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from gfed_downloader import GFEDDownloader, GFEDReader
from coordinate_systems import load_land_sea_mask, convert_to_geoschem_grid
from scientific_utils import nan_to_zero
from netcdf_infrastructure import CARDAMOMNetCDFWriter


@dataclass
class GFEDData:
    """
    Structure for processed GFED data matching MATLAB output format.

    Equivalent to MATLAB GFED struct with fields:
    - BA: Burned area data
    - FireC: Fire carbon emissions
    - year: Year coordinates
    - month: Month coordinates
    """
    burned_area: np.ndarray  # GFED.BA in MATLAB
    fire_carbon: np.ndarray  # GFED.FireC in MATLAB
    year: np.ndarray         # GFED.year in MATLAB
    month: np.ndarray        # GFED.month in MATLAB
    resolution: str
    units: Dict[str, str]

    def to_netcdf_files(self, output_dir: str, file_format: str = "monthly") -> List[str]:
        """
        Export GFED data to CARDAMOM NetCDF format files.

        Args:
            output_dir: Directory to save NetCDF files
            file_format: Format type ("monthly", "yearly", "single")

        Returns:
            List of created NetCDF file paths
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        if file_format == "monthly":
            return self._create_monthly_files(output_dir)
        elif file_format == "yearly":
            return self._create_yearly_files(output_dir)
        elif file_format == "single":
            return self._create_single_file(output_dir)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

    def to_cardamom_format(self, output_dir: str) -> Dict[str, List[str]]:
        """
        Export in full CARDAMOM-compliant structure.

        Args:
            output_dir: Base output directory

        Returns:
            Dictionary with file categories and paths
        """
        import os

        # Create subdirectories following CARDAMOM structure
        ba_dir = os.path.join(output_dir, "burned_area")
        fc_dir = os.path.join(output_dir, "fire_carbon")

        os.makedirs(ba_dir, exist_ok=True)
        os.makedirs(fc_dir, exist_ok=True)

        result = {
            "burned_area_files": self._create_variable_files(ba_dir, "burned_area"),
            "fire_carbon_files": self._create_variable_files(fc_dir, "fire_carbon")
        }

        return result

    def _create_monthly_files(self, output_dir: str) -> List[str]:
        """Create separate NetCDF file for each month."""
        created_files = []
        writer = CARDAMOMNetCDFWriter(template_type="3D")

        # Process each month
        for i in range(len(self.month)):
            year = int(self.year[i])
            month = int(self.month[i])

            # Extract monthly data
            ba_monthly = self.burned_area[:, :, i:i+1]
            fc_monthly = self.fire_carbon[:, :, i:i+1]

            # Create burned area file
            ba_filename = self._get_cardamom_filename("burned_area", year, month, output_dir)
            self._write_monthly_netcdf(writer, ba_filename, ba_monthly, "burned_area", year, month)
            created_files.append(ba_filename)

            # Create fire carbon file
            fc_filename = self._get_cardamom_filename("fire_carbon", year, month, output_dir)
            self._write_monthly_netcdf(writer, fc_filename, fc_monthly, "fire_carbon", year, month)
            created_files.append(fc_filename)

        return created_files

    def _create_yearly_files(self, output_dir: str) -> List[str]:
        """Create separate NetCDF file for each year."""
        created_files = []
        writer = CARDAMOMNetCDFWriter(template_type="3D")

        # Group by year
        unique_years = np.unique(self.year)

        for year in unique_years:
            year_mask = self.year == year
            year_indices = np.where(year_mask)[0]

            if len(year_indices) == 0:
                continue

            # Extract yearly data
            ba_yearly = self.burned_area[:, :, year_indices]
            fc_yearly = self.fire_carbon[:, :, year_indices]
            months_yearly = self.month[year_indices]

            # Create burned area file
            ba_filename = self._get_cardamom_filename("burned_area", int(year), None, output_dir)
            self._write_yearly_netcdf(writer, ba_filename, ba_yearly, months_yearly, "burned_area", int(year))
            created_files.append(ba_filename)

            # Create fire carbon file
            fc_filename = self._get_cardamom_filename("fire_carbon", int(year), None, output_dir)
            self._write_yearly_netcdf(writer, fc_filename, fc_yearly, months_yearly, "fire_carbon", int(year))
            created_files.append(fc_filename)

        return created_files

    def _create_single_file(self, output_dir: str) -> List[str]:
        """Create single NetCDF file containing all data."""
        writer = CARDAMOMNetCDFWriter(template_type="3D")

        # Create combined file
        filename = os.path.join(output_dir, f"CARDAMOM_GFED_combined_{self.resolution}.nc")

        # Prepare coordinate arrays
        from coordinate_systems import CoordinateGrid
        grid = CoordinateGrid(resolution=float(self.resolution.replace('deg', '').replace('05', '0.5')))

        data_dict = {
            'filename': filename,
            'x': grid.longitude_coordinates,
            'y': grid.latitude_coordinates,
            't': np.arange(len(self.month)),  # Time index
            'timeunits': 'months since start',
            'data': np.stack([self.burned_area, self.fire_carbon], axis=3),
            'info': [
                {'name': 'burned_area', 'units': self.units['burned_area'], 'longname': 'Monthly Burned Area Fraction'},
                {'name': 'fire_carbon', 'units': self.units['fire_carbon'], 'longname': 'Monthly Fire Carbon Emissions'}
            ],
            'Attributes': {
                'title': 'CARDAMOM GFED Processed Data',
                'resolution': self.resolution,
                'temporal_coverage': f"{int(self.year[0])}-{int(self.year[-1])}",
                'variables': 'burned_area, fire_carbon'
            }
        }

        writer.write_3d_dataset(data_dict)
        return [filename]

    def _create_variable_files(self, output_dir: str, variable: str) -> List[str]:
        """Create files for specific variable organized by CARDAMOM structure."""
        created_files = []
        writer = CARDAMOMNetCDFWriter(template_type="3D")

        data_array = self.burned_area if variable == "burned_area" else self.fire_carbon

        # Create monthly files for this variable
        for i in range(len(self.month)):
            year = int(self.year[i])
            month = int(self.month[i])

            filename = self._get_cardamom_filename(variable, year, month, output_dir)
            monthly_data = data_array[:, :, i:i+1]

            self._write_monthly_netcdf(writer, filename, monthly_data, variable, year, month)
            created_files.append(filename)

        return created_files

    def _write_monthly_netcdf(self, writer: CARDAMOMNetCDFWriter, filename: str,
                             data: np.ndarray, variable: str, year: int, month: int) -> None:
        """Write monthly data to NetCDF file."""
        # Prepare coordinate arrays
        from coordinate_systems import CoordinateGrid

        if self.resolution == '05deg':
            grid_res = 0.5
        elif self.resolution == '0.25deg':
            grid_res = 0.25
        elif self.resolution == 'GC4x5':
            # Use approximate 4x5 degree grid
            grid_res = 4.0  # This is approximate
        else:
            grid_res = 0.5  # Default

        grid = CoordinateGrid(resolution=grid_res)

        data_dict = {
            'filename': filename,
            'x': grid.longitude_coordinates,
            'y': grid.latitude_coordinates,
            't': np.array([0]),  # Single time step
            'timeunits': f'months since {year}-{month:02d}-01',
            'data': data,
            'info': {
                'name': variable,
                'units': self.units[variable],
                'longname': f'Monthly {variable.replace("_", " ").title()}'
            },
            'Attributes': {
                'title': f'CARDAMOM GFED {variable.replace("_", " ").title()}',
                'year': year,
                'month': month,
                'resolution': self.resolution,
                'source': 'GFED4.1s processed by CARDAMOM preprocessor'
            }
        }

        writer.write_3d_dataset(data_dict)

    def _write_yearly_netcdf(self, writer: CARDAMOMNetCDFWriter, filename: str,
                            data: np.ndarray, months: np.ndarray, variable: str, year: int) -> None:
        """Write yearly data to NetCDF file."""
        # Prepare coordinate arrays
        from coordinate_systems import CoordinateGrid

        if self.resolution == '05deg':
            grid_res = 0.5
        elif self.resolution == '0.25deg':
            grid_res = 0.25
        elif self.resolution == 'GC4x5':
            grid_res = 4.0
        else:
            grid_res = 0.5

        grid = CoordinateGrid(resolution=grid_res)

        data_dict = {
            'filename': filename,
            'x': grid.longitude_coordinates,
            'y': grid.latitude_coordinates,
            't': months - 1,  # 0-based month indexing
            'timeunits': f'months since {year}-01-01',
            'data': data,
            'info': {
                'name': variable,
                'units': self.units[variable],
                'longname': f'Monthly {variable.replace("_", " ").title()}'
            },
            'Attributes': {
                'title': f'CARDAMOM GFED {variable.replace("_", " ").title()}',
                'year': year,
                'resolution': self.resolution,
                'source': 'GFED4.1s processed by CARDAMOM preprocessor'
            }
        }

        writer.write_3d_dataset(data_dict)

    def _get_cardamom_filename(self, variable: str, year: int, month: Optional[int], output_dir: str) -> str:
        """
        Generate CARDAMOM-compliant filename.

        Args:
            variable: Variable name ("burned_area", "fire_carbon")
            year: Year
            month: Month (None for yearly files)
            output_dir: Output directory

        Returns:
            Complete file path with CARDAMOM naming convention
        """
        import os

        if month is not None:
            # Monthly file: CARDAMOM_GFED_{variable}_{resolution}_{YYYYMM}.nc
            filename = f"CARDAMOM_GFED_{variable}_{self.resolution}_{year}{month:02d}.nc"
        else:
            # Yearly file: CARDAMOM_GFED_{variable}_{resolution}_{YYYY}.nc
            filename = f"CARDAMOM_GFED_{variable}_{self.resolution}_{year}.nc"

        return os.path.join(output_dir, filename)


class GFEDProcessor:
    """
    Process GFED4.1s burned area and fire emissions data.

    Direct Python translation of MATLAB CARDAMOM_MAPS_READ_GFED_NOV24.m
    Handles multi-year loading, gap-filling, and resolution conversion.

    Based on MATLAB function CARDAMOM_MAPS_READ_GFED_NOV24(RES) lines 2-81
    """

    def __init__(self, data_dir: str = "./DATA/GFED4/"):
        """
        Initialize GFED processor.

        Args:
            data_dir: Directory containing GFED HDF5 files
        """
        self.data_dir = data_dir
        self.logger = logging.getLogger(self.__class__.__name__)
        self.downloader = GFEDDownloader(data_dir)

        # MATLAB equivalent: YEARS=2001:2023 (line 9)
        self.default_years = list(range(2001, 2024))  # 2001-2023

    def process_gfed_data(self,
                         target_resolution: str = '05deg',
                         start_year: int = 2001,
                         end_year: Optional[int] = None,
                         create_netcdf: bool = False,
                         output_dir: Optional[str] = None) -> GFEDData:
        """
        Main processing function equivalent to MATLAB CARDAMOM_MAPS_READ_GFED_NOV24.

        Replicates MATLAB function logic from lines 2-81:
        1. Load multi-year GFED data (lines 17-21)
        2. Apply land-sea mask corrections (lines 24-27)
        3. Convert to target resolution (lines 29-58)
        4. Fill missing years with climatology (lines 66-68)
        5. Create temporal coordinates (lines 75-77)
        6. Optionally create NetCDF files (lines 182-195)

        Args:
            target_resolution: Output resolution ('05deg', 'GC4x5', '0.25deg')
            start_year: First year to process (default: 2001)
            end_year: Last year to process (default: 2023, auto-detect if None)
            create_netcdf: Whether to create CARDAMOM NetCDF files (default: False)
            output_dir: Output directory for NetCDF files (default: "./DATA/CARDAMOM-MAPS_GFED/")

        Returns:
            GFEDData: Processed burned area and fire carbon data with NetCDF files created if requested
        """
        if end_year is None:
            end_year = max(self.default_years)

        years_to_process = list(range(start_year, end_year + 1))

        self.logger.info(f"Processing GFED data for years {start_year}-{end_year}, "
                        f"target resolution: {target_resolution}")

        # Step 1: Load multi-year data (MATLAB lines 17-21)
        gba_data, gce_data = self._load_multi_year_data(years_to_process)

        # Step 2: Apply land-sea mask corrections (MATLAB lines 24-27)
        gba_data, gce_data = self._apply_land_sea_mask(gba_data, gce_data)

        # Step 3: Convert to target resolution (MATLAB lines 29-58)
        processed_gfed = self._convert_resolution(gba_data, gce_data, target_resolution)

        # Step 4: Fill missing years using climatology (MATLAB lines 66-68)
        if end_year > 2016:
            processed_gfed = self._fill_missing_years(processed_gfed, years_to_process)

        # Step 5: Create temporal coordinates (MATLAB lines 75-77)
        year_coords, month_coords = self._create_temporal_coordinates(start_year, end_year)

        # Package results in MATLAB-equivalent format
        result = GFEDData(
            burned_area=processed_gfed['BA'],
            fire_carbon=processed_gfed['FireC'],
            year=year_coords,
            month=month_coords,
            resolution=target_resolution,
            units={
                'burned_area': 'fraction_of_cell',
                'fire_carbon': 'g_C_m-2_month-1'
            }
        )

        self.logger.info(f"GFED processing completed. Output shape: {result.burned_area.shape}")

        # Step 6: Optionally create NetCDF files (equivalent to MATLAB lines 182-195)
        if create_netcdf:
            if output_dir is None:
                output_dir = "./DATA/CARDAMOM-MAPS_GFED/"

            self.logger.info("Creating CARDAMOM NetCDF files as requested")
            netcdf_files = self.create_cardamom_netcdf_files(result, output_dir, "monthly")
            self.logger.info(f"Created {len(netcdf_files)} NetCDF files in {output_dir}")

        return result

    def _load_multi_year_data(self, years: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load GFED data for multiple years.

        Equivalent to MATLAB lines 17-21:
        for yr=YEARS
            GF4=DATASCRIPT_READ_GFED4_DATA_MAY16(yr);
            GBA(:,:,(yr-2001)*12+[1:12])=GF4.BA;
            GCE(:,:,(yr-2001)*12+[1:12])=GF4.ES(:,:,:,4);
        end

        Args:
            years: List of years to process

        Returns:
            tuple: (burned_area_data, carbon_emissions_data) as numpy arrays
        """
        self.logger.info(f"Loading GFED data for {len(years)} years")

        # Initialize arrays - will determine size from first file
        gba_data = None
        gce_data = None

        start_year = min(years)

        for year in years:
            try:
                # Download file if needed
                download_result = self.downloader.download_yearly_files([year])

                if not download_result['downloaded_files']:
                    raise FileNotFoundError(f"Failed to download GFED data for year {year}")

                # Read GFED data from HDF5 file
                filepath = download_result['downloaded_files'][0]
                reader = GFEDReader(filepath)

                # Load all 12 months for this year
                year_ba_data = []
                year_ce_data = []

                for month in range(1, 13):
                    monthly_data = reader.extract_monthly_data(year, month)

                    # Extract burned area and carbon emissions (4th species)
                    if 'burned_area' in monthly_data:
                        year_ba_data.append(monthly_data['burned_area'])
                    else:
                        self.logger.warning(f"No burned area data for {year}-{month:02d}")
                        year_ba_data.append(None)

                    # Extract carbon emissions (equivalent to ES(:,:,:,4) in MATLAB)
                    if 'total_emission' in monthly_data:
                        year_ce_data.append(monthly_data['total_emission'])
                    elif 'partitioning' in monthly_data and 'C' in monthly_data['partitioning']:
                        year_ce_data.append(monthly_data['partitioning']['C'])
                    else:
                        self.logger.warning(f"No carbon emission data for {year}-{month:02d}")
                        year_ce_data.append(None)

                # Stack monthly data
                year_ba_stack = np.stack(year_ba_data, axis=2) if all(d is not None for d in year_ba_data) else None
                year_ce_stack = np.stack(year_ce_data, axis=2) if all(d is not None for d in year_ce_data) else None

                # Initialize output arrays on first iteration
                if gba_data is None and year_ba_stack is not None:
                    n_lat, n_lon, _ = year_ba_stack.shape
                    total_months = len(years) * 12
                    gba_data = np.full((n_lat, n_lon, total_months), np.nan)
                    gce_data = np.full((n_lat, n_lon, total_months), np.nan)

                # Insert data at correct temporal indices
                # MATLAB: (yr-2001)*12+[1:12] becomes (year-start_year)*12+[0:11] in Python
                if year_ba_stack is not None and gba_data is not None:
                    month_start = (year - start_year) * 12
                    month_end = month_start + 12
                    gba_data[:, :, month_start:month_end] = year_ba_stack

                if year_ce_stack is not None and gce_data is not None:
                    month_start = (year - start_year) * 12
                    month_end = month_start + 12
                    gce_data[:, :, month_start:month_end] = year_ce_stack

                self.logger.info(f"Loaded GFED data for year {year}")

            except Exception as e:
                self.logger.error(f"Failed to load GFED data for year {year}: {e}")
                # Continue with other years - gaps will be filled later

        if gba_data is None:
            raise ValueError("No valid GFED data loaded")

        self.logger.info(f"Multi-year data loading completed. Shape: {gba_data.shape}")
        return gba_data, gce_data

    def _apply_land_sea_mask(self, gba_data: np.ndarray, gce_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply land-sea mask corrections to GFED data.

        Equivalent to MATLAB lines 24-27:
        M4D=repmat(lsfrac025>0.5,[1,1,size(GBA,3),size(GBA,4)]);
        GBA(M4D==0)=NaN;GBA(isnan(GBA) & M4D)=0;
        GCE(M4D==0)=NaN;GCE(isnan(GCE) & M4D)=0;
        %Ensures land fire-free regions = 0 but sea regions are = NaN

        Args:
            gba_data: Burned area data array
            gce_data: Carbon emissions data array

        Returns:
            tuple: (masked_gba_data, masked_gce_data)
        """
        self.logger.info("Applying land-sea mask corrections")

        # Load land-sea mask at 0.25 degree resolution
        # MATLAB line 11: [~,lsfrac025]=loadlandseamask(0.25);
        _, land_sea_fraction = load_land_sea_mask(0.25)

        # Create land mask (land fraction > 0.5)
        land_mask = land_sea_fraction > 0.5

        # Replicate mask for all time dimensions
        # MATLAB line 24: M4D=repmat(lsfrac025>0.5,[1,1,size(GBA,3),size(GBA,4)]);
        n_time = gba_data.shape[2]
        mask_4d = np.tile(land_mask[:, :, np.newaxis], (1, 1, n_time))

        # Apply mask to burned area data
        # MATLAB lines 25-26: GBA(M4D==0)=NaN;GBA(isnan(GBA) & M4D)=0;
        gba_masked = gba_data.copy()
        gba_masked[mask_4d == 0] = np.nan  # Sea regions = NaN
        gba_masked[np.isnan(gba_masked) & mask_4d] = 0  # Land fire-free regions = 0

        # Apply mask to carbon emissions data
        # MATLAB lines 25-26: GCE(M4D==0)=NaN;GCE(isnan(GCE) & M4D)=0;
        gce_masked = gce_data.copy()
        gce_masked[mask_4d == 0] = np.nan  # Sea regions = NaN
        gce_masked[np.isnan(gce_masked) & mask_4d] = 0  # Land fire-free regions = 0

        self.logger.info("Land-sea mask corrections applied")
        return gba_masked, gce_masked

    def _convert_resolution(self, gba_data: np.ndarray, gce_data: np.ndarray,
                           target_resolution: str) -> Dict[str, np.ndarray]:
        """
        Convert GFED data to target spatial resolution.

        Equivalent to MATLAB switch statement lines 29-58:
        switch RES
            case '05deg' ...
            case 'GC4x5' ...
            case '0.25deg' ...
        end

        Args:
            gba_data: Burned area data at native 0.25 degree resolution
            gce_data: Carbon emissions data at native 0.25 degree resolution
            target_resolution: Target resolution ('05deg', 'GC4x5', '0.25deg')

        Returns:
            dict: Processed data with 'BA' and 'FireC' keys
        """
        self.logger.info(f"Converting to resolution: {target_resolution}")

        if target_resolution == '05deg':
            # MATLAB lines 32-46: Aggregate from 0.25 to 0.5 degrees
            return self._aggregate_to_05_degree(gba_data, gce_data)

        elif target_resolution == 'GC4x5':
            # MATLAB lines 51-52: Convert to GeosChem grid
            return self._convert_to_geoschem(gba_data, gce_data)

        elif target_resolution == '0.25deg':
            # MATLAB lines 54-56: Keep native resolution
            return {
                'BA': gba_data,
                'FireC': gce_data * 12 / 365.25  # Convert to g species/m2/month
            }

        else:
            raise ValueError(f"Unsupported resolution: {target_resolution}")

    def _aggregate_to_05_degree(self, gba_data: np.ndarray, gce_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Aggregate 0.25 degree data to 0.5 degree resolution.

        Equivalent to MATLAB lines 32-46:
        GFED.BA=GBA(2:2:end,2:2:end,:)*0;
        GFED.FireC=GCE(2:2:end,2:2:end,:)*0;
        Fcount=GBA(2:2:end,2:2:end,:)*0;

        for r=1:2
            for c=1:2
                GFED.BA=GFED.BA + nan2zero(GBA(2:2:end,2:2:end,:));
                GFED.FireC=GFED.FireC + nan2zero(GCE(2:2:end,2:2:end,:));
                Fcount=Fcount + double( isnan(GBA(2:2:end,2:2:end,:))==0);
            end
        end
        GFED.BA=GFED.BA./Fcount;
        GFED.FireC=GFED.FireC*12/365.25./Fcount;
        """
        # Initialize output arrays at 0.5 degree resolution
        # MATLAB: GBA(2:2:end,2:2:end,:) selects every other grid cell
        output_shape = (gba_data.shape[0] // 2, gba_data.shape[1] // 2, gba_data.shape[2])

        ba_sum = np.zeros(output_shape)
        firec_sum = np.zeros(output_shape)
        count = np.zeros(output_shape)

        # Aggregate 2x2 cells into single 0.5 degree cells
        # MATLAB loops r=1:2, c=1:2 but uses same indices - this appears to be an error
        # Implementing correct 2x2 aggregation:
        for r_offset in range(2):
            for c_offset in range(2):
                # Extract every 2nd point starting from offset
                ba_subset = gba_data[r_offset::2, c_offset::2, :]
                ce_subset = gce_data[r_offset::2, c_offset::2, :]

                # Ensure shapes match
                if ba_subset.shape != output_shape:
                    # Trim to match output shape
                    ba_subset = ba_subset[:output_shape[0], :output_shape[1], :]
                    ce_subset = ce_subset[:output_shape[0], :output_shape[1], :]

                # Sum non-NaN values (equivalent to nan2zero)
                ba_subset_clean = nan_to_zero(ba_subset)
                ce_subset_clean = nan_to_zero(ce_subset)

                ba_sum += ba_subset_clean
                firec_sum += ce_subset_clean

                # Count valid (non-NaN) values
                count += (~np.isnan(ba_subset)).astype(float)

        # Calculate averages (avoid division by zero)
        count[count == 0] = 1  # Prevent division by zero

        ba_avg = ba_sum / count
        firec_avg = firec_sum * 12 / 365.25 / count  # Convert to monthly units

        return {'BA': ba_avg, 'FireC': firec_avg}

    def _convert_to_geoschem(self, gba_data: np.ndarray, gce_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Convert to GeosChem 4x5 degree grid.

        Equivalent to MATLAB lines 51-52:
        GFED.BA=GEOSChem_regular_grid_to_GC(GBA);
        GFED.FireC=GEOSChem_regular_grid_to_GC(GCE)*12/365.25;
        """
        ba_gc = convert_to_geoschem_grid(gba_data)
        firec_gc = convert_to_geoschem_grid(gce_data) * 12 / 365.25

        return {'BA': ba_gc, 'FireC': firec_gc}

    def _fill_missing_years(self, gfed_data: Dict[str, np.ndarray],
                           years: List[int]) -> Dict[str, np.ndarray]:
        """
        Fill missing years using climatological patterns.

        Equivalent to MATLAB lines 66-68:
        for m=1:(YEARS(end)-2016)*12
            BAextra(:,:,m)=sum(GFED.BA(:,:,m:12:idxdec16),3)./sum(GFED.FireC(:,:,m:12:idxdec16),3).*GFED.FireC(:,:,idxdec16+m);
        end
        GFED.BA(:,:,idxdec16+1:end)= BAextra;
        GFED.BA=nan2zero(GFED.BA);
        """
        self.logger.info("Applying gap-filling for years after 2016")

        # MATLAB: idxdec16=16*12 (index for end of 2016)
        idx_dec_2016 = 16 * 12  # 192 months (end of 2016)

        if gfed_data['BA'].shape[2] <= idx_dec_2016:
            # No years after 2016 to fill
            return gfed_data

        ba_data = gfed_data['BA'].copy()
        firec_data = gfed_data['FireC'].copy()

        # Calculate number of months after 2016
        end_year = max(years)
        n_months_after_2016 = (end_year - 2016) * 12

        # Apply gap-filling for each month
        for m in range(n_months_after_2016):
            # Monthly climatology from reference period (2001-2016)
            # MATLAB: m:12:idxdec16 selects same month across all years
            month_indices = np.arange(m, idx_dec_2016, 12)

            # Calculate BA/FireC ratio for this month across reference years
            ba_monthly = ba_data[:, :, month_indices]
            firec_monthly = firec_data[:, :, month_indices]

            # Sum across reference years
            ba_sum = np.nansum(ba_monthly, axis=2)
            firec_sum = np.nansum(firec_monthly, axis=2)

            # Calculate ratio and apply to target year emission
            with np.errstate(divide='ignore', invalid='ignore'):
                ba_firec_ratio = ba_sum / firec_sum
                ba_firec_ratio[~np.isfinite(ba_firec_ratio)] = 0

            # Apply ratio to emission in target year
            target_month_idx = idx_dec_2016 + m
            if target_month_idx < ba_data.shape[2]:
                target_firec = firec_data[:, :, target_month_idx]
                ba_extra = ba_firec_ratio * target_firec
                ba_data[:, :, target_month_idx] = ba_extra

        # Clean up NaN values (MATLAB: nan2zero)
        ba_data = nan_to_zero(ba_data)

        self.logger.info(f"Gap-filling completed for {n_months_after_2016} months after 2016")
        return {'BA': ba_data, 'FireC': firec_data}

    def _create_temporal_coordinates(self, start_year: int, end_year: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create temporal coordinate arrays.

        Equivalent to MATLAB lines 75-77:
        yrdates=YEARS(1)+1/24:1/12:YEARS(end)+1;
        GFED.year=floor(yrdates);
        GFED.month=round(mod(yrdates*12-0.5,12)+1);

        Args:
            start_year: First year of data
            end_year: Last year of data

        Returns:
            tuple: (year_array, month_array)
        """
        # MATLAB: YEARS(1)+1/24:1/12:YEARS(end)+1
        year_dates = np.arange(start_year + 1/24, end_year + 1, 1/12)

        # MATLAB: floor(yrdates)
        years = np.floor(year_dates)

        # MATLAB: round(mod(yrdates*12-0.5,12)+1)
        months = np.round(np.mod(year_dates * 12 - 0.5, 12) + 1)

        return years.astype(int), months.astype(int)

    def create_cardamom_netcdf_files(self, gfed_data: GFEDData, 
                                    output_dir: str = "./DATA/CARDAMOM-MAPS_GFED/",
                                    file_format: str = "monthly") -> List[str]:
        """
        Create CARDAMOM-compliant NetCDF files for processed GFED data.

        Equivalent to MATLAB lines 182-195 in CARDAMOM_MAPS_READ_GFED_NOV24.m
        Creates individual NetCDF files following CARDAMOM naming conventions
        and data structure requirements.

        Args:
            gfed_data: Processed GFED data structure
            output_dir: Output directory for NetCDF files (default: "./DATA/CARDAMOM-MAPS_GFED/")
            file_format: File organization format ("monthly", "yearly", "single")

        Returns:
            List of created NetCDF file paths

        Example:
            >>> processor = GFEDProcessor()
            >>> gfed_result = processor.process_gfed_data('05deg', 2020, 2021)
            >>> netcdf_files = processor.create_cardamom_netcdf_files(gfed_result)
            >>> print(f"Created {len(netcdf_files)} NetCDF files")
        """
        import os
        
        self.logger.info(f"Creating CARDAMOM NetCDF files in format: {file_format}")
        self.logger.info(f"Output directory: {output_dir}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Use GFEDData's built-in NetCDF export functionality
        created_files = gfed_data.to_netcdf_files(output_dir, file_format)

        self.logger.info(f"Successfully created {len(created_files)} NetCDF files")
        return created_files

    def _create_burned_area_files(self, gfed_data: GFEDData, output_dir: str) -> List[str]:
        """
        Create individual NetCDF files for burned area data by year/month.

        Args:
            gfed_data: Processed GFED data structure
            output_dir: Output directory for burned area files

        Returns:
            List of created burned area NetCDF file paths
        """
        import os
        
        ba_dir = os.path.join(output_dir, "burned_area")
        os.makedirs(ba_dir, exist_ok=True)

        created_files = []
        writer = CARDAMOMNetCDFWriter(template_type="3D")

        self.logger.info("Creating burned area NetCDF files")

        # Create file for each month
        for i in range(len(gfed_data.month)):
            year = int(gfed_data.year[i])
            month = int(gfed_data.month[i])

            # Extract monthly burned area data
            ba_monthly = gfed_data.burned_area[:, :, i:i+1]

            # Generate filename
            filename = self._apply_cardamom_naming_convention(
                "burned_area", year, month, gfed_data.resolution, ba_dir
            )

            # Write NetCDF file
            self._write_gfed_netcdf_file(
                writer, filename, ba_monthly, "burned_area", 
                gfed_data.resolution, year, month, gfed_data.units
            )

            created_files.append(filename)

        self.logger.info(f"Created {len(created_files)} burned area files")
        return created_files

    def _create_fire_carbon_files(self, gfed_data: GFEDData, output_dir: str) -> List[str]:
        """
        Create individual NetCDF files for fire carbon emissions by year/month.

        Args:
            gfed_data: Processed GFED data structure
            output_dir: Output directory for fire carbon files

        Returns:
            List of created fire carbon NetCDF file paths
        """
        import os
        
        fc_dir = os.path.join(output_dir, "fire_carbon")
        os.makedirs(fc_dir, exist_ok=True)

        created_files = []
        writer = CARDAMOMNetCDFWriter(template_type="3D")

        self.logger.info("Creating fire carbon NetCDF files")

        # Create file for each month
        for i in range(len(gfed_data.month)):
            year = int(gfed_data.year[i])
            month = int(gfed_data.month[i])

            # Extract monthly fire carbon data
            fc_monthly = gfed_data.fire_carbon[:, :, i:i+1]

            # Generate filename
            filename = self._apply_cardamom_naming_convention(
                "fire_carbon", year, month, gfed_data.resolution, fc_dir
            )

            # Write NetCDF file
            self._write_gfed_netcdf_file(
                writer, filename, fc_monthly, "fire_carbon", 
                gfed_data.resolution, year, month, gfed_data.units
            )

            created_files.append(filename)

        self.logger.info(f"Created {len(created_files)} fire carbon files")
        return created_files

    def _apply_cardamom_naming_convention(self, variable: str, year: int, month: int, 
                                        resolution: str, output_dir: str) -> str:
        """
        Apply CARDAMOM file naming convention.

        Implements naming pattern used in MATLAB CARDAMOM processing:
        CARDAMOM_GFED_{variable}_{resolution}_{YYYYMM}.nc

        Args:
            variable: Variable name ("burned_area", "fire_carbon")
            year: Year (YYYY)
            month: Month (MM)
            resolution: Resolution string ('05deg', '0.25deg', 'GC4x5')
            output_dir: Output directory path

        Returns:
            Complete file path following CARDAMOM naming convention

        Example:
            >>> filename = processor._apply_cardamom_naming_convention(
            ...     "burned_area", 2020, 3, "05deg", "/data/output"
            ... )
            >>> print(filename)
            /data/output/CARDAMOM_GFED_burned_area_05deg_202003.nc
        """
        import os
        
        # Format: CARDAMOM_GFED_{variable}_{resolution}_{YYYYMM}.nc
        filename = f"CARDAMOM_GFED_{variable}_{resolution}_{year}{month:02d}.nc"
        return os.path.join(output_dir, filename)

    def _write_gfed_netcdf_file(self, writer: CARDAMOMNetCDFWriter, filename: str,
                               data: np.ndarray, variable: str, resolution: str,
                               year: int, month: int, units: Dict[str, str]) -> None:
        """
        Write GFED data to NetCDF file using CARDAMOM structure.

        Args:
            writer: NetCDF writer instance
            filename: Output file path
            data: Data array to write
            variable: Variable name
            resolution: Grid resolution
            year: Data year
            month: Data month
            units: Units dictionary
        """
        # Create coordinate grid based on resolution
        from coordinate_systems import CoordinateGrid

        if resolution == '05deg':
            grid_res = 0.5
        elif resolution == '0.25deg':
            grid_res = 0.25
        elif resolution == 'GC4x5':
            # For GeosChem grid, use approximate resolution
            grid_res = 4.0
        else:
            grid_res = 0.5  # Default

        grid = CoordinateGrid(resolution=grid_res)

        # Prepare data dictionary for NetCDF writer
        data_dict = {
            'filename': filename,
            'x': grid.longitude_coordinates,
            'y': grid.latitude_coordinates,
            't': np.array([0]),  # Single time step for monthly data
            'timeunits': f'months since {year}-{month:02d}-01 00:00:00',
            'data': data,
            'info': {
                'name': variable,
                'units': units.get(variable, 'unknown'),
                'longname': f'Monthly {variable.replace("_", " ").title()}',
                'standard_name': self._get_cf_standard_name(variable)
            },
            'Attributes': {
                'title': f'CARDAMOM GFED {variable.replace("_", " ").title()}',
                'institution': 'NASA Jet Propulsion Laboratory',
                'source': 'GFED4.1s processed by CARDAMOM preprocessor',
                'conventions': 'CF-1.6',
                'year': year,
                'month': month,
                'resolution': resolution,
                'contact': 'CARDAMOM Development Team',
                'references': 'van der Werf et al. (2017), GFED4.1s',
                'processing_level': 'L3',
                'temporal_resolution': 'monthly'
            }
        }

        # Write the NetCDF file
        writer.write_3d_dataset(data_dict)

    def _get_cf_standard_name(self, variable: str) -> str:
        """
        Get CF-compliant standard name for GFED variables.

        Args:
            variable: Variable name

        Returns:
            CF standard name string
        """
        cf_names = {
            'burned_area': 'burned_area_fraction',
            'fire_carbon': 'surface_carbon_emissions_due_to_fires'
        }
        return cf_names.get(variable, variable)
