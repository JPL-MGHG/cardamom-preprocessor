# Phase 4: Diurnal Flux Processing

## Overview
Create comprehensive diurnal flux processing system based on `PROJSCRIPT_DIURNAL_CMS_C_FLUXES_AUG25.m`. Handles CONUS carbon flux downscaling from monthly to hourly resolution using meteorological drivers and fire patterns.

## 4.1 Core Diurnal Processor (`diurnal_processor.py`)

### Main DiurnalProcessor Class
```python
class DiurnalProcessor:
    """
    Process CONUS carbon fluxes from monthly to diurnal (hourly) resolution.
    Based on MATLAB PROJSCRIPT_DIURNAL_CMS_C_FLUXES_AUG25.

    Handles downscaling of GPP, REC, FIR, NEE, and NBE fluxes using:
    - Solar radiation patterns for GPP
    - Temperature patterns for respiration (REC)
    - GFED fire timing for fire emissions (FIR)
    """

    def __init__(self, config_file=None):
        self.config = self._load_config(config_file)
        self.aux_data = self._setup_auxiliary_data()
        self.years_range = (2015, 2020)  # Default from MATLAB
        self.region_bounds = [60, -130, 20, -50]  # CONUS: N, W, S, E

    def _setup_auxiliary_data(self):
        """
        Setup auxiliary data paths and coordinate systems.
        Equivalent to MATLAB AUX structure setup.
        """
        return {
            'destination_path': {
                1: 'DUMPFILES/CARDAMOM_CONUS_DIURNAL_FLUXES_JUL25_EXP1/',
                2: 'DUMPFILES/CARDAMOM_CONUS_DIURNAL_FLUXES_JUL25/'
            },
            'lon_range': [-124.7500, -65.2500],
            'lat_range': [24.7500, 60.2500]
        }

    def process_diurnal_fluxes(self, experiment_number, years=None):
        """
        Main processing function for diurnal flux creation.

        Args:
            experiment_number: 1 or 2 (different CMS experiments)
            years: List of years to process (default: 2015-2020)
        """
        if years is None:
            years = list(range(*self.years_range))

        # Load monthly CMS fluxes
        monthly_fluxes = self.load_cms_monthly_fluxes(experiment_number)

        for year in years:
            for month in range(1, 13):
                print(f"Processing Month {month}, Year {year}")
                self._process_single_month(monthly_fluxes, year, month, experiment_number)

    def _process_single_month(self, fluxes, year, month, experiment_number):
        """Process diurnal fluxes for a single month"""
        month_index = month + (year - 2001) * 12

        # Write monthly fluxes
        self._write_monthly_fluxes(fluxes, year, month, month_index, experiment_number)

        # Generate diurnal patterns
        self._generate_diurnal_patterns(fluxes, year, month, month_index, experiment_number)
```

### Monthly Flux Processing
```python
def _write_monthly_fluxes(self, fluxes, year, month, month_index, experiment_number):
    """
    Write monthly flux files with uncertainties.
    Equivalent to MATLAB write_monthlyflux_to_geoschem_format calls.
    """
    flux_types = ['GPP', 'NBE', 'NEE', 'REC', 'FIR']

    for flux_type in flux_types:
        flux_data = fluxes[flux_type][:, :, month_index] * 1e3 / 24 / 3600  # Unit conversion
        flux_unc = fluxes[f'{flux_type}unc'][:, :, month_index] * 1e3 / 24 / 3600

        self.write_monthly_flux_to_geoschem_format(
            flux_data, flux_unc, year, month,
            self.aux_data, flux_type, experiment_number
        )

def _generate_diurnal_patterns(self, fluxes, year, month, month_index, experiment_number):
    """Generate hourly diurnal patterns for the month"""

    # Load meteorological drivers
    ssrd, skt = self.load_era5_diurnal_fields(month, year)

    # Load fire diurnal patterns
    co2_diurnal = self.load_gfed_diurnal_fields(month, year)

    # Calculate diurnal fluxes
    diurnal_fluxes = self._calculate_diurnal_fluxes(
        fluxes, month_index, ssrd, skt, co2_diurnal
    )

    # Write hourly files
    self._write_hourly_fluxes(diurnal_fluxes, year, month, experiment_number)
```

## 4.2 CMS Monthly Flux Loader (`cms_flux_loader.py`)

### CMS Data Reader
```python
class CMSFluxLoader:
    """
    Load and process monthly CMS flux data from NetCDF files.
    Based on MATLAB load_erens_cms_monthly_fluxes function.
    """

    def __init__(self, data_dir="./DATA/DATA_FROM_EREN/CMS_CONUS_JUL25/"):
        self.data_dir = data_dir
        self.experiment_files = {
            1: {
                'mean': 'Outputmean_exp1redo5.nc',
                'std': 'Outputstd_exp1redo5.nc'
            },
            2: {
                'mean': 'Outputmean_exp2redo5.nc',
                'std': 'Outputstd_exp2redo5.nc'
            }
        }

    def load_monthly_fluxes(self, experiment_number):
        """
        Load monthly CMS fluxes for specified experiment.

        Returns:
            Dictionary with flux arrays and uncertainties
        """
        files = self.experiment_files[experiment_number]
        mean_file = os.path.join(self.data_dir, files['mean'])
        std_file = os.path.join(self.data_dir, files['std'])

        # Load flux data (permute to match MATLAB [2,1,3] order)
        fluxes = {}
        with xr.open_dataset(mean_file) as ds:
            fluxes['GPP'] = ds['GPP'].values.transpose(1, 0, 2)
            fluxes['REC'] = ds['Resp_eco'].values.transpose(1, 0, 2)
            fluxes['FIR'] = ds['Fire'].values.transpose(1, 0, 2)
            fluxes['NEE'] = ds['NEE'].values.transpose(1, 0, 2)
            fluxes['NBE'] = ds['NBE'].values.transpose(1, 0, 2)

        # Load uncertainties
        with xr.open_dataset(std_file) as ds:
            fluxes['GPPunc'] = ds['GPP'].values.transpose(1, 0, 2)
            fluxes['RECunc'] = ds['Resp_eco'].values.transpose(1, 0, 2)
            fluxes['FIRunc'] = ds['Fire'].values.transpose(1, 0, 2)
            fluxes['NEEunc'] = ds['NEE'].values.transpose(1, 0, 2)
            fluxes['NBEunc'] = ds['NBE'].values.transpose(1, 0, 2)

        # Apply spatial interpolation for missing values
        return self.patch_missing_values(fluxes)

    def patch_missing_values(self, fluxes):
        """
        Fill missing values using spatial interpolation.
        Equivalent to MATLAB scattered interpolation logic.
        """
        # Load coordinate and land-sea mask data
        land_sea_mask = self._load_conus_land_sea_mask()
        x_coords, y_coords = self._get_coordinate_grids()

        # Identify finite and missing data points
        flux_sum = np.zeros_like(fluxes['GPP'][:, :, 0])
        for flux_name in ['GPP', 'REC', 'FIR', 'NEE', 'NBE']:
            flux_sum += fluxes[flux_name].mean(axis=2)

        valid_points = np.isfinite(flux_sum) & (land_sea_mask > 0)
        missing_points = (~np.isfinite(flux_sum)) & (land_sea_mask > 0)

        # Apply interpolation to all flux variables
        for flux_name in fluxes.keys():
            fluxes[flux_name] = self._interpolate_flux_field(
                fluxes[flux_name], valid_points, missing_points, x_coords, y_coords
            )

        return fluxes
```

### Spatial Interpolation
```python
def _interpolate_flux_field(self, flux_data, valid_points, missing_points, x_coords, y_coords):
    """
    Interpolate missing flux values using scattered interpolation.
    Equivalent to MATLAB scatteredInterpolant usage.
    """
    from scipy.interpolate import griddata

    n_timesteps = flux_data.shape[2]
    interpolated_data = flux_data.copy()

    for t in range(n_timesteps):
        flux_slice = flux_data[:, :, t]

        # Get valid data points
        valid_values = flux_slice[valid_points]
        valid_x = x_coords[valid_points]
        valid_y = y_coords[valid_points]

        # Interpolate to missing points
        missing_x = x_coords[missing_points]
        missing_y = y_coords[missing_points]

        interpolated_values = griddata(
            (valid_x, valid_y), valid_values,
            (missing_x, missing_y), method='linear', fill_value='nearest'
        )

        interpolated_data[missing_points, t] = interpolated_values

    return interpolated_data
```

## 4.3 Meteorological Driver Loader (`met_driver_loader.py`)

### ERA5 Diurnal Field Loader
```python
class ERA5DiurnalLoader:
    """
    Load ERA5 diurnal meteorological fields for flux downscaling.
    Based on MATLAB load_era5_diurnal_fields_new function.
    """

    def __init__(self, data_dir="./DATA/ERA5_CUSTOM/CONUS_2015_2020_DIURNAL/"):
        self.data_dir = data_dir
        self.file_pattern = "ECMWF_CARDAMOM_HOURLY_DRIVER_{var}_{month:02d}{year}.nc"

    def load_diurnal_fields(self, month, year):
        """
        Load skin temperature and solar radiation for specified month/year.

        Returns:
            Tuple of (SSRD, SKT) arrays with hourly data
        """
        # Construct file paths
        skt_file = os.path.join(
            self.data_dir,
            self.file_pattern.format(var='SKT', month=month, year=year)
        )
        ssrd_file = os.path.join(
            self.data_dir,
            self.file_pattern.format(var='SSRD', month=month, year=year)
        )

        # Load data and reorient to match MATLAB processing
        skt = self._load_and_reorient(skt_file, 'skt')  # Skin temperature in K
        ssrd = self._load_and_reorient(ssrd_file, 'ssrd')  # Solar radiation in J/m²

        return ssrd, skt

    def _load_and_reorient(self, filepath, variable):
        """
        Load NetCDF data and reorient to match MATLAB conventions.
        Equivalent to MATLAB: flipud(permute(data, [2,1,3]))
        """
        with xr.open_dataset(filepath) as ds:
            data = ds[variable].values
            # Reorient: permute [2,1,3] then flip vertically
            data_reoriented = np.transpose(data, (1, 0, 2))
            data_reoriented = np.flipud(data_reoriented)
            return data_reoriented
```

### Data Validation
```python
def validate_met_data(self, ssrd, skt):
    """Validate meteorological data ranges and consistency"""

    # Check temperature ranges (should be in Kelvin, reasonable values)
    if np.any(skt < 200) or np.any(skt > 350):
        raise ValueError("Skin temperature values outside reasonable range")

    # Check radiation values (should be non-negative)
    if np.any(ssrd < 0):
        raise ValueError("Solar radiation contains negative values")

    # Check for consistent spatial dimensions
    if ssrd.shape[:2] != skt.shape[:2]:
        raise ValueError("Spatial dimensions of SSRD and SKT do not match")
```

## 4.4 Diurnal Pattern Calculation (`diurnal_calculator.py`)

### Flux Downscaling Engine
```python
class DiurnalCalculator:
    """
    Calculate diurnal (hourly) flux patterns from monthly means.
    Implements the core downscaling algorithms from MATLAB script.
    """

    def __init__(self):
        self.q10_factor = 1.4  # Q10 temperature sensitivity (optional)

    def calculate_diurnal_fluxes(self, monthly_fluxes, month_index, ssrd, skt, co2_diurnal):
        """
        Calculate hourly flux patterns from monthly means and drivers.

        Args:
            monthly_fluxes: Dictionary with monthly flux data
            month_index: Time index for current month
            ssrd: Solar radiation diurnal patterns
            skt: Skin temperature diurnal patterns
            co2_diurnal: Fire diurnal patterns from GFED

        Returns:
            Dictionary with hourly flux patterns
        """

        # Extract monthly means for this time period
        gpp_monthly = monthly_fluxes['GPP'][:, :, month_index]
        rec_monthly = monthly_fluxes['REC'][:, :, month_index]
        fir_monthly = monthly_fluxes['FIR'][:, :, month_index]

        # Calculate diurnal patterns
        gpp_diurnal = self._calculate_gpp_diurnal(gpp_monthly, ssrd)
        rec_diurnal = self._calculate_rec_diurnal(rec_monthly, skt)
        fir_diurnal = self._calculate_fir_diurnal(fir_monthly, co2_diurnal)

        # Calculate composite fluxes
        nbe_diurnal = rec_diurnal - gpp_diurnal + fir_diurnal
        nee_diurnal = rec_diurnal - gpp_diurnal

        return {
            'GPP': gpp_diurnal,
            'REC': rec_diurnal,
            'FIR': fir_diurnal,
            'NBE': nbe_diurnal,
            'NEE': nee_diurnal
        }

    def _calculate_gpp_diurnal(self, gpp_monthly, ssrd):
        """
        Calculate GPP diurnal pattern using solar radiation.
        Equivalent to MATLAB: SSRD.*repmat(GPP_monthly./mean(SSRD,3), [1,1,size(SSRD,3)])
        """
        # Calculate mean solar radiation over time dimension
        ssrd_mean = np.mean(ssrd, axis=2, keepdims=True)

        # Avoid division by zero
        ssrd_mean[ssrd_mean == 0] = np.nan

        # Scale solar radiation by monthly GPP ratio
        gpp_scaling = gpp_monthly[:, :, np.newaxis] / ssrd_mean
        gpp_diurnal = ssrd * gpp_scaling

        # Convert units: gC/m²/day to Kg C/Km²/sec
        return gpp_diurnal * 1e3 / 24 / 3600

    def _calculate_rec_diurnal(self, rec_monthly, skt):
        """
        Calculate respiration diurnal pattern using temperature.
        Equivalent to MATLAB: SKT.*repmat(REC_monthly./mean(SKT,3), [1,1,size(SKT,3)])
        """
        # Calculate mean temperature over time dimension
        skt_mean = np.mean(skt, axis=2, keepdims=True)

        # Scale temperature by monthly REC ratio
        rec_scaling = rec_monthly[:, :, np.newaxis] / skt_mean
        rec_diurnal = skt * rec_scaling

        # Convert units: gC/m²/day to Kg C/Km²/sec
        return rec_diurnal * 1e3 / 24 / 3600

    def _calculate_fir_diurnal(self, fir_monthly, co2_diurnal):
        """
        Calculate fire diurnal pattern using GFED fire timing.
        Handles 3-hourly to hourly conversion.
        """
        # Calculate mean fire pattern over time
        co2_mean = np.mean(co2_diurnal, axis=2, keepdims=True)
        co2_mean[co2_mean == 0] = np.nan

        # Scale fire timing by monthly fire emissions
        fir_scaling = fir_monthly[:, :, np.newaxis] / co2_mean
        fir_diurnal_3h = self._nan_to_zero(co2_diurnal * fir_scaling)

        # Convert 3-hourly to hourly by triplication
        fir_diurnal = self._triplicate_3hourly_to_hourly(fir_diurnal_3h)

        # Convert units: gC/m²/day to Kg C/Km²/sec
        return fir_diurnal * 1e3 / 24 / 3600
```

### Time Resolution Conversion
```python
def _triplicate_3hourly_to_hourly(self, data_3h):
    """
    Convert 3-hourly fire data to hourly by triplication.
    Equivalent to MATLAB logic for FIRdiurnal expansion.
    """
    n_times_3h = data_3h.shape[2]
    n_times_1h = n_times_3h * 3

    # Create hourly array
    data_1h = np.zeros((*data_3h.shape[:2], n_times_1h))

    # Triplicate each 3-hourly timestep
    for i in range(3):
        data_1h[:, :, i::3] = data_3h

    return data_1h

def _nan_to_zero(self, data):
    """Convert NaN values to zero, equivalent to MATLAB nan2zero"""
    data_copy = data.copy()
    data_copy[np.isnan(data_copy)] = 0
    return data_copy
```

## 4.5 GFED Diurnal Pattern Loader (`gfed_diurnal_loader.py`)

### GFED Diurnal Data Reader
```python
class GFEDDiurnalLoader:
    """
    Load GFED diurnal fire patterns for specific months.
    Based on MATLAB load_gfed_diurnal_fields_05deg function.
    """

    def __init__(self, gfed_data_dir="./DATA/GFED4/", region_bounds=None):
        self.gfed_data_dir = gfed_data_dir
        self.region_bounds = region_bounds or [-124.75, -65.25, 24.75, 60.25]  # CONUS
        self.emission_factors = self._setup_emission_factors()

    def load_diurnal_fields(self, month, year, target_region=None):
        """
        Load GFED diurnal fire patterns for specified month and year.

        Returns:
            CO2 diurnal emissions array for the target region
        """
        # Determine file path (beta version for years >= 2017)
        beta_suffix = '_beta' if year >= 2017 else ''
        gfed_file = os.path.join(
            self.gfed_data_dir,
            f'GFED4.1s_{year}{beta_suffix}.hdf5'
        )

        # Load GFED data for the month
        gfed_data = self._load_monthly_gfed_data(gfed_file, month)

        # Extract regional subset
        co2_diurnal = self._extract_regional_co2_diurnal(gfed_data, target_region)

        return co2_diurnal

    def _load_monthly_gfed_data(self, filepath, month):
        """Load monthly GFED data from HDF5 file"""

        with h5py.File(filepath, 'r') as f:
            # Get number of days in month
            year = int(os.path.basename(filepath).split('_')[1][:4])
            days_in_month = self._get_days_in_month(year, month)

            # Load dry matter fractions by vegetation type
            vegetation_types = ['SAVA', 'BORF', 'TEMF', 'DEFO', 'PEAT', 'AGRI']
            dm_fractions = np.zeros((720, 1440, 6))  # 0.25° global grid

            for i, veg_type in enumerate(vegetation_types):
                field_path = f'/emissions/{month:02d}/partitioning/DM_{veg_type}'
                dm_fractions[:, :, i] = np.flipud(f[field_path][:].T)

            # Load daily fractions
            daily_fractions = np.zeros((720, 1440, days_in_month))
            for day in range(1, days_in_month + 1):
                field_path = f'/emissions/{month:02d}/daily_fraction/day_{day}'
                daily_fractions[:, :, day-1] = np.flipud(f[field_path][:].T)

            # Load diurnal fractions (8 × 3-hour periods)
            diurnal_fractions = np.zeros((720, 1440, 8))
            for hour_group in range(8):
                start_hour = hour_group * 3
                end_hour = (hour_group + 1) * 3
                field_path = f'/emissions/{month:02d}/diurnal_cycle/UTC_{start_hour}-{end_hour}h'
                diurnal_fractions[:, :, hour_group] = np.flipud(f[field_path][:].T)

        return {
            'dm_fractions': dm_fractions,
            'daily_fractions': daily_fractions,
            'diurnal_fractions': diurnal_fractions,
            'days_in_month': days_in_month
        }
```

### Emission Calculation
```python
def _calculate_co2_emissions(self, gfed_data):
    """
    Calculate CO2 emissions from GFED dry matter and emission factors.
    Equivalent to complex nested loop in MATLAB (lines 598-609).
    """
    days = gfed_data['days_in_month']
    co2_emissions = np.zeros((720, 1440, days * 8, 4))  # 4 species: CO2, CO, CH4, C

    # Get emission factors
    ef_matrix = self.emission_factors.get_factor_matrix()

    for day in range(days):
        for hour_group in range(8):  # 8 × 3-hour periods per day
            for species in range(4):  # CO2, CO, CH4, C
                for veg_type in range(6):  # 6 vegetation types

                    time_index = hour_group + day * 8

                    # Calculate emissions per unit carbon
                    emission_rate = (
                        gfed_data['dm_fractions'][:, :, veg_type] *
                        ef_matrix[species, veg_type] *
                        gfed_data['daily_fractions'][:, :, day] *
                        gfed_data['diurnal_fractions'][:, :, hour_group] /
                        ef_matrix[3, veg_type]  # Normalize by carbon content
                    )

                    co2_emissions[:, :, time_index, species] += emission_rate

    return co2_emissions[:, :, :, 0]  # Return only CO2 emissions

def _extract_regional_co2_diurnal(self, gfed_data, target_region):
    """Extract CO2 diurnal patterns for target region"""

    # Calculate full CO2 emissions
    co2_global = self._calculate_co2_emissions(gfed_data)

    # Aggregate to 0.5° and extract region
    co2_05deg = self._aggregate_025_to_05_degree(co2_global)

    # Extract regional subset
    if target_region:
        region_indices = self._get_region_indices(target_region)
        co2_regional = co2_05deg[region_indices]
    else:
        co2_regional = co2_05deg

    return co2_regional
```

## 4.6 NetCDF Output Writers (`diurnal_output_writers.py`)

### Monthly Flux Writer
```python
class DiurnalFluxWriter:
    """
    Write diurnal flux data to NetCDF files in GeosChem-compatible format.
    Based on MATLAB write functions.
    """

    def __init__(self, output_base_dir):
        self.output_base_dir = output_base_dir

    def write_monthly_flux_to_geoschem_format(self, flux_data, uncertainty_data,
                                            year, month, aux_data, flux_name, experiment):
        """
        Write monthly flux with uncertainty to NetCDF.
        Equivalent to MATLAB write_monthlyflux_to_geoschem_format.
        """

        # Setup directory structure
        output_dir = self._setup_monthly_output_directory(flux_name, experiment)
        filename = os.path.join(output_dir, f"{year}", f"{month:02d}.nc")

        # Skip if file already exists
        if os.path.exists(filename):
            return

        # Prepare data arrays
        uncertainty_factor = self._calculate_uncertainty_factor(flux_data, uncertainty_data)

        # Create NetCDF file
        with netCDF4.Dataset(filename, 'w') as nc:
            self._create_monthly_netcdf_structure(nc, aux_data, flux_data.shape)

            # Write data
            nc.variables['CO2_Flux'][:] = self._nan_to_zero(flux_data)
            nc.variables['Uncertainty'][:] = uncertainty_factor

            # Add metadata
            self._add_monthly_metadata(nc, flux_name, year, month)

    def write_hourly_flux_to_geoschem_format(self, flux_data, year, month,
                                           aux_data, flux_name, experiment):
        """
        Write hourly flux data to daily NetCDF files.
        Equivalent to MATLAB write_hourly_flux_to_geoschem_format.
        """

        # Setup directory structure
        output_dir = self._setup_diurnal_output_directory(flux_name, experiment)
        month_dir = os.path.join(output_dir, f"{year}", f"{month:02d}")
        os.makedirs(month_dir, exist_ok=True)

        # Get days in month
        days_in_month = self._get_days_in_month(year, month)

        # Write daily files
        for day in range(1, days_in_month + 1):
            day_filename = os.path.join(month_dir, f"{day:02d}.nc")

            # Extract 24 hours for this day
            hour_indices = slice((day-1)*24, day*24)
            daily_flux = flux_data[:, :, hour_indices]

            # Create daily NetCDF file
            with netCDF4.Dataset(day_filename, 'w') as nc:
                self._create_hourly_netcdf_structure(nc, aux_data, daily_flux.shape)
                nc.variables['CO2_Flux'][:] = self._nan_to_zero(daily_flux)
                self._add_hourly_metadata(nc, flux_name, year, month, day)
```

### NetCDF Structure Creation
```python
def _create_monthly_netcdf_structure(self, nc_dataset, aux_data, data_shape):
    """Create NetCDF structure for monthly files"""

    # Create dimensions
    nc_dataset.createDimension('longitude', data_shape[1])
    nc_dataset.createDimension('latitude', data_shape[0])
    nc_dataset.createDimension('time', 1)
    nc_dataset.createDimension('data_vars', 2)  # flux and uncertainty

    # Create coordinate variables
    lon_var = nc_dataset.createVariable('longitude', 'f4', ('longitude',))
    lat_var = nc_dataset.createVariable('latitude', 'f4', ('latitude',))
    time_var = nc_dataset.createVariable('time', 'f4', ('time',))

    # Create data variables
    flux_var = nc_dataset.createVariable('CO2_Flux', 'f4',
                                       ('latitude', 'longitude', 'time'))
    unc_var = nc_dataset.createVariable('Uncertainty', 'f4',
                                      ('latitude', 'longitude', 'time'))

    # Set coordinate data
    lon_var[:] = aux_data['x']
    lat_var[:] = aux_data['y']
    time_var[:] = 1

    # Add units
    lon_var.units = 'degrees_east'
    lat_var.units = 'degrees_north'
    time_var.units = 'N/A'
    flux_var.units = 'Kg C/Km^2/sec'
    unc_var.units = 'factor'

def _create_hourly_netcdf_structure(self, nc_dataset, aux_data, data_shape):
    """Create NetCDF structure for hourly files"""

    # Create dimensions
    nc_dataset.createDimension('longitude', data_shape[1])
    nc_dataset.createDimension('latitude', data_shape[0])
    nc_dataset.createDimension('time', 24)  # 24 hours

    # Create coordinate variables
    lon_var = nc_dataset.createVariable('longitude', 'f4', ('longitude',))
    lat_var = nc_dataset.createVariable('latitude', 'f4', ('latitude',))
    time_var = nc_dataset.createVariable('time', 'f4', ('time',))

    # Create data variable
    flux_var = nc_dataset.createVariable('CO2_Flux', 'f4',
                                       ('latitude', 'longitude', 'time'))

    # Set coordinate data
    lon_var[:] = aux_data['x']
    lat_var[:] = aux_data['y']
    time_var[:] = np.arange(0.5, 24, 1)  # Hour centers

    # Add units
    lon_var.units = 'degrees_east'
    lat_var.units = 'degrees_north'
    time_var.units = 'hour'
    flux_var.units = 'Kg C/Km^2/sec'
```

## 4.7 Testing and Validation

### Test Structure
```
tests/diurnal/
├── test_diurnal_processor.py
├── test_cms_flux_loader.py
├── test_met_driver_loader.py
├── test_diurnal_calculator.py
├── test_gfed_diurnal_loader.py
├── test_diurnal_output_writers.py
└── fixtures/
    ├── sample_cms_fluxes.nc
    ├── sample_era5_diurnal.nc
    ├── sample_gfed_diurnal.hdf5
    └── expected_outputs/
```

### Validation Against MATLAB
```python
def test_diurnal_calculation_accuracy(self):
    """Test diurnal flux calculations against MATLAB results"""

def test_file_output_format(self):
    """Verify output files match MATLAB format exactly"""

def test_temporal_consistency(self):
    """Ensure temporal continuity in hourly outputs"""
```

## 4.8 Success Criteria

### Functional Requirements
- [ ] Accurately reproduce MATLAB diurnal downscaling algorithms
- [ ] Process all five flux types (GPP, REC, FIR, NEE, NBE)
- [ ] Generate both monthly and hourly output files
- [ ] Handle missing data through spatial interpolation

### Data Quality Requirements
- [ ] Preserve monthly flux totals in hourly disaggregation
- [ ] Maintain physical relationships between flux components
- [ ] Apply appropriate unit conversions throughout pipeline
- [ ] Generate realistic diurnal patterns based on drivers

### Performance Requirements
- [ ] Process 6 years of CONUS data efficiently
- [ ] Support parallel processing for multiple months
- [ ] Manage memory usage for large hourly arrays
- [ ] Provide progress tracking for long operations

### Integration Requirements
- [ ] Compatible with existing CARDAMOM infrastructure
- [ ] Support multiple CMS experiments
- [ ] Generate GeosChem-compatible output format
- [ ] Maintain CARDAMOM metadata standards