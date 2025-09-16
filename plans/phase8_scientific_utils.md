# Phase 8: Scientific Functions Library

## Overview
Create a comprehensive library of scientific utility functions that replicate MATLAB functionality while providing enhanced capabilities for atmospheric science, carbon cycle modeling, and meteorological data processing. This library serves as the foundation for all scientific calculations across the CARDAMOM preprocessing system.

## 8.1 Atmospheric Science Functions (`atmospheric_science.py`)

### Water Vapor and Humidity Calculations
```python
import numpy as np
from scipy import constants

class AtmosphericCalculations:
    """
    Atmospheric science calculations for water vapor, pressure, and humidity.
    Replicates and extends MATLAB SCIFUN functions.
    """

    def __init__(self):
        self.R_v = 461.5  # Specific gas constant for water vapor (J/kg/K)
        self.R_d = 287.04  # Specific gas constant for dry air (J/kg/K)
        self.epsilon = self.R_d / self.R_v  # Ratio of gas constants

    def saturation_pressure_water(self, temperature_kelvin, method='tetens'):
        """
        Calculate saturation pressure of water vapor.
        Equivalent to MATLAB SCIFUN_H2O_SATURATION_PRESSURE.

        Args:
            temperature_kelvin: Temperature in Kelvin (scalar or array)
            method: Calculation method ('tetens', 'magnus', 'wmo')

        Returns:
            Saturation pressure in Pa
        """

        T = np.asarray(temperature_kelvin)
        T_celsius = T - 273.15

        if method == 'tetens':
            # Tetens formula (widely used, good for 0-50°C)
            e_sat = 610.78 * np.exp(17.27 * T_celsius / (T_celsius + 237.3))

        elif method == 'magnus':
            # Magnus formula (more accurate over wider range)
            e_sat = 610.94 * np.exp(17.625 * T_celsius / (T_celsius + 243.04))

        elif method == 'wmo':
            # WMO recommended formula (Goff-Gratch)
            e_sat = self._goff_gratch_formula(T)

        else:
            raise ValueError(f"Unknown method: {method}")

        return e_sat

    def _goff_gratch_formula(self, temperature_kelvin):
        """
        Goff-Gratch formula for saturation pressure (WMO recommendation).
        More accurate but computationally intensive.
        """

        T = np.asarray(temperature_kelvin)
        T0 = 273.16  # Triple point temperature

        # Over water (T > 273.16 K)
        log10_ew = (-7.90298 * (373.16 / T - 1) +
                    5.02808 * np.log10(373.16 / T) -
                    1.3816e-7 * (10**(11.344 * (1 - T / 373.16)) - 1) +
                    8.1328e-3 * (10**(-3.49149 * (373.16 / T - 1)) - 1) +
                    np.log10(1013.246))

        # Over ice (T < 273.16 K)
        log10_ei = (-9.09718 * (T0 / T - 1) -
                    3.56654 * np.log10(T0 / T) +
                    0.876793 * (1 - T / T0) +
                    np.log10(6.1071))

        # Choose appropriate formula based on temperature
        e_sat = np.where(T >= T0, 10**log10_ew, 10**log10_ei) * 100  # Convert hPa to Pa

        return e_sat

    def calculate_vpd(self, temperature_max_kelvin, dewpoint_temperature_kelvin,
                     output_units='hPa'):
        """
        Calculate Vapor Pressure Deficit.
        Replicates MATLAB VPD calculation from line 202 of main script.

        Args:
            temperature_max_kelvin: Maximum temperature in Kelvin
            dewpoint_temperature_kelvin: Dewpoint temperature in Kelvin
            output_units: Output units ('Pa', 'hPa', 'kPa')

        Returns:
            VPD in specified units
        """

        # Calculate saturation pressure at maximum temperature
        e_sat_tmax = self.saturation_pressure_water(temperature_max_kelvin)

        # Calculate saturation pressure at dewpoint temperature
        e_sat_dewpoint = self.saturation_pressure_water(dewpoint_temperature_kelvin)

        # VPD is the difference
        vpd_pa = e_sat_tmax - e_sat_dewpoint

        # Convert units
        if output_units == 'Pa':
            return vpd_pa
        elif output_units == 'hPa':
            return vpd_pa / 100  # Convert Pa to hPa
        elif output_units == 'kPa':
            return vpd_pa / 1000  # Convert Pa to kPa
        else:
            raise ValueError(f"Unknown output units: {output_units}")

    def calculate_specific_humidity(self, pressure_pa, mixing_ratio):
        """
        Calculate specific humidity from mixing ratio.

        Args:
            pressure_pa: Atmospheric pressure in Pa
            mixing_ratio: Water vapor mixing ratio (kg/kg)

        Returns:
            Specific humidity (kg/kg)
        """
        return mixing_ratio / (1 + mixing_ratio)

    def calculate_mixing_ratio(self, vapor_pressure_pa, pressure_pa):
        """
        Calculate water vapor mixing ratio.

        Args:
            vapor_pressure_pa: Water vapor pressure in Pa
            pressure_pa: Total atmospheric pressure in Pa

        Returns:
            Mixing ratio (kg/kg)
        """
        return self.epsilon * vapor_pressure_pa / (pressure_pa - vapor_pressure_pa)

    def calculate_relative_humidity(self, vapor_pressure_pa, temperature_kelvin):
        """
        Calculate relative humidity.

        Args:
            vapor_pressure_pa: Actual water vapor pressure in Pa
            temperature_kelvin: Temperature in Kelvin

        Returns:
            Relative humidity as fraction (0-1)
        """
        e_sat = self.saturation_pressure_water(temperature_kelvin)
        return vapor_pressure_pa / e_sat
```

### Radiation and Energy Balance
```python
class RadiationCalculations:
    """Radiation and energy balance calculations"""

    def __init__(self):
        self.stefan_boltzmann = constants.Stefan_Boltzmann  # W m^-2 K^-4
        self.solar_constant = 1361  # W m^-2 (current solar constant)

    def calculate_solar_zenith_angle(self, latitude, longitude, year, day_of_year,
                                   hour_utc):
        """
        Calculate solar zenith angle for photosynthesis calculations.

        Args:
            latitude: Latitude in degrees (-90 to 90)
            longitude: Longitude in degrees (-180 to 180)
            year: Year (for orbital variations)
            day_of_year: Day of year (1-365/366)
            hour_utc: Hour in UTC (0-24)

        Returns:
            Solar zenith angle in radians
        """

        # Convert to radians
        lat_rad = np.radians(latitude)
        lon_rad = np.radians(longitude)

        # Solar declination angle
        declination = self._calculate_solar_declination(day_of_year)

        # Hour angle
        hour_angle = self._calculate_hour_angle(longitude, hour_utc, day_of_year)

        # Solar zenith angle
        cos_zenith = (np.sin(lat_rad) * np.sin(declination) +
                     np.cos(lat_rad) * np.cos(declination) * np.cos(hour_angle))

        zenith_angle = np.arccos(np.clip(cos_zenith, -1, 1))

        return zenith_angle

    def _calculate_solar_declination(self, day_of_year):
        """Calculate solar declination angle"""
        return np.radians(23.45) * np.sin(np.radians(360 * (284 + day_of_year) / 365))

    def _calculate_hour_angle(self, longitude, hour_utc, day_of_year):
        """Calculate hour angle"""
        # Equation of time correction (simplified)
        equation_of_time = 4 * (longitude - 15 * self._calculate_solar_time_correction(day_of_year))

        # Local solar time
        local_solar_time = hour_utc + equation_of_time / 60

        # Hour angle (15 degrees per hour)
        return np.radians(15 * (local_solar_time - 12))

    def _calculate_solar_time_correction(self, day_of_year):
        """Calculate solar time correction factor"""
        B = np.radians(360 * (day_of_year - 1) / 365)
        return (9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)) / 60

    def calculate_clear_sky_radiation(self, zenith_angle, elevation_m=0):
        """
        Calculate clear-sky solar radiation.

        Args:
            zenith_angle: Solar zenith angle in radians
            elevation_m: Elevation above sea level in meters

        Returns:
            Clear-sky radiation in W m^-2
        """

        # Atmospheric pressure adjustment for elevation
        pressure_ratio = np.exp(-elevation_m / 8400)  # Scale height ~8.4 km

        # Air mass calculation
        air_mass = pressure_ratio / np.cos(zenith_angle)

        # Atmospheric transmission (simplified model)
        transmission = 0.7 ** (air_mass ** 0.678)

        # Clear-sky radiation
        clear_sky_rad = self.solar_constant * np.cos(zenith_angle) * transmission

        # Set to zero for zenith angles > 90 degrees (sun below horizon)
        clear_sky_rad = np.where(zenith_angle > np.pi/2, 0, clear_sky_rad)

        return clear_sky_rad

    def convert_radiation_units(self, radiation_data, input_units, output_units,
                              time_step_hours=1):
        """
        Convert between different radiation units.

        Args:
            radiation_data: Radiation values
            input_units: Input units ('W_m2', 'J_m2', 'MJ_m2_day', 'kJ_m2_day')
            output_units: Output units (same options as input_units)
            time_step_hours: Time step in hours for flux conversions

        Returns:
            Converted radiation values
        """

        # Convert to standard units (W m^-2)
        if input_units == 'W_m2':
            standard_flux = radiation_data
        elif input_units == 'J_m2':
            standard_flux = radiation_data / (time_step_hours * 3600)
        elif input_units == 'MJ_m2_day':
            standard_flux = radiation_data * 1e6 / (24 * 3600)
        elif input_units == 'kJ_m2_day':
            standard_flux = radiation_data * 1e3 / (24 * 3600)
        else:
            raise ValueError(f"Unknown input units: {input_units}")

        # Convert from standard units to output units
        if output_units == 'W_m2':
            return standard_flux
        elif output_units == 'J_m2':
            return standard_flux * time_step_hours * 3600
        elif output_units == 'MJ_m2_day':
            return standard_flux * 24 * 3600 / 1e6
        elif output_units == 'kJ_m2_day':
            return standard_flux * 24 * 3600 / 1e3
        else:
            raise ValueError(f"Unknown output units: {output_units}")
```

## 8.2 Carbon Cycle and Biogeochemistry (`carbon_cycle.py`)

### Photosynthesis and Respiration Models
```python
class CarbonCycleCalculations:
    """Carbon cycle calculations for CARDAMOM preprocessing"""

    def __init__(self):
        self.molar_mass_c = 12.011  # g/mol
        self.molar_mass_co2 = 44.01  # g/mol
        self.seconds_per_day = 86400

    def calculate_light_use_efficiency(self, temperature_celsius, vpd_kpa,
                                     par_mol_m2_s, method='physiological'):
        """
        Calculate light use efficiency for photosynthesis.

        Args:
            temperature_celsius: Temperature in Celsius
            vpd_kpa: Vapor pressure deficit in kPa
            par_mol_m2_s: Photosynthetically active radiation in mol m^-2 s^-1
            method: Calculation method ('physiological', 'empirical')

        Returns:
            Light use efficiency (mol CO2 / mol photons)
        """

        if method == 'physiological':
            return self._physiological_lue(temperature_celsius, vpd_kpa, par_mol_m2_s)
        elif method == 'empirical':
            return self._empirical_lue(temperature_celsius, vpd_kpa)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _physiological_lue(self, temperature_celsius, vpd_kpa, par_mol_m2_s):
        """Physiologically-based light use efficiency"""

        # Temperature response (optimum around 25°C)
        t_opt = 25.0
        t_response = np.exp(-((temperature_celsius - t_opt) / 15.0)**2)

        # VPD stress response
        vpd_response = np.exp(-vpd_kpa / 2.0)

        # Light saturation response
        light_response = par_mol_m2_s / (par_mol_m2_s + 500)  # Half-saturation ~500 µmol m^-2 s^-1

        # Maximum LUE (typical C3 plants)
        max_lue = 0.08  # mol CO2 / mol photons

        return max_lue * t_response * vpd_response * light_response

    def _empirical_lue(self, temperature_celsius, vpd_kpa):
        """Empirical light use efficiency model"""

        # Simple empirical relationships
        t_factor = np.clip((temperature_celsius - 0) / 30.0, 0, 1)
        vpd_factor = np.exp(-vpd_kpa / 3.0)

        max_lue = 0.06  # mol CO2 / mol photons
        return max_lue * t_factor * vpd_factor

    def calculate_q10_temperature_response(self, temperature_celsius,
                                         reference_temp=10.0, q10=2.0):
        """
        Calculate Q10 temperature response for respiration.
        Used in diurnal respiration downscaling.

        Args:
            temperature_celsius: Temperature in Celsius
            reference_temp: Reference temperature in Celsius
            q10: Q10 factor (default 2.0 for respiration)

        Returns:
            Temperature response factor
        """

        delta_t = temperature_celsius - reference_temp
        return q10 ** (delta_t / 10.0)

    def convert_carbon_flux_units(self, flux_data, input_units, output_units,
                                molecular_weight=None):
        """
        Convert between different carbon flux units.

        Args:
            flux_data: Flux values
            input_units: Input units (see options below)
            output_units: Output units (see options below)
            molecular_weight: Molecular weight for species conversions

        Unit options:
        - 'gC_m2_day': grams C per m² per day
        - 'gC_m2_s': grams C per m² per second
        - 'kgC_km2_s': kilograms C per km² per second
        - 'molC_m2_s': moles C per m² per second
        - 'gCO2_m2_day': grams CO2 per m² per day
        - 'umolCO2_m2_s': micromoles CO2 per m² per second
        """

        # Define conversion factors to standard units (gC m^-2 s^-1)
        input_factors = {
            'gC_m2_day': 1.0 / self.seconds_per_day,
            'gC_m2_s': 1.0,
            'kgC_km2_s': 1.0,  # kg/km² = g/m²
            'molC_m2_s': self.molar_mass_c,
            'gCO2_m2_day': (self.molar_mass_c / self.molar_mass_co2) / self.seconds_per_day,
            'umolCO2_m2_s': self.molar_mass_c / 1e6
        }

        output_factors = {
            'gC_m2_day': self.seconds_per_day,
            'gC_m2_s': 1.0,
            'kgC_km2_s': 1.0,
            'molC_m2_s': 1.0 / self.molar_mass_c,
            'gCO2_m2_day': (self.molar_mass_co2 / self.molar_mass_c) * self.seconds_per_day,
            'umolCO2_m2_s': 1e6 / self.molar_mass_c
        }

        # Convert to standard units, then to output units
        standard_flux = flux_data * input_factors[input_units]
        converted_flux = standard_flux * output_factors[output_units]

        return converted_flux

    def calculate_ecosystem_respiration(self, soil_temperature_celsius,
                                      soil_moisture_fraction=None,
                                      base_respiration=None, q10=2.0):
        """
        Calculate ecosystem respiration based on environmental drivers.

        Args:
            soil_temperature_celsius: Soil temperature in Celsius
            soil_moisture_fraction: Soil moisture as fraction (0-1)
            base_respiration: Base respiration rate at 10°C
            q10: Temperature sensitivity factor

        Returns:
            Ecosystem respiration rate
        """

        # Temperature response
        temp_response = self.calculate_q10_temperature_response(
            soil_temperature_celsius, reference_temp=10.0, q10=q10
        )

        # Moisture response (optional)
        if soil_moisture_fraction is not None:
            # Simple moisture response function
            moisture_response = np.clip(soil_moisture_fraction / 0.5, 0, 1)
            moisture_response = np.where(soil_moisture_fraction > 0.8,
                                       1 - (soil_moisture_fraction - 0.8) / 0.2,
                                       moisture_response)
        else:
            moisture_response = 1.0

        # Base respiration (if not provided, use default)
        if base_respiration is None:
            base_respiration = 1.0  # gC m^-2 day^-1

        return base_respiration * temp_response * moisture_response
```

### Fire Emission Calculations
```python
class FireEmissionCalculations:
    """Fire emission calculations for GFED processing"""

    def __init__(self):
        self.emission_factors = self._initialize_emission_factors()
        self.vegetation_types = ['SAVA', 'BORF', 'TEMF', 'DEFO', 'PEAT', 'AGRI']
        self.species = ['CO2', 'CO', 'CH4', 'C']

    def _initialize_emission_factors(self):
        """
        Initialize emission factors matrix.
        From MATLAB emission_factors() function in GFED processing.
        """
        # Emission factors (g species / kg dry matter)
        ef_matrix = np.array([
            [1686, 1489, 1647, 1643, 1703, 1585],  # CO2
            [63, 127, 88, 93, 210, 102],           # CO
            [1.94, 5.96, 3.36, 5.07, 20.8, 5.82], # CH4
            [488.273, 464.989, 489.416, 491.751, 570.055, 480.352]  # C
        ])

        # Uncertainty matrix
        ef_uncertainty = np.array([
            [38, 121, 37, 58, 58, 100],          # CO2 uncertainty
            [17, 45, 19, 27, 68, 33],            # CO uncertainty
            [0.85, 3.14, 0.91, 1.98, 11.42, 3.56], # CH4 uncertainty
            [np.nan] * 6                          # C uncertainty (not provided)
        ])

        return {
            'factors': ef_matrix,
            'uncertainties': ef_uncertainty
        }

    def calculate_species_emissions(self, dry_matter_burned, vegetation_fractions,
                                  species='CO2'):
        """
        Calculate emissions for specific species from dry matter burned.

        Args:
            dry_matter_burned: Dry matter burned (kg DM m^-2)
            vegetation_fractions: Fractions of each vegetation type
            species: Species to calculate ('CO2', 'CO', 'CH4', 'C')

        Returns:
            Emissions in g species m^-2
        """

        species_index = self.species.index(species)
        species_ef = self.emission_factors['factors'][species_index, :]

        # Calculate emissions for each vegetation type
        emissions_by_veg = dry_matter_burned * vegetation_fractions * species_ef

        # Sum across vegetation types
        total_emissions = np.sum(emissions_by_veg, axis=-1)

        return total_emissions

    def apply_diurnal_fire_pattern(self, monthly_emissions, diurnal_fractions,
                                 daily_fractions):
        """
        Apply diurnal and daily patterns to monthly fire emissions.
        Used in diurnal flux processing.

        Args:
            monthly_emissions: Monthly total emissions
            diurnal_fractions: Hourly fractions (8 × 3-hour periods)
            daily_fractions: Daily fractions for each day of month

        Returns:
            Hourly emissions for the month
        """

        days_in_month = daily_fractions.shape[-1]
        hours_per_day = 24
        total_hours = days_in_month * hours_per_day

        hourly_emissions = np.zeros((*monthly_emissions.shape, total_hours))

        for day in range(days_in_month):
            for hour_group in range(8):  # 8 × 3-hour periods
                # Calculate emission rate for this 3-hour period
                emission_rate = (monthly_emissions *
                               daily_fractions[..., day] *
                               diurnal_fractions[..., hour_group])

                # Assign to 3 consecutive hours
                hour_start = day * 24 + hour_group * 3
                hour_end = hour_start + 3

                for h in range(hour_start, min(hour_end, total_hours)):
                    hourly_emissions[..., h] = emission_rate / 3  # Divide by 3 hours

        return hourly_emissions

    def calculate_emission_uncertainty(self, emissions, species, vegetation_type):
        """
        Calculate uncertainty in fire emissions.

        Args:
            emissions: Emission values
            species: Species name
            vegetation_type: Vegetation type index

        Returns:
            Emission uncertainty (same units as emissions)
        """

        species_index = self.species.index(species)
        uncertainty_factor = self.emission_factors['uncertainties'][species_index, vegetation_type]

        if np.isnan(uncertainty_factor):
            return np.zeros_like(emissions)
        else:
            return emissions * (uncertainty_factor / 100.0)  # Convert percentage to fraction
```

## 8.3 Statistical and Interpolation Functions (`statistics_utils.py`)

### Spatial Interpolation
```python
class SpatialInterpolation:
    """Spatial interpolation methods for data processing"""

    def __init__(self):
        from scipy.spatial import cKDTree
        from scipy.interpolate import griddata, RBFInterpolator
        self.griddata = griddata
        self.RBFInterpolator = RBFInterpolator

    def scattered_interpolation(self, x_known, y_known, values_known,
                               x_target, y_target, method='linear',
                               fill_value='extrapolate'):
        """
        Scattered data interpolation.
        Equivalent to MATLAB scatteredInterpolant functionality.

        Args:
            x_known, y_known: Known coordinate points
            values_known: Known values at coordinate points
            x_target, y_target: Target coordinate points
            method: Interpolation method ('linear', 'nearest', 'cubic')
            fill_value: How to handle extrapolation

        Returns:
            Interpolated values at target points
        """

        points_known = np.column_stack((x_known.flatten(), y_known.flatten()))
        points_target = np.column_stack((x_target.flatten(), y_target.flatten()))

        if fill_value == 'extrapolate':
            # Use nearest neighbor for extrapolation
            interpolated = self.griddata(
                points_known, values_known.flatten(),
                points_target, method=method, fill_value=np.nan
            )

            # Fill NaN values with nearest neighbor
            nan_mask = np.isnan(interpolated)
            if np.any(nan_mask):
                nearest_values = self.griddata(
                    points_known, values_known.flatten(),
                    points_target[nan_mask], method='nearest'
                )
                interpolated[nan_mask] = nearest_values

        else:
            interpolated = self.griddata(
                points_known, values_known.flatten(),
                points_target, method=method, fill_value=fill_value
            )

        return interpolated.reshape(x_target.shape)

    def rbf_interpolation(self, x_known, y_known, values_known,
                         x_target, y_target, function='thin_plate_spline'):
        """
        Radial basis function interpolation for smooth surfaces.

        Args:
            x_known, y_known: Known coordinate points
            values_known: Known values
            x_target, y_target: Target coordinates
            function: RBF function type

        Returns:
            Interpolated values
        """

        points_known = np.column_stack((x_known.flatten(), y_known.flatten()))
        points_target = np.column_stack((x_target.flatten(), y_target.flatten()))

        # Create RBF interpolator
        rbf = self.RBFInterpolator(points_known, values_known.flatten(),
                                  kernel=function)

        # Interpolate to target points
        interpolated = rbf(points_target)

        return interpolated.reshape(x_target.shape)

    def grid_aggregation(self, high_res_data, aggregation_factor, method='mean'):
        """
        Aggregate high-resolution grid to lower resolution.
        Used for resolution conversions (e.g., 0.25° to 0.5°).

        Args:
            high_res_data: High resolution input data
            aggregation_factor: Factor to aggregate by (e.g., 2 for 2x2 blocks)
            method: Aggregation method ('mean', 'sum', 'max', 'min')

        Returns:
            Aggregated lower resolution data
        """

        if len(high_res_data.shape) == 2:
            return self._aggregate_2d(high_res_data, aggregation_factor, method)
        elif len(high_res_data.shape) == 3:
            return self._aggregate_3d(high_res_data, aggregation_factor, method)
        else:
            raise ValueError("Only 2D and 3D arrays supported")

    def _aggregate_2d(self, data, factor, method):
        """Aggregate 2D data"""

        # Ensure dimensions are divisible by factor
        ny, nx = data.shape
        ny_new = (ny // factor) * factor
        nx_new = (nx // factor) * factor

        # Trim data if necessary
        data_trimmed = data[:ny_new, :nx_new]

        # Reshape for aggregation
        reshaped = data_trimmed.reshape(ny_new // factor, factor,
                                      nx_new // factor, factor)

        # Apply aggregation method
        if method == 'mean':
            return np.nanmean(reshaped, axis=(1, 3))
        elif method == 'sum':
            return np.nansum(reshaped, axis=(1, 3))
        elif method == 'max':
            return np.nanmax(reshaped, axis=(1, 3))
        elif method == 'min':
            return np.nanmin(reshaped, axis=(1, 3))
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def _aggregate_3d(self, data, factor, method):
        """Aggregate 3D data (time series)"""

        aggregated_slices = []
        for t in range(data.shape[2]):
            agg_slice = self._aggregate_2d(data[:, :, t], factor, method)
            aggregated_slices.append(agg_slice)

        return np.stack(aggregated_slices, axis=2)
```

### Time Series Analysis
```python
class TimeSeriesAnalysis:
    """Time series analysis and processing functions"""

    def __init__(self):
        pass

    def calculate_climatology(self, data, years, months, reference_period=None):
        """
        Calculate climatological means for gap-filling.
        Used in GFED processing for missing years.

        Args:
            data: Time series data (years × months)
            years: Array of years
            months: Array of months
            reference_period: Tuple of (start_year, end_year) for climatology

        Returns:
            Climatological means by month
        """

        if reference_period is None:
            reference_period = (years[0], years[-1])

        start_year, end_year = reference_period

        # Select reference period data
        ref_mask = (years >= start_year) & (years <= end_year)
        ref_data = data[ref_mask]
        ref_years = years[ref_mask]

        # Calculate monthly climatology
        climatology = np.zeros(12)
        for month in range(1, 13):
            month_mask = months[ref_mask] == month
            if np.any(month_mask):
                climatology[month-1] = np.nanmean(ref_data[month_mask])

        return climatology

    def fill_missing_values(self, data, method='climatology', **kwargs):
        """
        Fill missing values in time series.

        Args:
            data: Data array with missing values (NaN)
            method: Gap-filling method ('climatology', 'interpolation', 'regression')
            **kwargs: Additional parameters for specific methods

        Returns:
            Data with filled values
        """

        if method == 'climatology':
            return self._fill_with_climatology(data, **kwargs)
        elif method == 'interpolation':
            return self._fill_with_interpolation(data, **kwargs)
        elif method == 'regression':
            return self._fill_with_regression(data, **kwargs)
        else:
            raise ValueError(f"Unknown gap-filling method: {method}")

    def _fill_with_climatology(self, data, climatology=None):
        """Fill missing values using climatological means"""

        if climatology is None:
            # Calculate climatology from available data
            climatology = np.nanmean(data, axis=0)

        filled_data = data.copy()
        missing_mask = np.isnan(data)

        # Fill missing values with climatology
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if missing_mask[i, j]:
                    filled_data[i, j] = climatology[j]

        return filled_data

    def calculate_trends(self, data, years, method='linear'):
        """
        Calculate temporal trends in data.

        Args:
            data: Time series data
            years: Array of years
            method: Trend calculation method ('linear', 'theil_sen')

        Returns:
            Trend slopes and statistics
        """

        from scipy import stats

        if method == 'linear':
            slope, intercept, r_value, p_value, std_err = stats.linregress(years, data)
            return {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2,
                'p_value': p_value,
                'std_error': std_err
            }

        elif method == 'theil_sen':
            slope, intercept, low_slope, high_slope = stats.theilslopes(data, years)
            return {
                'slope': slope,
                'intercept': intercept,
                'slope_low': low_slope,
                'slope_high': high_slope
            }

        else:
            raise ValueError(f"Unknown trend method: {method}")

    def detrend_data(self, data, years, method='linear'):
        """
        Remove trends from time series data.

        Args:
            data: Time series data
            years: Array of years
            method: Detrending method ('linear', 'polynomial')

        Returns:
            Detrended data
        """

        if method == 'linear':
            # Linear detrending
            trend_stats = self.calculate_trends(data, years, method='linear')
            trend_line = trend_stats['slope'] * years + trend_stats['intercept']
            return data - trend_line

        elif method == 'polynomial':
            # Polynomial detrending (order 2)
            coeffs = np.polyfit(years, data, deg=2)
            trend_line = np.polyval(coeffs, years)
            return data - trend_line

        else:
            raise ValueError(f"Unknown detrending method: {method}")
```

## 8.4 Unit Conversion and Constants (`units_constants.py`)

### Physical Constants
```python
class PhysicalConstants:
    """Physical constants for atmospheric and carbon cycle calculations"""

    # Universal constants
    R_universal = 8.314472  # J mol^-1 K^-1
    N_avogadro = 6.02214076e23  # mol^-1
    stefan_boltzmann = 5.670374419e-8  # W m^-2 K^-4

    # Atmospheric constants
    standard_pressure = 101325  # Pa
    standard_temperature = 273.15  # K
    dry_air_molar_mass = 28.9647e-3  # kg mol^-1
    water_vapor_molar_mass = 18.01528e-3  # kg mol^-1

    # Gas constants
    R_dry_air = 287.0  # J kg^-1 K^-1
    R_water_vapor = 461.5  # J kg^-1 K^-1

    # Carbon cycle constants
    carbon_molar_mass = 12.011e-3  # kg mol^-1
    co2_molar_mass = 44.01e-3  # kg mol^-1
    ch4_molar_mass = 16.04e-3  # kg mol^-1

    # Earth constants
    earth_radius = 6.371e6  # m
    earth_surface_area = 5.101e14  # m^2
    solar_constant = 1361  # W m^-2

    # Time constants
    seconds_per_day = 86400
    seconds_per_year = 365.25 * 86400
    days_per_year = 365.25

class UnitConverter:
    """Comprehensive unit conversion utilities"""

    def __init__(self):
        self.constants = PhysicalConstants()

    def convert_temperature(self, temp_data, input_units, output_units):
        """
        Convert temperature between different units.

        Args:
            temp_data: Temperature values
            input_units: Input units ('K', 'C', 'F')
            output_units: Output units ('K', 'C', 'F')

        Returns:
            Converted temperature values
        """

        # Convert to Kelvin first
        if input_units == 'K':
            temp_k = temp_data
        elif input_units == 'C':
            temp_k = temp_data + 273.15
        elif input_units == 'F':
            temp_k = (temp_data - 32) * 5/9 + 273.15
        else:
            raise ValueError(f"Unknown input temperature units: {input_units}")

        # Convert from Kelvin to output units
        if output_units == 'K':
            return temp_k
        elif output_units == 'C':
            return temp_k - 273.15
        elif output_units == 'F':
            return (temp_k - 273.15) * 9/5 + 32
        else:
            raise ValueError(f"Unknown output temperature units: {output_units}")

    def convert_pressure(self, pressure_data, input_units, output_units):
        """
        Convert pressure between different units.

        Args:
            pressure_data: Pressure values
            input_units: Input units ('Pa', 'hPa', 'kPa', 'bar', 'atm', 'mmHg')
            output_units: Output units (same options)

        Returns:
            Converted pressure values
        """

        # Conversion factors to Pa
        to_pa = {
            'Pa': 1.0,
            'hPa': 100.0,
            'kPa': 1000.0,
            'bar': 100000.0,
            'atm': 101325.0,
            'mmHg': 133.322387415
        }

        # Convert to Pa, then to output units
        pressure_pa = pressure_data * to_pa[input_units]
        converted_pressure = pressure_pa / to_pa[output_units]

        return converted_pressure

    def convert_radiation(self, radiation_data, input_units, output_units,
                         time_step_hours=1):
        """
        Convert radiation between different units and time bases.

        Args:
            radiation_data: Radiation values
            input_units: Input units (see options below)
            output_units: Output units (see options below)
            time_step_hours: Time step for energy/flux conversions

        Unit options:
        - 'W_m2': Watts per square meter
        - 'J_m2_s': Joules per square meter per second
        - 'J_m2_h': Joules per square meter per hour
        - 'J_m2_d': Joules per square meter per day
        - 'MJ_m2_d': Megajoules per square meter per day
        - 'cal_cm2_min': Calories per square centimeter per minute
        - 'ly_min': Langleys per minute
        """

        # Convert to standard units (W m^-2)
        to_w_m2 = {
            'W_m2': 1.0,
            'J_m2_s': 1.0,
            'J_m2_h': 1.0 / 3600,
            'J_m2_d': 1.0 / 86400,
            'MJ_m2_d': 1e6 / 86400,
            'cal_cm2_min': 4.184 * 1e4 / 60,  # cal to J, cm² to m², min to s
            'ly_min': 4.184 * 1e4 / 60  # langley = cal cm^-2
        }

        # Convert from standard units
        from_w_m2 = {
            'W_m2': 1.0,
            'J_m2_s': 1.0,
            'J_m2_h': 3600.0,
            'J_m2_d': 86400.0,
            'MJ_m2_d': 86400.0 / 1e6,
            'cal_cm2_min': 60 / (4.184 * 1e4),
            'ly_min': 60 / (4.184 * 1e4)
        }

        # Perform conversion
        standard_radiation = radiation_data * to_w_m2[input_units]
        converted_radiation = standard_radiation * from_w_m2[output_units]

        return converted_radiation

    def convert_precipitation(self, precip_data, input_units, output_units,
                            density_water=1000):
        """
        Convert precipitation between different units.

        Args:
            precip_data: Precipitation values
            input_units: Input units (see options below)
            output_units: Output units (see options below)
            density_water: Water density in kg m^-3

        Unit options:
        - 'mm_d': Millimeters per day
        - 'mm_h': Millimeters per hour
        - 'mm_s': Millimeters per second
        - 'm_s': Meters per second
        - 'kg_m2_s': Kilograms per square meter per second
        - 'kg_m2_d': Kilograms per square meter per day
        """

        # Convert to standard units (m s^-1)
        to_m_s = {
            'mm_d': 1e-3 / 86400,
            'mm_h': 1e-3 / 3600,
            'mm_s': 1e-3,
            'm_s': 1.0,
            'kg_m2_s': 1.0 / density_water,
            'kg_m2_d': 1.0 / (density_water * 86400)
        }

        # Convert from standard units
        from_m_s = {
            'mm_d': 1e3 * 86400,
            'mm_h': 1e3 * 3600,
            'mm_s': 1e3,
            'm_s': 1.0,
            'kg_m2_s': density_water,
            'kg_m2_d': density_water * 86400
        }

        # Perform conversion
        standard_precip = precip_data * to_m_s[input_units]
        converted_precip = standard_precip * from_m_s[output_units]

        return converted_precip
```

## 8.5 Data Quality and Validation (`quality_control.py`)

### Data Range Validation
```python
class DataQualityControl:
    """Data quality control and validation functions"""

    def __init__(self):
        self.physical_ranges = self._setup_physical_ranges()
        self.statistical_thresholds = self._setup_statistical_thresholds()

    def _setup_physical_ranges(self):
        """Setup physically reasonable ranges for different variables"""
        return {
            'temperature_celsius': {'min': -100, 'max': 60},
            'temperature_kelvin': {'min': 173, 'max': 333},
            'precipitation_mm_day': {'min': 0, 'max': 1000},
            'radiation_w_m2': {'min': 0, 'max': 1600},
            'pressure_hpa': {'min': 500, 'max': 1100},
            'humidity_percent': {'min': 0, 'max': 100},
            'vpd_hpa': {'min': 0, 'max': 100},
            'co2_ppm': {'min': 300, 'max': 500},
            'wind_speed_m_s': {'min': 0, 'max': 100},
            'carbon_flux_gc_m2_day': {'min': -50, 'max': 50}
        }

    def _setup_statistical_thresholds(self):
        """Setup statistical thresholds for outlier detection"""
        return {
            'zscore_threshold': 4.0,  # Z-score threshold for outliers
            'iqr_multiplier': 2.5,    # IQR multiplier for outlier detection
            'mad_threshold': 3.0      # Median absolute deviation threshold
        }

    def validate_physical_ranges(self, data, variable_type, units=None):
        """
        Validate data against physical ranges.

        Args:
            data: Data array to validate
            variable_type: Type of variable (see physical_ranges keys)
            units: Units of the data (for unit-specific ranges)

        Returns:
            Dictionary with validation results
        """

        # Adjust variable type for units if needed
        if units and f"{variable_type}_{units}" in self.physical_ranges:
            range_key = f"{variable_type}_{units}"
        else:
            range_key = variable_type

        if range_key not in self.physical_ranges:
            return {
                'status': 'unknown_variable',
                'message': f"Unknown variable type: {variable_type}"
            }

        ranges = self.physical_ranges[range_key]
        min_val, max_val = ranges['min'], ranges['max']

        # Check for values outside physical ranges
        valid_data = data[~np.isnan(data)]
        below_min = np.sum(valid_data < min_val)
        above_max = np.sum(valid_data > max_val)
        total_valid = len(valid_data)

        return {
            'status': 'pass' if (below_min == 0 and above_max == 0) else 'fail',
            'total_points': len(data.flatten()),
            'valid_points': total_valid,
            'nan_points': len(data.flatten()) - total_valid,
            'below_minimum': below_min,
            'above_maximum': above_max,
            'data_range': {
                'min': float(np.nanmin(data)),
                'max': float(np.nanmax(data)),
                'mean': float(np.nanmean(data)),
                'std': float(np.nanstd(data))
            },
            'physical_range': ranges
        }

    def detect_outliers(self, data, method='zscore', threshold=None):
        """
        Detect outliers in data using various statistical methods.

        Args:
            data: Data array
            method: Detection method ('zscore', 'iqr', 'mad')
            threshold: Custom threshold (uses default if None)

        Returns:
            Boolean array indicating outliers
        """

        if method == 'zscore':
            return self._detect_zscore_outliers(data, threshold)
        elif method == 'iqr':
            return self._detect_iqr_outliers(data, threshold)
        elif method == 'mad':
            return self._detect_mad_outliers(data, threshold)
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

    def _detect_zscore_outliers(self, data, threshold=None):
        """Detect outliers using Z-score method"""
        if threshold is None:
            threshold = self.statistical_thresholds['zscore_threshold']

        valid_data = data[~np.isnan(data)]
        mean_val = np.mean(valid_data)
        std_val = np.std(valid_data)

        if std_val == 0:
            return np.zeros_like(data, dtype=bool)

        z_scores = np.abs((data - mean_val) / std_val)
        return z_scores > threshold

    def _detect_iqr_outliers(self, data, threshold=None):
        """Detect outliers using Interquartile Range method"""
        if threshold is None:
            threshold = self.statistical_thresholds['iqr_multiplier']

        valid_data = data[~np.isnan(data)]
        q1 = np.percentile(valid_data, 25)
        q3 = np.percentile(valid_data, 75)
        iqr = q3 - q1

        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        return (data < lower_bound) | (data > upper_bound)

    def _detect_mad_outliers(self, data, threshold=None):
        """Detect outliers using Median Absolute Deviation method"""
        if threshold is None:
            threshold = self.statistical_thresholds['mad_threshold']

        valid_data = data[~np.isnan(data)]
        median_val = np.median(valid_data)
        mad = np.median(np.abs(valid_data - median_val))

        if mad == 0:
            return np.zeros_like(data, dtype=bool)

        modified_z_scores = 0.6745 * (data - median_val) / mad
        return np.abs(modified_z_scores) > threshold

    def calculate_data_completeness(self, data, spatial_coverage_threshold=0.8,
                                  temporal_coverage_threshold=0.9):
        """
        Calculate data completeness statistics.

        Args:
            data: Data array (can be 2D or 3D)
            spatial_coverage_threshold: Minimum fraction of spatial points required
            temporal_coverage_threshold: Minimum fraction of time points required

        Returns:
            Dictionary with completeness statistics
        """

        total_points = data.size
        valid_points = np.sum(~np.isnan(data))
        overall_completeness = valid_points / total_points

        completeness_stats = {
            'overall_completeness': overall_completeness,
            'total_points': total_points,
            'valid_points': valid_points,
            'missing_points': total_points - valid_points
        }

        # Spatial completeness (if 3D data)
        if data.ndim == 3:
            spatial_completeness = np.sum(~np.isnan(data), axis=2) / data.shape[2]
            completeness_stats['spatial_completeness'] = {
                'mean': float(np.mean(spatial_completeness)),
                'min': float(np.min(spatial_completeness)),
                'max': float(np.max(spatial_completeness)),
                'below_threshold': np.sum(spatial_completeness < spatial_coverage_threshold)
            }

            # Temporal completeness
            temporal_completeness = np.sum(~np.isnan(data), axis=(0, 1)) / (data.shape[0] * data.shape[1])
            completeness_stats['temporal_completeness'] = {
                'mean': float(np.mean(temporal_completeness)),
                'min': float(np.min(temporal_completeness)),
                'max': float(np.max(temporal_completeness)),
                'below_threshold': np.sum(temporal_completeness < temporal_coverage_threshold)
            }

        return completeness_stats
```

## 8.6 Testing Framework

### Scientific Function Tests
```
tests/scientific_utils/
├── test_atmospheric_science.py
├── test_carbon_cycle.py
├── test_statistics_utils.py
├── test_units_constants.py
├── test_quality_control.py
└── fixtures/
    ├── test_data/
    ├── validation_outputs/
    └── matlab_references/
```

### Validation Against Known Results
```python
def test_vpd_calculation_accuracy():
    """Test VPD calculation against known meteorological values"""

def test_carbon_flux_unit_conversions():
    """Test carbon flux unit conversions maintain mass balance"""

def test_fire_emission_factors():
    """Test fire emission calculations against GFED documentation"""

def test_interpolation_accuracy():
    """Test spatial interpolation methods against analytical solutions"""
```

## 8.7 Success Criteria

### Accuracy Requirements
- [ ] Replicate MATLAB scientific function results within 0.1% tolerance
- [ ] Pass validation against published meteorological formulas
- [ ] Maintain physical conservation laws (mass, energy balance)
- [ ] Handle edge cases and extreme values appropriately

### Performance Requirements
- [ ] Efficient vectorized operations for large arrays
- [ ] Memory-efficient processing of global datasets
- [ ] Reasonable computation times for complex calculations
- [ ] Support for both scalar and array inputs

### Robustness Requirements
- [ ] Comprehensive input validation and error handling
- [ ] Graceful handling of missing data and NaN values
- [ ] Clear error messages and debugging information
- [ ] Extensive unit test coverage (>95%)

### Integration Requirements
- [ ] Seamless integration with all other phases
- [ ] Consistent interfaces and data structures
- [ ] Compatible with existing NumPy/SciPy ecosystem
- [ ] Support for different coordinate systems and projections