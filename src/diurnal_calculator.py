"""
Diurnal Pattern Calculation

Calculate diurnal (hourly) flux patterns from monthly means.
Implements the core downscaling algorithms from MATLAB script.

This module contains the scientific algorithms to downscale monthly
carbon fluxes to hourly resolution using meteorological drivers and
fire timing patterns.

Scientific Context:
Diurnal downscaling preserves monthly totals while adding realistic
within-day variability based on environmental drivers:
- GPP follows solar radiation patterns
- Respiration follows temperature patterns
- Fire emissions follow GFED diurnal timing
"""

import numpy as np
import logging
from typing import Dict, Optional

from scientific_utils import validate_carbon_flux_data


class DiurnalCalculator:
    """
    Calculate diurnal (hourly) flux patterns from monthly means.
    Implements the core downscaling algorithms from MATLAB script.

    MATLAB Reference: Core flux calculation sections in MATLAB script
    """

    def __init__(self, q10_factor: float = 1.4):
        """
        Initialize diurnal calculator.

        Args:
            q10_factor: Q10 temperature sensitivity factor (optional enhancement)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.q10_factor = q10_factor

        # Unit conversion constants
        self.unit_conversion_factor = 1e3 / 24 / 3600  # gC/m²/day to Kg C/Km²/sec

    def calculate_diurnal_fluxes(self,
                               monthly_fluxes: Dict[str, np.ndarray],
                               month_index: int,
                               ssrd: np.ndarray,
                               skt: np.ndarray,
                               co2_diurnal: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate hourly flux patterns from monthly means and drivers.

        MATLAB Reference: Main diurnal flux calculation section in MATLAB script

        Args:
            monthly_fluxes: Dictionary with monthly flux data
            month_index: Time index for current month
            ssrd: Solar radiation diurnal patterns (J/m² or W/m²)
            skt: Skin temperature diurnal patterns (K)
            co2_diurnal: Fire diurnal patterns from GFED (3-hourly)

        Returns:
            dict: Dictionary with hourly flux patterns
                Keys: 'GPP', 'REC', 'FIR', 'NBE', 'NEE'
        """
        self.logger.info("Calculating diurnal flux patterns")

        # Extract monthly means for this time period - MATLAB: monthly flux extraction
        gpp_monthly = monthly_fluxes['GPP'][:, :, month_index]
        rec_monthly = monthly_fluxes['REC'][:, :, month_index]
        fir_monthly = monthly_fluxes['FIR'][:, :, month_index]

        self.logger.debug(f"Monthly flux shapes - GPP: {gpp_monthly.shape}, REC: {rec_monthly.shape}, FIR: {fir_monthly.shape}")

        # Calculate individual flux diurnal patterns
        gpp_diurnal = self._calculate_gpp_diurnal(gpp_monthly, ssrd)
        rec_diurnal = self._calculate_rec_diurnal(rec_monthly, skt)
        fir_diurnal = self._calculate_fir_diurnal(fir_monthly, co2_diurnal)

        # Calculate composite fluxes - MATLAB: NEE and NBE calculations
        nee_diurnal = rec_diurnal - gpp_diurnal                    # NEE = REC - GPP
        nbe_diurnal = rec_diurnal - gpp_diurnal + fir_diurnal      # NBE = REC - GPP + FIR

        diurnal_fluxes = {
            'GPP': gpp_diurnal,
            'REC': rec_diurnal,
            'FIR': fir_diurnal,
            'NBE': nbe_diurnal,
            'NEE': nee_diurnal
        }

        # Validate results
        self._validate_diurnal_fluxes(diurnal_fluxes, monthly_fluxes, month_index)

        self.logger.info("Diurnal flux calculation completed successfully")
        return diurnal_fluxes

    def _calculate_gpp_diurnal(self, gpp_monthly: np.ndarray, ssrd: np.ndarray) -> np.ndarray:
        """
        Calculate GPP diurnal pattern using solar radiation.
        Equivalent to MATLAB: SSRD.*repmat(GPP_monthly./mean(SSRD,3), [1,1,size(SSRD,3)])

        MATLAB Reference: GPP diurnal calculation using solar radiation scaling

        Args:
            gpp_monthly: Monthly GPP values (gC/m²/day)
            ssrd: Solar radiation diurnal patterns

        Returns:
            ndarray: Hourly GPP patterns (Kg C/Km²/sec)
        """
        self.logger.debug("Calculating GPP diurnal patterns using solar radiation")

        # Calculate mean solar radiation over time dimension - MATLAB: mean(SSRD,3)
        ssrd_mean = np.mean(ssrd, axis=2, keepdims=True)

        # Avoid division by zero - set zero means to NaN
        ssrd_mean[ssrd_mean == 0] = np.nan

        # Scale solar radiation by monthly GPP ratio - MATLAB: GPP_monthly./mean(SSRD,3)
        gpp_scaling = gpp_monthly[:, :, np.newaxis] / ssrd_mean

        # Apply diurnal pattern - MATLAB: SSRD.*repmat(scaling, [1,1,size(SSRD,3)])
        gpp_diurnal = ssrd * gpp_scaling

        # Handle NaN values (set to zero for no solar radiation periods)
        gpp_diurnal = self._nan_to_zero(gpp_diurnal)

        # Convert units: gC/m²/day to Kg C/Km²/sec
        gpp_diurnal_converted = gpp_diurnal * self.unit_conversion_factor

        self.logger.debug(f"GPP diurnal range: [{np.nanmin(gpp_diurnal_converted):.6f}, {np.nanmax(gpp_diurnal_converted):.6f}] Kg C/Km²/sec")

        return gpp_diurnal_converted

    def _calculate_rec_diurnal(self, rec_monthly: np.ndarray, skt: np.ndarray) -> np.ndarray:
        """
        Calculate respiration diurnal pattern using temperature.
        Equivalent to MATLAB: SKT.*repmat(REC_monthly./mean(SKT,3), [1,1,size(SKT,3)])

        MATLAB Reference: REC diurnal calculation using temperature scaling

        Args:
            rec_monthly: Monthly respiration values (gC/m²/day)
            skt: Skin temperature diurnal patterns (K)

        Returns:
            ndarray: Hourly respiration patterns (Kg C/Km²/sec)
        """
        self.logger.debug("Calculating respiration diurnal patterns using temperature")

        # Calculate mean temperature over time dimension - MATLAB: mean(SKT,3)
        skt_mean = np.mean(skt, axis=2, keepdims=True)

        # Avoid division by zero
        skt_mean[skt_mean == 0] = np.nan

        # Scale temperature by monthly REC ratio - MATLAB: REC_monthly./mean(SKT,3)
        rec_scaling = rec_monthly[:, :, np.newaxis] / skt_mean

        # Apply diurnal pattern - MATLAB: SKT.*repmat(scaling, [1,1,size(SKT,3)])
        rec_diurnal = skt * rec_scaling

        # Handle NaN values
        rec_diurnal = self._nan_to_zero(rec_diurnal)

        # Convert units: gC/m²/day to Kg C/Km²/sec
        rec_diurnal_converted = rec_diurnal * self.unit_conversion_factor

        self.logger.debug(f"REC diurnal range: [{np.nanmin(rec_diurnal_converted):.6f}, {np.nanmax(rec_diurnal_converted):.6f}] Kg C/Km²/sec")

        return rec_diurnal_converted

    def _calculate_fir_diurnal(self, fir_monthly: np.ndarray, co2_diurnal: np.ndarray) -> np.ndarray:
        """
        Calculate fire diurnal pattern using GFED fire timing.
        Handles 3-hourly to hourly conversion.

        MATLAB Reference: Fire diurnal calculation using GFED timing patterns

        Args:
            fir_monthly: Monthly fire emissions (gC/m²/day)
            co2_diurnal: GFED diurnal fire patterns (3-hourly)

        Returns:
            ndarray: Hourly fire emission patterns (Kg C/Km²/sec)
        """
        self.logger.debug("Calculating fire diurnal patterns using GFED timing")

        # Calculate mean fire pattern over time - MATLAB: mean(CO2_diurnal,3)
        co2_mean = np.mean(co2_diurnal, axis=2, keepdims=True)

        # Avoid division by zero
        co2_mean[co2_mean == 0] = np.nan

        # Scale fire timing by monthly fire emissions - MATLAB: FIR_monthly./mean(CO2_diurnal,3)
        fir_scaling = fir_monthly[:, :, np.newaxis] / co2_mean

        # Apply diurnal pattern to 3-hourly data
        fir_diurnal_3h = self._nan_to_zero(co2_diurnal * fir_scaling)

        # Convert 3-hourly to hourly by triplication - MATLAB: FIRdiurnal expansion
        fir_diurnal = self._triplicate_3hourly_to_hourly(fir_diurnal_3h)

        # Convert units: gC/m²/day to Kg C/Km²/sec
        fir_diurnal_converted = fir_diurnal * self.unit_conversion_factor

        self.logger.debug(f"FIR diurnal range: [{np.nanmin(fir_diurnal_converted):.6f}, {np.nanmax(fir_diurnal_converted):.6f}] Kg C/Km²/sec")

        return fir_diurnal_converted

    def _triplicate_3hourly_to_hourly(self, data_3h: np.ndarray) -> np.ndarray:
        """
        Convert 3-hourly fire data to hourly by triplication.
        Equivalent to MATLAB logic for FIRdiurnal expansion.

        MATLAB Reference: 3-hourly to hourly conversion logic in MATLAB script

        Args:
            data_3h: 3-hourly data array

        Returns:
            ndarray: Hourly data array (triplicated)
        """
        n_times_3h = data_3h.shape[2]
        n_times_1h = n_times_3h * 3

        # Create hourly array
        data_1h = np.zeros((*data_3h.shape[:2], n_times_1h))

        # Triplicate each 3-hourly timestep - MATLAB: expansion to hourly
        for i in range(3):
            data_1h[:, :, i::3] = data_3h

        self.logger.debug(f"Converted 3-hourly data {data_3h.shape} to hourly {data_1h.shape}")

        return data_1h

    def _nan_to_zero(self, data: np.ndarray) -> np.ndarray:
        """
        Convert NaN values to zero, equivalent to MATLAB nan2zero.

        MATLAB Reference: nan2zero function usage in MATLAB script

        Args:
            data: Input array potentially containing NaN values

        Returns:
            ndarray: Array with NaN values replaced by zeros
        """
        data_copy = data.copy()
        data_copy[np.isnan(data_copy)] = 0
        return data_copy

    def _validate_diurnal_fluxes(self,
                               diurnal_fluxes: Dict[str, np.ndarray],
                               monthly_fluxes: Dict[str, np.ndarray],
                               month_index: int) -> None:
        """
        Validate that diurnal fluxes preserve monthly totals and are physically reasonable.

        Args:
            diurnal_fluxes: Calculated hourly fluxes
            monthly_fluxes: Original monthly fluxes
            month_index: Month index for comparison
        """
        self.logger.debug("Validating diurnal flux conservation and physical reasonableness")

        for flux_type in ['GPP', 'REC', 'FIR']:
            if flux_type in diurnal_fluxes and flux_type in monthly_fluxes:
                # Check mass conservation (hourly sum should equal monthly total)
                hourly_data = diurnal_fluxes[flux_type]
                monthly_data = monthly_fluxes[flux_type][:, :, month_index]

                # Convert hourly back to daily totals for comparison
                hours_per_month = hourly_data.shape[2]
                daily_equivalent = np.sum(hourly_data, axis=2) * 24 * 3600 / 1e3  # Convert back to gC/m²/day

                # Calculate relative error
                relative_error = np.abs(daily_equivalent - monthly_data) / (np.abs(monthly_data) + 1e-10)
                max_error = np.nanmax(relative_error)

                if max_error > 0.01:  # More than 1% error
                    self.logger.warning(f"{flux_type} mass conservation error: {max_error:.3f}")
                else:
                    self.logger.debug(f"{flux_type} mass conservation check passed: max error {max_error:.3f}")

                # Physical range validation
                try:
                    validate_carbon_flux_data(hourly_data.flatten(), flux_type.lower())
                except Exception as e:
                    self.logger.warning(f"{flux_type} physical validation warning: {e}")

    def calculate_enhanced_temperature_response(self,
                                              rec_monthly: np.ndarray,
                                              skt: np.ndarray,
                                              reference_temperature: float = 288.15) -> np.ndarray:
        """
        Calculate respiration with enhanced Q10 temperature response.
        Optional enhancement beyond basic linear scaling.

        Args:
            rec_monthly: Monthly respiration values
            skt: Skin temperature patterns
            reference_temperature: Reference temperature for Q10 scaling (K)

        Returns:
            ndarray: Enhanced respiration patterns
        """
        self.logger.debug("Calculating enhanced Q10 temperature response for respiration")

        # Basic linear scaling
        rec_basic = self._calculate_rec_diurnal(rec_monthly, skt)

        # Q10 enhancement: Q10^((T-Tref)/10)
        q10_factor = self.q10_factor ** ((skt - reference_temperature) / 10.0)
        q10_mean = np.mean(q10_factor, axis=2, keepdims=True)

        # Apply Q10 correction to basic scaling
        q10_correction = q10_factor / q10_mean
        rec_enhanced = rec_basic * q10_correction

        self.logger.debug("Enhanced Q10 temperature response calculation completed")
        return rec_enhanced

    def calculate_photosynthetically_active_radiation(self, ssrd: np.ndarray) -> np.ndarray:
        """
        Convert solar radiation to photosynthetically active radiation (PAR).

        Args:
            ssrd: Solar radiation (W/m² or J/m²)

        Returns:
            ndarray: PAR values (μmol photons/m²/s)
        """
        # Conversion factor: ~2.3 μmol photons per J for PAR wavelengths
        par_conversion_factor = 2.3

        # Assume PAR is ~45% of total solar radiation
        par_fraction = 0.45

        # Convert to PAR
        par_values = ssrd * par_fraction * par_conversion_factor

        self.logger.debug("Converted solar radiation to PAR")
        return par_values