#!/usr/bin/env python3
"""
DPS Wrapper for CARDAMOM ECMWF Downloader
Handles MAAP-specific parameter parsing and execution
"""

import os
import sys
import json
import logging
import subprocess
from typing import Dict, Any, List
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/tmp/algorithm.log')
    ]
)

logger = logging.getLogger(__name__)


class MAAPECMWFWrapper:
    """MAAP DPS wrapper for ECMWF downloader"""
    
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.root_dir = self.script_dir.parent
        self.run_script = self.script_dir / "run.sh"
        
    def parse_maap_inputs(self) -> Dict[str, Any]:
        """Parse MAAP DPS input parameters from environment variables"""
        
        # MAAP DPS provides inputs via environment variables
        inputs = {
            'download_mode': os.getenv('download_mode', 'cardamom-monthly'),
            'output_dir': os.getenv('output_dir', './output'),
            'years': os.getenv('years', '2020-2021'),
            'months': os.getenv('months', '1-12'),
            'variables': os.getenv('variables', '2m_temperature,total_precipitation'),
            'area': os.getenv('area', ''),
            'grid': os.getenv('grid', '0.5/0.5'),
            'format': os.getenv('format', 'netcdf')
        }
        
        logger.info(f"Parsed MAAP inputs: {inputs}")
        return inputs
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Validate input parameters"""
        
        valid_modes = ['hourly', 'monthly', 'cardamom-hourly', 'cardamom-monthly']
        if inputs['download_mode'] not in valid_modes:
            logger.error(f"Invalid download mode: {inputs['download_mode']}. Must be one of: {valid_modes}")
            return False
        
        valid_formats = ['netcdf', 'grib']
        if inputs['format'] not in valid_formats:
            logger.error(f"Invalid format: {inputs['format']}. Must be one of: {valid_formats}")
            return False
        
        # Validate years format
        years = inputs['years']
        if '-' in years:
            try:
                start, end = map(int, years.split('-'))
                if start > end or start < 1979 or end > 2024:
                    logger.error(f"Invalid years range: {years}")
                    return False
            except ValueError:
                logger.error(f"Invalid years format: {years}")
                return False
        else:
            try:
                year = int(years)
                if year < 1979 or year > 2024:
                    logger.error(f"Invalid year: {year}")
                    return False
            except ValueError:
                logger.error(f"Invalid year format: {years}")
                return False
        
        return True
    
    def setup_cds_credentials(self):
        """Set up ECMWF CDS credentials from MAAP secrets"""
        
        # In MAAP, credentials should be provided as environment variables
        cds_uid = os.getenv('ECMWF_CDS_UID')
        cds_key = os.getenv('ECMWF_CDS_KEY')
        
        if not cds_uid or not cds_key:
            logger.warning("ECMWF CDS credentials not found in environment variables")
            logger.info("Please ensure ECMWF_CDS_UID and ECMWF_CDS_KEY are set")
            return
        
        # Create .cdsapirc file
        cdsapi_rc = Path.home() / '.cdsapirc'
        with open(cdsapi_rc, 'w') as f:
            f.write(f"url: https://cds.climate.copernicus.eu/api/v2\n")
            f.write(f"key: {cds_uid}:{cds_key}\n")
        
        logger.info("ECMWF CDS credentials configured successfully")
    
    def run_algorithm(self, inputs: Dict[str, Any]) -> int:
        """Execute the ECMWF downloader algorithm"""
        
        # Prepare command arguments
        cmd = [
            str(self.run_script),
            inputs['download_mode'],
            inputs['output_dir'],
            inputs['years'],
            inputs['months'],
            inputs['variables'],
            inputs['area'],
            inputs['grid'],
            inputs['format']
        ]
        
        logger.info(f"Executing command: {' '.join(cmd)}")
        
        try:
            # Execute the run script
            result = subprocess.run(
                cmd,
                cwd=str(self.root_dir),
                check=True,
                capture_output=True,
                text=True
            )
            
            logger.info("Algorithm execution completed successfully")
            logger.info(f"stdout: {result.stdout}")
            
            return 0
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Algorithm execution failed with return code {e.returncode}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            return e.returncode
        except Exception as e:
            logger.error(f"Unexpected error during algorithm execution: {e}")
            return 1
    
    def create_output_manifest(self, output_dir: str):
        """Create a manifest of output files for MAAP"""
        
        output_path = Path(output_dir)
        if not output_path.exists():
            logger.warning(f"Output directory {output_dir} does not exist")
            return
        
        # Find all NetCDF files
        nc_files = list(output_path.glob("*.nc"))
        
        manifest = {
            "algorithm": "cardamom-ecmwf-downloader",
            "timestamp": str(pd.Timestamp.now()),
            "output_files": [str(f.relative_to(output_path)) for f in nc_files],
            "file_count": len(nc_files),
            "total_size_mb": sum(f.stat().st_size for f in nc_files) / (1024 * 1024)
        }
        
        manifest_file = output_path / "output_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Created output manifest: {manifest_file}")
        logger.info(f"Generated {manifest['file_count']} files, total size: {manifest['total_size_mb']:.2f} MB")


def main():
    """Main DPS wrapper function"""
    
    logger.info("Starting CARDAMOM ECMWF Downloader MAAP algorithm")
    
    wrapper = MAAPECMWFWrapper()
    
    try:
        # Parse MAAP inputs
        inputs = wrapper.parse_maap_inputs()
        
        # Validate inputs
        if not wrapper.validate_inputs(inputs):
            logger.error("Input validation failed")
            return 1
        
        # Set up ECMWF credentials
        wrapper.setup_cds_credentials()
        
        # Run the algorithm
        result = wrapper.run_algorithm(inputs)
        
        if result == 0:
            # Create output manifest
            wrapper.create_output_manifest(inputs['output_dir'])
            logger.info("Algorithm completed successfully")
        else:
            logger.error("Algorithm execution failed")
        
        return result
        
    except Exception as e:
        logger.error(f"Unexpected error in DPS wrapper: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())