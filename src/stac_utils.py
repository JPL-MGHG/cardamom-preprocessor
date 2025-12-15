"""
STAC (SpatioTemporal Asset Catalog) Utilities for CARDAMOM Preprocessor

This module provides utilities for creating and managing STAC Items and Collections
that describe the preprocessed meteorological datasets produced by CARDAMOM downloaders.

STAC enables the CARDAMOM ecosystem modeling system to discover, query, and retrieve
analysis-ready meteorological data through standardized metadata.

Scientific Context:
STAC catalogs organize satellite and remote sensing data with standardized metadata,
allowing downstream applications (like CARDAMOM's CBF generator) to automatically
discover available data files and their properties without manual file management.

References:
- STAC Specification: https://stacspec.org/
- pystac documentation: https://pystac.readthedocs.io/
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import pystac
from pystac import Item, Collection, Extent, SpatialExtent, TemporalExtent, Link, Asset, Catalog

logger = logging.getLogger(__name__)


# ========== STAC Collection Definitions ==========

# Define metadata for each CARDAMOM variable that gets its own STAC collection
CARDAMOM_STAC_COLLECTIONS: Dict[str, Dict[str, Any]] = {
    'cardamom-t2m-min': {
        'description': 'Monthly minimum 2-meter temperature from ERA5 reanalysis',
        'cbf_variable': 'T2M_MIN',
        'units': 'K',
        'data_source': 'era5',
        'processing_level': 'analysis-ready',
        'keywords': ['temperature', 'era5', 'meteorology', 'monthly'],
    },
    'cardamom-t2m-max': {
        'description': 'Monthly maximum 2-meter temperature from ERA5 reanalysis',
        'cbf_variable': 'T2M_MAX',
        'units': 'K',
        'data_source': 'era5',
        'processing_level': 'analysis-ready',
        'keywords': ['temperature', 'era5', 'meteorology', 'monthly'],
    },
    'cardamom-vpd': {
        'description': 'Vapor Pressure Deficit calculated from ERA5 temperature and dewpoint',
        'cbf_variable': 'VPD',
        'units': 'hPa',
        'data_source': 'era5',
        'processing_level': 'analysis-ready',
        'keywords': ['humidity', 'vapor-pressure', 'era5', 'meteorology', 'monthly'],
    },
    'cardamom-total-prec': {
        'description': 'Monthly total precipitation from ERA5 reanalysis',
        'cbf_variable': 'TOTAL_PREC',
        'units': 'mm',
        'data_source': 'era5',
        'processing_level': 'analysis-ready',
        'keywords': ['precipitation', 'era5', 'meteorology', 'monthly'],
    },
    'cardamom-ssrd': {
        'description': 'Surface solar radiation downwards from ERA5 reanalysis',
        'cbf_variable': 'SSRD',
        'units': 'W m-2',
        'data_source': 'era5',
        'processing_level': 'analysis-ready',
        'keywords': ['radiation', 'solar', 'era5', 'meteorology', 'monthly'],
    },
    'cardamom-strd': {
        'description': 'Surface thermal radiation downwards from ERA5 reanalysis',
        'cbf_variable': 'STRD',
        'units': 'W m-2',
        'data_source': 'era5',
        'processing_level': 'analysis-ready',
        'keywords': ['radiation', 'thermal', 'era5', 'meteorology', 'monthly'],
    },
    'cardamom-skt': {
        'description': 'Skin temperature from ERA5 reanalysis',
        'cbf_variable': 'SKT',
        'units': 'K',
        'data_source': 'era5',
        'processing_level': 'analysis-ready',
        'keywords': ['temperature', 'skin', 'era5', 'meteorology', 'monthly'],
    },
    'cardamom-snowfall': {
        'description': 'Monthly snowfall from ERA5 reanalysis',
        'cbf_variable': 'SNOWFALL',
        'units': 'mm',
        'data_source': 'era5',
        'processing_level': 'analysis-ready',
        'keywords': ['snow', 'precipitation', 'era5', 'meteorology', 'monthly'],
    },
    'cardamom-co2': {
        'description': 'Atmospheric CO2 concentration from NOAA Global Monitoring Laboratory',
        'cbf_variable': 'CO2',
        'units': 'ppm',
        'data_source': 'noaa',
        'processing_level': 'analysis-ready',
        'keywords': ['co2', 'greenhouse-gas', 'noaa', 'atmospheric'],
    },
    'cardamom-burned-area': {
        'description': 'Monthly burned area fraction from GFED4.1s fire emissions database',
        'cbf_variable': 'BURNED_AREA',
        'units': 'fraction',
        'data_source': 'gfed',
        'processing_level': 'analysis-ready',
        'keywords': ['fire', 'burned-area', 'gfed', 'disturbance'],
    },
}

# Global spatial extent for CARDAMOM
GLOBAL_SPATIAL_EXTENT = SpatialExtent(
    bboxes=[[-180, -90, 180, 90]]  # [min_lon, min_lat, max_lon, max_lat]
)


# ========== STAC Item and Collection Creation Functions ==========

def create_stac_collection(
    collection_id: str,
    description: str,
    keywords: List[str],
    temporal_start: datetime,
    temporal_end: Optional[datetime] = None,
    extra_properties: Optional[Dict[str, Any]] = None,
) -> Collection:
    """
    Create a STAC Collection for a CARDAMOM variable.

    A STAC Collection groups related data items (e.g., monthly temperature files)
    with common metadata and describes the spatial and temporal coverage.

    Scientific Context:
    This collection represents all instances of a particular meteorological variable
    (e.g., monthly minimum temperature) that have been processed for CARDAMOM.

    Args:
        collection_id (str): Unique collection identifier (e.g., 'cardamom-t2m-min')
        description (str): Human-readable description of the variable
        keywords (List[str]): Keywords for searching and categorizing the collection
        temporal_start (datetime): Start of temporal coverage
        temporal_end (Optional[datetime]): End of temporal coverage. If None, assumed ongoing
        extra_properties (Optional[Dict]): Additional custom properties to add

    Returns:
        pystac.Collection: STAC Collection object ready for serialization

    Example:
        >>> from datetime import datetime
        >>> collection = create_stac_collection(
        ...     collection_id='cardamom-t2m-min',
        ...     description='Monthly minimum 2-meter temperature',
        ...     keywords=['temperature', 'era5'],
        ...     temporal_start=datetime(2001, 1, 1),
        ...     temporal_end=datetime(2024, 12, 31)
        ... )
        >>> print(collection.id)
        cardamom-t2m-min
    """

    # Create temporal extent (may be open-ended if ongoing)
    temporal_extent = TemporalExtent(
        intervals=[[temporal_start, temporal_end]]
    )

    # Create the collection object
    collection = Collection(
        id=collection_id,
        description=description,
        extent=Extent(
            spatial=GLOBAL_SPATIAL_EXTENT,
            temporal=temporal_extent
        ),
        keywords=keywords,
        license='proprietary',  # Update based on ERA5/NOAA/GFED licensing
        providers=[
            # Add provider information for data attribution
            pystac.Provider(
                name='ECMWF',
                description='European Centre for Medium-Range Weather Forecasts',
                roles=['producer', 'processor'] if 'era5' in keywords else [],
                url='https://www.ecmwf.int/'
            )
        ],
    )

    # Add custom CARDAMOM-specific properties
    if extra_properties:
        for key, value in extra_properties.items():
            collection.extra_fields[key] = value

    return collection


def create_stac_item(
    variable_name: str,
    year: int,
    month: int,
    data_file_path: str,
    collection_id: str,
    bbox: Optional[List[float]] = None,
    properties: Optional[Dict[str, Any]] = None,
) -> Item:
    """
    Create a STAC Item representing a single data file (e.g., one month of data).

    A STAC Item describes a single file of analysis-ready data with its location,
    temporal range, and metadata properties.

    Scientific Context:
    Each monthly meteorological file becomes a STAC Item, allowing the CBF generator
    to query and discover specific months of specific variables without scanning
    the filesystem.

    Args:
        variable_name (str): CARDAMOM variable name (e.g., 'T2M_MIN')
        year (int): Year of data (e.g., 2020)
        month (int): Month of data (1-12)
        data_file_path (str): Relative or absolute path to NetCDF file
        collection_id (str): ID of the parent STAC Collection
        bbox (Optional[List[float]]): Bounding box [min_lon, min_lat, max_lon, max_lat].
            Defaults to global bbox if not provided.
        properties (Optional[Dict]): Additional custom properties

    Returns:
        pystac.Item: STAC Item object ready for serialization

    Example:
        >>> from datetime import datetime
        >>> item = create_stac_item(
        ...     variable_name='T2M_MIN',
        ...     year=2020,
        ...     month=1,
        ...     data_file_path='./data/t2m_min_2020_01.nc',
        ...     collection_id='cardamom-t2m-min'
        ... )
        >>> print(f"{item.id}: {item.datetime}")
        t2m_min_2020_01: 2020-01-01 00:00:00
    """

    # Use global bounding box if not specified
    if bbox is None:
        bbox = [-180, -90, 180, 90]

    # Calculate datetime range for this month
    start_datetime = datetime(year, month, 1)

    # Calculate end of month (first day of next month, or Dec 31 if December)
    if month == 12:
        end_datetime = datetime(year + 1, 1, 1)
    else:
        end_datetime = datetime(year, month + 1, 1)

    # Create unique item ID
    item_id = f"{variable_name.lower()}_{year}_{month:02d}"

    # Create the STAC Item
    item = Item(
        id=item_id,
        geometry={
            'type': 'Polygon',
            'coordinates': [
                [
                    [bbox[0], bbox[1]],  # min_lon, min_lat
                    [bbox[2], bbox[1]],  # max_lon, min_lat
                    [bbox[2], bbox[3]],  # max_lon, max_lat
                    [bbox[0], bbox[3]],  # min_lon, max_lat
                    [bbox[0], bbox[1]],  # close polygon
                ]
            ]
        },
        bbox=bbox,
        datetime=start_datetime,
        properties={
            'start_datetime': start_datetime.isoformat() + 'Z',
            'end_datetime': end_datetime.isoformat() + 'Z',
        },
    )

    # Add standard CARDAMOM properties
    item.properties['cardamom:variable'] = variable_name
    item.properties['cardamom:collection'] = collection_id

    # Add custom properties if provided
    if properties:
        for key, value in properties.items():
            item.properties[key] = value

    # Add the data asset
    item.add_asset(
        key='data',
        asset=Asset(
            href=data_file_path,
            media_type='application/x-netcdf',
            title=f'{variable_name} {year}-{month:02d}',
            roles=['data'],
        )
    )

    return item


def write_stac_output(
    collection: Collection,
    items: List[Item],
    output_dir: str,
) -> None:
    """
    Write STAC Collection and Items to the filesystem as JSON files.

    This function organizes STAC metadata in a standard directory structure:
    ```
    output_dir/
    ├── collection.json
    └── items/
        ├── {item_id_1}.json
        ├── {item_id_2}.json
        └── ...
    ```

    Scientific Context:
    The STAC JSON files enable the CBF generator to discover and query available
    meteorological data without direct filesystem access, supporting distributed
    data processing systems like MAAP.

    Args:
        collection (pystac.Collection): The STAC Collection to write
        items (List[pystac.Item]): List of STAC Items to write
        output_dir (str): Directory path where STAC files will be written

    Returns:
        None

    Raises:
        IOError: If unable to create directories or write files

    Example:
        >>> collection = create_stac_collection(...)
        >>> items = [create_stac_item(...) for ...]
        >>> write_stac_output(collection, items, './stac_output/cardamom-t2m-min')
    """

    # Create output directory structure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    items_dir = output_path / 'items'
    items_dir.mkdir(exist_ok=True)

    # Write collection.json
    collection_file = output_path / 'collection.json'

    # Set up collection links to items
    for item in items:
        item_rel_path = items_dir / f"{item.id}.json"
        collection.add_link(
            Link(
                rel='item',
                target=str(item_rel_path),
                title=f'{item.id}'
            )
        )

    # Serialize collection to JSON
    collection_dict = collection.to_dict()

    with open(collection_file, 'w') as f:
        json.dump(collection_dict, f, indent=2)

    print(f"Wrote STAC Collection to: {collection_file}")

    # Write individual item files
    for item in items:
        item_file = items_dir / f"{item.id}.json"
        item_dict = item.to_dict()

        with open(item_file, 'w') as f:
            json.dump(item_dict, f, indent=2)

    print(f"Wrote {len(items)} STAC Items to: {items_dir}")


def get_stac_collection_metadata(collection_id: str) -> Dict[str, Any]:
    """
    Get predefined metadata for a CARDAMOM STAC collection.

    This function provides quick access to standardized collection metadata
    defined in CARDAMOM_STAC_COLLECTIONS.

    Args:
        collection_id (str): Collection ID (e.g., 'cardamom-t2m-min')

    Returns:
        Dict[str, Any]: Metadata dictionary for the collection

    Raises:
        KeyError: If collection_id is not defined in CARDAMOM_STAC_COLLECTIONS

    Example:
        >>> metadata = get_stac_collection_metadata('cardamom-t2m-min')
        >>> print(metadata['cbf_variable'])
        T2M_MIN
    """

    if collection_id not in CARDAMOM_STAC_COLLECTIONS:
        available_collections = list(CARDAMOM_STAC_COLLECTIONS.keys())
        raise KeyError(
            f"Collection '{collection_id}' not found. "
            f"Available collections: {', '.join(available_collections)}"
        )

    return CARDAMOM_STAC_COLLECTIONS[collection_id]


def validate_stac_item(item: Item, strict: bool = False) -> Dict[str, Any]:
    """
    Validate a STAC Item for CARDAMOM compliance.

    Checks that the item has all required CARDAMOM-specific properties and
    that asset paths point to valid NetCDF files.

    Scientific Context:
    Validation ensures that downstream applications (CBF generator) can reliably
    discover and load data from STAC metadata without runtime errors.

    Args:
        item (pystac.Item): STAC Item to validate
        strict (bool): If True, raise exceptions on validation failures.
            If False, return validation results without raising.

    Returns:
        Dict[str, Any]: Validation results with keys:
            - 'is_valid': bool
            - 'errors': List[str]
            - 'warnings': List[str]

    Example:
        >>> item = create_stac_item(...)
        >>> results = validate_stac_item(item)
        >>> if results['is_valid']:
        ...     print("Item is valid for CARDAMOM use")
    """

    errors = []
    warnings = []

    # Check required CARDAMOM properties
    if 'cardamom:variable' not in item.properties:
        errors.append("Missing 'cardamom:variable' property")

    if 'cardamom:collection' not in item.properties:
        errors.append("Missing 'cardamom:collection' property")

    if 'start_datetime' not in item.properties:
        errors.append("Missing 'start_datetime' property")

    if 'end_datetime' not in item.properties:
        errors.append("Missing 'end_datetime' property")

    # Check that data asset exists
    data_asset = item.get_asset('data')
    if not data_asset:
        errors.append("Missing 'data' asset")
    else:
        # Check if file exists (if absolute path)
        file_path = Path(data_asset.href)
        if file_path.is_absolute() and not file_path.exists():
            warnings.append(f"Data file not found at {data_asset.href}")

    is_valid = len(errors) == 0

    if strict and not is_valid:
        raise ValueError(f"STAC Item validation failed: {errors}")

    return {
        'is_valid': is_valid,
        'errors': errors,
        'warnings': warnings,
    }
