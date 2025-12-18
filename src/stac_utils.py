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


# ========== Helper Functions ==========

def _normalize_path_separators(file_path: str) -> str:
    """
    Normalize path separators to forward slashes for STAC compliance.

    STAC requires URLs with forward slashes, regardless of OS.
    This is called during item creation; actual relative path computation
    happens during serialization in write_stac_output() where file locations
    are known.

    Args:
        file_path (str): File path with potentially OS-specific separators

    Returns:
        str: Path with forward slashes
    """
    return str(file_path).replace('\\', '/')


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
        data_file_path (str): Relative or absolute path to NetCDF file.
            Will be converted to relative path when serializing.
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
    # Store the file path as-is; it will be converted to relative path during serialization
    asset_href = _normalize_path_separators(data_file_path)
    item.add_asset(
        key='data',
        asset=Asset(
            href=asset_href,
            media_type='application/x-netcdf',
            title=f'{variable_name} {year}-{month:02d}',
            roles=['data'],
        )
    )

    return item


def read_existing_collection(collection_path: str) -> Optional[Collection]:
    """
    Read existing STAC collection from filesystem if it exists.

    This function enables incremental updates to STAC catalogs by loading
    previously created collections, allowing downloaders to append new data
    without losing historical metadata.

    Scientific Context:
    Meteorological data processing often occurs in stages across different time
    periods. This function preserves collection metadata (temporal extent, spatial
    coverage, item counts) across multiple processing runs, maintaining a complete
    historical record of available data.

    Args:
        collection_path (str): Full path to collection.json file

    Returns:
        Optional[Collection]: pystac.Collection object if file exists and is valid,
            None if file doesn't exist or is malformed

    Example:
        >>> collection = read_existing_collection('./stac/cardamom-t2m-min/collection.json')
        >>> if collection:
        >>>     print(f"Found {len(collection.get_all_items())} existing items")
    """

    collection_file = Path(collection_path)

    # File doesn't exist - this is expected for first run
    if not collection_file.exists():
        return None

    # Try to load collection
    try:
        collection = Collection.from_file(str(collection_file))
        logger.info(f"Found existing STAC collection: {collection.id}")
        return collection
    except Exception as error:
        logger.warning(
            f"STAC collection at {collection_path} is malformed. "
            f"Creating fresh collection instead. Error: {error}"
        )
        return None


def read_existing_items(items_directory: str) -> Dict[str, Item]:
    """
    Read all existing STAC items from the items directory.

    This function loads existing item metadata files to enable merging with
    new downloads, preventing loss of historical data when processing new
    time periods or reprocessing existing periods with updated algorithms.

    Scientific Context:
    Each STAC item represents a single meteorological variable for a specific
    time period (e.g., T2M_MIN for January 2020). Preserving existing items
    allows scientists to incrementally build datasets spanning multiple years
    without manually managing individual files.

    Args:
        items_directory (str): Path to items/ subdirectory containing item JSON files

    Returns:
        Dict[str, Item]: Dictionary mapping item IDs to pystac.Item objects
            Empty dict if directory doesn't exist

    Example:
        >>> items = read_existing_items('./stac/cardamom-t2m-min/items')
        >>> print(f"Loaded {len(items)} existing items")
        >>> print(f"Item IDs: {list(items.keys())}")
    """

    items_dir = Path(items_directory)
    existing_items_dict = {}

    # Directory doesn't exist - this is expected for first run
    if not items_dir.exists():
        return existing_items_dict

    # Load each item JSON file
    for item_file in items_dir.glob('*.json'):
        try:
            item = Item.from_file(str(item_file))
            existing_items_dict[item.id] = item
        except Exception as error:
            logger.warning(
                f"Unable to read STAC item {item_file.name}. "
                f"Skipping this item. Error: {error}"
            )
            continue

    if existing_items_dict:
        logger.info(f"Loaded {len(existing_items_dict)} existing items")

    return existing_items_dict


def merge_stac_items(
    existing_items: Dict[str, Item],
    new_items: List[Item],
    duplicate_policy: str = 'update'
) -> List[Item]:
    """
    Merge new STAC items with existing items based on duplicate policy.

    This function combines existing and new items, handling duplicates (items with
    the same ID) according to the specified policy. This enables flexible workflows
    for reprocessing data or resuming interrupted downloads.

    Scientific Context:
    Meteorological reanalysis data may be updated with improved algorithms or
    corrected observations. The 'update' policy allows reprocessing specific time
    periods while preserving other periods' data. The 'skip' policy supports
    resuming interrupted downloads without unnecessary reprocessing.

    Args:
        existing_items (Dict[str, Item]): Dictionary of existing items by ID
        new_items (List[Item]): List of new items to merge
        duplicate_policy (str): How to handle duplicate item IDs
            - 'update': Replace existing item with new item (default)
            - 'skip': Keep existing item, ignore new item
            - 'error': Raise error for user decision

    Returns:
        List[Item]: Merged list of items with duplicates resolved

    Raises:
        ValueError: If duplicate_policy='error' and duplicates are found

    Example:
        >>> existing = {'t2m_min_2020_01': item1}
        >>> new = [item2_for_january, item3_for_february]
        >>> merged = merge_stac_items(existing, new, policy='update')
        >>> # Result: January item updated, February item added
    """

    # Create merged dictionary starting with existing items
    merged_dict = existing_items.copy()

    # Track statistics for logging
    items_added = 0
    items_updated = 0
    duplicate_ids = []

    # Process each new item
    for new_item in new_items:
        if new_item.id in merged_dict:
            # Duplicate detected
            duplicate_ids.append(new_item.id)

            if duplicate_policy == 'update':
                merged_dict[new_item.id] = new_item
                items_updated += 1
            elif duplicate_policy == 'skip':
                # Keep existing item, do nothing
                pass
            elif duplicate_policy == 'error':
                raise ValueError(
                    f"Duplicate STAC items detected with policy='error': {duplicate_ids}. "
                    f"These items already exist in the collection. "
                    f"Options: (1) Use --stac-duplicate-policy update to replace them, "
                    f"(2) Use --stac-duplicate-policy skip to keep existing, "
                    f"(3) Remove existing items manually before rerunning."
                )
            else:
                raise ValueError(f"Invalid duplicate_policy: {duplicate_policy}. Must be 'update', 'skip', or 'error'")
        else:
            # New item (not a duplicate)
            merged_dict[new_item.id] = new_item
            items_added += 1

    # Log merge statistics
    items_total = len(merged_dict)
    logger.info(
        f"Merging {len(new_items)} new items (policy: {duplicate_policy})"
    )
    logger.info(
        f"Merge complete: {items_added} added, {items_updated} updated, {items_total} total"
    )

    return list(merged_dict.values())


def update_collection_extent(
    collection: Collection,
    items: List[Item]
) -> Collection:
    """
    Update collection's spatial and temporal extent based on all items.

    This function ensures collection metadata accurately reflects the full
    spatiotemporal coverage of available data after merging items from
    multiple download runs.

    Scientific Context:
    Collection extent metadata is critical for data discovery queries.
    When scientists search for data covering a specific region and time period,
    the collection extent determines whether this collection appears in search
    results. Accurate extent updates enable proper data discovery across
    incrementally built datasets.

    Args:
        collection (Collection): STAC Collection to update
        items (List[Item]): All items in the collection (after merging)

    Returns:
        Collection: Collection with updated extent metadata

    Example:
        >>> # Collection initially covers 2020-01
        >>> # After adding items for 2020-02 and 2020-03:
        >>> updated_collection = update_collection_extent(collection, all_items)
        >>> # Collection now covers 2020-01 to 2020-04
    """

    if not items:
        # No items - keep existing extent
        return collection

    # Extract temporal bounds from all items
    start_times = []
    end_times = []

    for item in items:
        props = item.properties

        # Get start_datetime (required by STAC spec)
        if 'start_datetime' in props:
            start_times.append(datetime.fromisoformat(props['start_datetime'].replace('Z', '+00:00')))
        elif 'datetime' in props:
            start_times.append(datetime.fromisoformat(props['datetime'].replace('Z', '+00:00')))

        # Get end_datetime (optional)
        if 'end_datetime' in props:
            end_times.append(datetime.fromisoformat(props['end_datetime'].replace('Z', '+00:00')))
        elif 'datetime' in props:
            # If no end_datetime, use datetime as end
            end_times.append(datetime.fromisoformat(props['datetime'].replace('Z', '+00:00')))

    # Calculate overall temporal extent
    if start_times and end_times:
        temporal_start = min(start_times)
        temporal_end = max(end_times)

        # Update collection extent
        collection.extent.temporal = TemporalExtent(
            intervals=[[temporal_start, temporal_end]]
        )

        logger.info(
            f"Updated collection temporal extent: "
            f"{temporal_start.strftime('%Y-%m-%d')} to {temporal_end.strftime('%Y-%m-%d')}"
        )

    # Spatial extent: Keep global bbox for CARDAMOM (all data is global or CONUS)
    # If needed in future, could compute union of item bboxes here

    return collection


def create_root_catalog(
    stac_output_root: str,
    collection_ids: List[str],
    catalog_id: str = 'cardamom-preprocessor',
    description: str = 'CARDAMOM Preprocessor Data Catalog',
    include_item_links: bool = True
) -> None:
    """
    Create a root STAC catalog linking all collections and optionally all items.

    This function creates a master catalog.json file at the output root that provides
    a single entry point for discovering all CARDAMOM preprocessor outputs, enabling
    STAC browsers and automated tools to traverse the complete data catalog. It supports
    a hybrid structure linking both collections and individual items for flexible discovery.

    Scientific Context:
    A root catalog enables scientists to discover all available meteorological
    variables (ERA5, NOAA CO2, GFED burned area) through a single query point.
    Item links provide direct access to specific data files without traversing
    the full collection hierarchy, improving discoverability for tools that query
    the catalog directly.

    Args:
        stac_output_root (str): Path to output root directory (where catalog.json will be created)
        collection_ids (List[str]): Collection IDs to link (e.g., ['cardamom-t2m-min', 'cardamom-co2'])
        catalog_id (str): Root catalog identifier (default: 'cardamom-preprocessor')
        description (str): Human-readable catalog description
        include_item_links (bool): If True, add direct links to all items in addition to collection links.
            Creates hybrid catalog structure linking both collections and items for easier discovery.
            Default: True

    Returns:
        None (writes catalog.json to disk)

    Example:
        >>> create_root_catalog(
        >>>     stac_output_root='./output',
        >>>     collection_ids=['cardamom-era5-variables', 'cardamom-co2']
        >>> )
        >>> # Creates ./output/catalog.json linking collections and all items in ./output/{collection_id}/items/
    """

    # Create root catalog
    catalog = Catalog(
        id=catalog_id,
        description=description,
        title='CARDAMOM Preprocessor STAC Catalog',
    )

    # Add links to each collection (collections are directly in stac_output_root)
    stac_root_path = Path(stac_output_root)
    valid_collections = 0
    total_items_linked = 0

    for collection_id in collection_ids:
        collection_path = stac_root_path / collection_id / 'collection.json'

        if collection_path.exists():
            # Create relative link to collection (relative to catalog.json location)
            catalog.add_link(
                Link(
                    rel='child',
                    target=f'{collection_id}/collection.json',
                    media_type='application/json',
                    title=collection_id,
                )
            )
            valid_collections += 1
        else:
            logger.warning(
                f"Collection {collection_id} not found at {collection_path}. "
                f"Skipping from root catalog."
            )

    # Add item links if requested (hybrid catalog structure)
    if include_item_links:
        for collection_id in collection_ids:
            items_dir = stac_root_path / collection_id / 'items'

            # Skip if items directory doesn't exist
            if not items_dir.exists():
                continue

            # Reuse existing item discovery function to load all items
            existing_items = read_existing_items(str(items_dir))

            # Add link for each item discovered
            for item_id in existing_items.keys():
                # Construct relative path from catalog.json to item file
                item_rel_href = f'{collection_id}/items/{item_id}.json'

                catalog.add_link(
                    Link(
                        rel='item',
                        target=item_rel_href,
                        media_type='application/json',
                        title=item_id,
                    )
                )
                total_items_linked += 1

    # Write root catalog
    catalog_file = stac_root_path / 'catalog.json'

    # Add self link to root catalog so pystac can traverse and resolve relative paths
    catalog.set_self_href(str(catalog_file))

    catalog_dict = catalog.to_dict()

    with open(catalog_file, 'w') as f:
        json.dump(catalog_dict, f, indent=2)

    logger.info(
        f"Created root STAC catalog at {catalog_file} "
        f"linking {valid_collections} collections and {total_items_linked} items"
    )


def write_stac_output(
    collection: Collection,
    items: List[Item],
    output_dir: str,
    incremental: bool = True,
    duplicate_policy: str = 'update',
) -> Dict[str, Any]:
    """
    Write STAC Collection and Items to filesystem with optional incremental updates.

    This function organizes STAC metadata in a standard directory structure:
    ```
    output_dir/
    ├── collection.json
    └── items/
        ├── {item_id_1}.json
        ├── {item_id_2}.json
        └── ...
    ```

    When incremental mode is enabled (default), this function reads existing
    collection and items, merges new items with existing ones, and updates
    collection extent to span all data. This prevents loss of historical
    metadata when processing new time periods.

    Scientific Context:
    The STAC JSON files enable the CBF generator to discover and query available
    meteorological data without direct filesystem access, supporting distributed
    data processing systems like MAAP. Incremental updates allow scientists to
    build comprehensive multi-year datasets through iterative processing runs
    without manual metadata management.

    Args:
        collection (pystac.Collection): The STAC Collection to write
        items (List[pystac.Item]): List of STAC Items to write
        output_dir (str): Directory path where STAC files will be written
        incremental (bool): Enable incremental updates by merging with existing
            catalog (default: True). Set to False for legacy overwrite behavior.
        duplicate_policy (str): How to handle duplicate items when incremental=True
            - 'update': Replace existing item with new item (default)
            - 'skip': Keep existing item, ignore new item
            - 'error': Raise error for user decision

    Returns:
        Dict[str, Any]: Merge statistics containing:
            - 'items_added': Number of new items added
            - 'items_updated': Number of existing items updated
            - 'items_total': Total number of items in collection
            - 'collection_created': True if new collection, False if updated

    Raises:
        IOError: If unable to create directories or write files
        ValueError: If duplicate_policy='error' and duplicates are found

    Example:
        >>> # Incremental mode (default) - merges with existing catalog
        >>> stats = write_stac_output(collection, items, './stac/cardamom-t2m-min')
        >>> print(f"Added {stats['items_added']}, updated {stats['items_updated']}")
        >>>
        >>> # Non-incremental mode - overwrites existing catalog
        >>> stats = write_stac_output(collection, items, './stac/cardamom-t2m-min', incremental=False)
    """

    # Setup paths
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    items_dir = output_path / 'items'
    items_dir.mkdir(exist_ok=True)

    collection_file = output_path / 'collection.json'

    # Incremental mode: merge with existing catalog
    if incremental:
        # Read existing collection and items
        existing_collection = read_existing_collection(str(collection_file))
        existing_items = read_existing_items(str(items_dir))

        # Merge items
        merged_items = merge_stac_items(existing_items, items, duplicate_policy)

        # Update collection extent to span all items
        if existing_collection:
            # Use existing collection and update its extent
            collection = update_collection_extent(existing_collection, merged_items)
        else:
            # First run - update new collection extent
            collection = update_collection_extent(collection, merged_items)

        # Use merged items for writing
        items = merged_items

        # Calculate statistics
        stats = {
            'items_added': len([i for i in items if i.id not in existing_items]),
            'items_updated': len([i for i in items if i.id in existing_items]),
            'items_total': len(items),
            'collection_created': existing_collection is None,
        }
    else:
        # Non-incremental mode: use original logic (backward compatibility)
        stats = {
            'items_added': len(items),
            'items_updated': 0,
            'items_total': len(items),
            'collection_created': True,
        }

    # Clear existing links before adding new ones
    # (Important for incremental updates to avoid duplicate links)
    collection.links = [link for link in collection.links if link.rel != 'item']

    # Set up collection links to items (relative to collection.json)
    for item in items:
        # Use relative path from collection.json location to items directory
        item_rel_href = f"./items/{item.id}.json"
        collection.add_link(
            Link(
                rel='item',
                target=item_rel_href,
                title=f'{item.id}'
            )
        )

    # Add self link to collection so pystac can resolve relative paths
    collection.set_self_href(str(collection_file))

    # Serialize collection to JSON
    collection_dict = collection.to_dict()

    try:
        with open(collection_file, 'w') as f:
            json.dump(collection_dict, f, indent=2)

        logger.info(f"Wrote STAC Collection to: {collection_file}")
    except IOError as error:
        logger.error(
            f"Failed to write STAC collection to {collection_file}. "
            f"Check directory permissions and available disk space. Error: {error}"
        )
        raise

    # Compute main output directory
    # With stac_subdir empty, structure is: output_directory/collection_id/items/
    # So we go up one level from collection to get to main output
    main_output_dir = output_path.parent

    # Write individual item files
    for item in items:
        item_file = items_dir / f"{item.id}.json"

        # Add self link to item so pystac can resolve relative asset paths
        item.set_self_href(str(item_file))

        item_dict = item.to_dict()

        # Ensure asset hrefs are relative to the item's location
        # Items are in items/ directory, so compute relative paths from there
        if 'assets' in item_dict:
            for asset_key, asset_info in item_dict['assets'].items():
                if 'href' in asset_info:
                    asset_href = asset_info['href']

                    # Convert to absolute path if needed
                    asset_path = Path(asset_href)
                    if not asset_path.is_absolute():
                        # If relative, it's relative to the main output directory
                        asset_path = main_output_dir / asset_href

                    # Compute relative path from item file location to asset
                    try:
                        relative_asset_path = asset_path.relative_to(items_dir)
                    except ValueError:
                        # If can't compute relative path, try alternative approach
                        try:
                            # Get relative path from main output directory
                            rel_from_main = asset_path.relative_to(main_output_dir)
                            # Then go up from items directory to main directory
                            relative_asset_path = Path('../..') / rel_from_main
                        except ValueError:
                            # Fall back to just the filename
                            logger.warning(
                                f"Could not compute relative path for asset {asset_href}. "
                                f"Using filename only."
                            )
                            relative_asset_path = asset_path.name

                    # Normalize path separators to forward slashes for STAC compliance
                    relative_asset_href = str(relative_asset_path).replace('\\', '/')
                    item_dict['assets'][asset_key]['href'] = relative_asset_href

        try:
            with open(item_file, 'w') as f:
                json.dump(item_dict, f, indent=2)
        except IOError as error:
            logger.error(
                f"Failed to write STAC item to {item_file}. "
                f"Check directory permissions. Error: {error}"
            )
            raise

    logger.info(f"Wrote {len(items)} STAC Items to: {items_dir}")

    return stats


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
