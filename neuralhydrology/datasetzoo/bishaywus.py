from functools import reduce
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import xarray

from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.utils.config import Config


class BishayWUS(BaseDataset):
    """Dataset class based on file structure of BishayWUS dataset derived from downscaled CMIP6 data.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    is_train : bool
        Defines if the dataset is used for training or evaluating. If True (training), means/stds for each feature
        are computed and stored to the run directory. If one-hot encoding is used, the mapping for the one-hot encoding
        is created and also stored to disk. If False, a `scaler` input is expected and similarly the `id_to_int` input
        if one-hot encoding is used.
    period : {'train', 'validation', 'test'}
        Defines the period for which the data will be loaded
    basin : str, optional
        If passed, the data for only this basin will be loaded. Otherwise the basin(s) are read from the appropriate
        basin file, corresponding to the `period`.
    additional_features : List[Dict[str, pd.DataFrame]], optional
        List of dictionaries, mapping from a basin id to a pandas DataFrame. This DataFrame will be added to the data
        loaded from the dataset and all columns are available as 'dynamic_inputs', 'evolving_attributes' and
        'target_variables'
    id_to_int : Dict[str, int], optional
        If the config argument 'use_basin_id_encoding' is True in the config and period is either 'validation' or
        'test', this input is required. It is a dictionary, mapping from basin id to an integer (the one-hot encoding).
    scaler : Dict[str, Union[pd.Series, xarray.DataArray]], optional
        If period is either 'validation' or 'test', this input is required. It contains the centering and scaling
        for each feature and is stored to the run directory during training (train_data/train_data_scaler.yml).
    """

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = {}):
        super(BishayWUS, self).__init__(cfg=cfg,
                                        is_train=is_train,
                                        period=period,
                                        basin=basin,
                                        additional_features=additional_features,
                                        id_to_int=id_to_int,
                                        scaler=scaler)

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """Load input and output data. """
        df = load_timeseries(data_dir=self.cfg.data_dir,
                             ensemble_member=self.cfg.ensemble_member,
                             basin=basin,
                             time_period=self.cfg.time_period,
                             scale=self.cfg.scale)
        return df

    def _load_attributes(self) -> pd.DataFrame:
        """Load static catchment attributes."""
        return load_attributes(self.cfg.data_dir, self.cfg.ensemble_member, self.basins, self.cfg.time_period)

def load_attributes(data_dir: Path,
                    ensemble_member: str = None,
                    basins: List[str] = None,
                    time_period: str = None) -> pd.DataFrame:
    """Load static attributes from one or more CSV files.

    Parameters
    ----------
    data_dir : Path
        Path to the root data directory. Must contain a 'basins' subfolder with attribute CSV files.
    ensemble_member : str, optional
        If provided, filters columns related to snow variables (DSD, DSO, MAX_SWE).
    basins : List[str], optional
        List of basin IDs (8-digit USGS identifiers) to include in the result. If not provided, all available basins are included.
    Returns
    -------
    pd.DataFrame
        DataFrame indexed by basin STAID, with attributes as columns.
        If multiple attribute files are present, the returned DataFrame combines the files as outlined below.
    """

    # Set path to attributes folder
    attributes_path = data_dir / 'attributes'
    if not attributes_path.exists():
        raise FileNotFoundError(f"Attributes folder not found at {attributes_path}")

    if not time_period:
        raise NameError(f"Time period of interest must be specified.")
    if not ensemble_member:
        raise NameError(f"Ensemble member/GCM of interest (or 'Hist-Daymet-USGS-UA') must be specified.")

    # Get list of all files in attribute folder
    files = list(attributes_path.glob(f'*.csv'))
    if not files:
        raise FileNotFoundError(f"No attributes files found.")

    # Select files for time period
    selected_files = [f for f in files if time_period in f.name]
    # Select files for ensemble member
    selected_files = [f for f in selected_files if ensemble_member in f.name]
    # Add GAGES-II file to list
    selected_files.append([f for f in files if "GAGES-II" in f.name][0])

    # Read-in attributes into one big dataframe. Sort by both axes so we can check for identical axes.
    dfs = []
    for f in selected_files:
        df = pd.read_csv(f, dtype={0: str}, delimiter="\t")  # make sure we read the basin id as str
        df = df.set_index(df.columns[0]).sort_index(axis=0).sort_index(axis=1)
        df.index = df.index.map(lambda x: x.zfill(8))
        if df.index.has_duplicates or df.columns.has_duplicates:
            raise ValueError(f'Attributes file {f} contains duplicate basin ids or features.')
        dfs.append(df)

    if len(dfs) == 1:
        df = dfs[0]
    else:
        if len(reduce(lambda idx, other_idx: idx.intersection(other_idx), (df.index for df in dfs))) > 0:
            # basin intersection is non-empty -> concatenate attributes, keep intersection of basins
            if np.any(np.unique(np.concatenate([df.columns for df in dfs]), return_counts=True)[1] > 1):
                raise ValueError('If attributes dataframes refer to the same basins, no attribute name may occur '
                                 'multiple times across the different attributes files.')
            concat_axis = 1
        elif len(reduce(lambda cols, other_cols: cols.intersection(other_cols), (df.columns for df in dfs))) > 0:
            # attributes intersection is non-empty -> concatenate basins, keep intersection of attributes
            # no need to check for basin duplicates, since then we'd have had a non-empty basin intersection.
            concat_axis = 0
        else:
            raise ValueError('Attribute files must overlap on either the index or the columns.')

        df = pd.concat(dfs, axis=concat_axis, join='inner')

    if isinstance(basins, list) and basins:
        missing_basins = [b for b in basins if b not in df.index]
        if len(missing_basins)>0:
            raise ValueError(f"The following basins are missing attribute information: {missing_basins}")
        df = df.loc[basins]

    return df

def load_timeseries(data_dir: Path,
                    ensemble_member: str = None,
                    basin: str = None,
                    time_period: str = '370',
                    scale: str = 'lumped') -> pd.DataFrame:
    """Load time series data from netCDF files into pandas DataFrame.

    Parameters
    ----------
    data_dir : Path
        Path to the root data directory. This folder must contain a subfolder 'forcings/<scale>/<ensemble_member>'
        containing the time series data files.
    time_period : str, default '370'
        The Shared Socioeconomic Pathway (SSP) scenario code to match in the filename (e.g., '370').
    scale : str, default 'lumped'
        The spatial scale subdirectory to use within 'forcings' (e.g., 'lumped').
    ensemble_member : str
        The ensemble member identifier subdirectory to use within 'forcings/<scale>'.
    basin : str
        8-digit USGS identifier of the basin.

    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame containing basin time series data.
    """
    if not ensemble_member:
        raise ValueError('Ensemble member name (or "Hist-Daymet-USGS-UA") must be specified in config file.')
    if not basin:
        raise ValueError('At least one basin must be specified via filenames in config file.')

    files_dir = data_dir / "forcings" / scale / ensemble_member
    files = list(files_dir.glob(f'*.txt'))
    basin_files = [f for f in files if basin in f.stem]

    if len(basin_files) == 0:
        raise FileNotFoundError(f"No forcing file found for basin {basin} in {files_dir}.")
    if len(basin_files) > 1:
        raise ValueError(f"Multiple forcing files found for basin {basin} in {files_dir}.")

    df = pd.read_csv(basin_files[0], sep='\t')
    df.time = pd.to_datetime(df.time, format="%Y-%m-%d")
    df = df.rename(columns={"time": "date"})
    df = df.set_index("date")

    return df

