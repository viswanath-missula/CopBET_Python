import json
import matlab
import matlab.engine
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import inspect
import os
import pathlib as pl
import pickle
from scipy.stats import ttest_rel

# HELPER FUNCTIONS
def file_exists(filepath):
    """    
    Determines whether a file exists by checking the filesystem at the specified path.
    This is a wrapper around pathlib's Path.is_file() method for convenient file existence
    validation throughout the pipeline.
    
    Parameters
    ----------
    filepath : str
        The file system path to check for file existence. Can be a relative or absolute path.
    
    Returns
    -------
    bool
        True if a file exists at the given filepath, False otherwise.
    """
    return pl.Path(filepath).is_file()

def atlas_to_matlab(atlas_mat, logical=False):
    """
    Converts a NumPy array to a MATLAB-compatible object, preserving or converting the data type
    as needed. Supports multiple numeric types including integer and floating-point arrays.
    The function detects the input array's dtype and wraps it in the corresponding MATLAB type.

    Parameters
    ----------
    atlas_mat : numpy.ndarray
        A 3D NumPy array representing an atlas or mask with integer or boolean values.
    logical : bool, optional
        If True, convert the array to a MATLAB logical type. If False, convert based on the
        array's dtype (default is False).
    
    Returns
    -------
    matlab object
        A MATLAB-compatible object. Returns matlab.logical if logical=True, or matlab.<dtype>
        corresponding to the input array's numpy dtype (uint8, int16, uint16, int32, uint32, or double).
    
    Raises
    ------
    ValueError
        If the array's dtype is not supported (not one of the handled types).
    """

    temp_mat = atlas_mat.tolist()
    
    if logical:
        # wrap in MATLAB logical
        return matlab.logical(temp_mat)
    
    dt = atlas_mat.dtype
    if dt == np.uint8:
        return matlab.uint8(temp_mat)
    elif dt == np.int16:
        return matlab.int16(temp_mat)
    elif dt == np.uint16:
        return matlab.uint16(temp_mat)
    elif dt == np.int32:
        return matlab.int32(temp_mat)
    elif dt == np.uint32:
        return matlab.uint32(temp_mat)
    elif dt == np.float64:
        return matlab.double(temp_mat)
    else:
        raise ValueError(f"Unsupported NumPy dtype {dt}")
    
def engine_init(path):
    """
    Starts a MATLAB engine session and sets up the execution environment by adding paths
    to MATLAB's search path. This allows Python to call MATLAB functions and pass data to
    a running MATLAB instance. The working directory is set to the specified path.

    Parameters
    ----------
    path : str
        Filesystem path to the MATLAB working directory containing .m function files.
        This directory and all its subdirectories will be added to MATLAB's search path.
    
    Returns
    -------
    matlab.engine.MatlabEngine
        An initialized and configured MATLAB engine session with the working directory set
        and all subdirectories added to the MATLAB path for function discovery.
    """

    engine = matlab.engine.start_matlab()
    engine.addpath(path)
    engine.cd(path, nargout=0)
    engine.addpath(engine.genpath(engine.pwd()), nargout=0)
    return engine

def json_to_df(table):
    """
    Parses a JSON-formatted string (typically output from MATLAB functions) and converts
    it into a pandas DataFrame for easier data manipulation and analysis in Python.
    Assumes the JSON represents an array of objects or a table-like structure.

    Parameters
    ----------
    table : str
        A JSON-formatted string representing tabular data. Typically an array of dictionaries
        where each dictionary represents a row with column names as keys.
    
    Returns
    -------
    pandas.DataFrame
        A DataFrame with data parsed from the JSON string. Column names correspond to
        the keys from the JSON objects.
    """

    return pd.DataFrame(json.loads(table))

def df_to_json(table, matlab_escaped=False):
    """
    Converts a pandas DataFrame to a JSON string representation suitable for passing
    to MATLAB functions via the Python-MATLAB engine. Optionally applies MATLAB-specific
    character escaping for compatibility with MATLAB's string parsing.

    Parameters
    ----------
    table : pandas.DataFrame
        The DataFrame to convert to JSON format.
    matlab_escaped : bool, optional
        If True, escape single quotes and backslashes for MATLAB compatibility.
        If False, return standard JSON without MATLAB-specific escaping (default is False).
    
    Returns
    -------
    str
        JSON-formatted string where each row of the DataFrame is an object (dictionary).
        Compatible with MATLAB's jsondecode function if matlab_escaped=True.
    """
    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        return x

    table = table.map(convert)
    
    if not matlab_escaped:
        return json.dumps(table.to_dict(orient="records"))
    else:
        return json.dumps(table.to_dict(orient="records")).replace("'", "''").replace("\\", "\\\\")

def return_varargin(keepdata, parallel, NRU_specific):
    """
    Builds a formatted string of optional name-value pair arguments for passing to MATLAB
    functions via the CopBET Python engine. These flags control data retention, parallel
    computing, and lab-specific processing configurations.

    Parameters
    ----------
    keepdata : bool
        If True, retain intermediate computation data and outputs from MATLAB processing.
        If False, clean up temporary files after computation.
    parallel : bool
        If True, enable MATLAB's parallel computing toolbox for multi-threaded execution.
        If False, run computations serially on a single thread.
    NRU_specific : bool
        If True, apply Neurobiology Research Unit (NRU)-specific preprocessing parameters
        and default values. If False, use standard CopBET parameters.
    
    Returns
    -------
    str
        A formatted string containing MATLAB varargin arguments in the form:
        'keepdata',true/false,'parallel',true/false,'NRUspecific',true/false
        Ready to be inserted into MATLAB function call strings.
    """

    out = ""
    first = True
    ls = ["'keepdata'", "'parallel'", "'NRUspecific'"]
    ls2 = [keepdata, parallel, NRU_specific]
    for x in range(3):
        y = "true" if ls2[x] else "false"
        if x < 1:
            out += ls[x] + "," + y
        else:
            out += "," + ls[x] + "," + y
    return out

def save_dataframe(file_name, df, out_dir):
    """
    Writes a pandas DataFrame to a CSV file, creating the output directory if it does not exist.
    This is a convenience wrapper around pandas.DataFrame.to_csv() that handles directory
    creation and provides informative console output.

    Parameters
    ----------
    file_name : str
        Name of the output CSV file without the .csv extension. The extension will be added
        automatically.
    df : pandas.DataFrame
        The DataFrame object to save. Must be a valid pandas DataFrame instance.
    out_dir : str
        Path to the output directory where the CSV file will be saved. The directory is
        created recursively if it does not exist.
    
    Raises
    ------
    TypeError
        If df is not a pandas DataFrame instance.
    """
    out_dir = pl.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)  # ensure folder exists

    # Ensure df is actually a pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    # Construct file path
    out_path = out_dir / f"{file_name}.csv"

    # Save DataFrame directly to CSV
    df.to_csv(out_path, index=False)

    print(f"Saved DataFrame to {out_path}")

# USER FUNCTIONS
def load_atlas(atlas_path, logical=False):
    """
    Reads a NIfTI (.nii or .nii.gz) atlas file and converts it to a NumPy array for
    computational processing. Optionally converts the atlas to a binary logical mask
    where all non-zero voxels are set to 1.

    Parameters
    ----------
    atlas_path : str
        Filesystem path to the NIfTI atlas file (.nii or .nii.gz format).
    logical : bool, optional
        If True, return a binary array where all non-zero voxels become 1 and zero voxels
        remain 0 (creating a logical mask). If False, return the raw atlas values as-is
        (default is False).
    
    Returns
    -------
    numpy.ndarray
        A 3D or 4D NumPy array (int32 dtype) representing the atlas. Shape depends on the
        input NIfTI file dimensions. If logical=True, values are binary (0 or 1).
    """

    out = np.asanyarray(nib.load(atlas_path).dataobj)
    out = (out!=0).astype(np.uint8) if logical else out
    return out.astype(np.int32)

def nifti_to_timeseries(fmri_path, atlas_path, brain_mask_path=None, resample=True):
    """
    Loads a 4D fMRI image and a volumetric atlas, aligns them to the same voxel grid
    (resampling if necessary), and extracts mean time series for each ROI. Handles both
    discrete and probabilistic atlases. Returns the time series and corresponding ROI labels.

    Parameters
    ----------
    fmri_path : str
        Filesystem path to the 4D fMRI NIfTI file (dimensions: X, Y, Z, T).
    atlas_path : str
        Filesystem path to the NIfTI atlas file. Can be 3D (discrete labels) or 4D (probabilistic maps).
    brain_mask_path : str, optional
        Path to a brain mask NIfTI file. Currently unused but reserved for future masking operations
        (default is None).
    resample : bool, optional
        If True, automatically resample the atlas to match the fMRI grid if dimensions differ.
        If False, raise an error if dimensions do not match (default is True).
    
    Returns
    -------
    V_roi : numpy.ndarray
        Time series array of shape (T, n_rois) where T is the number of timepoints and n_rois
        is the number of ROIs. Each column contains the mean time series for that ROI.
    labels : numpy.ndarray
        1D array of ROI labels (integers) corresponding to each column in V_roi. Background
        label (0) is excluded.
    
    Raises
    ------
    ValueError
        If atlas and fMRI dimensions do not match and resample=False, or if there is a voxel
        count mismatch after processing.
    RuntimeError
        If atlas needs resampling but nilearn is not installed.
    """
    # Load fMRI
    fmri_img = nib.load(fmri_path)
    fmri_data = fmri_img.get_fdata()  # shape (X, Y, Z, T)
    X, Y, Z, T = fmri_data.shape
    n_fmri_vox = X * Y * Z
    fmri_2d = fmri_data.reshape(n_fmri_vox, T)

    # Load atlas
    atlas_img = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata()

    # Handle probabilistic (4D) atlas → discrete labels
    if atlas_data.ndim == 4:
        print("Atlas is 4D: taking argmax across maps to make discrete labels.")
        atlas_data = np.argmax(atlas_data, axis=3).astype(int)

    # Resample if needed
    if atlas_data.shape != fmri_data.shape[:3]:
        if not resample:
            raise ValueError(f"Atlas shape {atlas_data.shape} != fMRI shape {fmri_data.shape[:3]}")
        try:
            from nilearn.image import resample_to_img
            _HAS_NILEARN = True
        except ImportError:
            _HAS_NILEARN = False
        if not _HAS_NILEARN:
            raise RuntimeError("Atlas and fMRI grids differ and nilearn is not installed.")
        atlas_res = resample_to_img(nib.Nifti1Image(atlas_data, atlas_img.affine),
                                    fmri_img, interpolation='nearest', force_resample=True, copy_header=True)
        atlas_data = atlas_res.get_fdata().astype(int)

    atlas_flat = atlas_data.ravel().astype(int)

    # Final sanity check
    if atlas_flat.size != n_fmri_vox:
        raise ValueError(f"Atlas voxels {atlas_flat.size} != fMRI voxels {n_fmri_vox}")

    # Extract ROI time series
    labels = np.unique(atlas_flat)
    labels = labels[labels != 0]  # drop background
    ts_list = []
    for roi in labels:
        mask_bool = (atlas_flat == roi)
        if mask_bool.sum() == 0:
            roi_ts = np.full((T,), np.nan)
        else:
            roi_ts = fmri_2d[mask_bool, :].mean(axis=0)
        ts_list.append(roi_ts)

    V_roi = np.column_stack(ts_list)  # shape (T, n_rois)
    return V_roi, labels

def convert_atlas_to_4D(numpy_atlas, drop_zero=True):
    """
    Transforms a 3D atlas where each voxel contains an integer label into a 4D array where
    each layer (4th dimension) is a binary mask for a specific ROI label. This representation
    is useful for certain computational pipelines and mask-based operations.

    Parameters
    ----------
    numpy_atlas : numpy.ndarray
        A 3D array of shape (X, Y, Z) where each voxel contains an integer ROI label or 0
        for background.
    drop_zero : bool, optional
        If True, exclude label 0 (background) from the output, creating masks only for
        non-background regions. If False, include all labels including 0 (default is True).
    
    Returns
    -------
    atlas_4D : numpy.ndarray
        A 4D array of shape (X, Y, Z, N) where N is the number of unique labels (excluding
        0 if drop_zero=True). Each layer contains a binary mask (uint8: 0 or 1) for the
        corresponding ROI.
    labels : numpy.ndarray
        1D array of the unique labels corresponding to each layer in atlas_4D. If drop_zero=True,
        label 0 is excluded.
    
    Raises
    ------
    ValueError
        If the input array is not 3-dimensional.
    """
    if numpy_atlas.ndim != 3:
        raise ValueError("Input atlas must be a 3D array (X, Y, Z).")

    labels = np.unique(numpy_atlas)
    if drop_zero:
        labels = labels[labels != 0]

    masks = [(numpy_atlas == lab).astype(np.uint8) for lab in labels]
    atlas_4D = np.stack(masks, axis=-1)

    return atlas_4D, labels

def make_directory(dir_path):
    """
    Create a directory at the specified path, including any parent directories.
    
    Creates a directory structure, recursively making all parent directories as needed.
    If the directory already exists, no error is raised (idempotent operation).

    Parameters
    ----------
    dir_path : str
        Path to the directory to create. Can be relative or absolute.
    """
    folder = pl.Path(dir_path)
    folder.mkdir(parents=True, exist_ok=True)

def get_timeseries(target_files, atlas_path, timeseries_path):
    """
    Checks if a cached timeseries pickle file exists. If it does, loads and returns it.
    If not, computes ROI time series from the input fMRI files using the provided atlas,
    saves the results to disk, and returns them. This function handles caching to avoid
    redundant computational work.

    Parameters
    ----------
    target_files : list of str
        List of file paths to 4D fMRI NIfTI files to process.
    atlas_path : str
        Path to the brain atlas NIfTI file used for ROI definition.
    timeseries_path : str
        Path where the computed timeseries will be cached as a pickle file. If the file
        already exists, it will be loaded instead of recomputing.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with columns 'target' (input fMRI file paths) and 'timeseries' (extracted
        time series data). Each row represents one fMRI file.
    """
    ts_data = []

    if not file_exists(timeseries_path):
        print("Timeseries file not found, computing new timeseries...")
        total = len(target_files)

        for count, t in enumerate(target_files, start=1):
            print(f"Target ({count}/{total}): {t}")
            timeseries = nifti_to_timeseries(t, atlas_path)

            # Store both target name and data
            ts_data.append({"target": t, "timeseries": timeseries})

        # Save computed timeseries
        with open(timeseries_path, "wb") as f:
            try:
                pickle.dump(ts_data, f)
            except FileNotFoundError:
                print("The subdirectory for saving the timeseries does not exist. Use CopBET.make_directory(path) to create it first.")

        print("Computed and saved timeseries.")
    else:
        print("Timeseries file found, loading existing data...")
        with open(timeseries_path, "rb") as f:
            ts_data = pickle.load(f)

    # Convert to DataFrame for convenience
    tbl = pd.DataFrame(ts_data)

    # Ensure 'target' column exists even if data was loaded
    if "target" not in tbl.columns:
        tbl["target"] = target_files[:len(tbl)]

    tbl['timeseries'] = tbl['timeseries'].apply(lambda x: x[0] if isinstance(x, tuple) else x)
    tbl['timeseries'] = tbl['timeseries'].apply(lambda x: x.tolist() if isinstance(x[0], np.ndarray) else x)

    return tbl

def return_target_files(subjects, criteria, filetype=".nii"):
    """
    Searches through a list of subject directories and returns paths to files that match
    all specified criteria. Useful for batch collection of fMRI or other neuroimaging files
    with specific naming patterns.

    Parameters
    ----------
    subjects : list of str
        List of subject directory paths to search within.
    criteria : list
        List of search criteria. Each criterion can be a string (must appear in filename)
        or a list of strings (at least one must appear in filename for that criterion to match).
    filetype : str, optional
        Required file extension (e.g., '.nii', '.nii.gz'). Only files containing this
        extension will be included (default is '.nii').
    
    Returns
    -------
    list of str
        List of full file paths for files matching all criteria and the filetype.
    """
    targets = []
    for s in subjects:
        for f in os.listdir(s):
            if all((any(opt in f for opt in c) if isinstance(c, list) else c in f) for c in criteria) and filetype in f:
                try:
                    targets.append(s+"/"+f)
                except TypeError:
                    print("TypeError")
                    print(f"Target file: {f}")
                    continue
    return targets

class CopBET_computer:
    """
    This class provides a Python interface to CopBET (Complexity in Brain Entropy Toolbox),
    allowing computation of various entropy and complexity metrics from fMRI time series data.
    It manages MATLAB engine communication, data format conversions, and result caching.
    """
    
    def __init__(self, matlab_path, df_timeseries, df_filenames, results_path, overwrite=False, keepdata=True, parallel=True, NRU_specific=False):
        """
        Initialize the CopBET_computer with MATLAB engine and configuration parameters.
        
        Sets up the Python-MATLAB interface and configures the computation pipeline with
        timeseries data, file metadata, and processing options.

        Parameters
        ----------
        matlab_path : str
            Filesystem path to the CopBET MATLAB code directory.
        df_timeseries : pandas.DataFrame
            DataFrame containing the fMRI time series data for each subject/scan.
        df_filenames : pandas.DataFrame
            DataFrame containing file paths and metadata for each scan (must have 'targets' column).
        results_path : str
            Output directory path where computed entropy results will be saved as CSV files.
        overwrite : bool, optional
            If True, recompute and overwrite existing result files. If False, skip computation
            if the result file already exists (default is False).
        keepdata : bool, optional
            If True, retain intermediate MATLAB computation files. If False, clean up temporary
            files after computation (default is True).
        parallel : bool, optional
            If True, enable MATLAB parallel computing. If False, run serially (default is True).
        NRU_specific : bool, optional
            If True, apply Neurobiology Research Unit-specific parameters. If False, use standard
            CopBET settings (default is False).
        """
        self.engine = engine_init(matlab_path)
        self.df_timeseries = df_timeseries
        self.df_filenames = df_filenames
        self.results_path = results_path
        self.overwrite = overwrite
        
        self.keepdata = keepdata
        self.parallel = parallel
        self.NRU_specific = NRU_specific

    def von_Neumann_entropy(self, savedata=True):
        """
        Calculates the von Neumann entropy, a quantum information-theoretic measure of
        complexity, from the covariance matrix of fMRI time series data. Results are cached
        to avoid redundant computation.

        Parameters
        ----------
        savedata : bool, optional
            If True, save the computed results to a CSV file in the results directory.
            If False, return results without saving (default is True).
        
        Returns
        -------
        pandas.DataFrame
            DataFrame with von Neumann entropy values. Includes a 'target' column with the
            original fMRI file paths. If savedata=True and the file already exists with
            overwrite=False, loads and returns the cached results instead of recomputing.
        """

        filename = f"{inspect.currentframe().f_code.co_name}_{self.results_path}"
        out_path = self.results_path + "/" + filename + ".csv"

        if savedata and not self.overwrite and file_exists(out_path):
            print("File already exists and overwrite is False. Skipping computation.")
            return pd.read_csv(out_path)

        inp = df_to_json(self.df_timeseries)
        varargin = return_varargin(self.keepdata, self.parallel, self.NRU_specific)
        self.engine.workspace["json_in"] = inp
        eval_statement = f"jsonencode(CopBET_von_Neumann_entropy(struct2table(jsondecode(json_in)),{varargin}));"
        json_out = self.engine.eval(eval_statement, nargout=1)
        df_out = json_to_df(json_out)
        df_out["target"] = self.df_filenames["targets"]

        if savedata:
            if file_exists(out_path):
                if self.overwrite:
                    save_dataframe(filename, df_out, self.results_path)
            else:
                save_dataframe(filename, df_out, self.results_path)

        return df_out
        
    def metastate_series_complexity(self, savedata=True):
        """        
        Analyzes the complexity and information content of brain state transitions derived
        from dynamic fMRI data. Computes metrics characterizing the temporal patterns of
        metastable network configurations.

        Parameters
        ----------
        savedata : bool, optional
            If True, save the computed results to a CSV file. If False, return results without
            saving (default is True).
        
        Returns
        -------
        pandas.DataFrame
            DataFrame with metastate complexity metrics. Includes a 'target' column with the
            original fMRI file paths. Returns cached results if savedata=True and file exists
            with overwrite=False.
        """

        filename = f"{inspect.currentframe().f_code.co_name}_{self.results_path}"
        out_path = self.results_path + "/" + filename + ".csv"

        if savedata and not self.overwrite and file_exists(out_path):
            print("File already exists and overwrite is False. Skipping computation.")
            return pd.read_csv(out_path)

        inp = df_to_json(self.df_timeseries, matlab_escaped=True)
        varargin = return_varargin(self.keepdata, self.parallel, self.NRU_specific)
        self.engine.workspace["json_in"] = inp
        eval_statement = f"jsonencode(CopBET_metastate_series_complexity(struct2table(jsondecode(json_in)),{varargin}));"
        json_out = self.engine.eval(eval_statement, nargout=1)
        df_out = json_to_df(json_out)
        df_out["target"] = self.df_filenames["targets"]

        if savedata:
            if file_exists(out_path):
                if self.overwrite:
                    save_dataframe(filename, df_out, self.results_path)
            else:
                save_dataframe(filename, df_out, self.results_path)

        return df_out

    def degree_distribution_entropy(self, savedata=True):
        """        
        Calculates the entropy of the degree distribution of a functional brain network
        constructed from fMRI time series. Measures the uniformity and complexity of how
        network connectivity is distributed across brain nodes.

        Parameters
        ----------
        savedata : bool, optional
            If True, save the computed results to a CSV file. If False, return results without
            saving (default is True).
        
        Returns
        -------
        pandas.DataFrame
            DataFrame with degree distribution entropy values per subject. Includes a 'target'
            column with the original fMRI file paths. Returns cached results if savedata=True
            and file exists with overwrite=False.
        """

        filename = f"{inspect.currentframe().f_code.co_name}_{self.results_path}"
        out_path = self.results_path + "/" + filename + ".csv"

        if savedata and not self.overwrite and file_exists(out_path):
            print("File already exists and overwrite is False. Skipping computation.")
            return pd.read_csv(out_path)
        
        inp = df_to_json(self.df_timeseries, matlab_escaped=True)
        varargin = return_varargin(self.keepdata, self.parallel, self.NRU_specific)
        self.engine.workspace["json_in"] = inp
        eval_statement = f"jsonencode(CopBET_degree_distribution_entropy(struct2table(jsondecode(json_in)),{varargin}));"
        json_out = self.engine.eval(eval_statement, nargout=1)
        df_out = json_to_df(json_out)
        df_out["target"] = self.df_filenames["targets"]

        if savedata:
            if file_exists(out_path):
                if self.overwrite:
                    save_dataframe(filename, df_out, self.results_path)
            else:
                save_dataframe(filename, df_out, self.results_path)

        return df_out

    def diversity_coefficient(self, savedata=True):
        """        
        Calculates the diversity coefficient, a measure of how each brain node participates
        in inter-module connectivity relative to within-module connectivity. High values
        indicate hub-like nodes with connections across multiple brain networks.

        Parameters
        ----------
        savedata : bool, optional
            If True, save the computed results to a CSV file. If False, return results without
            saving (default is True).
        
        Returns
        -------
        pandas.DataFrame
            DataFrame with diversity coefficient values per node or region. Includes a 'target'
            column with the original fMRI file paths. Returns cached results if savedata=True
            and file exists with overwrite=False.
        """

        filename = f"{inspect.currentframe().f_code.co_name}_{self.results_path}"
        out_path = self.results_path + "/" + filename + ".csv"

        if savedata and not self.overwrite and file_exists(out_path):
            print("File already exists and overwrite is False. Skipping computation.")
            return pd.read_csv(out_path)
        
        inp = df_to_json(self.df_timeseries, matlab_escaped=True)
        varargin = return_varargin(self.keepdata, self.parallel, self.NRU_specific)
        self.engine.workspace["json_in"] = inp
        eval_statement = f"jsonencode(CopBET_diversity_coefficient(struct2table(jsondecode(json_in)),{varargin}));"
        json_out = self.engine.eval(eval_statement, nargout=1)
        df_out = json_to_df(json_out)
        df_out["target"] = self.df_filenames["targets"]

        if savedata:
            if file_exists(out_path):
                if self.overwrite:
                    save_dataframe(filename, df_out, self.results_path)
            else:
                save_dataframe(filename, df_out, self.results_path)

        return df_out

    def geodesic_entropy(self, savedata=True):
        """        
        Calculates the entropy of geodesic paths in the functional connectivity network,
        measuring the complexity of information routing patterns across brain regions.
        Reflects the stability and predictability of signal propagation paths.

        Parameters
        ----------
        savedata : bool, optional
            If True, save the computed results to a CSV file. If False, return results without
            saving (default is True).
        
        Returns
        -------
        pandas.DataFrame
            DataFrame with geodesic entropy estimates per subject. Includes a 'target' column
            with the original fMRI file paths. Returns cached results if savedata=True and file
            exists with overwrite=False.
        """

        filename = f"{inspect.currentframe().f_code.co_name}_{self.results_path}"
        out_path = self.results_path + "/" + filename + ".csv"

        if savedata and not self.overwrite and file_exists(out_path):
            print("File already exists and overwrite is False. Skipping computation.")
            return pd.read_csv(out_path)
        
        inp = df_to_json(self.df_timeseries, matlab_escaped=True)
        varargin = return_varargin(self.keepdata, self.parallel, self.NRU_specific)
        self.engine.workspace["json_in"] = inp
        eval_statement = f"jsonencode(CopBET_geodesic_entropy(struct2table(jsondecode(json_in)),{varargin}));"
        json_out = self.engine.eval(eval_statement, nargout=1)
        df_out = json_to_df(json_out)
        df_out["target"] = self.df_filenames["targets"]

        if savedata:
            if file_exists(out_path):
                if self.overwrite:
                    save_dataframe(filename, df_out, self.results_path)
            else:
                save_dataframe(filename, df_out, self.results_path)

        return df_out

    def LEiDA_transition_entropy(self, K, savedata=True):
        """        
        Performs LEiDA on fMRI time series to identify dominant network states, then computes
        entropy of transitions between these states. Quantifies the complexity and predictability
        of brain network state dynamics.

        Parameters
        ----------
        K : int
            Number of brain network states (clusters) to identify using LEiDA. Typical values
            range from 2-10 depending on data and research question.
        savedata : bool, optional
            If True, save the computed results to a CSV file. If False, return results without
            saving (default is True).
        
        Returns
        -------
        pandas.DataFrame
            DataFrame with LEiDA transition entropy values per subject. Includes a 'target' column
            with the original fMRI file paths. Returns cached results if savedata=True and file
            exists with overwrite=False.
        """

        filename = f"{inspect.currentframe().f_code.co_name}_{self.results_path}"
        out_path = self.results_path + "/" + filename + ".csv"

        if savedata and not self.overwrite and file_exists(out_path):
            print("File already exists and overwrite is False. Skipping computation.")
            return pd.read_csv(out_path)
        
        inp = df_to_json(self.df_timeseries, matlab_escaped=True)
        varargin = return_varargin(self.keepdata, self.parallel, self.NRU_specific)
        self.engine.workspace["json_in"] = inp
        eval_statement = f"jsonencode(CopBET_LEiDA_transition_entropy(struct2table(jsondecode(json_in)),{str(K)},{varargin}));"
        json_out = self.engine.eval(eval_statement, nargout=1)
        df_out = json_to_df(json_out)
        df_out["target"] = self.df_filenames["targets"]

        if savedata:
            if file_exists(out_path):
                if self.overwrite:
                    save_dataframe(filename, df_out, self.results_path)
            else:
                save_dataframe(filename, df_out, self.results_path)

        return df_out

    def temporal_entropy(self, TR, savedata=True):
        """        
        Calculates entropy measures that capture temporal dynamics and structure in the BOLD
        signal. Takes into account the repetition time (TR) for proper temporal scaling.

        Parameters
        ----------
        TR : float
            Repetition time (TR) of the fMRI acquisition in seconds. Used for proper temporal
            scale calibration of entropy measures.
        savedata : bool, optional
            If True, save the computed results to a CSV file. If False, return results without
            saving (default is True).
        
        Returns
        -------
        pandas.DataFrame
            DataFrame with temporal entropy values per subject. Includes a 'target' column with
            the original fMRI file paths. Returns cached results if savedata=True and file exists
            with overwrite=False.
        """

        filename = f"{inspect.currentframe().f_code.co_name}_{self.results_path}"
        out_path = self.results_path + "/" + filename + ".csv"

        if savedata and not self.overwrite and file_exists(out_path):
            print("File already exists and overwrite is False. Skipping computation.")
            return pd.read_csv(out_path)
        
        inp = df_to_json(self.df_timeseries, matlab_escaped=True)
        varargin = return_varargin(self.keepdata, self.parallel, self.NRU_specific)
        self.engine.workspace["json_in"] = inp
        eval_statement = f"jsonencode(CopBET_temporal_entropy(struct2table(jsondecode(json_in)),{str(TR)},{varargin}));"
        json_out = self.engine.eval(eval_statement, nargout=1)
        df_out = json_to_df(json_out)
        df_out["target"] = self.df_filenames["targets"]

        if savedata:
            if file_exists(out_path):
                if self.overwrite:
                    save_dataframe(filename, df_out, self.results_path)
            else:
                save_dataframe(filename, df_out, self.results_path)

        return df_out

    def DCC_entropy(self, compute, savedata=True):
        """        
        Estimates conditional covariance dynamics and computes entropy from the resulting
        Dynamic Conditional Correlation matrices. Can optionally compute DCC matrices from
        time series or load precomputed values.

        Parameters
        ----------
        compute : bool
            If True, compute DCC matrices from the time series and then entropy. If False,
            assume DCC matrices are precomputed and load them for entropy calculation.
        savedata : bool, optional
            If True, save the computed results to a CSV file. If False, return results without
            saving (default is True).
        
        Returns
        -------
        pandas.DataFrame
            DataFrame with DCC entropy values per subject. Includes a 'target' column with the
            original fMRI file paths. Returns cached results if savedata=True and file exists
            with overwrite=False.
        """

        filename = f"{inspect.currentframe().f_code.co_name}_{self.results_path}"
        out_path = self.results_path + "/" + filename + ".csv"

        if savedata and not self.overwrite and file_exists(out_path):
            print("File already exists and overwrite is False. Skipping computation.")
            return pd.read_csv(out_path)
        
        inp = df_to_json(self.df_timeseries, matlab_escaped=True)
        varargin = return_varargin(self.keepdata, self.parallel, self.NRU_specific)
        compute_str = "true" if compute else "false"
        self.engine.workspace["json_in"] = inp
        eval_statement = f"jsonencode(CopBET_DCC_entropy(struct2table(jsondecode(json_in)),{compute_str},{varargin}));"
        json_out = self.engine.eval(eval_statement, nargout=1)
        df_out = json_to_df(json_out)
        df_out["target"] = self.df_filenames["targets"]

        if savedata:
            if file_exists(out_path):
                if self.overwrite:
                    save_dataframe(filename, df_out, self.results_path)
            else:
                save_dataframe(filename, df_out, self.results_path)

        return df_out

    def intranetwork_synchrony(self, atlas, savedata=True):
        """        
        Calculates synchronization and coherence measures for functional networks defined by
        the provided brain atlas. Quantifies the coupling and coordinated activity within
        each network.

        Parameters
        ----------
        atlas : numpy.ndarray
            A 3D or 4D NumPy array representing the brain atlas with ROI labels or binary masks
            defining network boundaries.
        savedata : bool, optional
            If True, save the computed results to a CSV file. If False, return results without
            saving (default is True).
        
        Returns
        -------
        pandas.DataFrame
            DataFrame with intra-network synchrony metrics per network. Includes a 'target'
            column with the original fMRI file paths. Returns cached results if savedata=True
            and file exists with overwrite=False.
        """

        filename = f"{inspect.currentframe().f_code.co_name}_{self.results_path}"
        out_path = self.results_path + "/" + filename + ".csv"

        if savedata and not self.overwrite and file_exists(out_path):
            print("File already exists and overwrite is False. Skipping computation.")
            return pd.read_csv(out_path)
        
        inp = df_to_json(self.df_filenames, matlab_escaped=True)
        varargin = return_varargin(self.keepdata, self.parallel, self.NRU_specific)
        self.engine.workspace["atlas_synchrony"] = atlas_to_matlab(atlas, True)
        self.engine.workspace["json_in"] = inp
        eval_statement = f"jsonencode(CopBET_intranetwork_synchrony(struct2table(jsondecode(json_in)),atlas_synchrony,{varargin}));"
        json_out = self.engine.eval(eval_statement, nargout=1)
        df_out = json_to_df(json_out)
        df_out["target"] = self.df_filenames["targets"]

        if savedata:
            if file_exists(out_path):
                if self.overwrite:
                    save_dataframe(filename, df_out, self.results_path)
            else:
                save_dataframe(filename, df_out, self.results_path)

        return df_out

    def time_series_complexity(self, LZtype, savedata=True):
        """
        Compute Lempel-Ziv complexity from BOLD time series using CopBET.

        Parameters
        ----------
        engine : matlab.engine.MatlabEngine
            Active MATLAB engine session.
        df : pandas.DataFrame
            BOLD fMRI time series in long or wide table format.
        LZtype : str
            Type of Lempel-Ziv complexity measure to apply (e.g., 'LZ76', 'LZ78').
        keepdata : bool, optional
            Whether to retain intermediate files/data (default is True).
        parallel : bool, optional
            Use MATLAB's parallel toolbox (default is True).
        NRU_specific : bool, optional
            Use NRU lab’s specific parameters and thresholds (default is False).

        Returns
        -------
        pandas.DataFrame
            Complexity metrics computed from each subject’s time series.
        """

        filename = f"{inspect.currentframe().f_code.co_name}_{self.results_path}_{LZtype}"
        out_path = self.results_path + "/" + filename + ".csv"

        if savedata and not self.overwrite and file_exists(out_path):
            print("File already exists and overwrite is False. Skipping computation.")
            return pd.read_csv(out_path)
        
        inp = df_to_json(self.df_timeseries, matlab_escaped=True)
        varargin = return_varargin(self.keepdata, self.parallel, self.NRU_specific)
        self.engine.workspace["json_in"] = inp
        eval_statement = f"jsonencode(CopBET_time_series_complexity(struct2table(jsondecode(json_in)),'{LZtype}',{varargin}));"
        json_out = self.engine.eval(eval_statement, nargout=1)
        df_out = json_to_df(json_out)
        df_out["target"] = self.df_filenames["targets"]

        if savedata:
            if file_exists(out_path):
                if self.overwrite:
                    save_dataframe(filename, df_out, self.results_path)
            else:
                save_dataframe(filename, df_out, self.results_path)

        return df_out

    def NGSC(self, atlas_path, num_workers, savedata=True):
        """
        Computes the Normalized Global Synchronization Coefficient (NGSC) for brain regions
        defined by an atlas. NGSC quantifies global network synchronization by measuring
        the degree to which activity is coordinated across all regions. Results are optionally
        saved to a CSV file.

        Parameters
        ----------
        atlas_path : str
            The file system path to the brain atlas file (typically a NIfTI or similar format)
            that defines the ROI boundaries and labels used for regionalization.
        num_workers : int
            The number of parallel worker processes to use during computation. Higher values
            enable parallel processing for faster computation on multi-core systems.
        savedata : bool, optional
            If True, save the computed NGSC values to a CSV file in the results directory.
            If False, return results without saving (default is True).
        
        Returns
        -------
        pandas.DataFrame
            DataFrame containing NGSC values for each subject/session. Includes a 'target'
            column with the corresponding target values from the input data.
        """
        
        filename = f"{inspect.currentframe().f_code.co_name}_{self.results_path}"
        out_path = self.results_path + "/" + filename + ".csv"

        if savedata and not self.overwrite and file_exists(out_path):
            print("File already exists and overwrite is False. Skipping computation.")
            return pd.read_csv(out_path)
        
        inp = df_to_json(self.df_filenames, matlab_escaped=True)
        varargin = return_varargin(self.keepdata, self.parallel, self.NRU_specific)
        self.engine.workspace["json_in"] = inp
        eval_statement = f"jsonencode(CopBET_NGSC(struct2table(jsondecode(json_in)),'{atlas_path}',{str(num_workers)},{varargin}));"
        json_out = self.engine.eval(eval_statement, nargout=1)
        df_out = json_to_df(json_out)
        df_out["target"] = self.df_filenames["targets"]

        if savedata:
            if file_exists(out_path):
                if self.overwrite:
                    save_dataframe(filename, df_out, self.results_path)
            else:
                save_dataframe(filename, df_out, self.results_path)

        return df_out

    def sample_entropy(self, atlas, compute, savedata=True):
        """        
        Calculates sample entropy, a regularity statistic that quantifies the complexity
        and predictability of BOLD time series within atlas-defined regions. Measures how
        much the signal resists algorithmic compression.

        Parameters
        ----------
        atlas : numpy.ndarray
            A 3D or 4D NumPy array representing the brain atlas with ROI labels or binary masks.
        compute : bool
            If True, compute sample entropy from the time series. If False, attempt to load
            precomputed sample entropy values.
        savedata : bool, optional
            If True, save the computed results to a CSV file. If False, return results without
            saving (default is True).
        
        Returns
        -------
        pandas.DataFrame
            DataFrame with sample entropy values per ROI or subject. Includes a 'target' column
            with the original fMRI file paths. Returns cached results if savedata=True and file
            exists with overwrite=False.
        """

        filename = f"{inspect.currentframe().f_code.co_name}_{self.results_path}"
        out_path = self.results_path + "/" + filename + ".csv"

        if savedata and not self.overwrite and file_exists(out_path):
            print("File already exists and overwrite is False. Skipping computation.")
            return pd.read_csv(out_path)
            
        inp = df_to_json(self.df_filenames, matlab_escaped=True)
        varargin = return_varargin(self.keepdata, self.parallel, self.NRU_specific)
        compute_str = "true" if compute else "false"
        self.engine.workspace["atlas_sample"] = atlas_to_matlab(atlas)
        self.engine.workspace["json_in"] = inp
        eval_statement = f"jsonencode(CopBET_sample_entropy(struct2table(jsondecode(json_in)),atlas_sample,{compute_str},{varargin}));"
        json_out = self.engine.eval(eval_statement, nargout=1)
        df_out = json_to_df(json_out)
        df_out["target"] = self.df_filenames["targets"]

        if savedata:
            if file_exists(out_path):
                if self.overwrite:
                    save_dataframe(filename, df_out, self.results_path)
            else:
                save_dataframe(filename, df_out, self.results_path)

        return df_out

if __name__ == "__main__":
    pass