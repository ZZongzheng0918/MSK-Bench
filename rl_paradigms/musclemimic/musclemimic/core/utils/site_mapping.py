"""
Site index mapping utilities for environments with reduced trajectory storage.

This module provides utilities to map between model site IDs and trajectory site indices
for environments that store only mimic sites in trajectory data to save memory.
"""

import mujoco
import numpy as np
from mujoco import MjModel
from mujoco.mjx import Model
from typing import Dict, List, Optional, Tuple, Union


class BaseSiteMapper:
    """
    Base class for environment-specific site mappers.
    
    Handles mapping between model site IDs and trajectory site indices
    for environments that store only mimic sites in trajectory data to save memory.
    """
    
    
    def __init__(self, model: Union[MjModel, Model], env_class_name: str,
                 env_sites_for_mimic: Optional[List[str]] = None,
                 trajectory_site_names: Optional[List[str]] = None):
        """
        Initialize the site mapper.
        
        Args:
            model: MuJoCo model containing site definitions
            env_class_name: Environment class name to determine if mapping is needed
            env_sites_for_mimic: Environment's sites_for_mimic list (optional)
        """
        self.model = model
        self.env_class_name = env_class_name
        self.env_sites_for_mimic = env_sites_for_mimic or []
        self.requires_mapping = self._determine_mapping_requirement()
        
        # Create mapping dictionaries
        self._site_name_to_model_id = {}
        self._model_id_to_traj_index = {}
        self._traj_index_to_model_id = {}
        self._trajectory_sites: List[str] = []
        
        if self.requires_mapping:
            # Prefer actual trajectory site order if provided
            if trajectory_site_names:
                self._trajectory_sites = list(trajectory_site_names)
            else:
                # Fallback to environment configuration ordered by model site ID
                self._trajectory_sites = self._get_trajectory_sites_ordered()
            self._build_mappings()
    
    def _determine_mapping_requirement(self) -> bool:
        """Determine if this environment requires site mapping. Override in subclasses."""
        return False
    
    def _get_trajectory_sites_ordered(self) -> List[str]:
        """Get trajectory site order by sorting environment's sites_for_mimic by model site ID.
        
        This replicates the exact logic from loco_mujoco/trajectory/handler.py:
        Sites are ordered by ascending model site ID to match retargeting pipeline.
        """
        if not self.env_sites_for_mimic:
            return []
            
        # Get site IDs for environment's mimic sites
        site_id_pairs = []
        for site_name in self.env_sites_for_mimic:
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if site_id >= 0:  # Valid site ID
                site_id_pairs.append((site_name, site_id))
        
        # Sort by model site ID (ascending) - this matches trajectory handler logic  
        site_id_pairs.sort(key=lambda x: x[1])
        return [site_name for site_name, _ in site_id_pairs]
    
    def _build_mappings(self):
        """Build the mapping dictionaries between model IDs and trajectory indices."""
        # Clear any previous mappings
        self._site_name_to_model_id.clear()
        self._model_id_to_traj_index.clear()
        self._traj_index_to_model_id.clear()

        # First, build site name to model ID mapping
        for site_id in range(self.model.nsite):
            site_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, site_id)
            if site_name:
                self._site_name_to_model_id[site_name] = site_id
        
        # Build model ID to trajectory index mapping for mimic sites
        for traj_idx, site_name in enumerate(self._trajectory_sites):
            if site_name in self._site_name_to_model_id:
                model_id = self._site_name_to_model_id[site_name]
                self._model_id_to_traj_index[model_id] = traj_idx
                self._traj_index_to_model_id[traj_idx] = model_id

    def attach_trajectory_sites(self, trajectory_site_names: List[str]):
        """
        Attach the actual trajectory site order and rebuild mappings.

        This should be called once the trajectory is loaded so the mapper matches
        the trajectory data layout even when only a subset of mimic sites is used.
        """
        if not self.requires_mapping:
            return
        if not trajectory_site_names:
            return
        self._trajectory_sites = list(trajectory_site_names)
        self._build_mappings()
    
    def model_ids_to_traj_indices(self, model_site_ids: Union[np.ndarray, List[int]]) -> np.ndarray:
        """
        Convert model site IDs to trajectory site indices.
        
        Args:
            model_site_ids: Array or list of model site IDs
            
        Returns:
            Array of trajectory site indices
            
        Raises:
            ValueError: If any model site ID is not found in trajectory sites
        """
        if not self.requires_mapping:
            # For environments that don't need mapping, return model IDs as-is
            return np.array(model_site_ids)
        
        model_site_ids = np.asarray(model_site_ids)
        traj_indices = []
        
        for model_id in model_site_ids:
            if model_id in self._model_id_to_traj_index:
                traj_indices.append(self._model_id_to_traj_index[model_id])
            else:
                # Get site name for error message
                site_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, model_id)
                raise ValueError(
                    f"Model site ID {model_id} ('{site_name}') not found in trajectory sites. "
                    f"Env={self.env_class_name}. Trajectory sites: {self._trajectory_sites[:10]}..." \
                    if len(self._trajectory_sites) > 10 else \
                    f"Model site ID {model_id} ('{site_name}') not found in trajectory sites. "
                    f"Env={self.env_class_name}. Trajectory sites: {self._trajectory_sites}"
                )
        
        return np.array(traj_indices)
    
    def traj_indices_to_model_ids(self, traj_indices: Union[np.ndarray, List[int]]) -> np.ndarray:
        """
        Convert trajectory site indices to model site IDs.
        
        Args:
            traj_indices: Array or list of trajectory site indices
            
        Returns:
            Array of model site IDs
            
        Raises:
            ValueError: If any trajectory index is invalid
        """
        if not self.requires_mapping:
            # For environments that don't need mapping, return indices as-is  
            return np.array(traj_indices)
        
        traj_indices = np.asarray(traj_indices)
        model_ids = []
        
        for traj_idx in traj_indices:
            if traj_idx in self._traj_index_to_model_id:
                model_ids.append(self._traj_index_to_model_id[traj_idx])
            else:
                raise ValueError(
                    f"Trajectory index {traj_idx} is invalid. "
                    f"Valid range: 0-{len(self._trajectory_sites)-1} for sites: {self._trajectory_sites}"
                )
        
        return np.array(model_ids)
    
    def validate_site_access(self, model_site_ids: Union[np.ndarray, List[int]], 
                           data_shape: Tuple[int, ...]) -> bool:
        """
        Validate that site access will not cause out-of-bounds errors.
        
        Args:
            model_site_ids: Model site IDs to check
            data_shape: Shape of the data being accessed (e.g., trajectory site data)
            
        Returns:
            True if access is valid, False otherwise
        """
        if not self.requires_mapping:
            # For environments that don't need mapping, check direct access
            max_id = np.max(model_site_ids) if len(model_site_ids) > 0 else -1
            return max_id < data_shape[0]

        # Mapping required: ensure all model IDs exist in mapping
        model_site_ids = np.asarray(model_site_ids)
        if model_site_ids.size == 0:
            return True
        if any((mid not in self._model_id_to_traj_index) for mid in model_site_ids):
            return False
        traj_indices = np.array([self._model_id_to_traj_index[mid] for mid in model_site_ids])
        max_idx = np.max(traj_indices) if traj_indices.size > 0 else -1
        return max_idx < data_shape[0]
    
    def get_site_info(self) -> Dict[str, any]:
        """
        Get debugging information about the site mapping.
        
        Returns:
            Dictionary with mapping information
        """
        trajectory_sites = self._trajectory_sites
        return {
            "env_class": self.env_class_name,
            "requires_mapping": self.requires_mapping,
            "trajectory_sites": trajectory_sites,
            "model_to_traj_mapping": self._model_id_to_traj_index.copy(),
            "total_model_sites": self.model.nsite,
            "trajectory_site_count": len(trajectory_sites)
        }


class BimanualSiteMapper(BaseSiteMapper):
    """
    Maps MyoBimanualArm model site IDs to trajectory site indices.

    The mapper uses `traj.info.site_names` when available. If trajectory site
    names are unavailable, it falls back to `sites_for_mimic` ordered by model
    site ID.
    """
    
    def _determine_mapping_requirement(self) -> bool:
        """MyoBimanualArm environments require site mapping."""
        return "MyoBimanualArm" in self.env_class_name


class MyoFullBodySiteMapper(BaseSiteMapper):
    """
    Maps MyoFullBody model site IDs to trajectory site indices.

    The mapper uses `traj.info.site_names` when available. If trajectory site
    names are unavailable, it falls back to `sites_for_mimic` ordered by model
    site ID.
    """
    
    
    def _determine_mapping_requirement(self) -> bool:
        """MyoFullBody environments require site mapping."""
        return "MyoFullBody" in self.env_class_name


class NoOpSiteMapper(BaseSiteMapper):
    """
    No-operation site mapper for environments that don't need site mapping.
    Delegates all behavior to BaseSiteMapper with requires_mapping=False.
    """

    def _determine_mapping_requirement(self) -> bool:
        return False


def create_site_mapper(model: Union[MjModel, Model], env_class_name: str,
                       env_sites_for_mimic: Optional[List[str]] = None,
                       trajectory_site_names: Optional[List[str]] = None) -> BaseSiteMapper:
    """
    Factory function to create appropriate site mapper for environment.
    
    Args:
        model: MuJoCo model
        env_class_name: Environment class name
        env_sites_for_mimic: Environment's sites_for_mimic list (optional)
        
    Returns:
        BaseSiteMapper instance (BimanualSiteMapper, MyoFullBodySiteMapper, or NoOpSiteMapper)
    """
    if "MyoBimanualArm" in env_class_name:
        return BimanualSiteMapper(model, env_class_name, env_sites_for_mimic, trajectory_site_names)
    elif "MyoFullBody" in env_class_name:
        return MyoFullBodySiteMapper(model, env_class_name, env_sites_for_mimic, trajectory_site_names)
    else:
        return NoOpSiteMapper(model, env_class_name, env_sites_for_mimic, trajectory_site_names)


def safe_trajectory_site_access(trajectory_data, model_site_ids: np.ndarray, 
                               site_mapper: BaseSiteMapper, 
                               field_name: str = "site_xpos") -> np.ndarray:
    """
    Safely access trajectory site data using proper index mapping.
    
    Args:
        trajectory_data: Trajectory data object with site information
        model_site_ids: Model site IDs to access
        site_mapper: Site mapper for index conversion
        field_name: Name of site field to access ("site_xpos", "site_xmat", etc.)
        
    Returns:
        Site data array with proper indexing
        
    Raises:
        AttributeError: If field_name doesn't exist in trajectory_data
        ValueError: If site mapping fails
    """
    if not hasattr(trajectory_data, field_name):
        raise AttributeError(f"Trajectory data does not have field: {field_name}")
    
    site_data = getattr(trajectory_data, field_name)
    
    if site_mapper.requires_mapping:
        # Use trajectory indices for memory-optimized environments (MyoBimanualArm, MyoFullBody)
        traj_indices = site_mapper.model_ids_to_traj_indices(model_site_ids)
        return site_data[traj_indices]
    else:
        # Use model IDs directly for other environments
        return site_data[model_site_ids]
