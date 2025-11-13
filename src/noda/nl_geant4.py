import numpy as np
import os
import re
import uproot
from scipy.interpolate import interp1d
from numba import njit, prange, typed


SPECTRA_PATH = "/junofs/users/sindhu/NODA/noda/data/"
me = 0.51099893
"""
Defines the Geant4Model for calculating particle non-linearity.

This module provides a class that encapsulates the logic for determining the
light yield non-linearity (L/E) for electrons and positrons based on 
pre-computed Geant4 dE/dx simulation data.

The methodology is 99% based on the work of Beatrice Jelmini:
- Thesis: https://www.research.unipd.it/handle/11577/3512960
- Code: https://code.ihep.ac.cn/beatrice.jelmini/detector_response

The positron ('e+') model is implemented as a hybrid model:
- Quenching (L_Q): Calculated directly from the Geant4 e+ dE/dx histograms.
- Cherenkov (C): Calculated using a composite model: 
                  C(e+, Ek) = C(e-, Ek) + 2 * C(gamma, 511 keV)
- Instrumental (f_I): Modeled as f_I(E_dep) = 1.0 - kI * E_dep

- Total Light: L_tot = (L_Q + fC * C) * f_I(E_dep)

OPTIMIZATION:
The core calculation loops in _calculate_total_light have been ported to Numba
to accelerate the computation, as this was identified as the primary bottleneck.
Helper functions are jitted, and the main data is pre-processed into
Numba-compatible typed lists in the constructor.

PICKLE/DEEPCOPY FIX:
Added __getstate__ and __setstate__ methods to make the class compatible with
copy.deepcopy and pickling. Numba-typed objects are excluded from the state
and regenerated during __setstate__.

NUMERICAL STABILITY FIX (v3):
Replaced np.interp with a custom Numba-jitted linear interpolation function
that correctly implements linear extrapolation, mimicking the original
scipy.interp1d(..., fill_value="extrapolate") behavior.

PERFORMANCE FIX (v4):
Added a caching layer to `Geant4Model.eval()`. The function now stores the
final interpolator object and only recalculates it if the physics
parameters (kB, kC, fC, kI) have changed. This dramatically speeds up
sequential calls (e.g., in a visualization loop) from milliseconds
to microseconds.
"""

##############################################################
#               99% based on Bea's work                      #
#   https://www.research.unipd.it/handle/11577/3512960      #
# https://code.ihep.ac.cn/beatrice.jelmini/detector_response #
##############################################################

# A dictionary mapping human-readable histogram names to their numeric IDs
# in the Geant4 output ROOT files.
HISTOS = {
    "total_track_length" : "1",
    "nb_steps" : "2",
    "step_size" : "3",
    "total_edep" : "4",
    "e_charged_secondaries" : "5",
    "e_neutral_secondaries" : "6",
    "NIEL" : "7",
    "e_electrons" : "8",
    "e_positrons" : "9",
    "total_qedep" : "10",
    "nb_charged_secondaries" : "11",
    "nb_neutral_secondaries" : "12",
    "nb_secondary_electrons" : "13",
    "nb_secondaries_positrons" : "14",
    "de_dx_ionization" : "15",
    "de_dx_total" : "16",
    "total_e_tracks" : "17",
    "de_dx_ioni_deltarays" : "18",
    "de_dx_vs_de" : "19",
    "edep_charged" : "20",
    "edep_neutral" : "21",
    "de_dx" : "22"
}

DATA_PATH = os.path.join(SPECTRA_PATH, "geant4")
name_dict = {
    "gamma": "gamma", "e-": "electron", "e+": "positron",
    "n": "neutron", "p": "proton", "alpha": "alpha"
}

# ####################################################################
# # Numba-jitted helper functions for high-performance calculations
# ####################################################################

@njit(cache=True, fastmath=True)
def _interp1d_linear_extrap_nb(x_new, xp, fp):
    """
    Numba-jitted 1D linear interpolation with linear extrapolation.
    Equivalent to scipy.interp1d(..., fill_value="extrapolate").
    Assumes xp is sorted.
    """
    # Find indices using binary search
    idx = np.searchsorted(xp, x_new)
    
    n = len(xp)
    
    # Handle out-of-bounds (extrapolation)
    # Below lower bound
    idx = np.where(idx == 0, 1, idx)
    # Above upper bound
    idx = np.where(idx == n, n - 1, idx)

    # In-bounds mask (for branching, as Numba optimizes this)
    in_bounds = (x_new >= xp[0]) & (x_new <= xp[-1])
    
    # Slopes
    x1 = xp[idx]
    x0 = xp[idx - 1]
    y1 = fp[idx]
    y0 = fp[idx - 1]
    
    dx = x1 - x0
    # Avoid division by zero if xp has duplicates
    slope = np.where(dx == 0, 0.0, (y1 - y0) / dx)
    
    # Standard interpolation
    y_interp = y0 + slope * (x_new - x0)

    # Extrapolation (if needed)
    # Left extrapolation
    y_extrap_left = fp[0] + slope[1] * (x_new - xp[0])
    # Right extrapolation
    y_extrap_right = fp[-1] + slope[-1] * (x_new - xp[-1])
    
    # Combine
    y_out = np.where(in_bounds, y_interp, 
                         np.where(x_new < xp[0], y_extrap_left, y_extrap_right))
    
    return y_out

@njit(cache=True, fastmath=True)
def _calculate_quenched_light_nb(dedx_hist, dedx_centers, kB, kC):
    """
    Calculates the total quenched light yield from a dE/dx histogram. (Numba JIT)
    """
    quenching_factors = 1.0 / (1.0 + kB * dedx_centers + kC * dedx_centers**2)
    return np.sum(dedx_hist * quenching_factors)

@njit(cache=True, fastmath=True)
def _calculate_secondary_cherenkov_nb(sec_e_hist, sec_e_centers, cherenkov_x, cherenkov_y):
    """
    Calculates the Cherenkov light contribution from a secondary e- spectrum. (Numba JIT)
    Uses the custom extrapolation-enabled interpolation function.
    """
    # Get Cherenkov yield C(E) for each energy bin
    cherenkov_per_electron = _interp1d_linear_extrap_nb(sec_e_centers, cherenkov_x, cherenkov_y)
    
    # Total Cherenkov = Sum[ N_e(E) * C(E) ]
    return np.sum(sec_e_hist * cherenkov_per_electron)

@njit(cache=True, fastmath=True)
def _f_instrumental_nb(E_vis, kI):
    """
    Calculates the instrumental non-linearity. (Numba JIT)
    NOTE: This is applied to the visible light (E_vis), not deposited energy.
    """
    return 1.0 - kI * E_vis

@njit(parallel=True, cache=True, fastmath=True)
def _calculate_total_light_standard_nb(
    sim_energies_nb,
    dedx_hists_nb, dedx_centers_nb,
    sec_e_hists_nb, sec_e_centers_nb,
    cherenkov_x, cherenkov_y,
    kB, kC, fC, kI,
    particle_is_e_minus
):
    """
    Calculates the total light yield for standard particles (e.g., e-, gamma)
    using a Numba-parallelized loop.
    """
    n_points = len(sim_energies_nb)
    total_light_yield = np.empty(n_points, dtype=np.float64)

    for i_raw in prange(n_points):
        # Explicitly cast the index to int64 to resolve a
        # NumbaTypeSafetyWarning (uint64 to int64) on some systems.
        i = np.int64(i_raw) 
        
        # Quenching Calculation
        quenched_light = _calculate_quenched_light_nb(
            dedx_hists_nb[i], dedx_centers_nb[i], kB, kC
        )

        # Cherenkov Calculation
        if particle_is_e_minus:
            # C(E) for primary e- (yield includes primary track + deltas)
            # Use custom interp function
            cherenkov_light_arr = _interp1d_linear_extrap_nb(np.array([sim_energies_nb[i]]), cherenkov_x, cherenkov_y)
            cherenkov_light = cherenkov_light_arr[0]
        else: 
            # For gamma, n, etc., light is only from secondaries
            cherenkov_light = _calculate_secondary_cherenkov_nb(
                sec_e_hists_nb[i], sec_e_centers_nb[i], cherenkov_x, cherenkov_y
            )
        
        # Calculate visible light (pre-instrumental effect)
        light_vis = quenched_light + fC * cherenkov_light
        
        # Total Light = E_vis * f_I(E_vis)
        total_light = light_vis * _f_instrumental_nb(light_vis, kI)
        total_light_yield[i] = total_light
    
    return total_light_yield

@njit(parallel=True, cache=True, fastmath=True)
def _calculate_total_light_positron_nb(
    sim_energies_nb,
    dedx_hists_nb, dedx_centers_nb,
    cherenkov_x, cherenkov_y,
    annihilation_cherenkov_nb,
    kB, kC, fC, kI
):
    """
    Calculates the total light yield for the e+ hybrid model
    using a Numba-parallelized loop.
    """
    n_points = len(sim_energies_nb)
    total_light_yield = np.empty(n_points, dtype=np.float64)

    for i_raw in prange(n_points):
        # Explicitly cast the index to int64 to resolve a
        # NumbaTypeSafetyWarning (uint64 to int64) on some systems.
        i = np.int64(i_raw) 
        
        # Quenching: Calculated from the e+ dE/dx histogram
        quenched_light = _calculate_quenched_light_nb(
            dedx_hists_nb[i], dedx_centers_nb[i], kB, kC
        )
        
        # Cherenkov: Composite model
        # C(e+, Ek) = C(e-, Ek) + 2 * C(gamma, 511)
        
        # C(e-, Ek): Light from the primary e+ track (treated as e-)
        # Use custom interp function
        primary_cherenkov_arr = _interp1d_linear_extrap_nb(np.array([sim_energies_nb[i]]), cherenkov_x, cherenkov_y)
        primary_cherenkov = primary_cherenkov_arr[0]
        
        # Total Cherenkov = (Primary Track) + (Annihilation Gammas)
        total_cherenkov = primary_cherenkov + annihilation_cherenkov_nb
        
        # Total Light = (L_Q + fC * C_total) * f_I(E_dep)
        
        # Calculate visible light (pre-instrumental effect)
        light_vis = quenched_light + fC * total_cherenkov
        
        # Total Light = E_vis * f_I(E_vis)
        total_light = light_vis * _f_instrumental_nb(light_vis, kI)
        total_light_yield[i] = total_light
        
    return total_light_yield


class Geant4Model:
    """
    Manages Geant4 non-linearity calculations from pre-computed histograms.

    This class loads dE/dx and secondary electron histograms from Geant4 ROOT files.
    It calculates the light yield non-linearity (L/E) by combining the scintillator's
    quenched light output (Birks' law) with a scalable Cherenkov light contribution
    and an instrumental non-linearity.
    
    The positron ('e+') model is implemented as a hybrid model:
    - Quenching (L_Q): Calculated directly from the Geant4 e+ dE/dx histograms.
    - Cherenkov (C): Calculated using a composite model: 
                      C(e+, Ek) = C(e-, Ek) + 2 * C(gamma, 511 keV)
    - Instrumental (f_I): Modeled as f_I(E_dep) = 1.0 - kI * E_dep
    - Total Light: L_tot = (L_Q + fC * C) * f_I(E_dep)
    """

    def __init__(self, particle, data_base_path=None):
        """
        Initializes the model by loading simulation data and creating the
        Cherenkov light interpolation function.

        Args:
            particle (str): The primary particle type (e.g., 'e-', 'e+', 'gamma').
            data_base_path (str, optional): The base path to the Geant4 data directory.
        """

        self.particle = particle
        self.base_path = data_base_path or DATA_PATH
        self.nevents = 1000000  # TODO: This should not be hardcoded
        
        # Initialize the cache for the 'eval' function
        self._cached_params = None
        self._cached_interpolator = None

        # Create the Cherenkov Light Interpolation Data (Numba-compatible) ---
        
        # DYB
        cherenkov_energies = np.array([0., 0.19985692, 0.240201179, 0.25962539, 0.275677909, 0.294745057, 0.218886576, 0.320458404, 0.341831426, 0.394354839, 0.361602497, 0.439569688, 0.47528616, 0.510828564, 0.558324662, 0.61300078, 0.672312284, 0.744194644, 0.839337088, 0.94066077, 1.050728408, 1.176118626, 1.296305931, 1.416493236, 1.536680541, 1.656867846, 1.777055151, 1.897242456, 2.017429761, 2.137617066, 2.25780437, 2.377991675, 2.49817898, 2.618366285, 2.73855359, 2.858740895, 2.9789282, 3.099115505, 3.21930281, 3.339490114, 3.459677419, 3.579864724, 3.700052029, 3.820239334, 3.940426639, 4.060613944, 4.180801249, 4.300988554, 4.421175858, 4.541363163, 4.661550468, 4.781737773, 4.901925078, 5.022112383, 5.142299688, 5.262486993, 5.382674298, 5.502861602, 5.623048907, 5.743236212, 5.863423517, 5.983610822, 6.103798127, 6.223985432, 6.344172737, 6.464360042, 6.584547347, 6.704734651, 6.824921956, 6.945109261, 7.065296566, 7.185483871, 7.305671176, 7.425858481, 7.546045786, 7.666233091, 7.786420395, 7.9066077, 8.026795005, 8.14698231, 8.267169615, 8.38735692, 8.507544225, 8.62773153, 8.747918835, 8.868106139, 8.988293444, 9.108480749, 9.228668054, 9.348855359, 9.469042664, 9.589229969, 9.709417274, 9.829604579, 9.938865765])
        cherenkov_yield = np.array([0., 8.13249E-05, 0.000766873, 0.001266486, 0.001860128, 0.002543045, 0.000367906, 0.003304245, 0.00409492, 0.006595886, 0.005074881, 0.008956401, 0.011021791, 0.013341224, 0.016187772, 0.020080121, 0.024146027, 0.02964739, 0.037093932, 0.045053896, 0.054738379, 0.06479271, 0.075101867, 0.086904814, 0.096754722, 0.107409243, 0.118162999, 0.128981544, 0.140313682, 0.152043227, 0.16145597, 0.171169609, 0.183260589, 0.194522115, 0.207152101, 0.218709268, 0.229491038, 0.240804321, 0.251948907, 0.262635302, 0.272987892, 0.28339867, 0.292763746, 0.306307051, 0.318381099, 0.329166401, 0.339199155, 0.350978267, 0.361973458, 0.373006148, 0.383085462, 0.393357009, 0.404043213, 0.416813696, 0.428221838, 0.43862389, 0.452728171, 0.461944607, 0.474655834, 0.485116096, 0.491957144, 0.503295238, 0.513456617, 0.524556166, 0.535895658, 0.545906342, 0.552467362, 0.562093636, 0.576758076, 0.588822268, 0.599574988, 0.605285505, 0.614819785, 0.627336009, 0.642732548, 0.648320597, 0.658261963, 0.674972496, 0.686450931, 0.69755054, 0.701592383, 0.712057699, 0.722679121, 0.739198546, 0.750842186, 0.760789493, 0.772137824, 0.782367261, 0.793384569, 0.805881731, 0.814545656, 0.823980232, 0.8301192, 0.840770916, 0.850634188])

        # Sort arrays by energy to ensure monotonicity for interpolation
        sort_idx = np.argsort(cherenkov_energies)
        self._cherenkov_x = cherenkov_energies[sort_idx].astype(np.float64)
        self._cherenkov_y = cherenkov_yield[sort_idx].astype(np.float64)

        # Load Geant4 simulation data
        if self.particle == 'e+':
            # For hybrid e+ model, we need BOTH e+ data and gamma data
            self.particle_sim_data = self._load_data_for_particle('e+') 
            self.gamma_annihilation_data = self._load_annihilation_gamma_data()
        else:
            # Standard loading for e-, gamma, n, etc.
            self.particle_sim_data = self._load_data_for_particle(self.particle)

        # Unpack simulation data into Numba-compatible structures
        # This is the logic that needs to be mirrored in __setstate__
        self._create_numba_structs()

    def _create_numba_structs(self):
        """Helper function to create Numba-typed lists from raw sim data."""
        # Check if raw data exists
        if not hasattr(self, 'particle_sim_data'):
            return
            
        self.sim_energies_nb = np.array([d['energy_mev'] for d in self.particle_sim_data], dtype=np.float64)
        
        # Use typed.List for ragged arrays (histograms of different lengths)
        self.dedx_hists_nb = typed.List(np.ascontiguousarray(d['dedx_hist'], dtype=np.float64) for d in self.particle_sim_data)
        self.dedx_centers_nb = typed.List(np.ascontiguousarray(d['dedx_centers'], dtype=np.float64) for d in self.particle_sim_data)
        self.sec_e_hists_nb = typed.List(np.ascontiguousarray(d['sec_e_hist'], dtype=np.float64) for d in self.particle_sim_data)
        self.sec_e_centers_nb = typed.List(np.ascontiguousarray(d['sec_e_centers'], dtype=np.float64) for d in self.particle_sim_data)

        if self.particle == 'e+':
            if not hasattr(self, 'gamma_annihilation_data'):
                return # Avoid error during pickling/unpickling
            g = self.gamma_annihilation_data
            gamma_sec_e_hist_nb = np.ascontiguousarray(g['sec_e_hist'], dtype=np.float64)
            gamma_sec_e_centers_nb = np.ascontiguousarray(g['sec_e_centers'], dtype=np.float64)
            
            self.annihilation_cherenkov_nb = 2.0 * _calculate_secondary_cherenkov_nb(
                gamma_sec_e_hist_nb, gamma_sec_e_centers_nb,
                self._cherenkov_x, self._cherenkov_y
            )
            
        # When we create new numba structs, any old cache is invalid.
        self._cached_params = None
        self._cached_interpolator = None

    def __getstate__(self):
        """
        Prepare the object's state for pickling/deepcopy.
        This method removes the un-picklable Numba objects and the cache.
        """
        state = self.__dict__.copy()
        # Remove the Numba-typed objects that cannot be pickled
        state.pop('sim_energies_nb', None)
        state.pop('dedx_hists_nb', None)
        state.pop('dedx_centers_nb', None)
        state.pop('sec_e_hists_nb', None)
        state.pop('sec_e_centers_nb', None)
        state.pop('annihilation_cherenkov_nb', None)
        
        # Remove cache objects
        state.pop('_cached_params', None)
        state.pop('_cached_interpolator', None)
        
        return state

    def __setstate__(self, state):
        """
        Restore the object's state from a pickled/deepcopied state.
        This method re-creates the Numba objects from the restored data.
        """
        # Restore the picklable state
        self.__dict__.update(state)
        
        # Re-initialize the cache
        self._cached_params = None
        self._cached_interpolator = None
        
        # Re-create the Numba-typed lists and other derived attributes
        self._create_numba_structs()


    def _load_data_for_particle(self, p_type):
        """
        Loads and processes all dE/dx and secondary electron histograms for a particle type.
        """
        p_name = name_dict.get(p_type)
        if not p_name:
            raise ValueError(f"Particle type '{p_type}' not found in name_dict.")
            
        path_to_data = os.path.join(self.base_path, f"{p_name}s")
        if not os.path.isdir(path_to_data):
            raise FileNotFoundError(f"Data directory not found: {path_to_data}")
        
        file_pattern = re.compile(rf"run_{p_name}_(\d+\.?\d*)keV_{self.nevents}evt.*\.root")
        
        found_files = []
        for fname in os.listdir(path_to_data):
            match = file_pattern.match(fname)
            if match:
                energy_kev = float(match.group(1))
                found_files.append({'energy_mev': energy_kev / 1000.0, 'path': os.path.join(path_to_data, fname)})

        if not found_files:
            raise FileNotFoundError(f"No simulation files found for particle '{p_name}' in '{path_to_data}'")

        found_files.sort(key=lambda f: f['energy_mev'])
        
        simulation_data = []
        for f_info in found_files:
            with uproot.open(f_info['path']) as file:
                hist_dedx = file[HISTOS["de_dx"]]
                hist_sec_e = file[HISTOS["e_electrons"]]
                
                sim_point = {
                    'energy_mev': f_info['energy_mev'],
                    'dedx_hist': hist_dedx.values() / self.nevents,
                    'dedx_centers': hist_dedx.axis(0).centers(),
                    'sec_e_hist': hist_sec_e.values() / self.nevents,
                    'sec_e_centers': hist_sec_e.axis(0).centers()
                }
                simulation_data.append(sim_point)
                
        return simulation_data

    def _load_annihilation_gamma_data(self):
        """
        Loads the specific 511 keV gamma simulation file for annihilation.
        """
        p_name = name_dict.get("gamma")
        path_to_data = os.path.join(self.base_path, f"{p_name}s")
        
        # Geant4 file pattern is in keV
        fname_pattern = re.compile(rf"run_gamma_511keV_{self.nevents}evt.*\.root") 
        
        found_file = None
        for fname in os.listdir(path_to_data):
            if fname_pattern.match(fname):
                found_file = os.path.join(path_to_data, fname)
                break
        
        if not found_file:
            raise FileNotFoundError(f"Annihilation gamma file (511keV) not found in {path_to_data}")

        with uproot.open(found_file) as file:
            hist_dedx = file[HISTOS["de_dx"]]
            hist_sec_e = file[HISTOS["e_electrons"]]
            
            gamma_data = {
                'energy_mev': 0.511,
                'dedx_hist': hist_dedx.values() / self.nevents,
                'dedx_centers': hist_dedx.axis(0).centers(),
                'sec_e_hist': hist_sec_e.values() / self.nevents,
                'sec_e_centers': hist_sec_e.axis(0).centers()
            }
        return gamma_data

    def _calculate_total_light(self, kB, kC, fC, kI):
        """
        Calculates the total light yield by dispatching to the appropriate
        Numba-jitted function based on the particle type.
        """
        
        if self.particle == 'e+':
            return _calculate_total_light_positron_nb(
                self.sim_energies_nb,
                self.dedx_hists_nb, self.dedx_centers_nb,
                self._cherenkov_x, self._cherenkov_y,
                self.annihilation_cherenkov_nb,
                kB, kC, fC, kI
            )
        else:
            particle_is_e_minus = (self.particle == 'e-')
            return _calculate_total_light_standard_nb(
                self.sim_energies_nb,
                self.dedx_hists_nb, self.dedx_centers_nb,
                self.sec_e_hists_nb, self.sec_e_centers_nb,
                self._cherenkov_x, self._cherenkov_y,
                kB, kC, fC, kI,
                particle_is_e_minus
            )

    def eval(self, E_dep, kB, kC=0.0, fC=0.0, kI=0.0, S=1.0):
        """
        Evaluates the raw, un-anchored non-linearity (L/E) for the specified particle.

        This function is now fast, as the expensive loop is JIT-compiled
        and the final interpolator is cached.

        Args:
            E_dep (float or np.ndarray): Deposited energy/energies in MeV.
            kB (float): Birks' constant for quenching.
            kC (float, optional): Second-order Birks' constant. Defaults to 0.0.
            fC (float, optional): Overall scaling factor for the Cherenkov light
                                    contribution. Defaults to 0.0.
            kI (float, optional): Instrumental non-linearity slope. Defaults to 0.0.
            S (float, optional): Overall light yield scale factor. Defaults to 1.0.
                                  (This parameter is accepted but not used here;
                                  scaling is applied in the NonLinearity class).

        Returns:
            float or np.ndarray: The calculated non-linearity (L/E).
        """
        is_scalar = np.isscalar(E_dep)
        E_dep_arr = np.atleast_1d(E_dep)
        
        # Check if the current parameters match the cached ones
        # LY is *not* part of the cache key, as it's applied by the wrapper class.
        current_params = (kB, kC, fC, kI)
        if current_params == self._cached_params and self._cached_interpolator is not None:
            # If they match, use the cached interpolator and return immediately
            nl_values = self._cached_interpolator(E_dep_arr)
            
            if self.particle == 'e+':
                # Apply hot fix for orsa_fitter
                nl_values[E_dep_arr < 2 * me] = self._cached_interpolator(2 * me)
                
            return nl_values[0] if is_scalar else nl_values

        # If no cache match, perform the expensive calculation
        
        # Calculate total light yield at each simulated energy point (NOW FAST)
        particle_l_total = self._calculate_total_light(kB, kC, fC, kI)
        
        # sim_energies are the KINETIC energies
        sim_energies = self.sim_energies_nb
        
        # For positrons, the deposited energy includes the annihilation energy
        if self.particle == 'e+':
            # E_dep = Kinetic Energy + Annihilation Energy
            particle_E_dep_sim = sim_energies + 2 * me
        else:
            particle_E_dep_sim = sim_energies
        
        # Calculate Raw Non-Linearity (L/E) at Simulated Energies
        particle_nl_sim = np.divide(
            particle_l_total, 
            particle_E_dep_sim, 
            out=np.zeros_like(particle_E_dep_sim), 
            where=particle_E_dep_sim != 0
        )
        
        # Interpolate the final L/E curve to the requested energy/energies
        kind = "linear" if self.particle == "n" else "quadratic"
        
        # Ensure data is sorted for interpolation
        sort_idx = np.argsort(particle_E_dep_sim)
        particle_E_dep_sim_sorted = particle_E_dep_sim[sort_idx]
        particle_nl_sim_sorted = particle_nl_sim[sort_idx]

        # Use scipy.interp1d as it's more robust than np.interp for this
        # final interpolation step (which is not in a hot loop).
        final_interp = interp1d(
            particle_E_dep_sim_sorted, 
            particle_nl_sim_sorted, 
            kind=kind, 
            fill_value="extrapolate"
        )
        
        # Store the newly created interpolator and parameters in the cache
        self._cached_params = current_params
        self._cached_interpolator = final_interp

        # Evaluate the new interpolator at the requested energies
        nl_values = final_interp(E_dep_arr)
        
        if self.particle == 'e+':
            # Set NL for energies below annihilation threshold
            # to the value *at* the threshold (hot fix for orsa_fitter)
            nl_values[E_dep_arr < 2 * me] = final_interp(2 * me) 
        
        return S*nl_values[0] if is_scalar else S*nl_values
