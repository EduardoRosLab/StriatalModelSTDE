# Example execution:
# python3 launch_masquelier_simulation.py seed=0 ...


# Importing libraries and defining classes


import os
import sys
from datetime import datetime
import time
import portalocker

# from pyedlut_STDE_hw import simulation_wrapper as pyedlut
from edlut import simulation_wrapper as pyedlut
import matplotlib.pyplot as plt
plt.rcParams.update({
    "figure.facecolor":  (1.0, 1.0, 1.0, 1.0),
    "axes.facecolor":    (1.0, 1.0, 1.0, 1.0),
    "savefig.facecolor": (1.0, 1.0, 1.0, 1.0),
})
import matplotlib.transforms as mtransforms
import matplotlib.patches as mpatches
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from scipy import interpolate

import pickle
import itertools


# Support objects


def moving_average(a, n=100) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def smooth_spikes(t,dt, spike_times, sigma_s=200.0, resolution=1000):
    sub = (spike_times>=t) * (spike_times<=(t+dt))
    spike_times = spike_times[sub]
    hist, times = np.histogram(spike_times, bins=resolution, range=(t,t+dt))
    signal = gaussian_filter1d(hist.astype('float'), sigma_s/dt)*resolution/dt
    return times[:-1], signal


class SimulationIterator:
    def __init__(self, stop_time, inputs):
        self.stop_time = stop_time
        self.inputs = inputs
        self.iteration = 0
        self.time_interval = [0.0, self.inputs[0]]

        times = np.cumsum(self.inputs)
        self._data_len = np.where(times>=self.stop_time)[0]
        if self._data_len.size==0:
            self._data_len = self.inputs.size-1
        else:
            self._data_len = self._data_len[0]
    def __iter__(self):
        return self
    def __next__(self):
        if self.time_interval[1]<self.stop_time and self.iteration<self.inputs.size:
            return_iteration = self.iteration
            return_time_interval = self.time_interval[:]

            self.iteration += 1
            self.time_interval[0] = self.time_interval[1]
            self.time_interval[1] += self.inputs[self.iteration]

            return return_iteration, return_time_interval
        raise StopIteration
    def __len__(self):
        return self._data_len

class EdlutUtils:
    def __init__(self):
        # Needed attributes for this class to be used
        self.simulation = None
        self.default_synapse_params = None
    def create_neuron_layer(self, number, params=None, model_name='LIFTimeDrivenModel', show=True, log_activity=False):
        if params is None: params = self.default_neuron_params
        return self.simulation.AddNeuronLayer(
            num_neurons = number,
            model_name = model_name,
            param_dict = params,
            log_activity = log_activity,
            output_activity = show
        )
    def create_noisy_layer(self, number, params=None, show=True):
        if params is None: params={}
        return self.simulation.AddNeuronLayer(
            num_neurons = number,
            model_name = 'PoissonGeneratorDeviceVector',
            param_dict = {'frequency': 1.0, **params},
            log_activity=False,
            output_activity=show)
    def connect_layers(self, a, b, params=None, random_weights=0.1):
        if params is None: params={}
        conns = None
        d = {**self.default_synapse_params, **params}

        if random_weights <= 0:
            conns = self.simulation.AddSynapticLayer(a, b, d)
        else:
            weights = (np.random.rand(len(a)) - 0.5) * 2.0
            weights = weights * d['weight'] * random_weights
            weights = weights + d['weight']
            conns = self.simulation.AddSynapticLayer(a, b, {**d, 'weight': weights.tolist()})
        return conns
    def connect_layers_all2all(self, a, b, params=None, autapses=False, random_weights=0.0):
        if params is None: params={}
        sources = []
        targets = []
        for src in a:
            for tgt in b:
                if src!=tgt or autapses is True:
                    sources.append(src)
                    targets.append(tgt)
        return self.connect_layers(sources, targets, params, random_weights)
    def connect_layers_fanin(self, a, b, params=None, fanin=50, autapses=False, random_weights=0.0):
        if params is None: params={}
        sources = []
        targets = []
        for tgt in b:
            if not autapses and tgt in a:
                _a = np.random.choice([n for n in a if n!=tgt], fanin, replace=False)
            else:
                _a = np.random.choice(a, fanin, replace=False)
            for src in _a:
                sources.append(src)
                targets.append(tgt)
        return self.connect_layers(sources, targets, params, random_weights)
    def connect_layers_fanout(self, a, b, params=None, fanout=50, autapses=False, random_weights=0.0):
        if params is None: params={}
        sources = []
        targets = []

        for src in a:
            if not autapses and src in b:
                _b = np.random.choice([n for n in b if n!=src], fanout, replace=False)
            else:
                _b = np.random.choice(b, fanout, replace=False)
            for tgt in _b:
                sources.append(src)
                targets.append(tgt)
        return self.connect_layers(sources, targets, params, random_weights)
    def set_status(self, neurons, status):
        if type(neurons) is int:
            self.simulation.SetSpecificNeuronParams(neurons, status)
        elif type(neurons) is list or type(neurons) is np.ndarray:
            for n in neurons:
                self.simulation.SetSpecificNeuronParams(n, status)
        else:
            raise RuntimeError()


# Network class

class MasquelierNetSin(EdlutUtils):
    def __init__(self):
        self._init_consts()
    def _init_consts(self):
        self.seed = 0
        self.stop_simulation_at = 1000.0

        self.n_inputs = 2000
        self.n_neurons_input_pattern = 1000

        self.min_input_current_factor = 0.87
        self.max_input_current_factor = 1.1
        self.current_freq = 8.0

        self.max_weight = 0.015 #0.075
        self.d1_initial_weight = None
        self.d2_initial_weight = None

        self.reward_delay = 2e-1 #0.1
        self.tau_eli = None
        self.tau_dop = 0.02
        self.dopa_basal_current = 16.0

        self.learning_rate = 0.002
        self.syn_pre_inc = None

        self.tau_thr = 10.0 #20.0
        self.tar_fir_rat = 0.2 #0.1

        self.reward_weights = [2.0,4.0]
        self.start_punishment_at = 0.0
        self.start_rewards_at = 0.0

        self.pattern_times_proportion = 0.8
        self.min_duration_pattern = 1e-1
        self.max_duration_pattern = 5e-1

        self.n_outputs = 16

        self.lateral_intra = True
        self.lateral_inter = True

        self.msn_d1_to_action = 300.0
        self.msn_d2_to_action = 300.0

        # The learning protocol is defined as a string that specifies the reward contingencies for
        # each action in response to different input patterns. This string implicitly determines the
        # number of input patterns and actions.
        #
        # Format: Each word in the string represents an input pattern. Within each word, the position 
        # of a letter corresponds to an action, and the letter itself (H, 0, L) indicates the type of 
        # reinforcement (High, None, Low) for that action. Words are separated by underscores ('_').
        #
        # Example:
        # 'H0L_L0H'
        # - Two words, separated by '_', represent two distinct input patterns.
        # - In the first pattern, the first action (H) is highly rewarded, the second action (0) 
        #   has no reinforcement, and the third action (L) is punished.
        # - In the second pattern, the first action is punished, the second has no reinforcement,
        #   and the third is highly rewarded.
        #
        # Another example (as used in the paper):
        # 'HL_HL_LH_LH_LL'
        # - Five words indicate five different input patterns.
        # - For the first two patterns, choosing action 1 (H) leads to a reward, while action 2 (L) 
        #   results in punishment.
        # - The next two patterns work oppositely: action 2 rewards, and action 1 punishes.
        # - The final pattern imposes a punishment for any action chosen.
        self.learning_protocol = 'HL_HL_LH_LH_LL'

        self.tau_plu = 0.032
        self.tau_min = 0.032

        # MSN D1
        self.stde_kernel_d1 = {
            'k_plu_hig':  1.0,
            'k_min_hig': -1.0,

            'k_plu_low': -1.0,
            'k_min_low': 0.0,
        }

        # MSN D2
        self.stde_kernel_d2 = {
            'k_plu_hig': -1.0,
            'k_min_hig': 0.0,

            'k_plu_low':  1.0,
            'k_min_low': -1.0,
        }

        self.default_neuron_params = {
            'c_m': 250.0,
            'e_exc': 0.0,
            'e_inh': -85.0,
            'e_leak': -65.0,
            'g_leak': 25.0,
            'tau_exc': 5.0,
            'tau_inh': 10.0,
            'tau_nmda': 20.0,
            'tau_ref': 1.0,
            'v_thr': -40.0}
        self.output_neuron_params = {
            **self.default_neuron_params,
            'c_m': 50.0,
            'g_leak': 10.0,
            'v_thr': -50.0,
            'tau_inh': 30.0,
            'tau_ref': 1000.0/65.0,
        }
        self.action_neuron_params = {
            **self.default_neuron_params,
            'c_m': 100.0,
            'tau_inh': 60.0, #~1000.0/16.0,
            'tau_ref': 15.0, #~1000.0/65.0,
        }
        self.check_neuron_params = {
            **self.default_neuron_params,
            'c_m': 2.0,
            'g_leak': 0.2,
            'tau_ref': 1000.0/10.0,}
        self.dopa_neuron_params = {
            **self.default_neuron_params,
            'c_m': 2.0,
            'g_leak': 0.2,
            'tau_exc': 100.0,
            'tau_inh': 100.0
        }

        self.edlut_time_step = 5e-4
        self.i_thr = 625.062506250625

        self.save_input_activity = False

    def _init_globals(self):

        # Parse the learning protocol
        input_policies = self.learning_protocol.split('_')
        self.n_input_patterns = len(input_policies)
        self.n_actions = len(input_policies[0])

        self.stimulus_actions_reinforcement = {
            (i+1): np.array([
                1  if a=='H' else
                -1 if a=='L' else
                0
                for a in act
            ]) for i, act in enumerate(self.learning_protocol.split('_'))
        }

        # Derived parameters

        self.tau_eli = self.reward_delay*2 if self.tau_eli is None else self.tau_eli

        # self.syn_pre_inc = self.learning_rate*4e-3 if self.syn_pre_inc is None else self.syn_pre_inc
        self.syn_pre_inc = 8e-6 if self.syn_pre_inc is None else self.syn_pre_inc
        # self.syn_pre_inc = 1e-4 if self.syn_pre_inc is None else self.syn_pre_inc

        self.d1_initial_weight = self.max_weight if self.d1_initial_weight is None else self.d1_initial_weight
        self.d2_initial_weight = self.max_weight*0.75 if self.d2_initial_weight is None else self.d2_initial_weight

        # Lateral connectivity patterns

        self.d2d1_intra = 6.0 if self.lateral_intra else 0.0
        self.d2d2_intra = 6.0 if self.lateral_intra else 0.0

        self.d1d1_inter = 3.0 if self.lateral_inter else 0.0
        self.d2d2_inter = 3.0 if self.lateral_inter else 0.0

        self.d1d1_intra = 0.0 #6.0
        self.d1d2_intra = 0.0 #6.0
        self.d1d2_inter = 0.0 #3.0
        self.d2d1_inter = 0.0 #3.0

        # Input parameters

        self.patterns_generated = int(self.stop_simulation_at/self.min_duration_pattern)

        self.sin_current_params = {
            'amplitude': self.i_thr * 0.15, #'Sinusoidal amplitude (pA): VECTOR',
            'frequency': self.current_freq, #'Sinusoidal frequency (Hz): VECTOR',
            'offset': 0.0, #'Sinusoidal offset (pA): VECTOR',
            'phase': 0.0} #'Sinusoidal phase (rad): VECTOR'

        self.min_input_current = self.i_thr * self.min_input_current_factor # 0.8 #0.95
        self.max_input_current = self.i_thr * self.max_input_current_factor #1.07

        # Synaptic parameters

        self.default_synapse_params = {
            'weight': self.d1_initial_weight,
            'max_weight': self.max_weight,
            'type': 0,
            'delay': 0.001,
            'wchange': -1,
            'trigger_wchange': -1,}


        self.output_neuron_params = {
            **self.output_neuron_params,
            'tau_thr': self.tau_thr,
            'tar_fir_rat': self.tar_fir_rat,
        }

        self.stde_common_learning_rule_params = {
            'tau_plu': self.tau_plu, #0.008,
            'tau_min': self.tau_min, #0.008, #0.032,
            'tau_eli': self.tau_eli,
            'tau_dop': self.tau_dop,
            'inc_dop': 1.0/self.tau_dop,
            'dop_max': 350.0,
            'dop_min': 50.0,
            'syn_pre_inc': self.syn_pre_inc,
        }

        self.stde_d1_learning_rule_params = {
            **self.stde_common_learning_rule_params,
            'k_plu_hig': self.stde_kernel_d1['k_plu_hig']*self.learning_rate,
            'k_plu_low': self.stde_kernel_d1['k_plu_low']*self.learning_rate,
            'k_min_hig': self.stde_kernel_d1['k_min_hig']*self.learning_rate,
            'k_min_low': self.stde_kernel_d1['k_min_low']*self.learning_rate,
        }

        self.stde_d2_learning_rule_params = {
            **self.stde_common_learning_rule_params,
            'k_plu_hig': self.stde_kernel_d2['k_plu_hig']*self.learning_rate,
            'k_plu_low': self.stde_kernel_d2['k_plu_low']*self.learning_rate,
            'k_min_hig': self.stde_kernel_d2['k_min_hig']*self.learning_rate,
            'k_min_low': self.stde_kernel_d2['k_min_low']*self.learning_rate,
        }
    def _init_inputs(self):
        # Generate the random patterns (as a sequence of firing rates)
        np.random.seed(self.seed)
        self.input_firing_rate_patterns = np.random.rand(self.n_inputs, self.patterns_generated)
        self.input_firing_rate_patterns *= (self.max_input_current-self.min_input_current)
        self.input_firing_rate_patterns += self.min_input_current

        # Each stimulus pattern cover different subsets of the input
        np.random.seed(self.seed+1)
        self.input_pattern_bool_mask = np.zeros((self.n_input_patterns, self.n_inputs), dtype=bool)
        for r in range(self.n_input_patterns):
            ids = np.random.choice(range(self.n_inputs), size=self.n_neurons_input_pattern, replace=False)
            self.input_pattern_bool_mask[r, ids] = True

        # Generate random time intervals for every random pattern
        np.random.seed(self.seed+2)
        self.input_duration_patterns = np.random.rand(self.patterns_generated)**2
        self.input_duration_patterns = self.input_duration_patterns * (
            self.max_duration_pattern-self.min_duration_pattern
        )
        self.input_duration_patterns += self.min_duration_pattern
        self.cumsum_input_duration_patterns = np.cumsum(self.input_duration_patterns)
        self.cumsum_input_duration_patterns = np.insert(self.cumsum_input_duration_patterns, 0, 0)

        # Decide when to show the stimulus patterns
        np.random.seed(self.seed+3)
        self.show_pattern = np.random.rand(self.patterns_generated) < self.pattern_times_proportion
        for i in range(1, self.show_pattern.size):
            if self.show_pattern[i-1]==1 and self.show_pattern[i]==1:
                self.show_pattern[i]=0
        self.show_pattern = self.show_pattern * np.random.randint(
            1, self.n_input_patterns+1, size=self.show_pattern.size)

        # Reduce the generated arrays to fit only the simulated time
        n_patterns_considered = np.where(self.cumsum_input_duration_patterns>=self.stop_simulation_at)[0][0]
        self.input_firing_rate_patterns = self.input_firing_rate_patterns[:, :n_patterns_considered]
        self.input_duration_patterns = self.input_duration_patterns[:n_patterns_considered]
        self.cumsum_input_duration_patterns = self.cumsum_input_duration_patterns[:n_patterns_considered]
        self.show_pattern = self.show_pattern[:n_patterns_considered]
    def _init_edlut(self):
        self.simulation = pyedlut.PySimulation_API()

        # Define the neuron model parameters for the output layer

        self.integration_method = pyedlut.PyModelDescription(
            model_name='RK4',
            params_dict={'step': self.edlut_time_step})

        self.default_neuron_params['int_meth'] = self.integration_method
        self.output_neuron_params['int_meth'] = self.integration_method
        self.dopa_neuron_params['int_meth'] = self.integration_method

        # Define the synapse models

        self.stde_d1_rule = self.simulation.AddLearningRule('DopamineSTDP', self.stde_d1_learning_rule_params)
        self.stde_d2_rule = self.simulation.AddLearningRule('DopamineSTDP', self.stde_d2_learning_rule_params)

        self.stde_d1_params = {**self.default_synapse_params, 'wchange': self.stde_d1_rule, 'weight': self.d1_initial_weight}
        self.stde_d2_params = {**self.default_synapse_params, 'wchange': self.stde_d2_rule, 'weight': self.d2_initial_weight}
        self.trigger_stde_d1_params = {
            **self.default_synapse_params,
            'trigger_wchange': self.stde_d1_rule,
            'weight': 0.0, 'max_weight': 0.0,
            }
        self.trigger_stde_d2_params = {
            **self.default_synapse_params,
            'trigger_wchange': self.stde_d2_rule,
            'weight': 0.0, 'max_weight': 0.0,
            }
        self.excitatory_params = {**self.default_synapse_params,  'type': 0}
        self.inhibitory_params = {**self.default_synapse_params,  'type': 1}
        self.current_syn_params = {**self.default_synapse_params, 'type': 3}
    def init_simulation(self):
        self._init_globals()
        self._init_inputs()
        self._init_edlut()
        self._init_network()
        self.simulation.Initialize()
    def _init_network(s):
        # Regular network

        s.input_layer = s.create_neuron_layer(s.n_inputs, show=s.save_input_activity)
        s.output_layer = s.create_neuron_layer(
            s.n_outputs*s.n_actions,
            model_name='LIFTimeDrivenModel_DA',
            params=s.output_neuron_params,
        )
        s.input_sin_currents = s.create_neuron_layer(
            1, model_name='SinCurrentDeviceVector', params=s.sin_current_params, show=False)
        s.input_currents = s.create_neuron_layer(
            s.n_inputs, model_name='InputCurrentNeuronModel', params={}, show=False)

        _ = s.simulation.AddSynapticLayer(
            s.input_sin_currents*s.n_inputs, s.input_layer,
            {**s.default_synapse_params, 'type':3})
        _ = s.simulation.AddSynapticLayer(
            s.input_currents, s.input_layer,
            {**s.default_synapse_params, 'type':3})

        # Input -> MSN D1
        s.weights = []
        source, target = [], []
        for i in s.input_layer:
            for a in range(s.n_actions):
                for o_i in range(a*s.n_outputs, (a+1)*s.n_outputs, 2):
                    o = s.output_layer[o_i]
                    source.append(i)
                    target.append(o)
        params = {
            **s.stde_d1_params,
            'weight': (np.random.rand(len(source))**10 * s.d1_initial_weight).tolist(),
#             'delay': ((np.random.rand(len(s.input_layer)) * 10e-3) + 1e-3).tolist()
        }
        s.weights = s.simulation.AddSynapticLayer(source, target, params)
        # Input -> MSN D2
        s.weights = []
        source, target = [], []
        for i in s.input_layer:
            for a in range(s.n_actions):
                for o_i in range(a*s.n_outputs+1, (a+1)*s.n_outputs, 2):
                    o = s.output_layer[o_i]
                    source.append(i)
                    target.append(o)
        params = {
            **s.stde_d2_params,
            'weight': (np.random.rand(len(source))**10 * s.d2_initial_weight).tolist(),
#             'delay': ((np.random.rand(len(s.input_layer)) * 10e-3) + 1e-3).tolist()
        }
        s.weights += s.simulation.AddSynapticLayer(source, target, params)


        # Lateral inhibition
        # d1d1_intra: D1->D1 within same action group
        # d1d1_inter: D1->D1 to other action groups
        # d1d2_intra: D1->D2 within same action group
        # d1d2_inter: D1->D2 to other action groups
        # d2d1_intra: D2->D1 within same action group
        # d2d1_inter: D2->D1 to other action groups
        # d2d2_intra: D2->D2 within same action group
        # d2d2_inter: D2->D2 to other action groups

        source, target, weights = [], [], []

        # Possible combinations of lateral inhibition
        combinations = [
            list(range(s.n_actions)), #From action module
            list(range(s.n_actions)), #To action module
            [0, 1], #From type D1 (0) or type D2 (1)
            [0, 1], #To type D1 (0) or type D2 (1)
        ]

        # Compute the list of connections
        for a1, a2, t1, t2 in itertools.product(*combinations):
            # Select the appropriate weight for this connection
            w = 0.0
            if   a1==a2:
                if   t1==0 and t2==0: w=s.d1d1_intra/s.n_outputs
                elif t1==0 and t2==1: w=s.d1d2_intra/s.n_outputs
                elif t1==1 and t2==0: w=s.d2d1_intra/s.n_outputs
                elif t1==1 and t2==1: w=s.d2d2_intra/s.n_outputs
            elif a1!=a2:
                if   t1==0 and t2==0: w=s.d1d1_inter/s.n_outputs
                elif t1==0 and t2==1: w=s.d1d2_inter/s.n_outputs
                elif t1==1 and t2==0: w=s.d2d1_inter/s.n_outputs
                elif t1==1 and t2==1: w=s.d2d2_inter/s.n_outputs
            if w==0.0: continue

            # Sources, targets and weights
            for i in range(a1*s.n_outputs+t1, (a1+1)*s.n_outputs, 2):
                ni = s.output_layer[i]
                for o in range(a2*s.n_outputs+t2, (a2+1)*s.n_outputs, 2):
                    no = s.output_layer[o]
                    weights.append(w)
                    source.append(ni)
                    target.append(no)

        # Create the connections if needed
        if len(source)>0:
          _ = s.simulation.AddSynapticLayer(
              source, target,
              {**s.inhibitory_params, 'weight': weights, 'max_weight': np.max(weights)})



        # Action neurons and experiment logic: place here neurons related to the action decision
        #dopa_neuron_params
        s.action_layer = s.create_neuron_layer(s.n_actions, params=s.action_neuron_params)
        s.is_rewarded_current = s.create_neuron_layer(s.n_actions, model_name='InputCurrentNeuronModel', params={}, show=False)
        s.is_punished_current = s.create_neuron_layer(s.n_actions, model_name='InputCurrentNeuronModel', params={}, show=False)
        s.action_rewarded_layer = s.create_neuron_layer(s.n_actions, params=s.check_neuron_params)
        s.action_punished_layer = s.create_neuron_layer(s.n_actions, params=s.check_neuron_params)
        s.reward_neuron = s.create_neuron_layer(1, params=s.check_neuron_params)
        s.punish_neuron = s.create_neuron_layer(1, params=s.check_neuron_params)

        # Excitatory connection between MSN action groups and its own action neuron
        # MSN D1
        source, target = [], []
        for a in range(s.n_actions):
            for i_idx in range(a*s.n_outputs, (a+1)*s.n_outputs, 2):
                source.append(s.output_layer[i_idx])
                target.append(s.action_layer[a])
        _ = s.simulation.AddSynapticLayer(
            source, target,
            {**s.excitatory_params, 'weight':s.msn_d1_to_action/s.n_outputs, 'max_weight':s.msn_d1_to_action/s.n_outputs})
        # MSN D2
        source, target = [], []
        for a in range(s.n_actions):
            for i_idx in range(a*s.n_outputs+1, (a+1)*s.n_outputs, 2):
                source.append(s.output_layer[i_idx])
                target.append(s.action_layer[a])
        _ = s.simulation.AddSynapticLayer(
            source, target,
            {**s.inhibitory_params, 'weight':s.msn_d2_to_action/s.n_outputs, 'max_weight':s.msn_d2_to_action/s.n_outputs})

        # Inhibitory connection between MSN action groups and the other action neurons
#         source, target = [], []
#         for a1 in range(s.n_actions):
#             for i_idx in range(a1*s.n_outputs, (a1+1)*s.n_outputs):
#                 for a2 in range(s.n_actions):
#                     if a1==a2: continue
#                     source.append(s.output_layer[i_idx])
#                     target.append(s.action_layer[a2])
#         _ = s.simulation.AddSynapticLayer(
#             source, target,
#             {**s.inhibitory_params, 'weight':10.0/s.n_outputs, 'max_weight':10.0/s.n_outputs})

        # Excitatory connection between action neurons and action-value neurons
        _ = s.simulation.AddSynapticLayer(
            s.action_layer, s.action_rewarded_layer,
            {**s.excitatory_params, 'weight':1.0, 'max_weight':1.0})
        _ = s.simulation.AddSynapticLayer(
            s.action_layer, s.action_punished_layer,
            {**s.excitatory_params, 'weight':1.0, 'max_weight':1.0})

        # Inhibitory current connection between trial status and action-value neurons
        _ = s.simulation.AddSynapticLayer(
            s.is_rewarded_current, s.action_punished_layer, s.current_syn_params)
        _ = s.simulation.AddSynapticLayer(
            s.is_punished_current, s.action_rewarded_layer, s.current_syn_params)

        # Connection to reward-punishment neurons
        _ = s.simulation.AddSynapticLayer(
            s.action_rewarded_layer, s.reward_neuron*len(s.action_rewarded_layer),
            {**s.excitatory_params, 'weight':1.0, 'max_weight':1.0})
        _ = s.simulation.AddSynapticLayer(
            s.action_punished_layer, s.punish_neuron*len(s.action_punished_layer),
            {**s.excitatory_params, 'weight':1.0, 'max_weight':1.0})

        # Dopaminergic parts

        s.dopa_current = s.create_neuron_layer(
            1, model_name='InputCurrentNeuronModel', params={}, show=False)
        s.dopa_neuron = s.create_neuron_layer(1, params=s.dopa_neuron_params)

        # Constant current connection to dopa neuron
        _ = s.simulation.AddSynapticLayer(s.dopa_current, s.dopa_neuron, s.current_syn_params)

        # Reward-punish connections
        _ = s.simulation.AddSynapticLayer(s.reward_neuron, s.dopa_neuron,
            {**s.excitatory_params,
             'weight':s.reward_weights[0], 'max_weight':s.reward_weights[0],
             'delay': s.reward_delay,})
        _ = s.simulation.AddSynapticLayer(s.punish_neuron, s.dopa_neuron,
            {**s.inhibitory_params,
             'weight':s.reward_weights[1], 'max_weight':s.reward_weights[1],
             'delay': s.reward_delay,})

        # Modulation connection to MSN neurons
        # MSN D1
        source, target = [], []
        for i in s.dopa_neuron:
            for o in s.output_layer[::2]:
                source.append(i)
                target.append(o)
        _ = s.simulation.AddSynapticLayer(source, target, s.trigger_stde_d1_params)
        # MSN D2
        source, target = [], []
        for i in s.dopa_neuron:
            for o in s.output_layer[1::2]:
                source.append(i)
                target.append(o)
        _ = s.simulation.AddSynapticLayer(source, target, s.trigger_stde_d2_params)
    def run(s, show_progress_bar=True):
        s.weights_history_list = []
        s.output_times = []
        s.output_index = []

        s.simulation.AddExternalCurrentActivity([0.0], s.dopa_current, [s.dopa_basal_current])


        for i, (from_time, to_time) in tqdm(
            SimulationIterator(s.stop_simulation_at, s.input_duration_patterns),
            disable=not show_progress_bar
        ):

            # Stimulus presentation

            input_pattern = s.input_firing_rate_patterns[:, i].copy()
            if s.show_pattern[i] > 0:
                mask = s.input_pattern_bool_mask[s.show_pattern[i]-1, :]
                input_pattern[mask] = (
                    s.input_firing_rate_patterns[mask, s.show_pattern[i]]
                )
            s.simulation.AddExternalCurrentActivity(
                np.full(s.n_inputs, from_time),
                s.input_currents,
                input_pattern
            )

            # Reward delivery

            current_policy = s.stimulus_actions_reinforcement.get(s.show_pattern[i], np.zeros(s.n_actions))
            times = [from_time] * s.n_actions

            value = np.array([-100.0] * s.n_actions)
            value[current_policy>0] = 0
            s.simulation.AddExternalCurrentActivity(times, s.is_punished_current, value)

            value = np.array([-100.0] * s.n_actions)
            value[current_policy<0] = 0
            s.simulation.AddExternalCurrentActivity(times, s.is_rewarded_current, value)

            # Saving the weights

            w = s.simulation.GetSelectedWeights(s.weights)
            s.weights_history_list.append(w)

            # Next simulation steps

            s.simulation.RunSimulation(to_time)

            # Saving the spikes

            ot, oi = s.simulation.GetSpikeActivity()
            s.output_times += ot
            s.output_index += oi

    def save_data(self):
        # Retrieve output spike activity and weight history
        self.ot = np.array(self.output_times)
        self.oi = np.array(self.output_index)
        self.weights_history = np.array(self.weights_history_list).T

    def get_accuracy(self):
        # Cálculo de la acción tomada

        action_taken = np.zeros((self.n_actions+1, self.show_pattern.size))

        for n_action in range(1, self.n_actions+1):
            spikes_action = self.oi==self.action_layer[n_action-1]
            for spike_time in self.ot[spikes_action]:
        #           time_bin = int(spike_time//self.max_duration_pattern)
                time_bin = np.searchsorted(self.cumsum_input_duration_patterns, spike_time)-1
                action_taken[n_action, time_bin] += 1
        action_taken[0, :] = (np.sum(action_taken[1:,:], axis=0)==0)# * np.max(action_taken)

        # Matriz de patrón a acción

        map_matrix_time = np.zeros((self.n_input_patterns+1, self.n_actions+1, self.show_pattern.size))
        for tak in range(action_taken.shape[0]):
            for t, (pat,spks) in enumerate(zip(self.show_pattern, action_taken[tak,:])):
                if pat is None: continue
                map_matrix_time[pat,tak,t] += spks>0

        # Cálculo de la acción deseada

        pattern_to_action_map = {
            (i+1): [ai+1 for ai, a in enumerate(act) if a=='H'] if 'H' in act else [0]
            for i, act in enumerate(self.learning_protocol.split('_'))
        }
        action_desired = []
        for i, pattern in enumerate(self.show_pattern):
          # print(i, type(i), pattern, type(pattern))
          action_desired.append(pattern_to_action_map.get(pattern))


        # Cálculo del tensor de confusion

        confusion_matrix_time = np.zeros((self.n_actions+1, self.n_actions+1, self.show_pattern.size))
        confusion_matrix = np.zeros((self.n_actions+1, self.n_actions+1))
        for tak in range(action_taken.shape[0]):
            for t, (des,spks) in enumerate(zip(action_desired, action_taken[tak,:])):
                if not des: continue
                for d in des:
                    confusion_matrix[d,tak] += spks>0
                    confusion_matrix_time[d,tak,t] += spks>0


        # Cálculo del vector traza y del vector suma

        trace_time = np.zeros_like(self.show_pattern)
        all_time = np.zeros_like(self.show_pattern)
        for i in range(self.n_actions+1):
            trace_time = trace_time + confusion_matrix_time[i,i,:]
            for j in range(self.n_actions+1):
                all_time = all_time + confusion_matrix_time[i,j,:]


        # Calculo de la precisión

        import pandas as pd

        n = 100 #10 #100
        rm_tra = pd.Series(trace_time).rolling(n).mean()
        rm_all = pd.Series(all_time).rolling(n).mean()

        pre = rm_tra/rm_all
        time = self.cumsum_input_duration_patterns[pre.index]
        label = self.get_label()
        seed = self.seed
        ret_dict = {
            'time': time.copy(),
            'accuracy': pre.copy(),
            'label': label,
            'seed': seed,
            'confusion_matrix': confusion_matrix_time,
            'map_matrix': map_matrix_time
        }

        # A veces nos puede interesar guardar los rasters

        if hasattr(self, 'with_rasterplots') and self.with_rasterplots:
            save_i = np.in1d(self.oi, self.output_layer)
            ret_dict = {
                **ret_dict,
                'ot': self.ot[save_i],
                'oi': self.oi[save_i],
                'show_pattern': self.show_pattern,
                'action_desired': action_desired,
            }


        return ret_dict
    def get_label(self):
        txt = 'd1d1_intra={}, d1d1_inter={}, d1d2_intra={}, d1d2_inter={}, d2d1_intra={}, d2d1_inter={}, d2d2_intra={}, d2d2_inter={}'
        txt = txt.format(
            self.d1d1_intra,
            self.d1d1_inter,
            self.d1d2_intra,
            self.d1d2_inter,
            self.d2d1_intra,
            self.d2d1_inter,
            self.d2d2_intra,
            self.d2d2_inter
        )
        return txt



# Script functions

def text2number(v):
    try:
        o = int(v)
    except ValueError:
        try:
            o = float(v)
        except ValueError:
            o = v
    return o

def get_label(params):
    params_no_seed = {key: value for key,value in params.items() if key!='seed'}
    return str(params_no_seed)


def main():
    # Get parameters
    params = {}
    for arg in sys.argv[1:]:
        param_name, param_value = arg.split('=')
        param_value = text2number(param_value)
        params[param_name] = param_value
    script_keys = ['experiments_file_path', 'save_results']
    condition = {key: value for key, value in params.items() if key not in script_keys}

    # Create a network and set attributes
    net = MasquelierNetSin()
    for key, value in condition.items():
        setattr(net, key, value)

    net.init_simulation()
    net.run(show_progress_bar=params.get('show_progress_bar', False))

    if params.get('save_results', False):
        date = datetime.utcnow()

        # Reset the saved files?
        if not os.path.exists(params.get('experiments_file_path', 'data.pkl')):
            with portalocker.Lock(params.get('experiments_file_path', 'data.pkl'), mode='wb') as f:
                pickle.dump(
                    {
                        'parameter_space': [],
                        'remaining': [],
                        'results': []
                    }, f)

        with portalocker.Lock(params.get('experiments_file_path', 'data.pkl'), mode='rb+') as f:
            net.save_data()
            results = net.get_accuracy()
            experiment = {'condition':condition, 'results':results, 'date': date}

            experiments = pickle.load(f)
            f.seek(0)
            f.truncate()
            if condition in experiments['remaining']: experiments['remaining'].remove(condition)
            experiments['results'].append(experiment)
            pickle.dump(experiments, f)

if __name__ == '__main__':
    main()
