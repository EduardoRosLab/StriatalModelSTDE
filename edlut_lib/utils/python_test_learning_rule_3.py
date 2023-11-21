#!/usr/bin/python

#import pyedlut.pyedlut as pyedlut
import pyedlut as pyedlut

import matplotlib.pyplot as plt

# Declare the simulation object
simulation = pyedlut.PySimulation_API()

# # Get and print all the available Neuron Models in EDLUT
# NeuronModelList = simulation.GetAvailableNeuronModels()
# simulation.PrintAvailableNeuronModels()
#
# #Get and print information about all the Neuron Model in EDLUT
# for i in NeuronModelList:
#     print('-------------------')
#     simulation.PrintNeuronModelInfo(i)
#     #print('-------------------')
#     #NeuronModelInfo = simulation.GetNeuronModelInfo(i)
#     #for name,value in zip(NeuronModelInfo.keys(),NeuronModelInfo.values()):
#     #    print(name,'->',value)
#
#
# # Get and print all the available Integration Methods in CPU for EDLUT
# IntegrationMethodList = simulation.GetAvailableIntegrationMethods()
# simulation.PrintAvailableIntegrationMethods()
#
# # Get and print all the available Learning Rules in EDLUT
# LearningRuleList = simulation.GetAvailableLearningRules()
# simulation.PrintAvailableLearningRules()

N_synapses=1000
Initial_weight=0.02



# Create the input neuron layers (three input fibers for spikes and, input fiber for current and a sinusoiidal current generator)
input_spike_layer_1 = simulation.AddNeuronLayer(
        num_neurons=N_synapses,
        model_name='InputSpikeNeuronModel',
        param_dict={},
        log_activity=False,
        output_activity=False)
		
input_spike_layer_1d = simulation.AddNeuronLayer(
        num_neurons=1,
        model_name='InputSpikeNeuronModel',
        param_dict={},
        log_activity=False,
        output_activity=False)		

input_spike_layer_2 = simulation.AddNeuronLayer(
        num_neurons=1,
        model_name='InputSpikeNeuronModel',
        param_dict={},
        log_activity=False,
        output_activity=False)

# # Get the default parameter values of the neuron model
# default_params = simulation.GetNeuronModelDefParams('LIFTimeDrivenModel');
# print('Default parameters for LIF Time-driven model:')
# for key, value in zip (default_params.keys(), default_params.values()):
#     print(key, value)
#
# # Create the output neuron layer
# #default_output_layer = simulation.AddNeuronLayer(
# #	num_neurons= 2,
# #	model_name= 'LIFTimeDrivenModel',
# #	param_dict= default_params,
# #	log_activity = False,
# #	output_activity = True)
#
# # Get the default parameter values of the integration method
# default_im_param = simulation.GetIntegrationMethodDefParams('Euler')
# print('Default parameters for Euler integration method:')
# for key in default_im_param.keys():
#     print(key)

# Define the neuron model parameters for the output layer
integration_method = pyedlut.PyModelDescription(model_name='Euler', params_dict={'step': 0.0001})
output_params = {
        'tau_ref': 1.0,
        'v_thr': -40.0,
        'e_exc': 0.0,
        'e_inh': -80.0,
        'e_leak': -65.0,
        'g_leak': 0.2,
        'c_m': 2.0,
        'tau_exc': 0.5,
        'tau_inh': 10.0,
        'tau_nmda': 15.0,
        'int_meth': integration_method
}



# Create the output layer
output_layer = simulation.AddNeuronLayer(
        num_neurons = 1,
        model_name = 'LIFTimeDrivenModel',
        param_dict = output_params,
        log_activity = False,
        output_activity = True)

# Get the default parameter values of the neuron model
default_param_lrule = simulation.GetLearningRuleDefParams('DopamineSTDP');
print('Default parameters for dopamine STDP learning rule:')
for key in default_param_lrule.keys():
    print(key)



# Define the learning rule parameters
lrule_params = {
        'max_LTP':0.016,
        'tau_LTP':0.100,
	'max_LTD':0.033,
	'tau_LTD':0.100,
	'tau_chn':0.100,
	'tau_dop':0.100,
	'inc_dop':0.010,
	'bas_dop':0.003
     }
	 

# Create the learning rule
Dopamine_STDP_rule = simulation.AddLearningRule('DopamineSTDP', lrule_params)

# Define the synaptic parameters
source_neurons_1 = input_spike_layer_1
target_neurons_1 = []
for i in range(N_synapses):
    target_neurons_1.append(output_layer[0])

source_neurons_1d = []
source_neurons_1d.append(input_spike_layer_1d[0])
target_neurons_1d = output_layer

source_neurons_2 = []
source_neurons_2.append(input_spike_layer_2[0])
target_neurons_2 = output_layer

lr_synaptic_params = {
        'weight': Initial_weight,
        'max_weight': 0.040,
        'type': 1,
        'delay': 0.001,
        'wchange': Dopamine_STDP_rule,
        'trigger_wchange': -1
    }
	
lrd_synaptic_params = {
        'weight': 0.0,
        'max_weight': 100.0,
        'type': 1,
        'delay': 0.001,
        'wchange': -1,
        'trigger_wchange': Dopamine_STDP_rule
    }	

synaptic_params = {
        'weight': 20.0,
        'max_weight': 100.0,
        'type': 0,
        'delay': 0.001,
        'wchange': -1,
        'trigger_wchange': -1
    }

# Create the list of synapses
synaptic_layer_1 = simulation.AddSynapticLayer(source_neurons_1, target_neurons_1, lr_synaptic_params);
synaptic_layer_1d = simulation.AddSynapticLayer(source_neurons_1d, target_neurons_1d, lrd_synaptic_params);
synaptic_layer_2 = simulation.AddSynapticLayer(source_neurons_2, target_neurons_2, synaptic_params);
#
#
#DEFINE INPUT AND OUTPUT FILE DRIVERS
simulation.AddFileOutputSpikeActivityDriver('Output_spikes.txt')

# Initialize the network
simulation.Initialize()


# Inject input spikes to the network
spikes = { 'times':[], 'neurons':[] }
times = []
reference = []

step = 0.001
for i in range(N_synapses):
    times.append(i+step)
    reference.append(Initial_weight)
    spikes['times'].append(i*step)
    spikes['neurons'].append(input_spike_layer_1[i])
#    spikes['times'].append(1.05)
#    spikes['neurons'].append(input_spike_layer_1[i])
	
#dopamine spikes
spikes['times'].append(N_synapses*step*0.15)
spikes['neurons'].append(input_spike_layer_1d[0])	
spikes['times'].append(N_synapses*step*0.50)
spikes['neurons'].append(input_spike_layer_1d[0])	
spikes['times'].append(N_synapses*step*0.85)
spikes['neurons'].append(input_spike_layer_1d[0])	


#first output spike
spikes['times'].append(N_synapses*step*0.2)
spikes['neurons'].append(input_spike_layer_2[0])
#second output spike
spikes['times'].append(N_synapses*step*0.8)
spikes['neurons'].append(input_spike_layer_2[0])


simulation.AddExternalSpikeActivity(spikes['times'], spikes['neurons'])

# Run the simulation step-by-step
simulation.RunSimulation(N_synapses*step*1.1)

weights = simulation.GetSelectedWeights(synaptic_layer_1)
plt.plot(times,weights,times,reference)
plt.xlabel('time (ms)')
plt.ylabel('Synaptic weight (nS)')
plt.title('Dopamine STDP learning rule')
plt.show()



#
# print('Simulation finished')
#
# #PLOT RASTERPLOT
# rasterplot = plt.scatter(output_times, output_index, alpha=0.5)
# plt.xlim(0, total_simulation_time)
# plt.xlabel('time (s)')
# plt.ylabel('5 = spikes (AMPA, GABA and NMDA);   6 = square current (5 Hz);   7 = sin current (2 Hz)')
# plt.show()
