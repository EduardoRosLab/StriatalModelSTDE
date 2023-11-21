#!/usr/bin/env python
# coding: utf-8

# # AdEx Resonance with EDLUT

# In[1]:


import pyedlut as pyedlut
import numpy


# Create a new simulation object

# In[2]:


simulation = pyedlut.PySimulation_API()


# Stimulation parameters

# In[3]:


stimulation_freq = numpy.arange(1.0,15.0,1.0)
current_amp = 8e-12
current_off = 12e-12


# Neuron model parameters

# In[4]:


neuron_cm = 3.10e-12
neuron_deltat = 5.42e-3
neuron_el = -64.06e-3
neuron_vp = -13.49e-3 # In EDLUT Vpeak=vthres+6+deltat=-8.07
neuron_vreset = -70.28e-3
neuron_vthres = -40.59e-3
neuron_a = 0.26e-9
neuron_b = 0.19e-12#0.19e-9
neuron_gl = 0.49e-9
neuron_tauw = 327.25e-3


# ## Network Definition

# Create a layer of neurons stimulating with sinusoidal current to the network

# In[5]:

sin_params = {
        'frequency': 0.0,
        'amplitude': current_amp*1.e12,
        'offset': current_off*1.e12,
        'phase': 0.0,
}

sin_current_layer = simulation.AddNeuronLayer(
        num_neurons=len(stimulation_freq),
        model_name='SinCurrentDeviceVector',
        param_dict=sin_params,
        log_activity=False,
        output_activity=False)

print(simulation.GetVectorizableParameters('SinCurrentDeviceVector'))


# And now, the output layer. First, define the integration method to be used (Euler). Second, the neuron model parameters. And third, the neuron layer.

# In[6]:


simulation.PrintNeuronModelInfo('AdExTimeDrivenModel')

# Define the integration method params by creating a PyModelDescription object.
integration_method = pyedlut.PyModelDescription(model_name='Euler', params_dict={'step': 0.0001})
# Define the neuron model params by creating a dictionary
# (including the PyModelDescription object corresponding to the integration method object)
nm_params = {
    'a': neuron_a*1.e9,
    'b': neuron_b*1.e12,
    'c_m': neuron_cm*1.e12,
    'e_leak': neuron_el*1.e3,
    'e_reset': neuron_vreset*1.e3,
    'g_leak': neuron_gl*1.e9,
    'tau_w': neuron_tauw*1.e3,
    'thr_slo_fac': neuron_deltat*1.e3,
    'v_thr': neuron_vthres*1e3,
    'int_meth': integration_method
}

# Create the output layer with AdEx neurons with nm_params
output_layer = simulation.AddNeuronLayer(
    num_neurons = len(stimulation_freq),
    model_name = 'AdExTimeDrivenModel',
    param_dict = nm_params,
    log_activity = True, # The spikes emitted by this neuron layer will not be registered in the log file
    output_activity = True) # The spikes emitted by this neuron layer will be registered
                            # in the output devices (file,TCP/IP connector,ROS node,...)


# Connect the current generators to the output layer

# In[7]:


# Define the synaptic parameters
synaptic_params = {
    'weight': 1.0, # Initial weight
    'max_weight': 100.0, # Maximum weight that is allowed to grow.
    'type': 3,
    'delay': 0.001, # Synaptic delay
} # This synapse is not the trigger for another learning rule

# Create the list of synapses
synaptic_layer_1 = simulation.AddSynapticLayer(sin_current_layer, output_layer, synaptic_params);
print('Synaptic layer:', synaptic_layer_1)


#Load a neuron monitor
simulation.AddFileOutputMonitorDriver('Output_state.txt', True)

# Initialize the network. It will not be allowed to change after the initialize function is called

# In[8]:


simulation.Initialize()


# ## Defining External Stimulation

# Define sinusoidal currents from the generators

# In[ ]:


for idx,freq in enumerate(stimulation_freq):
    simulation.SetSpecificNeuronParams(sin_current_layer[idx],{'frequency':freq})
# ------------------------------------------------------
# Aquí tenemos un segmentation fault
# ------------------------------------------------------
#print(simulation.GetSpecificNeuronParams(sin_current_layer))
final_params_vector = simulation.GetSpecificNeuronParams(sin_current_layer)
for i in range (len(sin_current_layer)):
    print('- Sin current generator ', sin_current_layer[i])
    print (final_params_vector[i])

# ## Run the Simulation

# Run the simulation

# In[10]:


total_simulation_time = 3 # Run the simulation for 20seconds
simulation.RunSimulation(total_simulation_time)


# ## Visualize the Output Activity and Weight Evolution

# Retrieve output spike activity from those neurons with RecordActivity=True. Please, note that GetSpikeActivity function only returns the activity the first time it is called. After it has been done, the buffers of recorded activity will be empty until we call RunSimulation again.

# In[11]:


output_times, output_index = simulation.GetSpikeActivity()


# Print the generated activity

# In[12]:


# Print the output spike activity
print('Output activity: ')
for t, i in zip(output_times, output_index):
    print("Spike from neuron",i,"at time",t,"s")


# Plot the activity by using matplotlib as a rasterplot

# In[13]:


#get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
plt.figure()
plt.plot(output_times,output_index,'.')
plt.xlabel('Time (s)')
plt.ylabel('Neuron Index')
plt.title('Raster Plot')
plt.xlim([0,3.0])
plt.show()


# In[ ]:
