/***************************************************************************
 *                           LIFTimeDrivenModel_1_3_GPU_C_INTERFACE.cu     *
 *                           -------------------                           *
 * copyright            : (C) 2012 by Francisco Naveros                    *
 * email                : fnaveros@ugr.es                                  *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "../../include/neuron_model/LIFTimeDrivenModel_1_3_GPU_C_INTERFACE.cuh"
#include "../../include/neuron_model/LIFTimeDrivenModel_1_3_GPU2.cuh"
#include "../../include/neuron_model/VectorNeuronState.h"
#include "../../include/neuron_model/VectorNeuronState_GPU_C_INTERFACE.cuh"
#include "../../include/neuron_model/CurrentSynapseModel.h"

#include <iostream>
#include <cmath>
#include <string>

#include "../../include/spike/EDLUTFileException.h"
#include "../../include/spike/Neuron.h"
#include "../../include/spike/InternalSpike.h"
#include "../../include/spike/PropagatedSpike.h"
#include "../../include/spike/Interconnection.h"

#include "../../include/simulation/Utils.h"

#include "../../include/openmp/openmp.h"

#include "../../include/cudaError.h"
//Library for CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "integration_method/IntegrationMethodFactory_GPU_C_INTERFACE.cuh"


LIFTimeDrivenModel_1_3_GPU_C_INTERFACE::LIFTimeDrivenModel_1_3_GPU_C_INTERFACE() : TimeDrivenNeuronModel_GPU_C_INTERFACE(SecondScale), e_exc(0), e_inh(0), e_leak(0), g_leak(0), c_m(0), v_thr(0), tau_exc(0), tau_inh(0),
tau_ref(0),  NeuronModel_GPU2(0){
	std::map<std::string, boost::any> param_map = LIFTimeDrivenModel_1_3_GPU_C_INTERFACE::GetDefaultParameters();
	param_map["name"] = LIFTimeDrivenModel_1_3_GPU_C_INTERFACE::GetName();
	this->SetParameters(param_map);

	this->State = (VectorNeuronState_GPU_C_INTERFACE *) new VectorNeuronState_GPU_C_INTERFACE(N_NeuronStateVariables);
}

LIFTimeDrivenModel_1_3_GPU_C_INTERFACE::~LIFTimeDrivenModel_1_3_GPU_C_INTERFACE(void){
	DeleteClassGPU2();
}

VectorNeuronState * LIFTimeDrivenModel_1_3_GPU_C_INTERFACE::InitializeState(){
	return this->GetVectorNeuronState();
}


InternalSpike * LIFTimeDrivenModel_1_3_GPU_C_INTERFACE::ProcessInputSpike(Interconnection * inter, double time){
	this->State_GPU->AuxStateCPU[inter->GetType()*State_GPU->GetSizeState() + inter->GetTargetNeuronModelIndex()] += inter->GetWeight();

	return 0;
}


__global__ void LIFTimeDrivenModel_1_3_GPU_C_INTERFACE_UpdateState(LIFTimeDrivenModel_1_3_GPU2 ** NeuronModel_GPU2, double CurrentTime){
	(*NeuronModel_GPU2)->UpdateState(CurrentTime);
}

		
bool LIFTimeDrivenModel_1_3_GPU_C_INTERFACE::UpdateState(int index, double CurrentTime){
	//update input current values from electrical coupling synapses.
	for (int i = 0; i < State_GPU->GetSizeState(); i++){
		this->State_GPU->AuxStateCPU[(N_TimeDependentNeuronState - 1)*State_GPU->GetSizeState() + i] = this->CurrentSynapses->GetTotalCurrent(i);
	}

	if(prop.canMapHostMemory){
		LIFTimeDrivenModel_1_3_GPU_C_INTERFACE_UpdateState<<<N_block,N_thread>>>(NeuronModel_GPU2, CurrentTime);
	}else{
		HANDLE_ERROR(cudaMemcpy(State_GPU->AuxStateGPU,State_GPU->AuxStateCPU,this->N_TimeDependentNeuronState*State_GPU->SizeStates*sizeof(float),cudaMemcpyHostToDevice));
		LIFTimeDrivenModel_1_3_GPU_C_INTERFACE_UpdateState<<<N_block,N_thread>>>(NeuronModel_GPU2, CurrentTime);
		HANDLE_ERROR(cudaMemcpy(State_GPU->InternalSpikeCPU,State_GPU->InternalSpikeGPU,State_GPU->SizeStates*sizeof(bool),cudaMemcpyDeviceToHost));
	}
	
	if(this->GetVectorNeuronState()->Get_Is_Monitored()){
		HANDLE_ERROR(cudaMemcpy(State_GPU->VectorNeuronStates,State_GPU->VectorNeuronStates_GPU,State_GPU->GetNumberOfVariables()*State_GPU->SizeStates*sizeof(float),cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(State_GPU->LastUpdate,State_GPU->LastUpdateGPU,State_GPU->SizeStates*sizeof(double),cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(State_GPU->LastSpikeTime,State_GPU->LastSpikeTimeGPU,State_GPU->SizeStates*sizeof(double),cudaMemcpyDeviceToHost));
	} 

	HANDLE_ERROR(cudaEventRecord(stop, 0)); 
	HANDLE_ERROR(cudaEventSynchronize(stop));

	memset(State_GPU->AuxStateCPU,0,N_TimeDependentNeuronState*State_GPU->SizeStates*sizeof(float));

	return false;
}


enum NeuronModelOutputActivityType LIFTimeDrivenModel_1_3_GPU_C_INTERFACE::GetModelOutputActivityType(){
	return OUTPUT_SPIKE;
}

enum NeuronModelInputActivityType LIFTimeDrivenModel_1_3_GPU_C_INTERFACE::GetModelInputActivityType(){
	return INPUT_SPIKE_AND_CURRENT;
}


ostream & LIFTimeDrivenModel_1_3_GPU_C_INTERFACE::PrintInfo(ostream & out){
	out << "- Leaky Time-Driven Model GPU: " << LIFTimeDrivenModel_1_3_GPU_C_INTERFACE::GetName() << endl;
	out << "\tExcitatory reversal potential (e_exc): " << this->e_exc << "V" << endl;
	out << "\tInhibitory reversal potential (e_inh): " << this->e_inh << "V" << endl;
	out << "\tEffective leak potential (e_leak): " << this->e_leak << "V" << endl;
	out << "\tMembrane capacitance (c_m): " << this->c_m << "F" << endl;
	out << "\tEffective threshold potential (v_thr): " << this->v_thr << "V" << endl;
	out << "\tAMPA (excitatory) receptor time constant (tau_exc): " << this->tau_exc << "s" << endl;
	out << "\tGABA (inhibitory) receptor time constant (tau_inh): " << this->tau_inh << "s" << endl;
	out << "\tRefractory period (tau_ref): " << this->tau_ref << "s" << endl;
	out << "\tLeak conductance (g_leak): " << this->g_leak << "S" << endl;
	return out;
}	




void LIFTimeDrivenModel_1_3_GPU_C_INTERFACE::InitializeStates(int N_neurons, int OpenMPQueueIndex){
	//Select the correnpondent device. 
	HANDLE_ERROR(cudaSetDevice(GPUsIndex[OpenMPQueueIndex % NumberOfGPUs]));  
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaGetDeviceProperties( &prop, GPUsIndex[OpenMPQueueIndex % NumberOfGPUs]));

	this->State_GPU = (VectorNeuronState_GPU_C_INTERFACE *) this->State;
	
	//Initialize neural state variables.
	float initialization[] = {e_leak,0.0f,0.0f,0.0f};
	State_GPU->InitializeStatesGPU(N_neurons, initialization, N_TimeDependentNeuronState, prop);

	//INITIALIZE CLASS IN GPU
	this->InitializeClassGPU2(N_neurons);

	InitializeVectorNeuronState_GPU2();

	//Initialize the array that stores the number of input current synapses for each neuron in the model
	this->CurrentSynapses = new CurrentSynapseModel(N_neurons);
}




__global__ void LIFTimeDrivenModel_1_3_GPU_C_INTERFACE_InitializeClassGPU2(LIFTimeDrivenModel_1_3_GPU2 ** NeuronModel_GPU2, 
		float e_exc, float e_inh, float e_leak, float v_thr, float c_m, float tau_exc, float tau_inh, float tau_ref, float g_leak, 
		char const* integrationName, int N_neurons, void ** Buffer_GPU){
	if(blockIdx.x==0 && threadIdx.x==0){
		(*NeuronModel_GPU2)=new LIFTimeDrivenModel_1_3_GPU2(e_exc, e_inh, e_leak, 
			v_thr, c_m, tau_exc, tau_inh, tau_ref, g_leak, integrationName, N_neurons, Buffer_GPU);
	}
}


void LIFTimeDrivenModel_1_3_GPU_C_INTERFACE::InitializeClassGPU2(int N_neurons){
	cudaMalloc(&NeuronModel_GPU2, sizeof(LIFTimeDrivenModel_1_3_GPU2 **));
	
	char * integrationNameGPU;
	cudaMalloc((void **)&integrationNameGPU,32*4);
//REVISAR
	HANDLE_ERROR(cudaMemcpy(integrationNameGPU, &integration_method_GPU->name[0], 32 * 4, cudaMemcpyHostToDevice));

	this->N_thread = 128;
	this->N_block=prop.multiProcessorCount*16;
	if((N_neurons+N_thread-1)/N_thread < N_block){
		N_block = (N_neurons+N_thread-1)/N_thread;
	}
	int Total_N_thread=N_thread*N_block;

	integration_method_GPU->InitializeMemoryGPU(N_neurons, Total_N_thread);

	LIFTimeDrivenModel_1_3_GPU_C_INTERFACE_InitializeClassGPU2<<<1,1>>>(NeuronModel_GPU2,e_exc, e_inh, e_leak, v_thr, 
		c_m, tau_exc, tau_inh, tau_ref, g_leak, integrationNameGPU, N_neurons, integration_method_GPU->Buffer_GPU);

	cudaFree(integrationNameGPU);
}



__global__ void initializeVectorNeuronState_GPU2(LIFTimeDrivenModel_1_3_GPU2 ** NeuronModel_GPU2, int NumberOfVariables, float * InitialStateGPU, float * AuxStateGPU, float * StateGPU, double * LastUpdateGPU, double * LastSpikeTimeGPU, bool * InternalSpikeGPU, int SizeStates){
	if(blockIdx.x==0 && threadIdx.x==0){
		(*NeuronModel_GPU2)->InitializeVectorNeuronState_GPU2(NumberOfVariables, InitialStateGPU, AuxStateGPU, StateGPU, LastUpdateGPU, LastSpikeTimeGPU, InternalSpikeGPU, SizeStates);
	}
}

void LIFTimeDrivenModel_1_3_GPU_C_INTERFACE::InitializeVectorNeuronState_GPU2(){
	VectorNeuronState_GPU_C_INTERFACE *state = (VectorNeuronState_GPU_C_INTERFACE *) State;
	initializeVectorNeuronState_GPU2<<<1,1>>>(NeuronModel_GPU2, state->NumberOfVariables, state->InitialStateGPU, state->AuxStateGPU, state->VectorNeuronStates_GPU, state->LastUpdateGPU, state->LastSpikeTimeGPU, state->InternalSpikeGPU, state->SizeStates);
}


__global__ void DeleteClass_GPU2(LIFTimeDrivenModel_1_3_GPU2 ** NeuronModel_GPU2){
	if(blockIdx.x==0 && threadIdx.x==0){
		delete (*NeuronModel_GPU2); 
	}
}


void LIFTimeDrivenModel_1_3_GPU_C_INTERFACE::DeleteClassGPU2(){
	if (NeuronModel_GPU2 != 0){
		DeleteClass_GPU2 << <1, 1 >> >(NeuronModel_GPU2);
		cudaFree(NeuronModel_GPU2);
	}
}


bool LIFTimeDrivenModel_1_3_GPU_C_INTERFACE::CheckSynapseType(Interconnection * connection){
	int Type = connection->GetType();

	if (Type < N_TimeDependentNeuronState && Type >= 0){
		NeuronModel * model = connection->GetSource()->GetNeuronModel();
		//Synapse types that process input spikes 
		if (Type < N_TimeDependentNeuronState - 1){
			if (model->GetModelOutputActivityType() == OUTPUT_SPIKE){
				return true;
			}
			else{
			cout << "Synapses type " << Type << " of neuron model " << LIFTimeDrivenModel_1_3_GPU_C_INTERFACE::GetName() << " must receive spikes. The source model generates currents." << endl;
				return false;
			}
		}
		//Synapse types that process input current 
		if (Type == N_TimeDependentNeuronState - 1){
			if (model->GetModelOutputActivityType() == OUTPUT_CURRENT){
				connection->SetSubindexType(this->CurrentSynapses->GetNInputCurrentSynapsesPerNeuron(connection->GetTarget()->GetIndex_VectorNeuronState()));
				this->CurrentSynapses->IncrementNInputCurrentSynapsesPerNeuron(connection->GetTarget()->GetIndex_VectorNeuronState());
				return true;
			}
			else{
				cout << "Synapses type " << Type << " of neuron model " << LIFTimeDrivenModel_1_3_GPU_C_INTERFACE::GetName() << " must receive current. The source model generates spikes." << endl;
				return false;
			}
		}
	}
	else{
		cout << "Neuron model " << LIFTimeDrivenModel_1_3_GPU_C_INTERFACE::GetName() << " does not support input synapses of type " << Type << ". Just defined " << N_TimeDependentNeuronState << " synapses types." << endl;
		return false;
	}
}


std::map<std::string, boost::any> LIFTimeDrivenModel_1_3_GPU_C_INTERFACE::GetParameters() const {
	// Return a dictionary with the parameters
	std::map<std::string, boost::any> newMap = TimeDrivenNeuronModel_GPU_C_INTERFACE::GetParameters();
	newMap["e_exc"] = boost::any(this->e_exc);
	newMap["e_inh"] = boost::any(this->e_inh);
	newMap["e_leak"] = boost::any(this->e_leak);
	newMap["c_m"] = boost::any(float(this->c_m));
	newMap["v_thr"] = boost::any(this->v_thr);
	newMap["tau_exc"] = boost::any(this->tau_exc);
	newMap["tau_inh"] = boost::any(this->tau_inh);
	newMap["tau_ref"] = boost::any(this->tau_ref);
	newMap["g_leak"] = boost::any(float(this->g_leak));
	return newMap;
}

void LIFTimeDrivenModel_1_3_GPU_C_INTERFACE::SetParameters(std::map<std::string, boost::any> param_map) throw (EDLUTException){

	// Search for the parameters in the dictionary
	std::map<std::string,boost::any>::iterator it=param_map.find("e_exc");
	if (it!=param_map.end()){
		float newe_exc = boost::any_cast<float>(it->second);
		this->e_exc = newe_exc;
		param_map.erase(it);
	}

	it=param_map.find("e_inh");
	if (it!=param_map.end()){
		float newe_inh = boost::any_cast<float>(it->second);
		this->e_inh = newe_inh;
		param_map.erase(it);
	}

	it=param_map.find("e_leak");
	if (it!=param_map.end()){
		float newe_leak = boost::any_cast<float>(it->second);
		this->e_leak = newe_leak;
		param_map.erase(it);
	}

	it=param_map.find("c_m");
	if (it!=param_map.end()){
		float newc_m = boost::any_cast<float>(it->second);
		this->c_m = newc_m;
		param_map.erase(it);
	}

	it=param_map.find("v_thr");
	if (it!=param_map.end()){
		float newv_thr = boost::any_cast<float>(it->second);
		this->v_thr = newv_thr;
		param_map.erase(it);
	}

	it=param_map.find("tau_exc");
	if (it!=param_map.end()){
		float newtau_exc = boost::any_cast<float>(it->second);
		this->tau_exc = newtau_exc;
		param_map.erase(it);
	}

	it=param_map.find("tau_inh");
	if (it!=param_map.end()){
		float newtau_inh = boost::any_cast<float>(it->second);
		this->tau_inh = newtau_inh;
		param_map.erase(it);
	}

	it=param_map.find("tau_ref");
	if (it!=param_map.end()){
		float newtau_ref = boost::any_cast<float>(it->second);
		this->tau_ref = newtau_ref;
		param_map.erase(it);
	}

	it = param_map.find("g_leak");
	if (it != param_map.end()){
		float newg_leak = boost::any_cast<float>(it->second);
		this->g_leak = newg_leak;
		param_map.erase(it);
	}

	// Search for the parameters in the dictionary
	TimeDrivenNeuronModel_GPU_C_INTERFACE::SetParameters(param_map);
	return;
}


IntegrationMethod_GPU_C_INTERFACE * LIFTimeDrivenModel_1_3_GPU_C_INTERFACE::CreateIntegrationMethod(ModelDescription imethodDescription) throw (EDLUTException){
	return IntegrationMethodFactory_GPU_C_INTERFACE<LIFTimeDrivenModel_1_3_GPU_C_INTERFACE>::CreateIntegrationMethod_GPU(imethodDescription, (LIFTimeDrivenModel_1_3_GPU_C_INTERFACE*) this);
}


std::map<std::string, boost::any> LIFTimeDrivenModel_1_3_GPU_C_INTERFACE::GetDefaultParameters() {
	// Return a dictionary with the parameters
	std::map<std::string, boost::any> newMap = TimeDrivenNeuronModel_GPU_C_INTERFACE::GetDefaultParameters<LIFTimeDrivenModel_1_3_GPU_C_INTERFACE>();
	newMap["e_exc"] = boost::any(0.0f);
	newMap["e_inh"] = boost::any(-80.0e-3f);
	newMap["e_leak"] = boost::any(-65.0f);
	newMap["c_m"] = boost::any(float(110.0e-9));
	newMap["v_thr"] = boost::any(-50.0e-3f);
	newMap["tau_exc"] = boost::any(5.0e-3f);
	newMap["tau_inh"] = boost::any(10.0e-3f);
	newMap["tau_ref"] = boost::any(1.0e-3f);
	newMap["g_leak"] = boost::any(float(10.0e-9));
	return newMap;
}

NeuronModel* LIFTimeDrivenModel_1_3_GPU_C_INTERFACE::CreateNeuronModel(ModelDescription nmDescription){
	LIFTimeDrivenModel_1_3_GPU_C_INTERFACE * nmodel = new LIFTimeDrivenModel_1_3_GPU_C_INTERFACE();
	nmodel->SetParameters(nmDescription.param_map);
	return nmodel;
}

ModelDescription LIFTimeDrivenModel_1_3_GPU_C_INTERFACE::ParseNeuronModel(std::string FileName) throw (EDLUTFileException){
	FILE *fh;
	ModelDescription nmodel;
	nmodel.model_name = LIFTimeDrivenModel_1_3_GPU_C_INTERFACE::GetName();
	long Currentline = 0L;
	fh = fopen(FileName.c_str(), "rt");
	if (!fh) {
		throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_1_3_GPU_LOAD, ERROR_NEURON_MODEL_OPEN, REPAIR_NEURON_MODEL_NAME, Currentline, FileName.c_str());
	}

	Currentline = 1L;
	skip_comments(fh, Currentline);

	float param;
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_1_3_GPU_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_1_3_E_EXC, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["e_exc"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_1_3_GPU_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_1_3_E_INH, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["e_inh"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_1_3_GPU_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_1_3_E_LEAK, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["e_leak"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_1_3_GPU_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_1_3_V_THR, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["v_thr"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1 || param <= 0.0f) {
		throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_1_3_GPU_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_1_3_C_M, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["c_m"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_1_3_GPU_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_1_3_TAU_EXC, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["tau_exc"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_1_3_GPU_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_1_3_TAU_INH, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["tau_inh"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_1_3_GPU_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_1_3_TAU_REF, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["tau_ref"] = boost::any(param);

	skip_comments(fh, Currentline);
	if (fscanf(fh, "%f", &param) != 1) {
		throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_1_3_GPU_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_1_3_G_LEAK, REPAIR_NEURON_MODEL_VALUES, Currentline, FileName.c_str());
	}
	nmodel.param_map["g_leak"] = boost::any(param);

	skip_comments(fh, Currentline);
	try {
		ModelDescription intMethodDescription = TimeDrivenNeuronModel_GPU_C_INTERFACE::ParseIntegrationMethod<LIFTimeDrivenModel_1_3_GPU_C_INTERFACE>(fh, Currentline);
		nmodel.param_map["integration_method"] = boost::any(intMethodDescription);
	}
	catch (EDLUTException exc) {
		throw EDLUTFileException(exc, Currentline, FileName.c_str());
	}

	nmodel.param_map["name"] = boost::any(LIFTimeDrivenModel_1_3_GPU_C_INTERFACE::GetName());

	fclose(fh);

	return nmodel;
}

std::string LIFTimeDrivenModel_1_3_GPU_C_INTERFACE::GetName(){
	return "LIFTimeDrivenModel_1_3_GPU";
}
