/***************************************************************************
 *                           VoglesSTDPWeightChange.cpp                    *
 *                           -------------------                           *
 * copyright            : (C) 2020 by Francisco Naveros and Álvaro González*
 * email                : fnaveros@ugr.es, alvarogr@ugr.es                 *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "../../include/learning_rules/VogelsSTDPWeightChange.h"

#include "../../include/learning_rules/VogelsSTDPState.h"

#include "../../include/spike/Interconnection.h"
#include "../../include/spike/Neuron.h"

#include "../../include/simulation/Utils.h"

#include "../../include/neuron_model/NeuronState.h"


VogelsSTDPWeightChange::VogelsSTDPWeightChange():WithPostSynaptic(){
	// Set the default values for the learning rule parameters
	this->SetParameters(VogelsSTDPWeightChange::GetDefaultParameters());
}

VogelsSTDPWeightChange::~VogelsSTDPWeightChange(){

}


void VogelsSTDPWeightChange::InitializeConnectionState(unsigned int NumberOfSynapses, unsigned int NumberOfNeurons){
	this->State = (ConnectionState *) new VogelsSTDPState(NumberOfSynapses, this->tauKernel);
}

void VogelsSTDPWeightChange::ApplyPreSynapticSpike(Interconnection * Connection,double SpikeTime){
	unsigned int LearningRuleIndex = Connection->GetLearningRuleIndex_withPost();

	// Apply synaptic activity decaying rule
	State->SetNewUpdateTime(LearningRuleIndex, SpikeTime, false);

	// Apply presynaptic spike
	State->ApplyPresynapticSpike(LearningRuleIndex);

	// Apply weight change


	Connection->IncrementWeight(ConstantChange + this->MaxChangeKernel*State->GetPostsynapticActivity(LearningRuleIndex));

	return;
}


void VogelsSTDPWeightChange::ApplyPostSynapticSpike(Neuron * neuron, double SpikeTime){
	for (int i = 0; i<neuron->GetInputNumberWithPostSynapticLearning(); ++i){
		Interconnection * interi = neuron->GetInputConnectionWithPostSynapticLearningAt(i);

		int LearningRuleIndex = neuron->IndexInputLearningConnections[0][i];

		// Apply synaptic activity decaying rule
		State->SetNewUpdateTime(LearningRuleIndex, SpikeTime, true);

		// Apply postsynaptic spike
		State->ApplyPostsynapticSpike(LearningRuleIndex);

		// Update synaptic weight
		interi->IncrementWeight(this->MaxChangeKernel*State->GetPresynapticActivity(LearningRuleIndex));
	}

}

ModelDescription VogelsSTDPWeightChange::ParseLearningRule(FILE * fh) throw (EDLUTException) {
	ModelDescription lrule;

	float lMaxChangeKernel, ltauKernel, lConstantChange;
	if (fscanf(fh, "%f", &lMaxChangeKernel) != 1 ||
		fscanf(fh, "%f", &ltauKernel) != 1 ||
		fscanf(fh, "%f", &lConstantChange) != 1){
		throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_LEARNING_RULE_LOAD, REPAIR_VOGELS_STDP_WEIGHT_CHANGE_LOAD);
	}
	if (ltauKernel <= 0){
		throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_VOGELS_STDP_WEIGHT_CHANGE_TAU, REPAIR_LEARNING_RULE_VALUES);
	}

	lrule.model_name = VogelsSTDPWeightChange::GetName();
	lrule.param_map["max_kernel_change"] = boost::any(lMaxChangeKernel);
	lrule.param_map["tau_kernel_change"] = boost::any(ltauKernel);
	lrule.param_map["const_change"] = boost::any(lConstantChange);

	return lrule;
}

void VogelsSTDPWeightChange::SetParameters(std::map<std::string, boost::any> param_map) throw (EDLUTException){

	// Search for the parameters in the dictionary
	std::map<std::string, boost::any>::iterator it = param_map.find("max_kernel_change");
	if (it != param_map.end()){
		this->MaxChangeKernel = boost::any_cast<float>(it->second);;
		param_map.erase(it);
	}
		
	it=param_map.find("tau_kernel_change");
	if (it!=param_map.end()){
		float newtauLTP = boost::any_cast<float>(it->second);
		if (newtauLTP<=0) {
			throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_VOGELS_STDP_WEIGHT_CHANGE_TAU, REPAIR_LEARNING_RULE_VALUES);
		}
		this->tauKernel = newtauLTP;
		param_map.erase(it);
	}

	it = param_map.find("const_change");
	if (it != param_map.end()){
		this->ConstantChange = boost::any_cast<float>(it->second);;
		param_map.erase(it);
	}

	WithPostSynaptic::SetParameters(param_map);
}


ostream & VogelsSTDPWeightChange::PrintInfo(ostream & out){
	out << "- STDP Learning Rule: " << endl;
	out << "\t Max change kernel:" << this->MaxChangeKernel << endl;
	out << "\t Tau kernel:" << this->tauKernel << endl;
	out << "\t Constant change:" << this->ConstantChange << endl;
	return out;
}


LearningRule* VogelsSTDPWeightChange::CreateLearningRule(ModelDescription lrDescription){
	VogelsSTDPWeightChange * lrule = new VogelsSTDPWeightChange();
	lrule->SetParameters(lrDescription.param_map);
	return lrule;
}

std::map<std::string,boost::any> VogelsSTDPWeightChange::GetParameters(){
	std::map<std::string,boost::any> newMap = WithPostSynaptic::GetParameters();
	newMap["max_kernel_change"] = boost::any(this->MaxChangeKernel);
	newMap["tau_kernel_change"] = boost::any(this->tauKernel);
	newMap["const_change"] = boost::any(this->ConstantChange);
	return newMap;
}

std::map<std::string,boost::any> VogelsSTDPWeightChange::GetDefaultParameters(){
	std::map<std::string,boost::any> newMap = WithPostSynaptic::GetDefaultParameters();
	newMap["max_kernel_change"] = boost::any(0.012f);
	newMap["tau_kernel_change"] = boost::any(0.020f);
	newMap["const_change"] = boost::any(0.001f);
	return newMap;
}
