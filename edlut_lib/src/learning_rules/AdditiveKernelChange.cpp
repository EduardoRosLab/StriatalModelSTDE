/***************************************************************************
 *                           AdditiveKernelChange.cpp                      *
 *                           -------------------                           *
 * copyright            : (C) 2010 by Jesus Garrido                        *
 * email                : jgarrido@atc.ugr.es                              *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "../../include/learning_rules/AdditiveKernelChange.h"

#include "../../include/simulation/NetworkDescription.h"
#include "../../include/spike/Interconnection.h"
#include "../../include/spike/Neuron.h"

#include <boost/any.hpp>

#include "../../include/simulation/Utils.h"

#include "../../include/openmp/openmp.h"

#include <cmath>


AdditiveKernelChange::AdditiveKernelChange():WithoutPostSynaptic(){
	this->SetParameters(AdditiveKernelChange::GetDefaultParameters());
}

AdditiveKernelChange::~AdditiveKernelChange(){
}


int AdditiveKernelChange::GetNumberOfVar() const{
	return 2;
}

void AdditiveKernelChange::SetParameters(std::map<std::string, boost::any> param_map) throw (EDLUTException){

	// Search for the parameters in the dictionary
	std::map<std::string,boost::any>::iterator it=param_map.find("kernel_peak");
	if (it!=param_map.end()){
		float newkernelpeak = boost::any_cast<float>(it->second);
		if (newkernelpeak <= 0){
			throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_ADDITIVE_KERNEL_CHANGE_VALUES, REPAIR_LEARNING_RULE_VALUES);
		}
		this->kernelpeak = newkernelpeak;
		param_map.erase(it);
	}

	// Search for the parameters in the dictionary
	it=param_map.find("fixed_change");
	if (it!=param_map.end()){
		this->fixwchange = boost::any_cast<float>(it->second);
		param_map.erase(it);
	}

	// Search for the parameters in the dictionary
	it=param_map.find("kernel_change");
	if (it!=param_map.end()){
		this->kernelwchange = boost::any_cast<float>(it->second);
		param_map.erase(it);
	}

	WithoutPostSynaptic::SetParameters(param_map);

}

ModelDescription AdditiveKernelChange::ParseLearningRule(FILE * fh) throw (EDLUTException) {
	float maxpos, fixwchange, kernelwchange;
	if(fscanf(fh,"%f",&maxpos)!=1 ||
	   fscanf(fh,"%f",&fixwchange)!=1 ||
	   fscanf(fh,"%f",&kernelwchange)!=1){
		throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_LEARNING_RULE_LOAD, REPAIR_ADDITIVE_KERNEL_CHANGE_LOAD);
	}
	if (maxpos <= 0){
		throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_ADDITIVE_KERNEL_CHANGE_VALUES, REPAIR_LEARNING_RULE_VALUES);
	}

	ModelDescription lrule;
	lrule.model_name = "AdditiveRule";
	lrule.param_map["kernel_peak"] = boost::any(maxpos);
	lrule.param_map["fixed_change"] = boost::any(fixwchange);
	lrule.param_map["kernel_change"] = boost::any(kernelwchange);
	return lrule;
}

void AdditiveKernelChange::ApplyPreSynapticSpike(Interconnection * Connection, double SpikeTime){


	if (Connection->GetTriggerConnection() == false){
		int LearningRuleIndex = Connection->GetLearningRuleIndex_withoutPost();

		// Second case: the weight change is linked to this connection
		Connection->IncrementWeight(this->fixwchange);

		// Update the presynaptic activity
		State->SetNewUpdateTime(LearningRuleIndex, SpikeTime, false);

		// Add the presynaptic spike influence
		State->ApplyPresynapticSpike(LearningRuleIndex);

	}
	else{
		Neuron * TargetNeuron = Connection->GetTarget();

		for (int i = 0; i<TargetNeuron->GetInputNumberWithoutPostSynapticLearning(); ++i){
			Interconnection * interi=TargetNeuron->GetInputConnectionWithoutPostSynapticLearningAt(i);

			if(interi->GetTriggerConnection()==false){
				// Apply sinaptic plasticity driven by teaching signal
				int LearningRuleIndex = interi->GetLearningRuleIndex_withoutPost();

				// Update the presynaptic activity
				State->SetNewUpdateTime(LearningRuleIndex, SpikeTime, false);
				// Update synaptic weight
				interi->IncrementWeight(this->kernelwchange*State->GetPresynapticActivity(LearningRuleIndex));
			}
		}
	}
}


std::map<std::string,boost::any> AdditiveKernelChange::GetParameters(){
	std::map<std::string,boost::any> newMap = WithoutPostSynaptic::GetParameters();
	newMap["kernel_peak"] = boost::any(this->kernelpeak);
	newMap["fixed_change"] = boost::any(this->fixwchange);
	newMap["kernel_change"] = boost::any(this->kernelwchange);
	return newMap;
}


std::map<std::string,boost::any> AdditiveKernelChange::GetDefaultParameters(){
	std::map<std::string,boost::any> newMap = WithoutPostSynaptic::GetDefaultParameters();
	newMap["kernel_peak"] = boost::any(0.100f);
	newMap["fixed_change"] = boost::any(0.001f);
	newMap["kernel_change"] = boost::any(0.010f);
	return newMap;
}


