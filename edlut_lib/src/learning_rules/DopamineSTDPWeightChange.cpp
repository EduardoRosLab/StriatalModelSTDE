/***************************************************************************
 *                           DopamineSTDPWeightChange.cpp                  *
 *                           -------------------                           *
 * copyright            : (C) 2020 by Álvaro González-Redondo and 		   *
 * 						  Francisco Naveros  							   *
 * email                : alvarogr@ugr.es, fnaveros@ugr.es                 *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "../../include/learning_rules/DopamineSTDPWeightChange.h"

#include "../../include/learning_rules/DopamineSTDPState.h"

#include "../../include/spike/Interconnection.h"
#include "../../include/spike/Neuron.h"

#include "../../include/simulation/Utils.h"

#include "../../include/neuron_model/NeuronState.h"


DopamineSTDPWeightChange::DopamineSTDPWeightChange():WithPostSynaptic(){
	// Set the default values for the learning rule parameters
	this->SetParameters(DopamineSTDPWeightChange::GetDefaultParameters());
}

DopamineSTDPWeightChange::~DopamineSTDPWeightChange(){}


void DopamineSTDPWeightChange::InitializeConnectionState(unsigned int NumberOfSynapses, unsigned int NumberOfNeurons){
	this->State=(ConnectionState *) new DopamineSTDPState(
		NumberOfSynapses
		, this->tau_p
		, this->kph
		, this->kpl
		, this->tau_m
		, this->kmh
		, this->kml
		, this->tau_eligibility
		, this->tau_dopamine
		, this->da_max
		, this->da_min
		, this->syn_pre_inc
	);

	for (int i = 0; i < NumberOfSynapses; i++){
		this->State->SetStateVariableAt(i, DopamineSTDPState::eligibility_pos_index, 0.0);
		this->State->SetStateVariableAt(i, DopamineSTDPState::eligibility_neg_index, 0.0);
		this->State->SetStateVariableAt(i, DopamineSTDPState::dopamine_index, (this->da_max+this->da_min)*0.5);
	}
}

void DopamineSTDPWeightChange::ApplyPreSynapticSpike(Interconnection * Connection, double SpikeTime){
	if (Connection->GetTriggerConnection() == false){

		unsigned int LearningRuleIndex = Connection->GetLearningRuleIndex_withPost();

		// Apply synaptic activity decaying rule
		State->SetNewUpdateTime(LearningRuleIndex, SpikeTime, false);

		// Apply presynaptic spike
		State->ApplyPresynapticSpike(LearningRuleIndex);

		// Apply weight change
		Connection->SetWeight(State->GetStateVariableAt(LearningRuleIndex, DopamineSTDPState::weight_index));
	}
	else{
		Neuron * neuron = Connection->GetTarget();
		for (int i = 0; i<neuron->GetInputNumberWithPostSynapticLearning(); ++i){
			Interconnection * interi = neuron->GetInputConnectionWithPostSynapticLearningAt(i);

			if (interi->GetTriggerConnection() == false){

				int LearningRuleIndex = neuron->IndexInputLearningConnections[0][i];

				// Apply synaptic activity decaying rule
				State->SetNewUpdateTime(LearningRuleIndex, SpikeTime, true);

				// Apply dopamine increment
				State->incrementStateVariableAt(LearningRuleIndex, DopamineSTDPState::dopamine_index, this->increment_dopamine);

				// Update synaptic weight
				interi->SetWeight(State->GetStateVariableAt(LearningRuleIndex, DopamineSTDPState::weight_index));
			}

		}
	}

	return;
}

void DopamineSTDPWeightChange::ApplyPostSynapticSpike(Neuron * neuron, double SpikeTime){
	for (int i = 0; i<neuron->GetInputNumberWithPostSynapticLearning(); ++i){
		Interconnection * interi = neuron->GetInputConnectionWithPostSynapticLearningAt(i);

		int LearningRuleIndex = neuron->IndexInputLearningConnections[0][i];

		// Apply synaptic activity decaying rule
		State->SetNewUpdateTime(LearningRuleIndex, SpikeTime, true);

		// Apply postsynaptic spike
		State->ApplyPostsynapticSpike(LearningRuleIndex);

		// Update synaptic weight
		interi->SetWeight(State->GetStateVariableAt(LearningRuleIndex, DopamineSTDPState::weight_index));
	}
}

ModelDescription DopamineSTDPWeightChange::ParseLearningRule(FILE * fh) throw (EDLUTException) {
	ModelDescription lrule;

	float lkph;
	float lkpl;
	float ltau_p;
	float lkmh;
	float lkml;
	float ltau_m;
	float ltau_eligibility;
	float ltau_dopamine;
	float lincrement_dopamine;
	float lda_max;
	float lda_min;
	float lsyn_pre_inc;

	if(fscanf(fh,"%f",&lkph)!=1
		|| fscanf(fh, "%f",&lkpl)!=1
		|| fscanf(fh, "%f",&ltau_p)!=1
		|| fscanf(fh, "%f",&lkmh)!=1
		|| fscanf(fh, "%f",&lkml)!=1
		|| fscanf(fh, "%f", &ltau_m) != 1
		|| fscanf(fh, "%f", &ltau_eligibility) != 1
		|| fscanf(fh, "%f", &ltau_dopamine) != 1
		|| fscanf(fh, "%f", &lincrement_dopamine) != 1
		|| fscanf(fh, "%f", &lda_max) != 1
		|| fscanf(fh, "%f", &lda_min) != 1
		|| fscanf(fh, "%f", &lsyn_pre_inc) != 1
	 ) {
		throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_LEARNING_RULE_LOAD, REPAIR_DOPAMINE_STDP_WEIGHT_CHANGE_LOAD);
	}
	if (ltau_p <= 0
		|| ltau_m <= 0
		|| ltau_eligibility <= 0
		|| ltau_dopamine <= 0
	) {
		throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_DOPAMINE_STDP_WEIGHT_CHANGE_TAUS, REPAIR_LEARNING_RULE_VALUES);
	}

	lrule.model_name = DopamineSTDPWeightChange::GetName();
	lrule.param_map["k_plu_hig"] = boost::any(lkph);
	lrule.param_map["k_plu_low"] = boost::any(lkpl);
	lrule.param_map["tau_plu"] = boost::any(ltau_p);
	lrule.param_map["k_min_hig"] = boost::any(lkmh);
	lrule.param_map["k_min_low"] = boost::any(lkml);
	lrule.param_map["tau_min"] = boost::any(ltau_m);
	lrule.param_map["tau_eli"] = boost::any(ltau_eligibility);
	lrule.param_map["tau_dop"] = boost::any(ltau_dopamine);
	lrule.param_map["inc_dop"] = boost::any(lincrement_dopamine);
	lrule.param_map["dop_max"] = boost::any(lda_max);
	lrule.param_map["dop_min"] = boost::any(lda_min);
	lrule.param_map["syn_pre_inc"] = boost::any(lsyn_pre_inc);
	return lrule;
}

/////////////////////////////////////////////////////////////////////////////////////7
void DopamineSTDPWeightChange::SetParameters(std::map<std::string, boost::any> param_map) throw (EDLUTException){

	// Search for the parameters in the dictionary
	std::map<std::string, boost::any>::iterator it = param_map.find("k_plu_hig");
	if (it != param_map.end()){
		this->kph = boost::any_cast<float>(it->second);;
		param_map.erase(it);
	}

	it = param_map.find("k_plu_low");
	if (it != param_map.end()){
		this->kpl = boost::any_cast<float>(it->second);;
		param_map.erase(it);
	}

	it=param_map.find("tau_plu");
	if (it!=param_map.end()){
		float new_tau_p = boost::any_cast<float>(it->second);
		if (new_tau_p<=0) {
			throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_DOPAMINE_STDP_WEIGHT_CHANGE_TAUS,
								 REPAIR_LEARNING_RULE_VALUES);
		}
		this->tau_p = new_tau_p;
		param_map.erase(it);
	}

	it = param_map.find("k_min_hig");
	if (it != param_map.end()){
		this->kmh = boost::any_cast<float>(it->second);;
		param_map.erase(it);
	}

	it = param_map.find("k_min_low");
	if (it != param_map.end()){
		this->kml = boost::any_cast<float>(it->second);;
		param_map.erase(it);
	}

	it=param_map.find("tau_min");
	if (it!=param_map.end()){
		float new_tau_m = boost::any_cast<float>(it->second);
		if (new_tau_m<=0) {
			throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_DOPAMINE_STDP_WEIGHT_CHANGE_TAUS,
								 REPAIR_LEARNING_RULE_VALUES);
		}
		this->tau_m = new_tau_m;
		param_map.erase(it);
	}

	it = param_map.find("tau_eli");
	if (it != param_map.end()){
		float newtaueli = boost::any_cast<float>(it->second);
		if (newtaueli <= 0) {
			throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_DOPAMINE_STDP_WEIGHT_CHANGE_TAUS,
				REPAIR_LEARNING_RULE_VALUES);
		}
		this->tau_eligibility = newtaueli;
		param_map.erase(it);
	}

	it = param_map.find("tau_dop");
	if (it != param_map.end()){
		float newtaudop = boost::any_cast<float>(it->second);
		if (newtaudop <= 0) {
			throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_DOPAMINE_STDP_WEIGHT_CHANGE_TAUS,
				REPAIR_LEARNING_RULE_VALUES);
		}
		this->tau_dopamine = newtaudop;
		param_map.erase(it);
	}

	it = param_map.find("inc_dop");
	if (it != param_map.end()){
		float newincdop = boost::any_cast<float>(it->second);
		if (newincdop <= 0) {
//			throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_DOPAMINE_STDP_WEIGHT_CHANGE_TAUS,
//				REPAIR_LEARNING_RULE_VALUES);
		}
		this->increment_dopamine = newincdop;
		param_map.erase(it);
	}

	it = param_map.find("dop_max");
	if (it != param_map.end()){
		float newdopmax = boost::any_cast<float>(it->second);
		if (newdopmax < 0) {
			//			throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_DOPAMINE_STDP_WEIGHT_CHANGE_TAUS,
			//				REPAIR_LEARNING_RULE_VALUES);
		}
		this->da_max = newdopmax;
		param_map.erase(it);
	}

	it = param_map.find("dop_min");
	if (it != param_map.end()){
		float newdopmin = boost::any_cast<float>(it->second);
		if (newdopmin < 0) {
			//			throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_DOPAMINE_STDP_WEIGHT_CHANGE_TAUS,
			//				REPAIR_LEARNING_RULE_VALUES);
		}
		this->da_min = newdopmin;
		param_map.erase(it);
	}

	it = param_map.find("syn_pre_inc");
	if (it != param_map.end()){
		float new_param_value = boost::any_cast<float>(it->second);
		if (new_param_value <= 0) {
			//			throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_DOPAMINE_STDP_WEIGHT_CHANGE_TAUS,
			//				REPAIR_LEARNING_RULE_VALUES);
		}
		this->syn_pre_inc = new_param_value;
		param_map.erase(it);
	}

	WithPostSynaptic::SetParameters(param_map);
}


ostream & DopamineSTDPWeightChange::PrintInfo(ostream & out){
	out << "- DopamineSTDP Learning Rule: " << endl;
	out << "\t k_plu_hig:" << this->kph << endl;
	out << "\t k_plu_low:" << this->kpl << endl;
	out << "\t tau_plu:" << this->tau_p << endl;
	out << "\t k_min_hig:" << this->kmh << endl;
	out << "\t k_min_low:" << this->kml << endl;
	out << "\t tau_min:" << this->tau_m << endl;
	out << "\t tau_eli:" << this->tau_eligibility << endl;
	out << "\t tau_dop:" << this->tau_dopamine << endl;
	out << "\t inc_dop:" << this->increment_dopamine << endl;
	out << "\t dop_max:" << this->da_max << endl;
	out << "\t dop_min:" << this->da_min << endl;
	out << "\t syn_pre_inc:" << this->syn_pre_inc << endl;

	return out;
}

// float DopamineSTDPWeightChange::GetMaxWeightChangeLTP() const{
// 	return this->MaxChangeLTP;
// }

// void DopamineSTDPWeightChange::SetMaxWeightChangeLTP(float NewMaxChange){
// 	this->MaxChangeLTP = NewMaxChange;
// }

// float DopamineSTDPWeightChange::GetMaxWeightChangeLTD() const{
// 	return this->MaxChangeLTD;
// }

// void DopamineSTDPWeightChange::SetMaxWeightChangeLTD(float NewMaxChange){
// 	this->MaxChangeLTD = NewMaxChange;
// }

LearningRule* DopamineSTDPWeightChange::CreateLearningRule(ModelDescription lrDescription){
	DopamineSTDPWeightChange * lrule = new DopamineSTDPWeightChange();
	lrule->SetParameters(lrDescription.param_map);
	return lrule;
}

std::map<std::string,boost::any> DopamineSTDPWeightChange::GetParameters(){
	std::map<std::string,boost::any> newMap = WithPostSynaptic::GetParameters();
	newMap["k_plu_hig"] = boost::any(this->kph);
	newMap["k_plu_low"] = boost::any(this->kpl);
	newMap["tau_plu"] = boost::any(this->tau_p);
	newMap["k_min_hig"] = boost::any(this->kmh);
	newMap["k_min_low"] = boost::any(this->kml);
	newMap["tau_min"] = boost::any(this->tau_m);
	newMap["tau_eli"] = boost::any(this->tau_eligibility);
	newMap["tau_dop"] = boost::any(this->tau_dopamine);
	newMap["inc_dop"] = boost::any(this->increment_dopamine);
	newMap["dop_max"] = boost::any(this->da_max);
	newMap["dop_min"] = boost::any(this->da_min);
	newMap["syn_pre_inc"] = boost::any(this->syn_pre_inc);
	return newMap;
}

std::map<std::string,boost::any> DopamineSTDPWeightChange::GetDefaultParameters(){
	std::map<std::string,boost::any> newMap = WithPostSynaptic::GetDefaultParameters();
	newMap["k_plu_hig"] = boost::any(1.2f);
	newMap["k_plu_low"] = boost::any(-0.3f);
	newMap["tau_plu"] = boost::any(0.100f);
	newMap["k_min_hig"] = boost::any(0.0f);
	newMap["k_min_low"] = boost::any(-0.4f);
	newMap["tau_min"] = boost::any(0.100f);
	newMap["tau_eli"] = boost::any(0.200f);
	newMap["tau_dop"] = boost::any(0.300f);
	newMap["inc_dop"] = boost::any(0.01f);
	newMap["dop_max"] = boost::any(30.0f);
	newMap["dop_min"] = boost::any(5.0f);
	newMap["syn_pre_inc"] = boost::any(0.0f);
	return newMap;
}
