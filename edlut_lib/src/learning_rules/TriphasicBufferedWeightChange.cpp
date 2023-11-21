/***************************************************************************
 *                           TriphasicBufferedWeightChange.cpp           *
 *                           -------------------                           *
 * copyright            : (C) 2016 by Francisco Naveros and Niceto Luque   *
 * email                : fnaveros@ugr.es nluque@ugr.es                    *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/


#include "../../include/learning_rules/TriphasicBufferedWeightChange.h"
#include "../../include/learning_rules/BufferedActivityTimes.h"

#include "../../include/spike/Interconnection.h"
#include "../../include/spike/Neuron.h"
#include <boost/any.hpp>

#include "../../include/simulation/Utils.h"

#include "../../include/openmp/openmp.h"

TriphasicBufferedWeightChange::TriphasicBufferedWeightChange() :WithPostSynaptic(),
		bufferedActivityTimesPresynaptic(0), bufferedActivityTimesPostsynaptic(0), kernelLookupTable(0){
	// Set the default values for the learning rule parameters
	this->SetParameters(TriphasicBufferedWeightChange::GetDefaultParameters());
}

TriphasicBufferedWeightChange::~TriphasicBufferedWeightChange(){
	if(bufferedActivityTimesPresynaptic!=0){
		delete bufferedActivityTimesPresynaptic;
	}
	if(bufferedActivityTimesPostsynaptic!=0){
		delete bufferedActivityTimesPostsynaptic;
	}
	if(kernelLookupTable!=0){
		delete kernelLookupTable;
	}
}


void TriphasicBufferedWeightChange::InitializeConnectionState(unsigned int NumberOfSynapses, unsigned int NumberOfNeurons){
	double step_size = 0.0001; //defined in s.
	double tolerance = 1e-6;
	this->maxTimeMeasured = 3.0f*alpha;
	while (1){
		this->maxTimeMeasured += step_size;
		if ( abs((1.0f - ((maxTimeMeasured*maxTimeMeasured)/(alpha*alpha))) * exp((-abs(maxTimeMeasured)/alpha))) < tolerance){
			break;
		}
	}
	
	this->N_elements = this->maxTimeMeasured / step_size + 1;
	kernelLookupTable = new float[this->N_elements];
	this->inv_maxTimeMeasured = 1.0f / this->maxTimeMeasured;
	//Precompute the kernel in the look-up table.
	for (int i = 0; i<N_elements; i++){
		double time = maxTimeMeasured*i / (N_elements);
		kernelLookupTable[i] = (1.0f - ((time*time)/(alpha*alpha))) * exp((-abs(time)/alpha));
	}

	//Inicitialize de buffer of activity
	bufferedActivityTimesPresynaptic = new BufferedActivityTimes(NumberOfNeurons);
	bufferedActivityTimesPostsynaptic = new BufferedActivityTimes(NumberOfNeurons);
}


ModelDescription TriphasicBufferedWeightChange::ParseLearningRule(FILE * fh) throw (EDLUTException) {
	ModelDescription lrule;

	float _a, _alpha;
	if(fscanf(fh,"%f",&_a)!=1 || fscanf(fh,"%f",&_alpha)!=1) {
// TODO: ARREGLAR ESTO
//throw EDLUTException(TASK_LEARNING_RULE_LOAD, ERROR_LEARNING_RULE_LOAD, REPAIR_COS_WEIGHT_CHANGE_LOAD);
	}

	lrule.model_name = TriphasicBufferedWeightChange::GetName();
	lrule.param_map["a"] = boost::any(_a);
	lrule.param_map["alpha"] = boost::any(_alpha);
	return lrule;
}

void TriphasicBufferedWeightChange::SetParameters(std::map<std::string, boost::any> param_map) throw (EDLUTException){

	// Search for the parameters in the dictionary
	std::map<std::string,boost::any>::iterator it=param_map.find("a");
	if (it!=param_map.end()){
		float newa = boost::any_cast<float>(it->second);
		this->a = newa;
		param_map.erase(it);
	}

	// Search for the parameters in the dictionary
	it=param_map.find("alpha");
	if (it!=param_map.end()){
		float newalpha = boost::any_cast<float>(it->second);
		this->alpha = newalpha;
		param_map.erase(it);
	}

	WithPostSynaptic::SetParameters(param_map);
}

void TriphasicBufferedWeightChange::ApplyPreSynapticSpike(Interconnection * Connection, double SpikeTime){

    // Insert presynaptic spike in buffer
    Neuron * TargetNeuron = Connection->GetTarget();
    int neuron_index = TargetNeuron->GetIndex();
    int synapse_index = Connection->LearningRuleIndex_withPost_insideTargetNeuron;
    this->bufferedActivityTimesPresynaptic->InsertElement(neuron_index, SpikeTime, SpikeTime - this->maxTimeMeasured - alpha, synapse_index);

    // Extract the postsynaptic spike times
    int N_spikes = bufferedActivityTimesPostsynaptic->ProcessElements(neuron_index, SpikeTime - this->maxTimeMeasured - alpha);
    SpikeData * spike_data = bufferedActivityTimesPostsynaptic->GetOutputSpikeData();
    
	float value = 0;
    for (int i = 0; i < N_spikes; i++){
        double ElapsedTime = SpikeTime - spike_data[i].time;
        int tableIndex = abs(ElapsedTime-alpha)*this->N_elements*this->inv_maxTimeMeasured;
        value += this->kernelLookupTable[tableIndex];
    }
	Connection->IncrementWeight(this->a * value);
}



void TriphasicBufferedWeightChange::ApplyPostSynapticSpike(Neuron * neuron, double SpikeTime){
    // Insert postsynaptic spike in buffer
    int neuron_index = neuron->GetIndex();
    this->bufferedActivityTimesPostsynaptic->InsertElement(neuron_index, SpikeTime, SpikeTime - this->maxTimeMeasured + alpha, 0);

	// Extract the presynaptic spike times
	int N_spikes = bufferedActivityTimesPresynaptic->ProcessElements(neuron_index, SpikeTime - this->maxTimeMeasured + alpha);
	SpikeData * spike_data = bufferedActivityTimesPresynaptic->GetOutputSpikeData();

	for (int j = 0; j < N_spikes; j++){
		double ElapsedTime = SpikeTime - spike_data[j].time;
		int tableIndex = abs(ElapsedTime + alpha)*this->N_elements*this->inv_maxTimeMeasured;
		neuron->GetInputConnectionWithPostSynapticLearningAt(spike_data[j].synapse_index)->IncrementWeight(this->a * this->kernelLookupTable[tableIndex]);
	}
}


ostream & TriphasicBufferedWeightChange::PrintInfo(ostream & out){

	out << "- TriphasicBufferedKernel Learning Rule: " << endl;
	out << "\t a:" << this->a << endl;
	out << "\t alpha:" << this->alpha << endl;
	return out;
}

LearningRule* TriphasicBufferedWeightChange::CreateLearningRule(ModelDescription lrDescription){
	TriphasicBufferedWeightChange * lrule = new TriphasicBufferedWeightChange();
    lrule->SetParameters(lrDescription.param_map);
	return lrule;
}

std::map<std::string,boost::any> TriphasicBufferedWeightChange::GetParameters(){
	std::map<std::string,boost::any> newMap = WithPostSynaptic::GetParameters();
	newMap["a"] = boost::any(this->a);
	newMap["alpha"] = boost::any(this->alpha);
	return newMap;
}


std::map<std::string,boost::any> TriphasicBufferedWeightChange::GetDefaultParameters(){
	std::map<std::string,boost::any> newMap = WithPostSynaptic::GetDefaultParameters();
	newMap["a"] = boost::any(0.100f);
	newMap["alpha"] = boost::any(0.005f);
	return newMap;
}
