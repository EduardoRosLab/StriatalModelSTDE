/***************************************************************************
 *                           CompressTableBasedModel.cpp                   *
 *                           -------------------                           *
 * copyright            : (C) 2015 by Francisco Naveros                    *
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

#include "../../include/neuron_model/CompressSynchronousTableBasedModel.h"
#include "../../include/neuron_model/CompressNeuronModelTable.h"
#include "../../include/neuron_model/VectorNeuronState.h"
#include "../../include/neuron_model/NeuronModel.h"

#include "../../include/spike/InternalSpike.h"
#include "../../include/spike/SynchronousTableBasedModelInternalSpike.h"
#include "../../include/spike/SynchronousTableBasedModelEvent.h"
#include "../../include/spike/EndRefractoryPeriodEvent.h"
#include "../../include/spike/Neuron.h"
#include "../../include/spike/Interconnection.h"
#include "../../include/spike/PropagatedSpike.h"

#include "../../include/simulation/Utils.h"

#include "../../include/openmp/openmp.h"

#include <cstring>


void CompressSynchronousTableBasedModel::LoadNeuronModel(string ConfigFile) throw (EDLUTFileException){
	FILE *fh;
	long Currentline = 0L;
	fh=fopen(ConfigFile.c_str(),"rt");
	if(fh){
		Currentline=1L;
		skip_comments(fh,Currentline);
		if(fscanf(fh,"%i",&this->NumStateVar)==1){
			unsigned int nv;

			// Initialize all vectors.
			this->StateVarTable = (CompressNeuronModelTable **) new CompressNeuronModelTable * [this->NumStateVar];
			this->StateVarOrderOriginalIndex = (unsigned int *) new unsigned int [this->NumStateVar];

			// Auxiliary table index
			this->TablesIndex = (unsigned int *) new unsigned int [this->NumStateVar];

			skip_comments(fh,Currentline);
			for(nv=0;nv<this->NumStateVar;nv++){
				if(fscanf(fh,"%i",&TablesIndex[nv])!=1){
					throw EDLUTFileException(TASK_TABLE_BASED_MODEL_LOAD, ERROR_TABLE_BASED_MODEL_INDEX, REPAIR_NEURON_MODEL_TABLE_TABLES_STRUCTURE, Currentline, ConfigFile.c_str());
				}
			}

			skip_comments(fh,Currentline);

			float InitValue;
			InitValues=new float[NumStateVar+1]();

			// Create a new initial state
       		this->State = (VectorNeuronState *) new VectorNeuronState(this->NumStateVar+1, false);

       		for(nv=0;nv<this->NumStateVar;nv++){
       			if(fscanf(fh,"%f",&InitValue)!=1){
					throw EDLUTFileException(TASK_TABLE_BASED_MODEL_LOAD, ERROR_TABLE_BASED_MODEL_INITIAL_VALUES, REPAIR_NEURON_MODEL_TABLE_TABLES_STRUCTURE, Currentline, ConfigFile.c_str());
       			} else {
					InitValues[nv+1]=InitValue;
       			}
       		}

			// Allocate temporal state vars

   			skip_comments(fh,Currentline);

   			if(fscanf(fh,"%i",&this->FiringIndex)==1){
   				skip_comments(fh,Currentline);
   				if(fscanf(fh,"%i",&this->EndFiringIndex)==1){
   					skip_comments(fh,Currentline);
   					if(fscanf(fh,"%i",&this->NumSynapticVar)==1){
               			skip_comments(fh,Currentline);

               			this->SynapticVar = (unsigned int *) new unsigned int [this->NumSynapticVar];
               			for(nv=0;nv<this->NumSynapticVar;nv++){
                  			if(fscanf(fh,"%i",&this->SynapticVar[nv])!=1){
								throw EDLUTFileException(TASK_TABLE_BASED_MODEL_LOAD, ERROR_TABLE_BASED_MODEL_SYNAPSE_INDEXS, REPAIR_NEURON_MODEL_TABLE_TABLES_STRUCTURE, Currentline, ConfigFile.c_str());
                  			}
                  		}

              			skip_comments(fh,Currentline);
              			if(fscanf(fh,"%i",&this->NumTables)==1){
              				unsigned int nt;
              				int tdeptables[MAXSTATEVARS];
              				int tstatevarpos,ntstatevarpos;

              				this->Tables = (CompressNeuronModelTable *) new CompressNeuronModelTable [this->NumTables];

              				// Update table links
              				for(nv=0;nv<this->NumStateVar;nv++){
								this->StateVarTable[nv] = this->Tables+TablesIndex[nv];
								this->StateVarTable[nv]->SetOutputStateVariableIndex(TablesIndex[nv]);
							}
              				this->FiringTable = this->Tables+FiringIndex;
              				this->EndFiringTable = this->Tables+EndFiringIndex;

              				for(nt=0;nt<this->NumTables;nt++){
								this->Tables[nt].LoadTableDescription(ConfigFile, fh, Currentline);
								this->Tables[nt].CalculateOutputTableDimensionIndex();
                   			}

              				this->NumTimeDependentStateVar = 0;
                 			for(nt=0;nt<this->NumStateVar;nt++){
         						for(nv=0;nv<this->StateVarTable[nt]->GetDimensionNumber() && this->StateVarTable[nt]->GetDimensionAt(nv)->statevar != 0;nv++);
            					if(nv<this->StateVarTable[nt]->GetDimensionNumber()){
            						tdeptables[nt]=1;
            						this->NumTimeDependentStateVar++;
            					}else{
               						tdeptables[nt]=0;
            					}
            				}

         					tstatevarpos=0;
         					ntstatevarpos=this->NumTimeDependentStateVar; // we place non-t-depentent variables in the end, so that they are evaluated afterwards
         					for(nt=0;nt<this->NumStateVar;nt++){
            					this->StateVarOrderOriginalIndex[(tdeptables[nt])?tstatevarpos++:ntstatevarpos++]=nt;
         					}

							
							//Load the neuron model time scale.
							skip_comments(fh,Currentline);
							char scale[MAXIDSIZE+1];
							if (fscanf(fh, " %"MAXIDSIZEC"[^ \n]", scale) == 1){
								if (strncmp(scale, "Milisecond", 10) == 0){
									//Millisecond scale
									this->SetTimeScale(MilisecondScale);
								}
								else if (strncmp(scale, "Second", 6) == 0){
									//Second scale
									this->SetTimeScale(SecondScale);
								}
								else{
									throw EDLUTFileException(TASK_TABLE_BASED_MODEL_LOAD, ERROR_TABLE_BASED_MODEL_TIME_SCALE, REPAIR_TABLE_BASED_MODEL_TIME_SCALE, Currentline, ConfigFile.c_str());
								}
							}
							else{
								throw EDLUTFileException(TASK_TABLE_BASED_MODEL_LOAD, ERROR_TABLE_BASED_MODEL_TIME_SCALE, REPAIR_TABLE_BASED_MODEL_TIME_SCALE, Currentline, ConfigFile.c_str());
							}

							//Load the optional parameter outputSpikeRestrictionTime
							skip_comments(fh, Currentline);
							if (fscanf(fh, "%lf", &this->SpikeRestrictionTime) == 1){
								if (this->SpikeRestrictionTime < 0.0){
									throw EDLUTFileException(TASK_TABLE_BASED_MODEL_LOAD, ERROR_TABLE_BASED_MODEL_SYNCHRONIZATION_PERIOD, REPAIR_NEURON_MODEL_VALUES, Currentline, ConfigFile.c_str());
								}
								else if (this->SpikeRestrictionTime == 0){
									this->inv_SpikeRestrictionTime = 0.0;
								}
								else{
									this->inv_SpikeRestrictionTime = 1.0 / this->SpikeRestrictionTime;
								}
							}
							else{
								printf("WARNING: The synchronization period has not been set in %s file. It has been fixed to zero seconds.\n", ConfigFile.c_str());
								this->SpikeRestrictionTime = 0.0;
								this->inv_SpikeRestrictionTime = 0.0;
							}
              			}else{
							throw EDLUTFileException(TASK_TABLE_BASED_MODEL_LOAD, ERROR_TABLE_BASED_MODEL_NUMBER_OF_TABLES, REPAIR_NEURON_MODEL_TABLE_TABLES_STRUCTURE, Currentline, ConfigFile.c_str());
      					}
      				}else{
						throw EDLUTFileException(TASK_TABLE_BASED_MODEL_LOAD, ERROR_TABLE_BASED_MODEL_NUMBER_OF_SYNAPSES, REPAIR_NEURON_MODEL_TABLE_TABLES_STRUCTURE, Currentline, ConfigFile.c_str());
          			}
				}else{
					throw EDLUTFileException(TASK_TABLE_BASED_MODEL_LOAD, ERROR_TABLE_BASED_MODEL_END_FIRING_INDEX, REPAIR_NEURON_MODEL_TABLE_TABLES_STRUCTURE, Currentline, ConfigFile.c_str());
          		}
			}else{
			throw EDLUTFileException(TASK_TABLE_BASED_MODEL_LOAD, ERROR_TABLE_BASED_MODEL_FIRING_INDEX, REPAIR_NEURON_MODEL_TABLE_TABLES_STRUCTURE, Currentline, ConfigFile.c_str());
			}
		}else{
		throw EDLUTFileException(TASK_TABLE_BASED_MODEL_LOAD, ERROR_TABLE_BASED_MODEL_NUMBER_OF_STATE_VARIABLES, REPAIR_NEURON_MODEL_TABLE_TABLES_STRUCTURE, Currentline, ConfigFile.c_str());
		}
	}else{
	throw EDLUTFileException(TASK_TABLE_BASED_MODEL_LOAD, ERROR_TABLE_BASED_MODEL_OPEN, REPAIR_NEURON_MODEL_TABLE_TABLES_STRUCTURE, Currentline, ConfigFile.c_str());
	}
	fclose(fh);

}


CompressSynchronousTableBasedModel::CompressSynchronousTableBasedModel(): CompressTableBasedModel(),
		SynchronousTableBasedModelTime(-1),SpikeRestrictionTime(0.0), inv_SpikeRestrictionTime(0.0) {
}

CompressSynchronousTableBasedModel::~CompressSynchronousTableBasedModel(){
}

void CompressSynchronousTableBasedModel::LoadNeuronModel() throw (EDLUTFileException){
	//Load neuron model parameters ("file.cfg")
	this->LoadNeuronModel(this->conf_filename);

	//Load neuron model look-up tables ("file.dat")
	this->LoadTables(this->tab_filename);

	this->CompactTables();
}

InternalSpike * CompressSynchronousTableBasedModel::GenerateInitialActivity(Neuron *  Cell){
	double Predicted = this->NextFiringPrediction(Cell->GetIndex_VectorNeuronState(),Cell->GetVectorNeuronState());
	InternalSpike * spike = 0;

	if(Predicted != NO_SPIKE_PREDICTED){
		Predicted += Cell->GetVectorNeuronState()->GetLastUpdateTime(Cell->GetIndex_VectorNeuronState());
		Predicted*=this->inv_time_scale;//REVISAR
		double virtual_predicted=GetSpikeRestrictionTimeMultiple(Predicted);
		spike=(InternalSpike *) new SynchronousTableBasedModelInternalSpike(Predicted,Cell->get_OpenMP_queue_index(),Cell,virtual_predicted);
	}

	this->GetVectorNeuronState()->SetNextPredictedSpikeTime(Cell->GetIndex_VectorNeuronState(),Predicted);

	return spike;
}


InternalSpike * CompressSynchronousTableBasedModel::ProcessInputSpike(Interconnection * inter, double time){

	InternalSpike * newEvent=0;

	int TargetIndex = inter->GetTargetNeuronModelIndex();
	Neuron * target = inter->GetTarget();


	// Update the neuron state until the current time
	if(time*this->time_scale != this->GetVectorNeuronState()->GetLastUpdateTime(TargetIndex)){
		if(time!=this->SynchronousTableBasedModelTime){
			this->SynchronousTableBasedModelTime=time;
			synchronousTableBasedModelEvent = new SynchronousTableBasedModelEvent(time,target->get_OpenMP_queue_index(), this->GetVectorNeuronState()->GetSizeState());
			newEvent=(InternalSpike*)synchronousTableBasedModelEvent;
		}
		this->UpdateState(TargetIndex,this->GetVectorNeuronState(),time*this->time_scale);
		synchronousTableBasedModelEvent->IncludeNewNeuron(target);
	}

	// Add the effect of the input spike
	this->SynapsisEffect(TargetIndex,inter);

	return newEvent;


}

InternalSpike * CompressSynchronousTableBasedModel::ProcessActivityAndPredictSpike(Neuron * target, double time){

	int TargetIndex=target->GetIndex_VectorNeuronState();

	InternalSpike * GeneratedSpike = 0;
	//check if the neuron is in the refractory period.
	if(time*this->time_scale>this->GetVectorNeuronState()->GetEndRefractoryPeriod(TargetIndex)){
		// Check if an spike will be fired
		double NextSpike = this->NextFiringPrediction(TargetIndex, this->GetVectorNeuronState());
		if (NextSpike != NO_SPIKE_PREDICTED){
			NextSpike += this->GetVectorNeuronState()->GetLastUpdateTime(TargetIndex);
			NextSpike*=this->inv_time_scale;
			if(NextSpike!=this->GetVectorNeuronState()->GetNextPredictedSpikeTime(TargetIndex)){
				double virtual_predicted=GetSpikeRestrictionTimeMultiple(NextSpike);
				GeneratedSpike=(InternalSpike *) new SynchronousTableBasedModelInternalSpike(NextSpike,target->get_OpenMP_queue_index(),target,virtual_predicted);
			}
		}
		this->GetVectorNeuronState()->SetNextPredictedSpikeTime(TargetIndex,NextSpike);
	}
	return GeneratedSpike;
}


EndRefractoryPeriodEvent * CompressSynchronousTableBasedModel::ProcessInternalSpike(InternalSpike *  OutputSpike){

	EndRefractoryPeriodEvent * endRefractoryPeriodEvent=0;

	Neuron * SourceCell = OutputSpike->GetSource();

	int SourceIndex=SourceCell->GetIndex_VectorNeuronState();

	VectorNeuronState * CurrentState = SourceCell->GetVectorNeuronState();

	this->UpdateState(SourceIndex,CurrentState,OutputSpike->GetTime()*this->time_scale);

	double EndRefractory = this->EndRefractoryPeriod(SourceIndex,CurrentState);

	if(this->FiringSubIndex!=this->EndFiringSubIndex){
		if(EndRefractory != NO_SPIKE_PREDICTED){
			EndRefractory += OutputSpike->GetTime()*this->time_scale; 
		}else{
			EndRefractory = (OutputSpike->GetTime()+DEF_REF_PERIOD)*this->time_scale;
		#ifdef _DEBUG
			cerr << "Warning: firing table and firing-end table discrepance (using def. ref. period)" << endl;
		#endif
		}
		endRefractoryPeriodEvent = new EndRefractoryPeriodEvent(EndRefractory, SourceCell->get_OpenMP_queue_index(), SourceCell);
	}
	CurrentState->SetEndRefractoryPeriod(SourceIndex,EndRefractory);

	return endRefractoryPeriodEvent;
}

InternalSpike * CompressSynchronousTableBasedModel::GenerateNextSpike(double time, Neuron * neuron){
	int SourceIndex=neuron->GetIndex_VectorNeuronState();

	VectorNeuronState * CurrentState = neuron->GetVectorNeuronState();

	this->UpdateState(SourceIndex,CurrentState,time*this->time_scale);

	double PredictedSpike = this->NextFiringPrediction(SourceIndex,CurrentState);

	InternalSpike * NextSpike = 0;

	if(PredictedSpike != NO_SPIKE_PREDICTED){
		PredictedSpike += CurrentState->GetEndRefractoryPeriod(SourceIndex);
		PredictedSpike*=this->inv_time_scale;
		if(PredictedSpike!=this->GetVectorNeuronState()->GetNextPredictedSpikeTime(SourceIndex)){
			double virtual_predicted=GetSpikeRestrictionTimeMultiple(PredictedSpike);
			NextSpike=(InternalSpike *) new SynchronousTableBasedModelInternalSpike(PredictedSpike,neuron->get_OpenMP_queue_index(),neuron,virtual_predicted);
		}

	}

	CurrentState->SetNextPredictedSpikeTime(SourceIndex,PredictedSpike);


	return NextSpike;

}


void CompressSynchronousTableBasedModel::InitializeStates(int N_neurons, int OpenMPQueueIndex){
	State->InitializeStates(N_neurons, InitValues);
}


double CompressSynchronousTableBasedModel::GetSpikeRestrictionTimeMultiple(double time){
	if(SpikeRestrictionTime>0){
		return ceil(time*inv_SpikeRestrictionTime)*SpikeRestrictionTime;
	}else{
		return time;
	}

}


std::map<std::string,boost::any> CompressSynchronousTableBasedModel::GetParameters() const {
	// Return a dictionary with the parameters
	std::map<std::string,boost::any> newMap = CompressTableBasedModel::GetParameters();
	return newMap;
}

std::map<std::string, boost::any> CompressSynchronousTableBasedModel::GetSpecificNeuronParameters(int index) const throw (EDLUTException){
	return GetParameters();
}

void CompressSynchronousTableBasedModel::SetParameters(std::map<std::string, boost::any> param_map) throw (EDLUTException){

	// Search for the parameters in the dictionary
	CompressTableBasedModel::SetParameters(param_map);

	//Load the configuration parameters and look-up tables from files.
	this->LoadNeuronModel();

	return;
}


std::map<std::string, boost::any> CompressSynchronousTableBasedModel::GetDefaultParameters() {
	// Return a dictionary with the parameters
	std::map<std::string, boost::any> newMap = CompressTableBasedModel::GetDefaultParameters();
	return newMap;
}

NeuronModel* CompressSynchronousTableBasedModel::CreateNeuronModel(ModelDescription nmDescription){
	CompressSynchronousTableBasedModel * nmodel = new CompressSynchronousTableBasedModel();
	nmodel->SetParameters(nmDescription.param_map);
	return nmodel;
}

ModelDescription CompressSynchronousTableBasedModel::ParseNeuronModel(std::string FileName) throw (EDLUTFileException){
	ModelDescription nmodel;
	nmodel.model_name = CompressSynchronousTableBasedModel::GetName();
	
	string new_conf_filename = FileName;
	nmodel.param_map["conf_filename"] = boost::any(new_conf_filename);

	string new_tab_filename = FileName.substr(0, FileName.length() - 3) + string("dat");
	nmodel.param_map["tab_filename"] = boost::any(new_tab_filename);

	nmodel.param_map["name"] = boost::any(CompressSynchronousTableBasedModel::GetName());

	return nmodel;
}

std::string CompressSynchronousTableBasedModel::GetName(){
	return "CompressSynchronousTableBasedModel";
}

std::map<std::string, std::string> CompressSynchronousTableBasedModel::GetNeuronModelInfo() {
	// Return a dictionary with the parameters
	std::map<std::string, std::string> newMap;
	newMap["info"] = std::string("CPU Event-driven neuron model. This model uses precomputed look-up tables to predict the neuron model behaviour");
	newMap["conf_filename"] = std::string("FILE.cfg: config filename");
	newMap["tab_filename"] = std::string("FILE.dat: look-up tables filename");
	return newMap;
}


