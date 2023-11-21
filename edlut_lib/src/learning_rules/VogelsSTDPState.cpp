/***************************************************************************
 *                           VogelsSTDPState.cpp                           *
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

#include "../../include/learning_rules/VogelsSTDPState.h"

#include "../../include/simulation/ExponentialTable.h"

#include <cmath>
#include <stdio.h>
#include <float.h>

VogelsSTDPState::VogelsSTDPState(int NumSynapses, float NewtauKernel) : ConnectionState(NumSynapses, 2), tauKernel(NewtauKernel){
	inv_tauKernel = 1.0f / NewtauKernel;
}

VogelsSTDPState::~VogelsSTDPState() {
}

unsigned int VogelsSTDPState::GetNumberOfPrintableValues(){
	return ConnectionState::GetNumberOfPrintableValues()+1;
}

double VogelsSTDPState::GetPrintableValuesAt(unsigned int index, unsigned int position){
	if (position<ConnectionState::GetNumberOfPrintableValues()){
		return ConnectionState::GetStateVariableAt(index, position);
	} else if (position==ConnectionState::GetNumberOfPrintableValues()) {
		return this->tauKernel;
	} else return -1;
}

//float VogelsSTDPState::GetPresynapticActivity(unsigned int index){
//	return this->GetStateVariableAt(index, 0);
//}

//float VogelsSTDPState::GetPostsynapticActivity(unsigned int index){
//	return this->GetStateVariableAt(index, 1);
//}


//void VogelsSTDPState::SetNewUpdateTime(unsigned int index, double NewTime, bool pre_post){
//	float PreActivity = this->GetPresynapticActivity(index);
//	float PostActivity = this->GetPostsynapticActivity(index);
//
//	float ElapsedTime=(float)(NewTime - this->GetLastUpdateTime(index));
//
//	//// Accumulate activity since the last update time
//	PreActivity *= exp(-ElapsedTime*this->inv_LTPTau);
//	PostActivity *= exp(-ElapsedTime*this->inv_LTDTau);
//
//	// Store the activity in state variables
//	//this->SetStateVariableAt(index, 0, PreActivity);
//	//this->SetStateVariableAt(index, 1, PostActivity);
//	this->SetStateVariableAt(index, 0, PreActivity, PostActivity);
//
//	this->SetLastUpdateTime(index, NewTime);
//}

void VogelsSTDPState::SetNewUpdateTime(unsigned int index, double NewTime, bool pre_post){
	float ElapsedTime=(float)(NewTime - this->GetLastUpdateTime(index));

    //Accumulate activity since the last update time
	this->multiplyStateVariableAt(index, 0, ExponentialTable::GetResult(-ElapsedTime*this->inv_tauKernel));
    //Accumulate activity since the last update time
	this->multiplyStateVariableAt(index, 1, ExponentialTable::GetResult(-ElapsedTime*this->inv_tauKernel));
	
	this->SetLastUpdateTime(index, NewTime);
}



void VogelsSTDPState::ApplyPresynapticSpike(unsigned int index){
	// Increment the activity in the state variable
	//this->SetStateVariableAt(index, 0, this->GetPresynapticActivity(index)+1.0f);
	this->incrementStateVariableAt(index, 0, 1.0f);
}

void VogelsSTDPState::ApplyPostsynapticSpike(unsigned int index){
	// Increment the activity in the state variable
	//this->SetStateVariableAt(index, 1, this->GetPostsynapticActivity(index)+1.0f);
	this->incrementStateVariableAt(index, 1, 1.0f); 
}

void VogelsSTDPState::SetWeight(unsigned int index, float weight, float max_weight){
}