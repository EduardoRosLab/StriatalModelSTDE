/***************************************************************************
 *                           DopamineSTDPState.cpp                         *
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

#include "../../include/learning_rules/DopamineSTDPState.h"

#include "../../include/simulation/ExponentialTable.h"

#include <cmath>
#include <stdio.h>
#include <float.h>

#include <iostream>
#include <iomanip>



//SIX STATES: 1) presynaptic activity, 2) postsynapti activity, 3) weight, 4) eligibility, 5) dopamine level and 6 max weight
DopamineSTDPState::DopamineSTDPState(
	int n_synapses
	, float new_tau_p
	, float new_kph
	, float new_kpl
	, float new_tau_m
	, float new_kmh
	, float new_kml
	, float new_tau_eligibility
	, float new_tau_dopamine
	, float new_da_max
	, float new_da_min
	, float new_syn_pre_inc
) :
	ConnectionState(n_synapses, N_states)
	, tau_p(new_tau_p)
	, kph(new_kph)
	, kpl(new_kpl)
	, tau_m(new_tau_m)
	, kmh(new_kmh)
	, kml(new_kml)
	, tau_eligibility(new_tau_eligibility)
	, tau_dopamine(new_tau_dopamine)
	, da_max(new_da_max)
	, da_min(new_da_min)
	, syn_pre_inc(new_syn_pre_inc)
{
	inv_tau_p=1.0f/new_tau_p;
	inv_tau_m=1.0f/new_tau_m;
	inv_tau_eligibility = 1.0f / tau_eligibility;
	inv_tau_dopamine = 1.0f / tau_dopamine;
}


DopamineSTDPState::~DopamineSTDPState() { }


unsigned int DopamineSTDPState::GetNumberOfPrintableValues() {
	return ConnectionState::GetNumberOfPrintableValues() + 2;
}


double DopamineSTDPState::GetPrintableValuesAt(unsigned int index, unsigned int position) {
	if (position<ConnectionState::GetNumberOfPrintableValues()) {
		return ConnectionState::GetStateVariableAt(index, position);
	} else if (position == ConnectionState::GetNumberOfPrintableValues()) {
		return this->tau_p;
	} else if (position == ConnectionState::GetNumberOfPrintableValues() + 1) {
		return this->tau_m;
	} else if (position == ConnectionState::GetNumberOfPrintableValues() + 2) {
		return this->tau_eligibility;
	} else if (position == ConnectionState::GetNumberOfPrintableValues() + 3) {
		return this->tau_dopamine;
	}
	else return -1;
}


void DopamineSTDPState::SetNewUpdateTime(unsigned int index, double NewTime, bool pre_post) {

	float dt = (float)(NewTime - this->GetLastUpdateTime(index));

	float mu = 1.0;
	float max_weight = this->GetStateVariableAt(index, max_weight_index);
	float min_weight = 0.0;

	float da_max = this->da_max;
	float da_min = this->da_min;

	float kph = this->kph; // D1: 1.2;	D2: 0.4
	float kpl = this->kpl; // D1: -0.3;	D2: 0.4
	float kmh = this->kmh; // D1: 0.0;	D2: -0.8
	float kml = this->kml; // D1: -0.4;	D2: 0.4

	float tau_g = this->tau_eligibility;
	float tau_d = this->tau_dopamine;

	float w_0 = this->GetWeight(index);
	float gp_0 = this->GetEligibilityPos(index);
	float gn_0 = this->GetEligibilityNeg(index);
	float d_0 = this->GetDopamine(index);

	if (d_0 < da_min) d_0 = da_min;
	if (d_0 > da_max*1.2) d_0 = da_max*1.2;

	float u0 = tau_g*mu;
	float u2 = tau_d+tau_g;
	float u3 = 1.0/tau_d;
	float u4 = 1.0/tau_g;
	float u5 = u3+u4;
	float u8 = exp(dt*u5);
	float u9 = exp(dt*u4);
	float u10 = exp(dt*u3);

	float u1_, u1, u6, u7;
	float hp, hm;
	float new_weight;

	u1_ = gp_0*u0;
	u1 = u1_*kph;
	u7 = u1_*kpl;
	u6 = kph-kpl;
	hp = (da_max*u2*u8*(-u7+u9*(w_0+u7))+u9*(d_0*gp_0*u6*tau_d*tau_g*(-1+u8)*mu-da_min*u2*u10*(-u1+u9*(w_0+u1)))) / ((da_max-da_min)*u2*u10*u9*u9);

	u1_ = gn_0*u0;
	u1 = u1_*kmh;
	u7 = u1_*kml;
	u6 = kmh-kml;
	hm = (da_max*u2*u8*(-u7+u9*(w_0+u7))+u9*(d_0*gn_0*u6*tau_d*tau_g*(-1+u8)*mu-da_min*u2*u10*(-u1+u9*(w_0+u1)))) / ((da_max-da_min)*u2*u10*u9*u9);

	new_weight = (hp+hm)*0.5; //hp + hm;

	if (new_weight < min_weight) new_weight = min_weight;
	if (new_weight > max_weight) new_weight = max_weight;

  	//Accumulate presynaptic activity since the last update time
	this->multiplyStateVariableAt(index, pre_index, ExponentialTable::GetResult(-dt*this->inv_tau_p));
  	//Accumulate postsynaptic activity since the last update time
	this->multiplyStateVariableAt(index, post_index, ExponentialTable::GetResult(-dt*this->inv_tau_m));

	//Set the new weight
	this->SetStateVariableAt(index, weight_index, new_weight);

	//Update eligibility value since the last update time
	this->multiplyStateVariableAt(index, eligibility_pos_index, ExponentialTable::GetResult(-dt*this->inv_tau_eligibility));
	this->multiplyStateVariableAt(index, eligibility_neg_index, ExponentialTable::GetResult(-dt*this->inv_tau_eligibility));
	//Update dopamine value since the last update time
	this->multiplyStateVariableAt(index, dopamine_index, ExponentialTable::GetResult(-dt*this->inv_tau_dopamine));

	//Set the last update time
	this->SetLastUpdateTime(index, NewTime);
}


void DopamineSTDPState::ApplyPresynapticSpike(unsigned int index) {
	// Increment synapse weight with every pre-synaptic spike
	this->incrementStateVariableAt(index, weight_index, this->syn_pre_inc);

	// Increment the activity in the state variable
	this->incrementStateVariableAt(index, pre_index, 1.0f);

	float eligibility_increment = this->GetStateVariableAt(index, post_index);
	this->incrementStateVariableAt(index, eligibility_neg_index, eligibility_increment);
}


void DopamineSTDPState::ApplyPostsynapticSpike(unsigned int index){
	// Increment the activity in the state variable
	this->incrementStateVariableAt(index, post_index, 1.0f);

	float eligibility_increment = this->GetStateVariableAt(index, pre_index);
	this->incrementStateVariableAt(index, eligibility_pos_index, eligibility_increment);
}


void DopamineSTDPState::SetWeight(unsigned int index, float weight, float max_weight){
	this->SetStateVariableAt(index, weight_index, weight);
	this->SetStateVariableAt(index, max_weight_index, max_weight);
}
