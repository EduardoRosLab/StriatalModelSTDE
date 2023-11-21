/***************************************************************************
 *                           DopamineSTDPState.h                           *
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

#ifndef DOPAMINESTDPSTATE_H_
#define DOPAMINESTDPSTATE_H_

#include "ConnectionState.h"

/*!
 * \file DopamineSTDPState.h
 *
 * \author Álvaro González-Redondo
 * \author Francisco Naveros
 * \date August 2020
 *
 * This file declares a class which abstracts the current state of a synaptic connection
 * with event-driven Dopamine STDP capabilities (https://doi.org/10.1093/cercor/bhl152).
 */

/*!
 * \class DopamineSTDPState
 *
 * \brief Synaptic connection current state.
 *
 * This class abstracts the state of a synaptic connection including Dopamine STDP and defines the state variables of
 * that connection.
 *
 * \author Álvaro González-Redondo
 * \author Francisco Naveros
 * \date August 2020
 */

class DopamineSTDPState : public ConnectionState{

	public:
		/*!
		 * LTP time constant.
		 */
	  float tau_p;
		float inv_tau_p;

		/*!
		 * LTD time constant.
		 */
		float tau_m;
		float inv_tau_m;

		/*!
		 * Kernel parameters
		 */
		float kph, kpl;
		float kmh, kml;

		/*!
		 * Dopamine parameters
		 */
		float tau_eligibility;
		float inv_tau_eligibility;
		float tau_dopamine;
		float inv_tau_dopamine;
		float da_max, da_min;

		/*!
		 * Synapse scaling parameters
		 */
		// float synapse_scaling_rate;
		// float target_firing_rate;
		// float min_weight_factor;
		// float tau_mean_pre, inv_tau_mean_pre;
		float syn_pre_inc;


		//index states
		static const int N_states = 7;
		static const int pre_index = 0;
		static const int post_index = 1;
		static const int weight_index = 2;
		static const int eligibility_pos_index = 3;
		static const int eligibility_neg_index = 4;
		static const int dopamine_index = 5;
		static const int max_weight_index = 6;


		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new state of a connection.
		 *
		 * \param tau_p Time constant of the LTP component.
		 * \param tau_m Time constant of the LTD component.
		 */
		DopamineSTDPState(
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
			// , float new_syn_sca_rat
			// , float new_tar_fir_rat
			// , float new_min_wei_fac
			// , float new_tau_mean_pre
			, float syn_pre_inc
		);

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~DopamineSTDPState();

		/*!
		 * \brief It gets the value of the accumulated presynaptic activity.
		 *
		 * It gets the value of the accumulated presynaptic activity.
		 *
		 * \return The accumulated presynaptic activity.
		 */
		//float GetPresynapticActivity(unsigned int index);
		inline float GetPresynapticActivity(unsigned int index){
			return this->GetStateVariableAt(index, pre_index);
		}

		/*!
		 * \brief It gets the value of the accumulated postsynaptic activity.
		 *
		 * It gets the value of the accumulated postsynaptic activity.
		 *
		 * \return The accumulated postsynaptic activity.
		 */
		//float GetPostsynapticActivity(unsigned int index);
		inline float GetPostsynapticActivity(unsigned int index){
			return this->GetStateVariableAt(index, post_index);
		}

		inline float GetWeight(unsigned int index){
			return this->GetStateVariableAt(index, weight_index);
		}

		inline float GetEligibilityPos(unsigned int index){
			return this->GetStateVariableAt(index, eligibility_pos_index);
		}

		inline float GetEligibilityNeg(unsigned int index){
			return this->GetStateVariableAt(index, eligibility_neg_index);
		}

		inline float GetDopamine(unsigned int index){
			return this->GetStateVariableAt(index, dopamine_index);
		}


		/*!
		 * \brief It gets the number of variables that you can print in this state.
		 *
		 * It gets the number of variables that you can print in this state.
		 *
		 * \return The number of variables that you can print in this state.
		 */
		virtual unsigned int GetNumberOfPrintableValues();

		/*!
		 * \brief It gets a value to be printed from this state.
		 *
		 * It gets a value to be printed from this state.
		 *
		 * \param index The synapse's index inside the learning rule.
		 * \param position Position inside each connection.
		 *
		 * \return The value at position-th position in this state.
		 */
		virtual double GetPrintableValuesAt(unsigned int index, unsigned int position);

		/*!
		 * \brief set new time to spikes.
		 *
		 * It set new time to spikes.
		 *
		 * \param index The synapse's index inside the learning rule.
		 * \param NewTime new time.
		 * \param pre_post In some learning rules (i.e. DopamineSTDPLS) this variable indicate wether the update affects the pre- or post- variables.
		 */
		virtual void SetNewUpdateTime(unsigned int index, double NewTime, bool pre_post);


		/*!
		 * \brief It implements the behaviour when it transmits a spike.
		 *
		 * It implements the behaviour when it transmits a spike. It must be implemented
		 * by any inherited class.
		 *
		 * \param index The synapse's index inside the learning rule.
		 */
		virtual void ApplyPresynapticSpike(unsigned int index);

		/*!
		 * \brief It implements the behaviour when the target cell fires a spike.
		 *
		 * It implements the behaviour when it the target cell fires a spike. It must be implemented
		 * by any inherited class.
		 *
		 * \param index The synapse's index inside the learning rule.
		 */
		virtual void ApplyPostsynapticSpike(unsigned int index);



		virtual void SetWeight(unsigned int index, float weight, float max_weight);

};

#endif /* NEURONSTATE_H_ */
