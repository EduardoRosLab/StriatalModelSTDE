/***************************************************************************
 *                           VogelsSTDPState.h                             *
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

#ifndef VOGELSSTDPSTATE_H_
#define VOGELSSTDPSTATE_H_

#include "ConnectionState.h"

/*!
 * \file VogelsSTDPState.h
 *
 * \author Francisco Naveros and Álvaro González
 * \date July 2020
 *
 * This file declares a class which abstracts the current state of a synaptic connection
 * defined in the article: https://doi.org/10.1126/science.1211095
 */

/*!
 * \class VoglesSTDPState
 *
 * \brief Synaptic connection current state.
 *
 * This class abstracts the state of a synaptic connection including Vogles STDP and defines the state variables of
 * that connection.
 *
 * \author Francisco Naveros and Álvaro González
 * \date July 2020
 */

class VogelsSTDPState : public ConnectionState{

	public:
		/*!
		* \brief Decay exponential parameter for simetric kernel
		*/
		float tauKernel;
		float inv_tauKernel;


		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new state of a connection.
		 *
		 * \param NumSynapses number of synapses associated with this learning rule.
		 * \param tauKernel Time constant of the kernel.
		 */
		VogelsSTDPState(int NumSynapses, float tauKernel);

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~VogelsSTDPState();

		/*!
		 * \brief It gets the value of the accumulated presynaptic activity.
		 *
		 * It gets the value of the accumulated presynaptic activity.
		 *
		 * \return The accumulated presynaptic activity.
		 */
		//float GetPresynapticActivity(unsigned int index);
		inline float GetPresynapticActivity(unsigned int index){
			return this->GetStateVariableAt(index, 0);
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
			return this->GetStateVariableAt(index, 1);
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
		 * \param pre_post In some learning rules (i.e. STDPLS) this variable indicate wether the update affects the pre- or post- variables.
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

#endif /* VOGELSSTDPSTATE_H_ */

