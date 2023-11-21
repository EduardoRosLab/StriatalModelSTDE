/***************************************************************************
 *                           TimeDrivenInputDevice.h                       *
 *                           -------------------                           *
 * copyright            : (C) 2020 by Francisco Naveros                    *
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

#ifndef TIMEDRIVENINPUTDEVICE_H_
#define TIMEDRIVENINPUTDEVICE_H_

/*!
 * \file TimeDrivenInputDevice.h
 *
 * \author Francisco Naveros
 * \date April 2020
 *
 * This file declares a class which abstracts an time-driven input devices (current and spike generators) in a CPU.
 */

#include "neuron_model/TimeDrivenModel.h"

#include "simulation/NetworkDescription.h"

#include <cstring>

using namespace std;

#include "simulation/Configuration.h"
class VectorNeuronState;
class InternalSpike;
class Interconnection;
//struct ModelDescription;

#include "simulation/Utils.h"



/*!
 * \class TimeDrivenInputDevice
 *
 * \brief Time-Driven Input device in CPU
 *
 * This class abstracts the behavior of time-driven input devices in spiking neural 
 * networks implemented in CPU.
 * It includes internal model functions which define the behavior of the device.
 * This is only a virtual function (an interface) which defines the functions of the
 * inherited classes.
 *
 * \author Francisco Naveros
 * \date April 2020
 */
class TimeDrivenInputDevice : public TimeDrivenModel {
	public:

        /*!
		 * \brief Constructor with parameters.
		 *
		 * It generates a new neuron model.
		 *
		 * \param new_time_scale Variable that indicate which time scale implement this neuron model.
		 */
		TimeDrivenInputDevice(TimeScale new_time_scale);


		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~TimeDrivenInputDevice();


		/*!
		 * \brief It initializes the neuron state to defined values.
		 *
		 * It initializes the neuron state to defined values.
		 *
		 */
		virtual VectorNeuronState * InitializeState() = 0;


		/*!
		 * \brief It processes a propagated spike (input spike in the cell).
		 *
		 * It processes a propagated spike (input spike in the cell).
		 *
		 * \note This function doesn't generate the next propagated spike. It must be externally done.
		 *
		 * \param inter the interconection which propagate the spike
		 * \param time the time of the spike.
		 *
		 * \return A new internal spike if someone is predicted. 0 if none is predicted.
		 */
		InternalSpike * ProcessInputSpike(Interconnection * inter, double time){
			return NULL;
		}

		/*!
		* \brief It processes a propagated current (input current in the cell).
		*
		* It processes a propagated current (input current in the cell).
		*
		* \param inter the interconection which propagate the spike
		* \param target the neuron which receives the spike
		* \param Current input current.
		*/
		void ProcessInputCurrent(Interconnection * inter, Neuron * target, float current){
		}


		/*!
		 * \brief Update the neuron state variables.
		 *
		 * It updates the neuron state variables.
		 *
		 * \param index The cell index inside the vector. if index=-1, updating all cell. 
		 * \param CurrentTime Current time.
		 *
		 * \return True if an output spike have been fired. False in other case.
		 */
		virtual bool UpdateState(int index, double CurrentTime) = 0;

		
		/*!
		 * \brief It gets the neuron model simulation method (event-driven, time-driven in CPU or time-driven in GPU).
		 *
		 * It gets the neuron model simulation method (event-driven, time-driven in CPU or time-driven in GPU).
		 *
		 * \return The simulation method of the neuron model.
		 */
		enum NeuronModelSimulationMethod GetModelSimulationMethod();

		/*!
		 * \brief It gets the neuron model type (an input device that inyects activity (spikes or currents) in the neural network or a neuron layer).
		 *
		 * It gets the neuron model type (an input device that inyects activity (spikes or currents) in the neural network or a neuron layer).
		 *
		 * \return The neuron model type
		 */
		enum NeuronModelType GetModelType();

		/*!
		 * \brief It gets the neuron output activity type (spikes or currents).
		 *
		 * It gets the neuron output activity type (spikes or currents).
		 *
		 * \return The neuron output activity type (spikes or currents).
		 */
		virtual enum NeuronModelOutputActivityType GetModelOutputActivityType() = 0;

		/*!
		 * \brief It gets the neuron input activity types (spikes and/or currents or none).
		 *
		 * It gets the neuron input activity types (spikes and/or currents or none).
		 *
		 * \return The neuron input activity types (spikes and/or currents or none).
		 */
		virtual enum NeuronModelInputActivityType GetModelInputActivityType();


		/*!
		 * \brief It prints the neuron model info.
		 *
		 * It prints the current neuron model characteristics.
		 *
		 * \param out The stream where it prints the information.
		 *
		 * \return The stream after the printer.
		 */
		virtual ostream & PrintInfo(ostream & out) = 0;

		/*!
		 * \brief It initialice VectorNeuronState.
		 *
		 * It initialice VectorNeuronState.
		 *
		 * \param N_neurons cell number inside the VectorNeuronState.
		 * \param OpenMPQueueIndex openmp index
		 */
		virtual void InitializeStates(int N_neurons, int OpenMPQueueIndex)=0;


		/*!
		 * \brief It Checks if the neuron model has this connection type.
		 *
		 * It Checks if the neuron model has this connection type.
		 *
		 * \param Type input connection type.
		 *
		 * \return If the neuron model supports this connection type
		 */
		bool CheckSynapseType(Interconnection * connection);


		/*!
		* \brief It calculates the number of electrical coupling synapses.
		*
		* It calculates the number for electrical coupling synapses.
		*
		* \param inter synapse that arrive to a neuron.
		*/
		void CalculateElectricalCouplingSynapseNumber(Interconnection * inter){
		};

		/*!
		* \brief It allocate memory for electrical coupling synapse dependencies.
		*
		* It allocate memory for electrical coupling synapse dependencies.
		*/
		virtual void InitializeElectricalCouplingSynapseDependencies(){
		};

		/*!
		* \brief It calculates the dependencies for electrical coupling synapses.
		*
		* It calculates the dependencies for electrical coupling synapses.
		*
		* \param inter synapse that arrive to a neuron.
		*/
		virtual void CalculateElectricalCouplingSynapseDependencies(Interconnection * inter){
		};

		/*!
		* \brief It initialieses the structure required to implement external input current synapsis.
		*
		* It initialieses the structure required to implement external input current synapsis.
		*/
		void InitializeInputCurrentSynapseStructure(){
		};

		/*!
		 * \brief It returns the neuron model parameters.
		 *
		 * It returns the neuron model parameters.
		 *
		 * \returns A dictionary with the neuron model parameters
		 */
		virtual std::map<std::string,boost::any> GetParameters() const;

		/*!
		* \brief It returns the neuron model parameters for a specific neuron once the neuron model has been initilized with the number of neurons.
		*
		* It returns the neuron model parameters for a specific neuron once the neuron model has been initilized with the number of neurons.
		*
		* \param index neuron index inside the neuron model.
		*
		* \returns A dictionary with the neuron model parameters
		*
		* NOTE: this function is accesible throgh the Simulatiob_API interface.
		*/
		virtual std::map<std::string, boost::any> GetSpecificNeuronParameters(int index) const throw (EDLUTException) = 0;

		/*!
		 * \brief It loads the neuron model properties.
		 *
		 * It loads the neuron model properties from parameter map.
		 *
		 * \param param_map The dictionary with the neuron model parameters.
		 *
		 * \throw EDLUTException If it happens a mistake with the parameters in the dictionary.
		 */
		virtual void SetParameters(std::map<std::string, boost::any> param_map) throw (EDLUTException);

		/*!
		 * \brief It returns the default parameters of the neuron model.
		 *
		 * It returns the default parameters of the neuron models. It may be used to obtained the parameters that can be
		 * set for this neuron model. 
		 *
		 * \returns A dictionary with the neuron model default parameters.
		 */
		static std::map<std::string, boost::any> GetDefaultParameters(){
			// Return a dictionary with the parameters
			std::map<std::string, boost::any> newMap = TimeDrivenModel::GetDefaultParameters();
			return newMap;
		}


		/*!
		 * \brief Comparison operator between neuron models.
		 *
		 * It compares two neuron models.
		 *
		 * \return True if the neuron models are of the same type and with the same parameters.
		 */
		virtual bool compare(const NeuronModel * rhs) const{
			if (!TimeDrivenModel::compare(rhs)){
				return false;
			}
			const TimeDrivenInputDevice * e = dynamic_cast<const TimeDrivenInputDevice *> (rhs);
			if (e == 0) return false;

			return true;
		};


};

#endif /* TIMEDRIVENINPUTDEVICE_H_ */
