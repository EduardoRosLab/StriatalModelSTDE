/***************************************************************************
 *                           TriphasicBufferedWeightChange.h               *
 *                           -------------------                           *
 * copyright            : (C) 2020 by Alvaro Gonzalez and Francisco Naveros*
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

#ifndef TRIPHASICBUFFEREDWEIGHTCHANGE_H_
#define TRIPHASICBUFFEREDWEIGHTCHANGE_H_

/*!
 * \file TriphasicBufferedWeightChange.h
 *
 * \author Alvaro Gonzalez
 * \author Francisco Naveros
 * \date August 2020
 *
 * This file declares a class which abstracts the behaviour of a triphasic STDP learning rule as in
 * https://doi.org/10.3389/fncom.2012.00088
 */
 
#include "./WithPostSynaptic.h"
#include "../simulation/NetworkDescription.h"

class BufferedActivityTimes;
 
/*!
 * \class TriphasicBufferedWeightChange
 *
 * \brief Triphasic learning rule.
 *
  * This class abstract the behaviour of a triphasic learning rule triphasic STDP learning rule as in
 * https://doi.org/10.3389/fncom.2012.00088
 *
 * \author Francisco Naveros
 * \author Alvaro Gonzalez
 * \date August 2020
 */ 

class TriphasicBufferedWeightChange: public WithPostSynaptic {

    private:

		/*!
		 * \brief Kernel amplitude (nS).
		 */
		float a;

		/*!
		 * \brief Kernel displacement and width (s)
		 */
		float alpha;

		/*!
		* Maximum time calculated in the look-up table.
		*/
		double maxTimeMeasured;
		double inv_maxTimeMeasured;

		/*!
		* Number of elements inside the look-up table.
		*/
		int N_elements;

		/*!
		* Look-up table for the kernel.
		*/
		float * kernelLookupTable; //medio kernel

		/*!
		* Buffer of presynaptic spikes
		*/
		BufferedActivityTimes * bufferedActivityTimesPresynaptic;

		/*!
		* Buffer of postsynaptic spikes
		*/
		BufferedActivityTimes * bufferedActivityTimesPostsynaptic;

	public:
		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new learning rule.
		 */
		TriphasicBufferedWeightChange();

		/*!
		 * \brief Object destructor.
		 *
		 * It remove the object.
		 */
		virtual ~TriphasicBufferedWeightChange();

		/*!
		 * \brief It initialize the state associated to the learning rule for all the synapses.
		 *
		 * It initialize the state associated to the learning rule for all the synapses.
		 *
		 * \param NumberOfSynapses the number of synapses that implement this learning rule.
		 * \param NumberOfNeurons the total number of neurons in the network
		 */
		void InitializeConnectionState(unsigned int NumberOfSynapses, unsigned int NumberOfNeurons);


		/*!
		 * \brief It loads the learning rule properties.
		 *
		 * It loads the learning rule properties.
		 *
		 * \param fh A file handler placed where the Learning rule properties are defined.
		 *
		 * \return The learning rule description object.
		 *
		 * \throw EDLUTException If something wrong happens in reading the learning rule properties.
		 */
		static ModelDescription ParseLearningRule(FILE * fh) throw (EDLUTException);

		/*!
   		 * \brief It applies the weight change function when a presynaptic spike arrives.
   		 *
   		 * It applies the weight change function when a presynaptic spike arrives.
   		 *
   		 * \param Connection The connection where the spike happened.
   		 * \param SpikeTime The spike time.
   		 */
   		virtual void ApplyPreSynapticSpike(Interconnection * Connection,double SpikeTime);

        /*!
        * \brief It applies the weight change function to all its input synapses when a postsynaptic spike arrives.
        *
        * It applies the weight change function to all its input synapses when a postsynaptic spike arrives.
        *
        * \param neuron The target neuron that manage the postsynaptic spike
        * \param SpikeTime The spike time of the postsynaptic spike.
        */
        virtual void ApplyPostSynapticSpike(Neuron * neuron, double SpikeTime);

        /*!
		 * \brief It prints the learning rule info.
		 *
		 * It prints the current learning rule characteristics.
		 *
		 * \param out The stream where it prints the information.
		 *
		 * \return The stream after the printer.
		 */
		virtual ostream & PrintInfo(ostream & out);

		/*!
		 * \brief It creates a new learning rule object of this type.
		 *
		 * It creates a new learning rule object of this type.
		 *
		 * \param param_map The learning rule description object.
		 *
		 * \return A newly created ExpWeightChange object.
		 */
		static LearningRule* CreateLearningRule(ModelDescription lrDescription);

		/*!
		 * \brief It provides the name of the learning rule
		 *
		 * It provides the name of the learning rule, i.e. the name that can be mentioned to use this learning rule.
		 *
		 * \return The name of the learning rule
		 */
		static std::string GetName(){
			return "TriphasicBufferedKernel";
		};

		/*!
		 * \brief It returns the learning rule parameters.
		 *
		 * It returns the learning rule parameters.
		 *
		 * \returns A dictionary with the learning rule parameters
		 */
		virtual std::map<std::string,boost::any> GetParameters();

		/*!
		 * \brief It loads the learning rule properties.
		 *
		 * It loads the learning rule properties from parameter map.
		 *
		 * \param param_map The dictionary with the learning rule parameters.
		 *
		 * \throw EDLUTFileException If it happens a mistake with the parameters in the dictionary.
		 */
		virtual void SetParameters(std::map<std::string, boost::any> param_map) throw (EDLUTException);

		/*!
		 * \brief It returns the default parameters of the learning rule.
		 *
		 * It returns the default parameters of the learning rule. It may be used to obtained the parameters that can be
		 * set for this learning rule.
		 *
		 * \returns A dictionary with the learning rule parameters.
		 */
		static std::map<std::string,boost::any> GetDefaultParameters();

	   		
};

#endif /*TRIPHASICBUFFEREDWEIGHTCHANGE_H_*/
