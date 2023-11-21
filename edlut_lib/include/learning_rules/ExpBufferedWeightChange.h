/***************************************************************************
 *                           ExpBufferedWeightChange.h                     *
 *                           -------------------                           *
 * copyright            : (C) 2019 by Francisco Naveros                    *
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

#ifndef EXPBUFFEREDWEIGHTCHANGE_H_
#define EXPBUFFEREDWEIGHTCHANGE_H_

/*!
 * \file ExpBufferedWeightChange.h
 *
 * \author Francisco Naveros
 * \date February 2019
 *
 * This file declares a class which abstracts a exponential additive learning rule precomputed in a look-up table.
 */

#include "./AdditiveKernelChange.h"

class BufferedActivityTimes;

/*!
 * \class ExpBufferedWeightChange
 *
 * \brief Sinuidal learning rule.
 *
 * This class abstract the behaviour of a exponential-sinusoidal additive learning rule precomputed in a look-up table.
 *
 * \author Francisco Naveros
 * \date April 2016
 */
class ExpBufferedWeightChange: public AdditiveKernelChange{
	private:

    /*!
		 * Initial time of the learning rule.
		 */
		float initTime;


		/*!
		* Maximum time calulated in the look-up table.
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
		float * kernelLookupTable;


		/*!
		* Buffer of spikes propagated by "no trigger" synapses
		*/
		BufferedActivityTimes * bufferedActivityTimesNoTrigger;

	public:
		/*!
		 * \brief Default constructor without parameters.
		 *
		 * It generates a new learning rule.
		 *
		 */
		ExpBufferedWeightChange();

		/*!
		 * \brief Object destructor.
		 *
		 * It remove the object.
		 */
		virtual ~ExpBufferedWeightChange();


		 /* \brief It initialize the state associated to the learning rule for all the synapses.
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
   		 * \brief It applies the weight change function when a presynaptic spike arrives.
   		 *
   		 * It applies the weight change function when a presynaptic spike arrives.
   		 *
   		 * \param Connection The connection where the spike happened.
   		 * \param SpikeTime The spike time.
   		 */
   		virtual void ApplyPreSynapticSpike(Interconnection * Connection,double SpikeTime);


		/*!
		 * \brief It gets the number of state variables that this learning rule needs.
		 *
		 * It gets the number of state variables that this learning rule needs.
		 *
		 * \return The number of state variables that this learning rule needs.
		 */
   		virtual int GetNumberOfVar() const;
   		
   		/*!
		 * \brief It gets the value of the initTime in the exp function.
		 * 
		 * It gets the value of the initTime in the exp function.
		 * 
		 * \return The value of the initTim in the exp function.
		 */
   		float GetInitTime() const;

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
			return "ExpBufferedAdditiveKernel";
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

#endif /*EXPOPTIMISEDBUFFEREDWEIGHTCHANGE_H_*/
