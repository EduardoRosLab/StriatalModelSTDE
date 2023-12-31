/***************************************************************************
 *                           LIFTimeDrivenModel_1_4_GPU2_NEW.h             *
 *                           -------------------                           *
 * copyright            : (C) 2016 by Francisco Naveros                    *
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

#ifndef LIFTIMEDRIVENMODEL_1_4_GPU2_NEW_H_
#define LIFTIMEDRIVENMODEL_1_4_GPU2_NEW_H_

/*!
 * \file LIFTimeDrivenModel_1_4_GPU_NEW.h
 *
 * \author Francisco Naveros
 * \date October 2016
 *
 * This file declares a class which abstracts a Leaky Integrate-And-Fire neuron model with one 
 * differential equation and four time dependent equations (conductances). This model is
 * implemented in GPU.This class is a redefinition of LIFTimeDrivenModel_1_4_GPU2 neuron 
 * model using mV, ms, nS and pF units.
 */

#include "./TimeDrivenNeuronModel_GPU2.h"
#include "../../include/integration_method/IntegrationMethod_GPU2.h"
#include "../../include/integration_method/LoadIntegrationMethod_GPU2.h"

//Library for CUDA
//#include <helper_cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/*!
 * \class LIFTimeDrivenModel_1_4_GPU2_NEW
 *
 *
 * \brief Leaky Integrate-And-Fire Time-Driven neuron model with a membrane potential and
 * four conductances. This model is implemented in GPU.
 *
 * This class abstracts the behavior of a neuron in a time-driven spiking neural network.
 * It includes internal model functions which define the behavior of the model
 * (initialization, update of the state, synapses effect, next firing prediction...).
 * This is only a virtual function (an interface) which defines the functions of the
 * inherited classes.
 *
 * \author Francisco Naveros
 * \date October 2016
 */

class LIFTimeDrivenModel_1_4_GPU2_NEW : public TimeDrivenNeuronModel_GPU2 {
	public:
		/*!
		 * \brief Excitatory reversal potential in mV units
		 */
		const float eexc;

		/*!
		 * \brief Inhibitory reversal potential in mV units
		 */
		const float einh;

		/*!
		 * \brief Resting potential in mV units
		 */
		const float erest;

		/*!
		 * \brief Firing threshold in mV units
		 */
		const float vthr;

		/*!
		 * \brief Membrane capacitance in pF units
		 */
		const float cm;
		const float inv_cm;

		/*!
		 * \brief AMPA receptor time constant in ms units
		 */
		const float tampa;
		const float inv_tampa;

		/*!
		 * \brief NMDA receptor time constant in ms units
		 */
		const float tnmda;
		const float inv_tnmda;
		
		/*!
		 * \brief GABA receptor time constant in ms units
		 */
		const float tinh;
		const float inv_tinh;

		/*!
		 * \brief Gap Junction time constant in ms units
		 */
		const float tgj;
		const float inv_tgj;

		/*!
		 * \brief Refractory period in ms units
		 */
		const float tref;

		/*!
		 * \brief Resting conductance in nS units
		 */
		const float grest;

		/*!
		 * \brief Gap junction factor in mV/nS units
		 */
		const float fgj;

		/*!
		 * \brief Number of state variables for each cell.
		*/
		static const int N_NeuronStateVariables=5;

		/*!
		 * \brief Number of state variables which are calculate with a differential equation for each cell.
		*/
		static const int N_DifferentialNeuronState=1;

		/*!
		 * \brief Number of state variables which are calculate with a time dependent equation for each cell.
		*/
		static const int N_TimeDependentNeuronState=4;


		/*!
		 * \brief constructor with parameters.
		 *
		 * It generates a new neuron model object.
		 *
		 * \param Eexc eexc.
		 * \param Einh einh.
		 * \param Erest erest.
		 * \param Vthr vthr.
		 * \param Cm cm.
		 * \param Tampa tampa.
		 * \param Tnmda tnmda.
		 * \param Tinh tinh.
		 * \param Tgj tgj.
		 * \param Tref tref.
		 * \param Grest grest.
		 * \param Fgj fgj.
		 * \param integrationName integration method type.
		 * \param N_neurons number of neurons.
		 * \param Total_N_thread total number of CUDA thread.
		 * \param Buffer_GPU Gpu auxiliar memory.
		 *
		 */
		__device__ LIFTimeDrivenModel_1_4_GPU2_NEW(float Eexc,float Einh,float Erest,float Vthr,float Cm,float Tampa,
			float Tnmda,float Tinh,float Tgj,float Tref,float Grest,float Fgj, char const* integrationName, int N_neurons,
			void ** Buffer_GPU):TimeDrivenNeuronModel_GPU2(MilisecondScale_GPU), eexc(Eexc),einh(Einh),erest(Erest),vthr(Vthr),cm(Cm),tampa(Tampa),
			tnmda(Tnmda),tinh(Tinh),tgj(Tgj), tref(Tref),grest(Grest),fgj(Fgj),inv_tampa(1.0f/tampa),inv_tnmda(1.0f/tnmda),inv_tinh(1.0f/tinh),
			inv_tgj(1.0f/tgj),inv_cm(1.0f/cm){
			integrationMethod_GPU2=LoadIntegrationMethod_GPU2::loadIntegrationMethod_GPU2(this, integrationName, N_NeuronStateVariables, N_DifferentialNeuronState, N_TimeDependentNeuronState, Buffer_GPU);
		
			integrationMethod_GPU2->Calculate_conductance_exp_values();	
		}

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		__device__ virtual ~LIFTimeDrivenModel_1_4_GPU2_NEW(){
			delete integrationMethod_GPU2;
		}


		/*!
		 * \brief Update the neuron state variables.
		 *
		 * It updates the neuron state variables.
		 *
		 * \param index The cell index inside the StateGPU. 
		 * \param AuxStateGPU Auxiliary incremental conductance vector.
		 * \param StateGPU Neural state variables.
		 * \param LastUpdateGPU Last update time
		 * \param LastSpikeTimeGPU Last spike time
		 * \param InternalSpikeGPU In this vector is stored if a neuron must generate an output spike.
		 * \param SizeStates Number of neurons
		 * \param CurrentTime Current time.
		 *
		 * \return True if an output spike have been fired. False in other case.
		 */
		__device__ void UpdateState(double CurrentTime)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			while (index<vectorNeuronState_GPU2->SizeStates){
				
				vectorNeuronState_GPU2->VectorNeuronStates_GPU[1*vectorNeuronState_GPU2->SizeStates + index]+=vectorNeuronState_GPU2->AuxStateGPU[index];                                          //gAMPA
				vectorNeuronState_GPU2->VectorNeuronStates_GPU[2*vectorNeuronState_GPU2->SizeStates + index]+=vectorNeuronState_GPU2->AuxStateGPU[vectorNeuronState_GPU2->SizeStates + index];     //gGABA
				vectorNeuronState_GPU2->VectorNeuronStates_GPU[3*vectorNeuronState_GPU2->SizeStates + index]+=vectorNeuronState_GPU2->AuxStateGPU[2*vectorNeuronState_GPU2->SizeStates + index];   //gNMDA
				vectorNeuronState_GPU2->VectorNeuronStates_GPU[4*vectorNeuronState_GPU2->SizeStates + index]+=vectorNeuronState_GPU2->AuxStateGPU[3*vectorNeuronState_GPU2->SizeStates + index];   //gGJ

				vectorNeuronState_GPU2->InternalSpikeGPU[index]=this->integrationMethod_GPU2->NextDifferentialEcuationValues(index, vectorNeuronState_GPU2->SizeStates, vectorNeuronState_GPU2->VectorNeuronStates_GPU);

				vectorNeuronState_GPU2->LastUpdateGPU[index]=CurrentTime;
				
				this->CheckValidIntegration(index);

				index+=blockDim.x*gridDim.x;
			}
		} 


		/*!
		 * \brief It evaluates if a neuron must spike.
		 *
		 * It evaluates if a neuron must spike.
		 *
		 * \param previous_V previous membrane potential
		 * \param NeuronState neuron state variables.
		 * \param index Neuron index inside the neuron model.
 		 * \param elapsedTimeInNeuronModelScale integration method step.
		 * \return It returns if a neuron must spike.
		 */
		__device__ virtual bool EvaluateSpikeCondition(float previous_V, float * NeuronState, int index, float elapsedTimeInNeuronModelScale){
			float vm_cou = NeuronState[index] + this->fgj * NeuronState[4*vectorNeuronState_GPU2->SizeStates + index];
			if (vm_cou > this->vthr){		
				NeuronState[index] = this->erest;
				vectorNeuronState_GPU2->LastSpikeTimeGPU[index]=0.0;
				this->integrationMethod_GPU2->resetState(index);
				return true;
			}
			return false;
		}

		/*!
		 * \brief It evaluates the differential equation in NeuronState and it stores the results in AuxNeuronState.
		 *
		 * It evaluates the differential equation in NeuronState and it stores the results in AuxNeuronState.
		 *
		 * \param index index inside the NeuronState vector.
		 * \param SizeStates number of element in NeuronState vector.
		 * \param NeuronState value of the neuron state variables where differential equations are evaluated.
		 * \param AuxNeuronState results of the differential equations evaluation.
		 */
		__device__ void EvaluateDifferentialEcuation(int index, int SizeStates, float * NeuronState, float * AuxNeuronState, float elapsed_time){
			if(vectorNeuronState_GPU2->LastSpikeTimeGPU[index] * this->GetTimeScale() > this->tref){
				float iampa = NeuronState[SizeStates + index]*(this->eexc-NeuronState[index]);
				float iinh = NeuronState[2*SizeStates + index]*(this->einh-NeuronState[index]);
				float gnmdainf = 1.0f/(1.0f + __expf(-0.062f*NeuronState[index])*(1.2f/3.57f));
				float inmda = NeuronState[3*SizeStates + index]*gnmdainf*(this->eexc-NeuronState[index]);
				AuxNeuronState[blockDim.x*blockIdx.x + threadIdx.x]=(iampa + iinh + inmda + this->grest* (this->erest-NeuronState[index]))*this->inv_cm;
			}else if ((vectorNeuronState_GPU2->LastSpikeTimeGPU[index] * this->GetTimeScale() + elapsed_time)>this->tref){
				float fraction = (this->vectorNeuronState_GPU2->LastSpikeTimeGPU[index] * this->GetTimeScale() + elapsed_time - this->tref) / elapsed_time;
				float iampa = NeuronState[SizeStates + index]*(this->eexc-NeuronState[index]);
				float iinh = NeuronState[2*SizeStates + index]*(this->einh-NeuronState[index]);
				float gnmdainf = 1.0f/(1.0f + __expf(-0.062f*NeuronState[index])*(1.2f/3.57f));
				float inmda = NeuronState[3*SizeStates + index]*gnmdainf*(this->eexc-NeuronState[index]);
				AuxNeuronState[blockDim.x*blockIdx.x + threadIdx.x]=fraction*(iampa + iinh + inmda + this->grest* (this->erest-NeuronState[index]))*this->inv_cm;
			}else{
				AuxNeuronState[blockDim.x*blockIdx.x + threadIdx.x]=0.0f;
			}
		}


		/*!
		 * \brief It evaluates the time depedendent ecuation in NeuronState for elapsed_time and it stores the results in NeuronState.
		 *
		 * It evaluates the time depedendent ecuation in NeuronState for elapsed_time and it stores the results in NeuronState.
		 *
		 * \param index index inside the NeuronState vector.
		 * \param SizeStates number of element in NeuronState vector.
		 * \param NeuronState value of the neuron state variables where time dependent equations are evaluated.
		 * \param elapsed_time integration time step.
		 * \param elapsed_time_index index inside the conductance_exp_values array.
		 */
		__device__ void EvaluateTimeDependentEcuation(int index, int SizeStates, float * NeuronState, float elapsed_time, int elapsed_time_index){
			float limit=1e-20;

			float * Conductance_values=this->Get_conductance_exponential_values(elapsed_time_index);

			if(NeuronState[this->N_DifferentialNeuronState*SizeStates + index]<limit){
				NeuronState[this->N_DifferentialNeuronState*SizeStates + index]=0.0f;
			}else{
				NeuronState[this->N_DifferentialNeuronState*SizeStates + index]*=  Conductance_values[0];
			}
			if(NeuronState[(this->N_DifferentialNeuronState+1)*SizeStates + index]<limit){
				NeuronState[(this->N_DifferentialNeuronState+1)*SizeStates + index]=0.0f;
			}else{
				NeuronState[(this->N_DifferentialNeuronState+1)*SizeStates + index]*=  Conductance_values[1];
			}
			if(NeuronState[(this->N_DifferentialNeuronState+2)*SizeStates + index]<limit){
				NeuronState[(this->N_DifferentialNeuronState+2)*SizeStates + index]=0.0f;
			}else{
				NeuronState[(this->N_DifferentialNeuronState+2)*SizeStates + index]*=  Conductance_values[2];
			}
			if(NeuronState[(this->N_DifferentialNeuronState+3)*SizeStates + index]<limit){
				NeuronState[(this->N_DifferentialNeuronState+3)*SizeStates + index]=0.0f;
			}else{
				NeuronState[(this->N_DifferentialNeuronState+3)*SizeStates + index]*=  Conductance_values[3];
			}
		}


		/*!
		 * \brief It calculates the conductace exponential value for an elapsed time.
		 *
		 * It calculates the conductace exponential value for an elapsed time.
		 *
		 * \param index elapsed time index .
		 * \param elapses_time elapsed time.
		 */
		__device__ void Calculate_conductance_exp_values(int index, float elapsed_time){
			//ampa synapse.
			Set_conductance_exp_values(index, 0, __expf(-elapsed_time*this->inv_tampa));
			//inhibitory synapse.
			Set_conductance_exp_values(index, 1, __expf(-elapsed_time*this->inv_tinh));
			//nmda synapse.
			Set_conductance_exp_values(index, 2, __expf(-elapsed_time*this->inv_tnmda));
			//gap junction synapse.
			Set_conductance_exp_values(index, 3, __expf(-elapsed_time*this->inv_tgj));

		}

};


#endif /* LIFTIMEDRIVENMODEL_1_4_GPU2_NEW_H_ */
