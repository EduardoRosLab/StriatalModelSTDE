/***************************************************************************
 *                           WithoutPostSynaptic.cpp                       *
 *                           -------------------                           *
 * copyright            : (C) 2013 by Francisco Naveros                    *
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

#include "../../include/learning_rules/WithoutPostSynaptic.h"


WithoutPostSynaptic::WithoutPostSynaptic():LearningRule(){
	this->SetParameters(WithoutPostSynaptic::GetDefaultParameters());
}

WithoutPostSynaptic::~WithoutPostSynaptic(){

}

bool WithoutPostSynaptic::ImplementPostSynaptic(){
	return false;
}

std::map<std::string,boost::any> WithoutPostSynaptic::GetParameters(){
	std::map<std::string,boost::any> newMap = LearningRule::GetParameters();
	return newMap;
}


std::map<std::string,boost::any> WithoutPostSynaptic::GetDefaultParameters(){
	std::map<std::string,boost::any> newMap = LearningRule::GetDefaultParameters();
	return newMap;
}




