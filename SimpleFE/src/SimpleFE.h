/*
 * fe.h
 *
 *  Created on: Sep 2, 2014
 *      Author: ireznice
 */

#ifndef FE_H_
#define FE_H_

#include "interfaces.h"

class SimpleFE : public IWorker
{
	void Process(protointerface::WorkRequest &request_response);
};

#endif /* FE_H_ */
