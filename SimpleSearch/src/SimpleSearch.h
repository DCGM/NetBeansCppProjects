/*
 * fe.h
 *
 *  Created on: Sep 2, 2014
 *      Author: ireznice
 */

#ifndef FESS_H_
#define FESS_H_

#include "interfaces.h"

class SimpleSearch : public IWorker
{
	public:
	std::vector<std::vector<float> > base;
	std::vector<std::string > base_descr;
	void Process(protointerface::WorkRequest &request_response);
	std::vector<std::pair<int, float> > Search(std::vector<float> &in, int nearest);
public:
	void Init(std::string basefile);
};

#endif /* FE_H_ */
