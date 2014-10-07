//============================================================================
// Name        : Filter.cpp
// Author      : Foo
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>

#include "RabbitMQCommunication.h"
#include "SimpleFE.h"
#include "OpenCVFE.h"
using namespace std;

int main()
{
	cout << "!!!Filter OpenCVFE!!!" << endl;
	//BowTranslate fe;
	OpenCVFE fe;
	RabbitMQCommunication comm(fe, "medusa", "testing", "its");
	comm.SetIncommingQueue("Search_work");
	comm.Start();
	return 0;
}
