//============================================================================
// Name        : Filter.cpp
// Author      : Foo
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>

#include "RabbitMQCommunication.h"
#include "SimpleSearch.h"

using namespace std;

int main()
{
	cout << "!!!Filter Search!!!" << endl;
	SimpleSearch fe;
	fe.Init("/home/ireznice/RMQProcessing/data/baseAll");

	RabbitMQCommunication comm(fe, "medusa", "testing", "its");
	comm.SetIncommingQueue("Search_bow");
	comm.Start();
	return 0;
}
