//============================================================================
// Name        : Filter.cpp
// Author      : Foo
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>

#include "RabbitMQCommunication.h"
#include "BowTranslate.h"

using namespace std;

int main()
{
	cout << "!!!Filter BOW!!!" << endl;
	BowTranslate bo;
	bo.InitBOW("/home/ireznice/RMQProcessing/data/rmq.1000.surf.cb");

	RabbitMQCommunication comm(bo, "medusa", "testing", "its");
	comm.SetIncommingQueue("Search_features");
	comm.Start();
	return 0;
}
