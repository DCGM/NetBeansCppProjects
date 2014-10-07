
#ifndef RABBITMQCOMMUNICATION__INCLUDED
#define RABBITMQCOMMUNICATION__INCLUDED

#include "interfaces.h"

class RabbitMQCommunication: public ICommunication
{
	std::string IncommingQueue;
public:
	RabbitMQCommunication(IWorker &worker, std::string server = "localhost", std::string username="guest", std::string password = "guest", int port = 5672);
	virtual ~RabbitMQCommunication();
	void Start();

	void SetIncommingQueue(std::string queue);
	std::string GetIncommingQueue();
};

#endif
