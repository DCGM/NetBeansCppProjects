
#ifndef INTERFACES__INCLUDED
#define INTERFACES__INCLUDED

#include <google/protobuf/message.h>
#include "interface.pb.h"

class IWorker
{
public:
	virtual ~IWorker();
	virtual void Process(protointerface::WorkRequest &request_response) = 0;
};

//-----------------------------------------------------------------------------

class ICommunication
{
	std::string server;
	std::string username;
	std::string password;
	int port;

public:
	ICommunication(IWorker &worker);
	ICommunication(IWorker &worker, std::string server);
	ICommunication(IWorker &worker, std::string server, int port, std::string user, std::string password);
	virtual ~ICommunication();
protected:
	IWorker &worker;
public:
	virtual void Start() = 0;

	std::string GetServer();
	void SetServer(std::string s);

	std::string GetUsername();
	void SetUsername(std::string s);

	std::string GetPassword();
	void SetPassword(std::string s);

	int GetPort();
	void SetPort(int port);
};

class Log
{
public:
	static void Message(protointerface::WorkRequest &request_response, std::string what);
};
#endif
