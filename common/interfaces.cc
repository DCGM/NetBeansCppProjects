#include "interfaces.h"

//-----------------------------------------------------------------------------

IWorker::~IWorker()
{
}

//-----------------------------------------------------------------------------

ICommunication::ICommunication(IWorker &worker):port(0), worker(worker)
{
}

//-----------------------------------------------------------------------------

ICommunication::ICommunication(IWorker &worker, std::string server, int port, std::string user, std::string password)
: worker(worker),
  username(user),
  password(password),
  port(port),
  server(server)
{
}

//-----------------------------------------------------------------------------

ICommunication::ICommunication(IWorker &worker, std::string server)
: server(server),
  port(0),
  worker(worker)
{
}

//-----------------------------------------------------------------------------

ICommunication::~ICommunication()
{
}

//-----------------------------------------------------------------------------

std::string ICommunication::GetServer()
{
	return this->server;
}

//-----------------------------------------------------------------------------

void ICommunication::SetServer(std::string s)
{
	this->server = s;
}

//-----------------------------------------------------------------------------

std::string ICommunication::GetUsername()
{
	return this->username;
}

//-----------------------------------------------------------------------------

void ICommunication::SetUsername(std::string s)
{
	this->username = s;
}

//-----------------------------------------------------------------------------

std::string ICommunication::GetPassword()
{
	return this->password;
}

//-----------------------------------------------------------------------------

void ICommunication::SetPassword(std::string s)
{
	this->password = s;
}

//-----------------------------------------------------------------------------

int ICommunication::GetPort()
{
	return this->port;
}

//-----------------------------------------------------------------------------

void ICommunication::SetPort(int port)
{
	this->port = port;
}

//-----------------------------------------------------------------------------
void Log::Message(protointerface::WorkRequest &request_response, std::string what)
{

}
