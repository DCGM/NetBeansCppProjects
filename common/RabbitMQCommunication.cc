
#include "RabbitMQCommunication.h"
#include <iostream>
#include <SimpleAmqpClient/SimpleAmqpClient.h>
#include "interface.pb.h"
#include <google/protobuf/repeated_field.h>
using namespace std;

//-----------------------------------------------------------------------------

RabbitMQCommunication::RabbitMQCommunication(IWorker &worker, std::string server, std::string username, std::string password, int port)
: ICommunication(worker, server, port, username, password)
{

}

//-----------------------------------------------------------------------------

void RabbitMQCommunication::Start()
{
	AmqpClient::Channel::ptr_t channel = AmqpClient::Channel::Create(this->GetServer(),this->GetPort(),this->GetUsername(),this->GetPassword());
	AmqpClient::Channel::ptr_t channel2 = AmqpClient::Channel::Create(this->GetServer(),this->GetPort(),this->GetUsername(),this->GetPassword());

	std::string consumer_tag = channel->BasicConsume(this->GetIncommingQueue(), "", true, false, false, 0);
	while(1)
	{
		//Get Message
		AmqpClient::Envelope::ptr_t envelope = channel->BasicConsumeMessage(consumer_tag);
		AmqpClient::BasicMessage::ptr_t message = envelope->Message();

		protointerface::WorkRequest request_response;
		request_response.ParseFromString(message->Body());

		try
		{
			this->worker.Process(request_response);
		}
		catch(exception &ex)
		{
			request_response.set_errorcode(10);
			Log::Message(request_response, ((std::string)"Exception: ") + ex.what());
		}

		try
		{
			request_response.mutable_pastconfiguration()->Add()->CopyFrom(request_response.configuration(0));
			request_response.mutable_configuration()->DeleteSubrange(0,1);
			//Log::Message(request_response, "Foo");
			//cerr << "==============\n===============\n\n";
			//cerr << request_response.DebugString();
			//cerr << "==============\n===============\n\n";
		}
		catch(exception &ex)
		{
			cerr << "RabbitMQCommunication: exception while configurations switching: " << ex.what() << "\n";
		}

		message->Body(request_response.SerializeAsString());
		try
		{
			//Publish Message
			string queue="";
			if(request_response.configuration_size()>0)
			{
				queue = request_response.configuration(0).queue();
			}
			else
			{
				queue = request_response.returnqueue();
			}
			channel2->BasicPublish("", queue, message);
			channel->BasicAck(envelope);
		}
		catch(exception &ex)
		{
			//message is invalid, reject it but not re-queue
			channel->BasicReject(envelope, false, false);
			cerr << "RabbitMQCommunication: exception while passing request to the output queue, probably queue does not exist \n";

		}
	}
}

//-----------------------------------------------------------------------------

RabbitMQCommunication::~RabbitMQCommunication()
{
}

//-----------------------------------------------------------------------------

void RabbitMQCommunication::SetIncommingQueue(std::string queue)
{
	this->IncommingQueue = queue;
}

//-----------------------------------------------------------------------------

std::string RabbitMQCommunication::GetIncommingQueue()
{
	return this->IncommingQueue;
}

//-----------------------------------------------------------------------------
