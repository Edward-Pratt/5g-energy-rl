//
// Copyright (C) 2020 Dominik S. Buse <buse@ccs-labs.org>, Max Schettler <schettler@ccs-labs.org>
//
// Documentation for these modules is at http://veins.car2x.org/
//
// SPDX-License-Identifier: GPL-2.0-or-later
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//

#include "GymConnection.h"
#include <cmath>

Define_Module(GymConnection);

static double safe(double x){
    return std::isfinite(x) ? x : 0.0;
}



void GymConnection::initialize()
{
    std::string host = par("host");
    int port = par("port");

    if (host == "") {
        if (std::getenv("VEINS_GYM_HOST") != nullptr)
            host = std::getenv("VEINS_GYM_HOST");
        else
            throw omnetpp::cRuntimeError("Gym host not configured!");
    }

    if (port < 0) {
        if (std::getenv("VEINS_GYM_PORT") != nullptr)
            port = std::atoi(std::getenv("VEINS_GYM_PORT"));
        else
            throw omnetpp::cRuntimeError("Gym port not configured!");
    }

    EV_INFO << "Connecting to server 'tcp://" << host << ":" << port << "'\n";
    socket.connect("tcp://" + host + ":" + std::to_string(port));

    veinsgym::proto::Request init_request;
    *(init_request.mutable_init()->mutable_observation_space_code()) = par("observation_space").stdstringValue();
    *(init_request.mutable_init()->mutable_action_space_code()) = par("action_space").stdstringValue();
    communicate(init_request);
    
    
    // ---- Subscribe to VoIP signals ----
    auto *top = getSystemModule();
    auto *app0 = top->getModuleByPath("ue[0].app[0]");   // VoipReceiver
    if (!app0) {
        EV_WARN << "Could not find ue[0].app[0]. Are you running VoIP-DL config?\n";
    } else {
        // registerSignal returns an ID (existing or new). Works even if already registered.
        sigThroughput = omnetpp::cComponent::registerSignal("voipReceivedThroughput");
        sigFrameDelay = omnetpp::cComponent::registerSignal("voipFrameDelay");
        sigJitter     = omnetpp::cComponent::registerSignal("voipJitter");
        sigLoss       = omnetpp::cComponent::registerSignal("voipFrameLoss"); // start with frame loss

        app0->subscribe(sigThroughput, this);
        app0->subscribe(sigFrameDelay, this);
        app0->subscribe(sigJitter, this);
        app0->subscribe(sigLoss, this);

        EV_INFO << "Subscribed to VoIP signals on ue[0].app[0]\n";
    }
    


    voipSender = top->getModuleByPath("server.app[0]");
    if (!voipSender) {
    	EV_WARN << "Could not find server.app[0] (VoipSender.)";
    }

    lastEnergyT = omnetpp::simTime();
    energyJ = 0.0;



    tick = new omnetpp::cMessage("gymTick");
    scheduleAt(omnetpp::simTime(), tick);
}



void GymConnection::receiveSignal(omnetpp::cComponent *, omnetpp::simsignal_t signalID,
                                  double value, omnetpp::cObject *)
{
    if (signalID == sigThroughput) lastThroughput = value;
    else if (signalID == sigFrameDelay) lastFrameDelay = value;
    else if (signalID == sigJitter) lastJitter = value;
    else if (signalID == sigLoss) lastLoss = value;
}


void GymConnection::receiveSignal(omnetpp::cComponent *, omnetpp::simsignal_t signalID,
                                  const omnetpp::SimTime& value, omnetpp::cObject *)
{
    // convert SimTime to seconds (double)
    double v = value.dbl();

    if (signalID == sigFrameDelay) lastFrameDelay = v;
    else if (signalID == sigJitter) lastJitter = v;
    // if any other SimTime-based signals appear later, handle them here too
}

void GymConnection::updateEnergy()
{
    auto now = omnetpp::simTime();
    auto dt = (now - lastEnergyT).dbl();
    if (dt <= 0) return;

    double p = 0.0;
    if (bsState == BsState::ACTIVE) p = par("pActive").doubleValue();
    else if (bsState == BsState::SLEEP) p = par("pSleep").doubleValue();
    else /*LOWPOWER*/ p = (par("pActive").doubleValue() + par("pSleep").doubleValue()) * 0.5;

    energyJ += p * dt;
    lastEnergyT = now;
}

int GymConnection::getNumUE() const
{
    // pragmatic start: just read the network parameter if present
    auto *top = getSystemModule();
    if (top->hasPar("numUe"))
        return top->par("numUe").intValue();
    return 0;
}



void GymConnection::handleMessage(omnetpp::cMessage *msg)
{
    if (msg != tick) return;

    

    updateEnergy();

    const double thr = lastThroughput;
    const double delay = lastFrameDelay;
    const double jitter = lastJitter;
    const double loss = lastLoss;
    const double numUe = (double)getNumUE();
    const double E = energyJ;
    
    veinsgym::proto::Request req;
    static uint64_t stepId = 1;
    req.set_id(stepId++);

    auto *step = req.mutable_step();


    auto *obs = step->mutable_observation();
    auto *box = obs->mutable_box();
    box->add_values(thr);
    box->add_values(delay);
    box->add_values(jitter);
    box->add_values(loss);
    box->add_values(numUe);
    box->add_values(E);

    double reward = (0.001 * thr) - (100.0 * delay) - (10.0 * loss) - (0.0001 * E);

    auto *rbox = step->mutable_reward()->mutable_box();
    rbox->add_values(reward);


    
    auto reply = communicate(req);
    // action handling: be defensive (Discrete(3) is what you want)
    if (reply.payload_case() == veinsgym::proto::Reply::kAction) {
        const auto &actSpace = reply.action();
        if (actSpace.value_case() == veinsgym::proto::Space::kDiscrete) {
            auto a = (int)actSpace.discrete().value();
            EV_INFO << "Gym action: " << a << "\n";

            // ---- Apply action ----
            if (a == 0) bsState = BsState::ACTIVE;
            else if (a == 1) bsState = BsState::SLEEP;
            else bsState = BsState::LOWPOWER;


	    

	    applyAction(a);
        }
    }

    scheduleAt(omnetpp::simTime() + par("tickInterval"), tick);
}

void GymConnection::applyAction(int a)
{
    auto *top = getSystemModule();
    auto *sender = top->getModuleByPath("server.app[0]"); // VoipSender in VoIP-DL
    if (!sender) {
        EV_WARN << "Could not find server.app[0] to control traffic\n";
        return;
    }

    bool pauseTraffic = (a == 2);
    sender->par("gymPaused").setBoolValue(pauseTraffic);
}


veinsgym::proto::Reply GymConnection::communicate(const veinsgym::proto::Request& request)
{
    std::string request_msg = request.SerializeAsString();
    socket.send(zmq::message_t(request_msg.data(), request_msg.size()), zmq::send_flags::none);

    zmq::message_t response_msg;
    socket.recv(response_msg, zmq::recv_flags::none);

    std::string response(static_cast<char*>(response_msg.data()), response_msg.size());
    veinsgym::proto::Reply reply;
    reply.ParseFromString(response);
    return reply;
}

GymConnection::~GymConnection()
{
    if (tick)
        cancelAndDelete(tick);

    veinsgym::proto::Request request;
    *(request.mutable_shutdown()) = {};
    communicate(request);
}

void GymConnection::finish()
{
    // If you want to record final metrics, do it here.
    // For now, nothing is required.
}

