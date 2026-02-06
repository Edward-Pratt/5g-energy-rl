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
    } else {
        baseSampling = voipSender->par("samplingTime").doubleValue();
        EV_INFO << "Base sampling time from server.app[0]: " << baseSampling << "s\n";
    }
    

    auto *sender0 = top->getModuleByPath("server.app[0]");
    if (!sender0) {
        EV_WARN << "Could not find server.app[0] for generated throughput subscription\n";
    } else {
        sigGenThroughput = omnetpp::cComponent::registerSignal("voipGeneratedThroughput");
        sender0->subscribe(sigGenThroughput, this);
    }


    lastEnergyT = omnetpp::simTime();
    energyJ = 0.0;
    lastThroughput = lastFrameDelay = lastJitter = lastLoss = 0.0;
    deliveredBits = 0.0;
    stepEnergyJ = 0.0;
    bsState = BsState::ACTIVE;



    tick = new omnetpp::cMessage("gymTick");
    scheduleAt(omnetpp::simTime(), tick);
}



void GymConnection::receiveSignal(omnetpp::cComponent *, omnetpp::simsignal_t signalID,
                                  double value, omnetpp::cObject *)
{
    if (signalID == sigGenThroughput) lastThroughput = value;   // <- use generated throughput as "thr"
    else if (signalID == sigThroughput) lastRxThroughput = value;
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
    stepEnergyJ = 0.0;

    auto now = omnetpp::simTime();
    auto dt = (now - lastEnergyT).dbl();
    if (dt <= 0) return;


    double pActive = par("pActive").doubleValue();
    double pSleep  = par("pSleep").doubleValue();

    // mAction in [0,2] -> p in [pSleep, pActive]
    double alpha = std::max(0.0, std::min(2.0, mAction)) / 2.0;
    double p = pSleep + alpha * (pActive - pSleep);
    
    stepEnergyJ = p * dt;
    energyJ += stepEnergyJ;
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

    const double thr = lastRxThroughput;
    const double delay = lastFrameDelay;
    const double jitter = lastJitter;
    const double loss = lastLoss;
    const double numUe = (double)getNumUE();
    const double E = stepEnergyJ;

    double dt = par("tickInterval").doubleValue();
    double stepBits = std::max(0.0, thr) * dt * 8.0;
    deliveredBits += stepBits;

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

    double reward = (0.001 * thr) - (100.0 * delay) - (10.0 * loss) - (0.05 * E);

    if (omnetpp::simTime() > 0.5 && thr < 1e-6) reward -= 1.0;


    auto *rbox = step->mutable_reward()->mutable_box();
    rbox->add_values(reward);


    
    auto reply = communicate(req);

    if (reply.payload_case() == veinsgym::proto::Reply::kAction) {
        const auto &actSpace = reply.action();

        if (actSpace.value_case() == veinsgym::proto::Space::kBox) {
            // Continuous action (Box)
            double m = 0.0;
            if (actSpace.box().values_size() > 0)
                m = actSpace.box().values(0);

            EV_INFO << "Gym multiplier action: " << m << "\n";
            applyMultiplier(m);
        }
        else if (actSpace.value_case() == veinsgym::proto::Space::kDiscrete) {
            // Optional fallback if you still sometimes run Discrete(3)
            int a = (int)actSpace.discrete().value();
            EV_INFO << "Gym discrete action: " << a << "\n";

            // map discrete -> multiplier if you want
            double m = (a == 0) ? 2.0 : (a == 1) ? 1.0 : 0.0;
            applyMultiplier(m);
        }
    }


    scheduleAt(omnetpp::simTime() + par("tickInterval"), tick);
}

void GymConnection::applyMultiplier(double m)
{
    mAction = std::max(0.0, std::min(2.0, m));

    if (!voipSender) return;

    // If m ~ 0: pause traffic
    bool pause = (mAction < 0.05);
    voipSender->par("gymPaused").setBoolValue(pause);

    if (!pause) {
        double newSampling = baseSampling / std::max(0.05, mAction);
        newSampling = std::max(minSampling, std::min(maxSampling, newSampling));
        voipSender->par("samplingTime").setDoubleValue(newSampling);
    }
}

void GymConnection::applyAction(int a)
{
    auto *top = getSystemModule();
    auto *sender = top->getModuleByPath("server.app[0]"); // VoipSender in VoIP-DL
    if (!sender) {
        EV_WARN << "Could not find server.app[0] to control traffic\n";
        return;
    }

    /*bool pauseTraffic = (a == 2);
    sender->par("gymPaused").setBoolValue(pauseTraffic);*/

    if (a == 0) { // ACTIVE: full rate
        sender->par("gymPaused").setBoolValue(false);
        sender->par("samplingTime").setDoubleValue(0.02); // 50 pkt/s
    }
    else if (a == 1) { // SLEEP: reduced rate
        sender->par("gymPaused").setBoolValue(false);
        sender->par("samplingTime").setDoubleValue(0.04); // 25 pkt/s
    }
    else { // LOWPOWER: pause traffic
        sender->par("gymPaused").setBoolValue(true);
    }
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
