#pragma once

#include <zmq.hpp>
#include <omnetpp.h>
#include "veinsgym.pb.h"

class GymConnection : public omnetpp::cSimpleModule, public omnetpp::cListener {
public:
    void initialize() override;
    void handleMessage(omnetpp::cMessage *msg) override;
    void finish() override;
    void receiveSignal(omnetpp::cComponent *source, omnetpp::simsignal_t signalID,
                       double value, omnetpp::cObject *details) override;
    void receiveSignal(omnetpp::cComponent *src, omnetpp::simsignal_t id,
                       const omnetpp::SimTime& value, omnetpp::cObject *details) override;
    
    ~GymConnection() override;

private:
    veinsgym::proto::Reply communicate(const veinsgym::proto::Request& request);

    zmq::context_t context = zmq::context_t(1);
    zmq::socket_t socket = zmq::socket_t(context, zmq::socket_type::req);
    omnetpp::cMessage* tick = nullptr;
    omnetpp::cModule* voipSender = nullptr;
    void applyAction(int a);
    bool trafficPaused = false;
    void setTrafficPaused(bool paused);

    // ---- VoIP metrics (latest observed values) ----
    double lastThroughput= 0.0;
    double lastFrameDelay = 0.0;
    double lastJitter = 0.0;
    double lastLoss = 0.0;

    // signal IDs (initialize to -1; avoid SIMSIGNAL_NULL headaches)
    omnetpp::simsignal_t sigThroughput = -1;
    omnetpp::simsignal_t sigFrameDelay = -1;
    omnetpp::simsignal_t sigJitter = -1;
    omnetpp::simsignal_t sigLoss = -1;   // pick one loss signal to start


    // ---- Energy Model ----
    enum class BsState { ACTIVE = 0, SLEEP = 1, LOWPOWER = 2};
    BsState bsState = BsState::ACTIVE;

    omnetpp::simtime_t lastEnergyT;
    double energyJ = 0.0;

    //Helper
    void updateEnergy();
    int getNumUE() const;
};

