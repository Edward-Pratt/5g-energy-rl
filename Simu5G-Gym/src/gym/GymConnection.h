#pragma once

#include <zmq.hpp>
#include <omnetpp.h>
#include "veinsgym.pb.h"
#include <vector>
#include <unordered_map>

class GymEnergyConsumer;
namespace simu5g { class LtePhyEnb; class LtePhyUe; }

class GymConnection : public omnetpp::cSimpleModule, public omnetpp::cListener {
public:
    void initialize() override;
    void handleMessage(omnetpp::cMessage *msg) override;
    void finish() override;
    void receiveSignal(omnetpp::cComponent *source, omnetpp::simsignal_t signalID,
                       double value, omnetpp::cObject *details) override;
    void receiveSignal(omnetpp::cComponent *source, omnetpp::simsignal_t signalID,
                       omnetpp::intval_t value, omnetpp::cObject *details) override;
    void receiveSignal(omnetpp::cComponent *source, omnetpp::simsignal_t signalID,
                       omnetpp::uintval_t value, omnetpp::cObject *details) override;
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
    double lastThroughput = 0.0;
    double lastFrameDelay = 0.0;
    double lastJitter = 0.0;
    double lastLoss = 0.0;
    double lastGenThroughput = 0.0;

    // ---- Signal IDs (registered in initialize()) ----
    struct SignalIds {
        omnetpp::simsignal_t genThroughput = -1;
        omnetpp::simsignal_t throughput = -1;
        omnetpp::simsignal_t frameDelay = -1;
        omnetpp::simsignal_t jitter = -1;
        omnetpp::simsignal_t loss = -1;
        omnetpp::simsignal_t sinrDl = -1;
        omnetpp::simsignal_t sinrUl = -1;
        omnetpp::simsignal_t measuredSinrDl = -1;
        omnetpp::simsignal_t measuredSinrUl = -1;
        omnetpp::simsignal_t cbrRxBytes = -1;
        omnetpp::simsignal_t cbrTxBytes = -1;
        omnetpp::simsignal_t cbrDelay = -1;
    } signals;

    double baseSampling = 0.02; // 50 pkt/s
    double minSampling = 0.005;
    double maxSampling = 0.2;

    double mAction = 1.0;
    void applyMultiplier(double m);

    // ---- Energy Model ----
    enum class BsState { ACTIVE = 0, SLEEP = 1, LOWPOWER = 2};
    BsState bsState = BsState::ACTIVE;

    omnetpp::simtime_t lastEnergyT;
    double energyJ = 0.0;
    double stepEnergyJ = 0.0;
    double deliveredBits = 0.0;
    double currentTxPowerDbm = 0.0;

    GymEnergyConsumer* energyConsumer = nullptr;
    simu5g::LtePhyEnb* gnbPhy = nullptr;
    std::vector<simu5g::LtePhyUe*> uePhys;

    std::unordered_map<const omnetpp::cComponent*, double> rxThrByComp;
    std::unordered_map<const omnetpp::cComponent*, double> delayByComp;
    std::unordered_map<const omnetpp::cComponent*, double> jitterByComp;
    std::unordered_map<const omnetpp::cComponent*, double> lossByComp;
    std::unordered_map<const omnetpp::cComponent*, double> sinrByComp;

    double rxBytesTotal = 0.0;
    double txBytesTotal = 0.0;
    double lastRxBytesTotal = 0.0;
    double lastTxBytesTotal = 0.0;
    double delaySum = 0.0;
    double delaySqSum = 0.0;
    int delayCount = 0;

    bool warnedNoSinr = false;
    bool warnedNoVoip = false;
    bool shutdownSent = false;

    //Helper
    void updateEnergy();
    int getNumUE() const;
};
