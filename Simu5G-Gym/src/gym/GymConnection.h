#pragma once

#include <zmq.hpp>
#include <omnetpp.h>
#include "protobuf/veinsgym.pb.h"
#include <vector>
#include <unordered_map>
#include <cstdint>

// Forward declarations
class GymEnergyConsumer;
namespace simu5g {
    class LtePhyEnb;
    class LtePhyUe;
}

class GymConnection : public omnetpp::cSimpleModule,
                      public omnetpp::cListener
{
public:
    // ── cSimpleModule interface ───────────────────────────────────────────────
    void initialize()    override;
    void handleMessage(omnetpp::cMessage *msg) override;
    void finish()        override;
    ~GymConnection()     override;

    // ── cListener interface (one overload per signal value type used) ─────────
    void receiveSignal(omnetpp::cComponent *source,
                       omnetpp::simsignal_t  signalID,
                       double               value,
                       omnetpp::cObject     *details) override;

    void receiveSignal(omnetpp::cComponent *source,
                       omnetpp::simsignal_t  signalID,
                       omnetpp::intval_t     value,
                       omnetpp::cObject     *details) override;

    void receiveSignal(omnetpp::cComponent *source,
                       omnetpp::simsignal_t  signalID,
                       omnetpp::uintval_t    value,
                       omnetpp::cObject     *details) override;

    void receiveSignal(omnetpp::cComponent       *source,
                       omnetpp::simsignal_t       signalID,
                       const omnetpp::SimTime    &value,
                       omnetpp::cObject          *details) override;

private:
    // ── ZMQ ──────────────────────────────────────────────────────────────────
    zmq::context_t context { 1 };
    zmq::socket_t  socket  { context, zmq::socket_type::req };

    veinsgym::proto::Reply communicate(const veinsgym::proto::Request &request);

    // ── Simulation clock ──────────────────────────────────────────────────────
    omnetpp::cMessage *tick = nullptr;

    // ── Step counter ──────────────────────────────────────────────────────────
    // Member (not static local) so it resets in every initialize() call
    // and does not drift across back-to-back episodes in the same process.
    uint64_t stepId = 1;

    // ── Signal IDs (registered once in initialize) ────────────────────────────
    struct SignalIds {
        omnetpp::simsignal_t genThroughput  = -1;
        omnetpp::simsignal_t throughput     = -1;
        omnetpp::simsignal_t frameDelay     = -1;
        omnetpp::simsignal_t jitter         = -1;
        omnetpp::simsignal_t loss           = -1;
        omnetpp::simsignal_t cbrRxBytes     = -1;
        omnetpp::simsignal_t cbrTxBytes     = -1;
        omnetpp::simsignal_t cbrDelay       = -1;
        omnetpp::simsignal_t sinrDl         = -1;
        omnetpp::simsignal_t sinrUl         = -1;
        omnetpp::simsignal_t measuredSinrDl = -1;
        omnetpp::simsignal_t measuredSinrUl = -1;
    } signals;

    // ── Per-component metric accumulators (cleared each tick) ─────────────────
    // VoIP / CBR receiver metrics — keyed by the app module that emitted them
    std::unordered_map<const omnetpp::cComponent *, double> rxThrByComp;
    std::unordered_map<const omnetpp::cComponent *, double> delayByComp;
    std::unordered_map<const omnetpp::cComponent *, double> jitterByComp;
    std::unordered_map<const omnetpp::cComponent *, double> lossByComp;

    // SINR — keyed by the channel-model module that emitted the sample
    std::unordered_map<const omnetpp::cComponent *, double> sinrByComp;

    // ── CBR byte / delay accumulators (reset each tick) ───────────────────────
    double rxBytesTotal     = 0.0;
    double txBytesTotal     = 0.0;
    double lastRxBytesTotal = 0.0;
    double lastTxBytesTotal = 0.0;

    double delaySum   = 0.0;   // sum of CBR one-way delay samples this tick
    double delaySqSum = 0.0;   // sum of squares (used to compute jitter)
    int    delayCount = 0;     // number of samples this tick

    // ── Last scalar metrics (carried between ticks for VoIP signals) ──────────
    double lastThroughput = 0.0;

    // ── Energy model ──────────────────────────────────────────────────────────
    omnetpp::simtime_t lastEnergyT;
    double energyJ         = 0.0;  // cumulative energy consumed (J)
    double stepEnergyJ     = 0.0;  // energy consumed in the last tick (J)
    double deliveredBits   = 0.0;  // cumulative bits delivered to UEs

    // ── Radio / hardware handles ──────────────────────────────────────────────
    // Only the gNB PHY is controlled by the agent.
    // UE TX power is fixed by omnetpp.ini (**.ueTxPower) and never modified.
    simu5g::LtePhyEnb              *gnbPhy        = nullptr;
    GymEnergyConsumer              *energyConsumer = nullptr;

    // UE PHY pointers are kept solely for SINR channel-model discovery.
    std::vector<simu5g::LtePhyUe *> uePhys;

    // ── TX power state ────────────────────────────────────────────────────────
    double mAction            = 1.0;   // last multiplier received from agent [0, 2]
    double currentTxPowerDbm  = 0.0;   // current gNB TX power (dBm)

    // ── One-time warning flags ────────────────────────────────────────────────
    bool warnedNoVoip  = false;
    bool warnedNoSinr  = false;
    bool shutdownSent  = false;

    // ── Private helpers ───────────────────────────────────────────────────────
    void applyMultiplier(double m);  // set gNB TX power from agent multiplier
    void applyAction(int a);         // discrete action → all server sender apps
    void updateEnergy();             // compute stepEnergyJ for this tick
    int  getNumUE() const;           // read numUe from top-level network module
};