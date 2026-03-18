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
#include "GymEnergyConsumer.h"
#include "simu5g/stack/phy/LtePhyEnb.h"
#include "simu5g/stack/phy/LtePhyUe.h"
#include "simu5g/stack/phy/channelmodel/LteChannelModel.h"
#include <cmath>
#include <algorithm>

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
    signals.throughput = omnetpp::cComponent::registerSignal("voipReceivedThroughput");
    signals.frameDelay = omnetpp::cComponent::registerSignal("voipFrameDelay");
    signals.jitter = omnetpp::cComponent::registerSignal("voipJitter");
    signals.loss = omnetpp::cComponent::registerSignal("voipFrameLoss");
    signals.cbrRxBytes = omnetpp::cComponent::registerSignal("cbrReceivedBytes");
    signals.cbrTxBytes = omnetpp::cComponent::registerSignal("cbrGeneratedBytes");
    signals.cbrDelay = omnetpp::cComponent::registerSignal("cbrFrameDelay");

    int receiverCount = 0;
    auto subscribeVoip = [&](omnetpp::cModule *app) {
        if (!app)
            return;
        app->subscribe(signals.throughput, this);
        app->subscribe(signals.frameDelay, this);
        app->subscribe(signals.jitter, this);
        app->subscribe(signals.loss, this);
        app->subscribe(signals.cbrRxBytes, this);
        app->subscribe(signals.cbrTxBytes, this);
        app->subscribe(signals.cbrDelay, this);
        receiverCount++;
    };


    // numResourceBlocks = par("numResourceBlocks").intValue();



    for (int i = 0;; ++i) {
        auto *ue = top->getSubmodule("ue", i);
        if (!ue)
            break;
        for (int j = 0;; ++j) {
            auto *app = ue->getSubmodule("app", j);
            if (!app)
                break;
            subscribeVoip(app);
        }
    }

    if (auto *server = top->getSubmodule("server")) {
        for (int j = 0;; ++j) {
            auto *app = server->getSubmodule("app", j);
            if (!app)
                break;
            subscribeVoip(app);
        }
    }

    if (receiverCount == 0) {
        EV_WARN << "No VoIP apps found to subscribe. Metrics will remain 0.\n";
        warnedNoVoip = true;
    } else {
        EV_INFO << "Subscribed to VoIP signals on " << receiverCount << " app modules\n";
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
        signals.genThroughput = omnetpp::cComponent::registerSignal("voipGeneratedThroughput");
        sender0->subscribe(signals.genThroughput, this);
    }

    lastEnergyT = omnetpp::simTime();
    energyJ = 0.0;
    lastThroughput = lastFrameDelay = lastJitter = lastLoss = 0.0;
    deliveredBits = 0.0;
    stepEnergyJ = 0.0;
    bsState = BsState::ACTIVE;

    const char *gnbPhyPath = par("gnbPhyPath");
    gnbPhy = dynamic_cast<simu5g::LtePhyEnb *>(findModuleByPath(gnbPhyPath));
    if (!gnbPhy) {
        EV_WARN << "Could not find gNB PHY at '" << gnbPhyPath << "'. TX power will not be controlled.\n";
    }

    const char *energyConsumerPath = par("energyConsumerPath");
    energyConsumer = dynamic_cast<GymEnergyConsumer *>(findModuleByPath(energyConsumerPath));
    if (!energyConsumer) {
        EV_WARN << "Could not find energy consumer at '" << energyConsumerPath << "'. INET energy tracking disabled.\n";
    }

    for (int i = 0;; ++i) {
        auto *ue = top->getSubmodule("ue", i);
        if (!ue)
            break;
        auto addPhy = [&](omnetpp::cModule *phyMod, const char *label) {
            if (!phyMod)
                return;
            auto *uePhy = dynamic_cast<simu5g::LtePhyUe *>(phyMod);
            if (uePhy) {
                if (std::find(uePhys.begin(), uePhys.end(), uePhy) == uePhys.end())
                    uePhys.push_back(uePhy);
            } else {
                EV_WARN << "Module " << ue->getFullPath() << "." << label << " is not a LtePhyUe-derived PHY\n";
            }
        };

        addPhy(ue->findModuleByPath("cellularNic.phy"), "cellularNic.phy");
        addPhy(ue->findModuleByPath("cellularNic.nrPhy"), "cellularNic.nrPhy");
    }
    if (uePhys.empty()) {
        EV_WARN << "No UE PHY modules found. UE TX power will not be controlled.\n";
    } else {
        EV_WARN << "UE PHY modules found: " << uePhys.size() << "\n";
    }

    const char *sinrDlPath = par("sinrDlPath");
    const char *sinrUlPath = par("sinrUlPath");
    signals.sinrDl = omnetpp::cComponent::registerSignal("rcvdSinrDl");
    signals.sinrUl = omnetpp::cComponent::registerSignal("rcvdSinrUl");
    signals.measuredSinrDl = omnetpp::cComponent::registerSignal("measuredSinrDl");
    signals.measuredSinrUl = omnetpp::cComponent::registerSignal("measuredSinrUl");

    std::vector<omnetpp::cModule*> sinrModules;
    auto addSinrModule = [&](omnetpp::cModule *mod) {
        if (!mod)
            return;
        if (std::find(sinrModules.begin(), sinrModules.end(), mod) == sinrModules.end())
            sinrModules.push_back(mod);
    };

    for (auto *uePhy : uePhys) {
        if (!uePhy)
            continue;
        const char *cmPath = uePhy->par("channelModelModule").stringValue();
        if (cmPath && *cmPath) {
            auto *cm = uePhy->findModuleByPath(cmPath);
            if (cm)
                addSinrModule(cm);
            else {
                auto *ue = uePhy->getParentModule();
                if (ue) {
                    addSinrModule(ue->findModuleByPath("cellularNic.nrChannelModel[0]"));
                    addSinrModule(ue->findModuleByPath("cellularNic.channelModel[0]"));
                }
            }
        }
    }

    if (sinrModules.empty()) {
        auto *sinrDlModule = (sinrDlPath && *sinrDlPath) ? findModuleByPath(sinrDlPath) : nullptr;
        auto *sinrUlModule = (sinrUlPath && *sinrUlPath) ? findModuleByPath(sinrUlPath) : nullptr;
        addSinrModule(sinrDlModule);
        addSinrModule(sinrUlModule);
    }

    if (sinrModules.empty()) {
        auto collectChannelModels = [&](omnetpp::cModule *mod, auto &self) -> void {
            for (omnetpp::cModule::SubmoduleIterator it(mod); !it.end(); ++it) {
                auto *sub = *it;
                if (!sub)
                    continue;
                if (dynamic_cast<simu5g::LteChannelModel *>(sub))
                    addSinrModule(sub);
                self(sub, self);
            }
        };
        collectChannelModels(top, collectChannelModels);
    }

    if (!sinrModules.empty()) {
        for (auto *mod : sinrModules) {
            mod->subscribe(signals.sinrDl, this);
            mod->subscribe(signals.sinrUl, this);
            mod->subscribe(signals.measuredSinrDl, this);
            mod->subscribe(signals.measuredSinrUl, this);
            EV_WARN << "Subscribed SINR signals on " << mod->getFullPath() << "\n";
        }
    } else {
        EV_WARN << "No SINR channel model modules found. SINR will remain 0.\n";
        for (auto *uePhy : uePhys) {
            if (!uePhy)
                continue;
            const char *cmPath = uePhy->par("channelModelModule").stringValue();
            EV_WARN << "UE PHY " << uePhy->getFullPath() << " channelModelModule='" << (cmPath ? cmPath : "") << "'\n";
        }
        if (sinrDlPath && *sinrDlPath)
            EV_WARN << "sinrDlPath override='" << sinrDlPath << "'\n";
        if (sinrUlPath && *sinrUlPath)
            EV_WARN << "sinrUlPath override='" << sinrUlPath << "'\n";
    }




    currentTxPowerDbm = par("txPowerMax").doubleValue();
    applyMultiplier(mAction);



    tick = new omnetpp::cMessage("gymTick");
    scheduleAt(omnetpp::simTime(), tick);
}



void GymConnection::receiveSignal(omnetpp::cComponent *source, omnetpp::simsignal_t signalID,
                                  double value, omnetpp::cObject *)
{
    if (signalID == signals.genThroughput) lastThroughput = value;
    else if (signalID == signals.throughput) rxThrByComp[source] = value;
    else if (signalID == signals.frameDelay) delayByComp[source] = value;
    else if (signalID == signals.jitter) jitterByComp[source] = value;
    else if (signalID == signals.loss) lossByComp[source] = value;
    else if (signalID == signals.sinrDl || signalID == signals.sinrUl || 
             signalID == signals.measuredSinrDl || signalID == signals.measuredSinrUl)
        sinrByComp[source] = value;

}

void GymConnection::receiveSignal(omnetpp::cComponent *, omnetpp::simsignal_t signalID,
                                  omnetpp::intval_t value, omnetpp::cObject *)
{
    if (signalID == signals.cbrRxBytes)
        rxBytesTotal += static_cast<double>(value);
    else if (signalID == signals.cbrTxBytes)
        txBytesTotal += static_cast<double>(value);
}

void GymConnection::receiveSignal(omnetpp::cComponent *, omnetpp::simsignal_t signalID,
                                  omnetpp::uintval_t value, omnetpp::cObject *)
{
    if (signalID == signals.cbrRxBytes)
        rxBytesTotal += static_cast<double>(value);
    else if (signalID == signals.cbrTxBytes)
        txBytesTotal += static_cast<double>(value);
}


void GymConnection::receiveSignal(omnetpp::cComponent *source, omnetpp::simsignal_t signalID,
                                  const omnetpp::SimTime& value, omnetpp::cObject *)
{
    // convert SimTime to seconds (double)
    double v = value.dbl();

    if (signalID == signals.frameDelay) delayByComp[source] = v;
    else if (signalID == signals.jitter) jitterByComp[source] = v;
    else if (signalID == signals.cbrDelay) {
        delaySum += v;
        delaySqSum += v * v;
        delayCount++;
    }
}

void GymConnection::updateEnergy()
{
    stepEnergyJ = 0.0;

    auto now = omnetpp::simTime();
    auto dt = (now - lastEnergyT).dbl();
    if (dt <= 0) return;

    double pActive = par("pActive").doubleValue();
    double pSleep  = par("pSleep").doubleValue();
    double pComputeBase = par("pComputeBase").doubleValue();
    double pComputePerUe = par("pComputePerUe").doubleValue();
    double pTxCoeff = par("pTxCoeff").doubleValue();

    // mAction in [0,2] -> base power in [pSleep, pActive]
    double alpha = std::max(0.0, std::min(2.0, mAction)) / 2.0;
    double pBase = pSleep + alpha * (pActive - pSleep);
    double pCompute = pComputeBase + pComputePerUe * getNumUE() * alpha;

    // tx power (dBm) -> mW
    double txPowerMw = std::pow(10.0, currentTxPowerDbm / 10.0);
    double pTx = pTxCoeff * txPowerMw;

    double p = pBase + pCompute + pTx;
    if (energyConsumer)
        energyConsumer->setPowerConsumptionW(p);

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

    auto sumMap = [](const std::unordered_map<const omnetpp::cComponent*, double> &m) {
        double s = 0.0;
        for (const auto &kv : m)
            s += kv.second;
        return s;
    };
    auto avgMap = [](const std::unordered_map<const omnetpp::cComponent*, double> &m) {
        if (m.empty())
            return 0.0;
        double s = 0.0;
        for (const auto &kv : m)
            s += kv.second;
        return s / static_cast<double>(m.size());
    };

    const double rxBytesDelta = rxBytesTotal - lastRxBytesTotal;
    const double txBytesDelta = txBytesTotal - lastTxBytesTotal;
    const bool hasCbr = (rxBytesDelta > 0.0) || (txBytesDelta > 0.0) || (delayCount > 0);

    double thr_cbr = 0.0;
    double delay_cbr = 0.0;
    double jitter_cbr = 0.0;
    double loss_cbr = 0.0;

    if (hasCbr) {
        double rxDelta = std::max(0.0, rxBytesDelta);
        double txDelta = std::max(0.0, txBytesDelta);
        double dt = par("tickInterval").doubleValue();
        thr_cbr = (dt > 0.0) ? (rxDelta * 8.0 / dt) : 0.0; // bps

        if (delayCount > 0) {
            delay_cbr = delaySum / delayCount;
            double meanSq = delaySqSum / delayCount;
            jitter_cbr = std::sqrt(std::max(0.0, meanSq - (delay_cbr * delay_cbr)));
        }

        if (txDelta > 0.0) {
            double ratio = rxDelta / txDelta;
            ratio = std::max(0.0, std::min(1.0, ratio));
            loss_cbr = 1.0 - ratio;
        }

        lastRxBytesTotal = rxBytesTotal;
        lastTxBytesTotal = txBytesTotal;
        delaySum = 0.0;
        delaySqSum = 0.0;
        delayCount = 0;
    }

    double thr_voip = sumMap(rxThrByComp) * 8.0; // convert B/s -> bps
    double delay_voip = avgMap(delayByComp);
    double jitter_voip = avgMap(jitterByComp);
    double loss_voip = avgMap(lossByComp);

    // Combine VoIP and CBR
    double thr = thr_cbr + thr_voip;
    double delay = (delay_cbr > 0 && delay_voip > 0) ? (delay_cbr + delay_voip) / 2.0 : std::max(delay_cbr, delay_voip);
    double jitter = (jitter_cbr > 0 && jitter_voip > 0) ? (jitter_cbr + jitter_voip) / 2.0 : std::max(jitter_cbr, jitter_voip);
    double loss = (loss_cbr > 0 && loss_voip > 0) ? (loss_cbr + loss_voip) / 2.0 : std::max(loss_cbr, loss_voip);

    const double sinr = avgMap(sinrByComp);

    // Clear maps to avoid stale values in the next tick
    rxThrByComp.clear();
    delayByComp.clear();
    jitterByComp.clear();
    lossByComp.clear();
    sinrByComp.clear();

    const double numUe = (double)getNumUE();
    const double E = stepEnergyJ;

    double dt = par("tickInterval").doubleValue();
    double stepBits = std::max(0.0, thr) * dt;
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
    box->add_values(currentTxPowerDbm);
    box->add_values(sinr);

    // IMPROVED REWARD FUNCTION (ver 3.0):
    // Goal: Minimize energy while maintaining High Quality of Service (QoS)
    // 
    // Rationale: Throughput is a CONSEQUENCE of control, not a control objective.
    // Instead, focus on delay/loss/jitter (actual QoS metrics per ITU-T G.131) 
    // combined with energy minimization.
    // 
    // QoS targets (ITU-T G.131 for VoIP):
    //  - Delay: <150ms ideal, <400ms acceptable
    //  - Loss: <1% excellent, <3% acceptable
    //  - Jitter: <5ms excellent
    
    double reward = 0.0;
    
    // ============ DELAY COMPONENT (Primary QoS Metric) ============
    // Reward good delay, heavily penalize poor delay
    double delay_reward = 0.0;
    if (delay < 0.05) {
        // Excellent: delay < 50ms
        delay_reward = 1.0;
    } else if (delay < 0.15) {
        // Good: 50-150ms (ideal ITU-T range)
        delay_reward = 0.5;
    } else if (delay < 0.30) {
        // Acceptable: 150-300ms (upper ITU-T range)
        delay_reward = 0.0;
    } else if (delay < 0.50) {
        // Poor: 300-500ms
        delay_reward = -1.0 * (delay - 0.30);
    } else {
        // Very poor: >500ms
        delay_reward = -1.0 - (2.0 * (delay - 0.50));
    }
    
    // ============ LOSS COMPONENT (Secondary QoS Metric) ============
    // Reward low loss, penalize high loss
    double loss_reward = 0.0;
    if (loss < 0.01) {
        // Excellent: <1% loss
        loss_reward = 0.5;
    } else if (loss < 0.03) {
        // Good: 1-3% loss
        loss_reward = 0.25;
    } else if (loss < 0.10) {
        // Acceptable: 3-10% loss
        loss_reward = 0.0;
    } else if (loss < 0.30) {
        // Poor: 10-30% loss
        loss_reward = -0.5 * loss;
    } else {
        // Very poor: >30% loss (network broken)
        loss_reward = -2.0;
    }
    
    // ============ JITTER COMPONENT (Tertiary QoS Metric) ============
    // Reward stable delay (low jitter)
    double jitter_reward = 0.0;
    if (jitter < 0.005) {
        // Excellent: jitter <5ms
        jitter_reward = 0.25;
    } else if (jitter < 0.020) {
        // Good: jitter 5-20ms
        jitter_reward = 0.1;
    } else if (jitter < 0.050) {
        // Acceptable: jitter 20-50ms
        jitter_reward = 0.0;
    } else {
        // Poor: jitter >50ms
        jitter_reward = -0.5 * (jitter - 0.050);
    }
    
    // ============ ENERGY COMPONENT (Minimization Objective) ============
    // Always penalize energy consumption
    // Scale: 3.5J typical operation = -0.35 penalty
    double energy_penalty = E / 10.0;
    
    // ============ COMBINE COMPONENTS ============
    // QoS is primary (sum of delay, loss, jitter)
    // Energy is secondary objective (minimize if QoS maintained)
    double qos_score = delay_reward + loss_reward + jitter_reward;
    
    // If QoS is poor, heavily penalize; otherwise apply energy penalty
    if (qos_score < -1.0) {
        // QoS severely degraded, don't bother with energy optimization
        reward = qos_score - 1.0;
    } else {
        // QoS is reasonable, apply energy penalty for optimization
        reward = qos_score - energy_penalty;
    }
    
    // ============ TERMINAL CONDITION PENALTY ============
    // Penalize if no throughput (stalled network)
    if (omnetpp::simTime() > 0.5 && thr < 1e-6) {
        reward -= 1.0;
    }

    if (!warnedNoSinr && sinrByComp.empty() && omnetpp::simTime() > 0.5) {
        EV_WARN << "No SINR samples received yet. Check channel model paths and collectSinrStatistics.\n";
        warnedNoSinr = true;
    }


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

            // Apply traffic rate control for discrete action
            applyAction(a);

            // map discrete -> power multiplier if you want
            double m = (a == 0) ? 2.0 : (a == 1) ? 1.0 : 0.0;
            applyMultiplier(m);
        }
    }

    scheduleAt(omnetpp::simTime() + par("tickInterval"), tick);
}

void GymConnection::applyMultiplier(double m)
{
    mAction = std::max(0.0, std::min(2.0, m));

    double alpha = mAction / 2.0;
    double txMin = par("txPowerMin").doubleValue();
    double txMax = par("txPowerMax").doubleValue();
    currentTxPowerDbm = txMin + alpha * (txMax - txMin);

    if (gnbPhy)
        gnbPhy->setTxPowerDbm(currentTxPowerDbm);

    for (auto *uePhy : uePhys)
        uePhy->setTxPowerDbm(currentTxPowerDbm);
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
    auto recv_result = socket.recv(response_msg, zmq::recv_flags::none);
    if (!recv_result) {
        EV_WARN << "ZMQ recv failed\n";
        return veinsgym::proto::Reply();
    }

    std::string response(static_cast<char*>(response_msg.data()), response_msg.size());
    veinsgym::proto::Reply reply;
    reply.ParseFromString(response);
    return reply;
}

GymConnection::~GymConnection()
{
    if (tick)
        cancelAndDelete(tick);

    if (!shutdownSent) {
        veinsgym::proto::Request request;
        *(request.mutable_shutdown()) = {};
        communicate(request);
        shutdownSent = true;
    }
}

void GymConnection::finish()
{
    if (!shutdownSent) {
        veinsgym::proto::Request request;
        *(request.mutable_shutdown()) = {};
        communicate(request);
        shutdownSent = true;
    }
}
