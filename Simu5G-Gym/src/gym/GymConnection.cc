//
// GymConnection.cc  – reviewed and corrected version
//

#include "GymConnection.h"
#include "GymEnergyConsumer.h"
#include "simu5g/stack/phy/LtePhyEnb.h"
#include "simu5g/stack/phy/LtePhyUe.h"
#include "simu5g/stack/phy/channelmodel/LteChannelModel.h"
#include <cmath>
#include <algorithm>

Define_Module(GymConnection);

// ── Helpers ───────────────────────────────────────────────────────────────────

static double safe(double x)
{
    return std::isfinite(x) ? x : 0.0;
}

static double avgMap(const std::unordered_map<const omnetpp::cComponent *, double> &m)
{
    if (m.empty()) return 0.0;
    double s = 0.0;
    for (const auto &kv : m) s += kv.second;
    return s / static_cast<double>(m.size());
}

static double sumMap(const std::unordered_map<const omnetpp::cComponent *, double> &m)
{
    double s = 0.0;
    for (const auto &kv : m) s += kv.second;
    return s;
}

// ── initialize ────────────────────────────────────────────────────────────────

void GymConnection::initialize()
{
    // ── ZMQ connection ────────────────────────────────────────────────────────
    std::string host = par("host").stdstringValue();
    int         port = par("port").intValue();

    if (host.empty()) {
        const char *env = std::getenv("VEINS_GYM_HOST");
        if (env) host = env;
        else throw omnetpp::cRuntimeError("Gym host not configured!");
    }
    if (port < 0) {
        const char *env = std::getenv("VEINS_GYM_PORT");
        if (env) port = std::atoi(env);
        else throw omnetpp::cRuntimeError("Gym port not configured!");
    }

    EV_INFO << "Connecting to tcp://" << host << ":" << port << "\n";
    socket.connect("tcp://" + host + ":" + std::to_string(port));

    veinsgym::proto::Request init_request;
    *(init_request.mutable_init()->mutable_observation_space_code()) =
        par("observation_space").stdstringValue();
    *(init_request.mutable_init()->mutable_action_space_code()) =
        par("action_space").stdstringValue();
    communicate(init_request);

    // ── Signal IDs ───────────────────────────────────────────────────────────
    signals.throughput    = omnetpp::cComponent::registerSignal("voipReceivedThroughput");
    signals.frameDelay    = omnetpp::cComponent::registerSignal("voipFrameDelay");
    signals.jitter        = omnetpp::cComponent::registerSignal("voipJitter");
    signals.loss          = omnetpp::cComponent::registerSignal("voipFrameLoss");
    signals.cbrRxBytes    = omnetpp::cComponent::registerSignal("cbrReceivedBytes");
    signals.cbrTxBytes    = omnetpp::cComponent::registerSignal("cbrGeneratedBytes");
    signals.cbrDelay      = omnetpp::cComponent::registerSignal("cbrFrameDelay");
    // FIX #6: register genThroughput exactly once, here.
    signals.genThroughput = omnetpp::cComponent::registerSignal("voipGeneratedThroughput");
    signals.sinrDl        = omnetpp::cComponent::registerSignal("rcvdSinrDl");
    signals.sinrUl        = omnetpp::cComponent::registerSignal("rcvdSinrUl");
    signals.measuredSinrDl = omnetpp::cComponent::registerSignal("measuredSinrDl");
    signals.measuredSinrUl = omnetpp::cComponent::registerSignal("measuredSinrUl");

    auto *top = getSystemModule();

    // ── Subscribe to app signals ──────────────────────────────────────────────
    // FIX #1 / #6: subscribe every app on every UE AND the server, and also
    // subscribe the server's genThroughput signal on every sender app (not
    // just app[0]).  This handles the new multi-phase / multi-UE scenarios.
    int subscribedCount = 0;

    auto subscribeApp = [&](omnetpp::cModule *app) {
        if (!app) return;
        app->subscribe(signals.throughput,    this);
        app->subscribe(signals.frameDelay,    this);
        app->subscribe(signals.jitter,        this);
        app->subscribe(signals.loss,          this);
        app->subscribe(signals.cbrRxBytes,    this);
        app->subscribe(signals.cbrTxBytes,    this);
        app->subscribe(signals.cbrDelay,      this);
        app->subscribe(signals.genThroughput, this);  // harmless on receivers
        ++subscribedCount;
    };

    for (int i = 0; ; ++i) {
        auto *ue = top->getSubmodule("ue", i);
        if (!ue) break;
        for (int j = 0; ; ++j) {
            auto *app = ue->getSubmodule("app", j);
            if (!app) break;
            subscribeApp(app);
        }
    }

    if (auto *server = top->getSubmodule("server")) {
        for (int j = 0; ; ++j) {
            auto *app = server->getSubmodule("app", j);
            if (!app) break;
            subscribeApp(app);
        }
    }

    if (subscribedCount == 0) {
        EV_WARN << "No app modules found to subscribe. All traffic metrics will be 0.\n";
        warnedNoVoip = true;
    } else {
        EV_INFO << "Subscribed to app signals on " << subscribedCount << " modules.\n";
    }

    // FIX #5: baseSampling was read but never used; removed to avoid dead code.
    // If you need it later, re-add it and document its purpose.

    // ── gNB PHY ───────────────────────────────────────────────────────────────
    const char *gnbPhyPath = par("gnbPhyPath").stringValue();
    gnbPhy = dynamic_cast<simu5g::LtePhyEnb *>(findModuleByPath(gnbPhyPath));
    if (!gnbPhy)
        EV_WARN << "gNB PHY not found at '" << gnbPhyPath << "'. TX power will not be controlled.\n";

    // ── Energy consumer ───────────────────────────────────────────────────────
    const char *ecPath = par("energyConsumerPath").stringValue();
    energyConsumer = dynamic_cast<GymEnergyConsumer *>(findModuleByPath(ecPath));
    if (!energyConsumer)
        EV_WARN << "Energy consumer not found at '" << ecPath << "'.\n";

    // ── Validate energy parameters at startup (FIX #4) ───────────────────────
    // Reading them now causes an early, clear error if the NED parameter is
    // missing, rather than a cryptic crash at runtime during the first tick.
    (void)par("pActive").doubleValue();
    (void)par("pSleep").doubleValue();
    (void)par("pComputeBase").doubleValue();
    (void)par("pComputePerUe").doubleValue();
    (void)par("pTxCoeff").doubleValue();

    // ── UE PHY modules ────────────────────────────────────────────────────────
    for (int i = 0; ; ++i) {
        auto *ue = top->getSubmodule("ue", i);
        if (!ue) break;

        auto tryAddPhy = [&](const char *relPath) {
            auto *phyMod = ue->findModuleByPath(relPath);
            if (!phyMod) return;
            auto *uePhy = dynamic_cast<simu5g::LtePhyUe *>(phyMod);
            if (uePhy) {
                if (std::find(uePhys.begin(), uePhys.end(), uePhy) == uePhys.end())
                    uePhys.push_back(uePhy);
            }
        };

        tryAddPhy("cellularNic.phy");
        tryAddPhy("cellularNic.nrPhy");
    }

    if (uePhys.empty())
        EV_WARN << "No UE PHY modules found for SINR channel model discovery.\n";
    else
        EV_INFO << "Found " << uePhys.size() << " UE PHY module(s) "
                   "(used for SINR subscription only, TX power not controlled).\n";

    // ── SINR channel model subscription ───────────────────────────────────────
    std::vector<omnetpp::cModule *> sinrModules;

    auto addSinrModule = [&](omnetpp::cModule *mod) {
        if (!mod) return;
        if (std::find(sinrModules.begin(), sinrModules.end(), mod) == sinrModules.end())
            sinrModules.push_back(mod);
    };

    // Prefer the channel model referenced by each UE PHY's parameter.
    for (auto *uePhy : uePhys) {
        const char *cmPath = uePhy->par("channelModelModule").stringValue();
        if (cmPath && *cmPath) {
            auto *cm = uePhy->findModuleByPath(cmPath);
            if (cm) {
                addSinrModule(cm);
                continue;
            }
        }
        // Fallback: sibling submodules.
        auto *ue = uePhy->getParentModule();
        if (ue) {
            addSinrModule(ue->findModuleByPath("cellularNic.nrChannelModel[0]"));
            addSinrModule(ue->findModuleByPath("cellularNic.channelModel[0]"));
        }
    }

    // Second fallback: explicit ini paths.
    if (sinrModules.empty()) {
        const char *dlPath = par("sinrDlPath").stringValue();
        const char *ulPath = par("sinrUlPath").stringValue();
        if (dlPath && *dlPath) addSinrModule(findModuleByPath(dlPath));
        if (ulPath && *ulPath) addSinrModule(findModuleByPath(ulPath));
    }

    // Third fallback: walk the whole tree.
    if (sinrModules.empty()) {
        std::function<void(omnetpp::cModule *)> walk = [&](omnetpp::cModule *mod) {
            for (omnetpp::cModule::SubmoduleIterator it(mod); !it.end(); ++it) {
                auto *sub = *it;
                if (dynamic_cast<simu5g::LteChannelModel *>(sub))
                    addSinrModule(sub);
                walk(sub);
            }
        };
        walk(top);
    }

    if (!sinrModules.empty()) {
        for (auto *mod : sinrModules) {
            mod->subscribe(signals.sinrDl,          this);
            mod->subscribe(signals.sinrUl,          this);
            mod->subscribe(signals.measuredSinrDl,  this);
            mod->subscribe(signals.measuredSinrUl,  this);
            EV_INFO << "Subscribed SINR signals on " << mod->getFullPath() << "\n";
        }
    } else {
        EV_WARN << "No SINR channel model modules found. SINR observations will be 0.\n";
    }

    // ── State initialisation ──────────────────────────────────────────────────
    lastThroughput = 0.0;
    deliveredBits  = 0.0;
    stepEnergyJ    = 0.0;

    // FIX #9: stepId reset to 1 per initialize() call so it does not drift
    // across back-to-back episodes in the same process.
    stepId = 1;

    currentTxPowerDbm = par("txPowerMax").doubleValue();
    applyMultiplier(mAction);

    tick = new omnetpp::cMessage("gymTick");
    scheduleAt(omnetpp::simTime(), tick);
}

// ── Signal reception ──────────────────────────────────────────────────────────

void GymConnection::receiveSignal(omnetpp::cComponent *source,
                                  omnetpp::simsignal_t  signalID,
                                  double value, omnetpp::cObject *)
{
    if      (signalID == signals.genThroughput) lastThroughput       = value;
    else if (signalID == signals.throughput)    rxThrByComp[source]  = value;
    else if (signalID == signals.loss)          lossByComp[source]   = value;
    // FIX #2: double-overload handles voipFrameDelay / voipJitter.
    // The SimTime overload below handles cbrFrameDelay only.
    // Previously both overloads wrote to delayByComp / jitterByComp,
    // causing the last-fired overload to silently win.
    else if (signalID == signals.frameDelay)    delayByComp[source]  = value;
    else if (signalID == signals.jitter)        jitterByComp[source] = value;
    else if (signalID == signals.sinrDl  || signalID == signals.sinrUl ||
             signalID == signals.measuredSinrDl || signalID == signals.measuredSinrUl)
        sinrByComp[source] = value;
}

void GymConnection::receiveSignal(omnetpp::cComponent *,
                                  omnetpp::simsignal_t  signalID,
                                  omnetpp::intval_t value, omnetpp::cObject *)
{
    if      (signalID == signals.cbrRxBytes) rxBytesTotal += static_cast<double>(value);
    else if (signalID == signals.cbrTxBytes) txBytesTotal += static_cast<double>(value);
}

void GymConnection::receiveSignal(omnetpp::cComponent *,
                                  omnetpp::simsignal_t  signalID,
                                  omnetpp::uintval_t value, omnetpp::cObject *)
{
    if      (signalID == signals.cbrRxBytes) rxBytesTotal += static_cast<double>(value);
    else if (signalID == signals.cbrTxBytes) txBytesTotal += static_cast<double>(value);
}

void GymConnection::receiveSignal(omnetpp::cComponent *,
                                  omnetpp::simsignal_t  signalID,
                                  const omnetpp::SimTime &value, omnetpp::cObject *)
{
    // FIX #2: only CBR delay is a SimTime signal — do not touch delayByComp
    // or jitterByComp here; those are handled in the double overload above.
    if (signalID == signals.cbrDelay) {
        double v = value.dbl();
        delaySum   += v;
        delaySqSum += v * v;
        ++delayCount;
    }
}

// ── Energy ────────────────────────────────────────────────────────────────────

void GymConnection::updateEnergy()
{
    stepEnergyJ = 0.0;

    auto now = omnetpp::simTime();
    double dt = (now - lastEnergyT).dbl();
    if (dt <= 0.0) return;

    // Parameters are validated once in initialize(); reads here are safe.
    double pActive      = par("pActive").doubleValue();
    double pSleep       = par("pSleep").doubleValue();
    double pComputeBase = par("pComputeBase").doubleValue();
    double pComputePerUe = par("pComputePerUe").doubleValue();
    double pTxCoeff     = par("pTxCoeff").doubleValue();

    double alpha    = std::max(0.0, std::min(2.0, mAction)) / 2.0;
    double pBase    = pSleep + alpha * (pActive - pSleep);
    double pCompute = pComputeBase + pComputePerUe * getNumUE() * alpha;
    double txMw     = std::pow(10.0, currentTxPowerDbm / 10.0);
    double pTx      = pTxCoeff * txMw;
    double p        = pBase + pCompute + pTx;

    if (energyConsumer)
        energyConsumer->setPowerConsumptionW(p);

    stepEnergyJ = p * dt;
    energyJ    += stepEnergyJ;
    lastEnergyT = now;
}

// FIX #10: warn explicitly if numUe is missing.
int GymConnection::getNumUE() const
{
    auto *top = getSystemModule();
    if (top->hasPar("numUe"))
        return top->par("numUe").intValue();
    EV_WARN << "numUe parameter not found on network; energy computation may be wrong.\n";
    return 0;
}

// ── handleMessage ─────────────────────────────────────────────────────────────

void GymConnection::handleMessage(omnetpp::cMessage *msg)
{
    if (msg != tick) return;

    updateEnergy();

    // ── Traffic metrics ───────────────────────────────────────────────────────
    const double rxBytesDelta = rxBytesTotal - lastRxBytesTotal;
    const double txBytesDelta = txBytesTotal - lastTxBytesTotal;
    const bool   hasCbr       = (rxBytesDelta > 0.0) ||
                                 (txBytesDelta > 0.0) ||
                                 (delayCount   > 0);

    double thr_cbr = 0.0, delay_cbr = 0.0, jitter_cbr = 0.0, loss_cbr = 0.0;

    if (hasCbr) {
        double rxDelta = std::max(0.0, rxBytesDelta);
        double txDelta = std::max(0.0, txBytesDelta);
        double dt      = par("tickInterval").doubleValue();

        thr_cbr = (dt > 0.0) ? (rxDelta * 8.0 / dt) : 0.0;

        if (delayCount > 0) {
            delay_cbr      = delaySum / delayCount;
            double meanSq  = delaySqSum / delayCount;
            jitter_cbr     = std::sqrt(std::max(0.0, meanSq - delay_cbr * delay_cbr));
        }
        if (txDelta > 0.0)
            loss_cbr = std::max(0.0, std::min(1.0, 1.0 - rxDelta / txDelta));

        lastRxBytesTotal = rxBytesTotal;
        lastTxBytesTotal = txBytesTotal;
        delaySum = delaySqSum = 0.0;
        delayCount = 0;
    }

    double thr_voip   = sumMap(rxThrByComp) * 8.0; // B/s → bps
    double delay_voip = avgMap(delayByComp);
    double jitter_voip = avgMap(jitterByComp);
    double loss_voip  = avgMap(lossByComp);

    // Combine CBR + VoIP metrics.
    double thr    = thr_cbr + thr_voip;
    double delay  = (delay_cbr  > 0.0 && delay_voip  > 0.0)
                    ? (delay_cbr  + delay_voip)  / 2.0
                    : std::max(delay_cbr,  delay_voip);
    double jitter = (jitter_cbr > 0.0 && jitter_voip > 0.0)
                    ? (jitter_cbr + jitter_voip) / 2.0
                    : std::max(jitter_cbr, jitter_voip);
    double loss   = (loss_cbr   > 0.0 && loss_voip   > 0.0)
                    ? (loss_cbr   + loss_voip)   / 2.0
                    : std::max(loss_cbr,   loss_voip);

    // FIX #3: snapshot sinrByComp BEFORE clearing it, then check the
    // snapshot for the warning — the original code cleared the map and
    // then checked the (now always-empty) map, so the warning fired
    // every single tick after the first.
    const bool sinrEmpty = sinrByComp.empty();
    const double sinr    = avgMap(sinrByComp);

    rxThrByComp.clear();
    delayByComp.clear();
    jitterByComp.clear();
    lossByComp.clear();
    sinrByComp.clear();

    if (!warnedNoSinr && sinrEmpty && omnetpp::simTime() > 0.5) {
        EV_WARN << "No SINR samples received. Check channel model paths "
                   "and collectSinrStatistics.\n";
        warnedNoSinr = true;
    }

    // ── Energy ────────────────────────────────────────────────────────────────
    const double numUe = static_cast<double>(getNumUE());
    const double E     = stepEnergyJ;

    double dt       = par("tickInterval").doubleValue();
    double stepBits = std::max(0.0, thr) * dt;
    deliveredBits  += stepBits;

    // ── Reward ────────────────────────────────────────────────────────────────
    //
    // Goal: minimise energy while maintaining QoS.
    //
    // Thresholds follow ITU-T G.131:
    //   Delay  : <150 ms ideal, <400 ms acceptable
    //   Loss   : <1 % excellent, <3 % acceptable
    //   Jitter : <5 ms excellent

    // --- Delay ---
    double delay_reward;
    if      (delay < 0.05)  delay_reward =  1.0;
    else if (delay < 0.15)  delay_reward =  0.5;
    else if (delay < 0.30)  delay_reward =  0.0;
    else if (delay < 0.50)  delay_reward = -1.0 * (delay - 0.30);
    else                    delay_reward = -1.0 - 2.0 * (delay - 0.50);

    // --- Loss ---
    double loss_reward;
    if      (loss < 0.01)   loss_reward =  0.5;
    else if (loss < 0.03)   loss_reward =  0.25;
    else if (loss < 0.10)   loss_reward =  0.0;
    else if (loss < 0.30)   loss_reward = -0.5 * loss;
    else                    loss_reward = -2.0;

    // --- Jitter ---
    double jitter_reward;
    if      (jitter < 0.005) jitter_reward =  0.25;
    else if (jitter < 0.020) jitter_reward =  0.10;
    else if (jitter < 0.050) jitter_reward =  0.0;
    else                     jitter_reward = -0.5 * (jitter - 0.050);

    // --- Energy ---
    double energy_penalty = E / 10.0;

    // --- Combine ---
    double qos_score = delay_reward + loss_reward + jitter_reward;
    double reward;
    if (qos_score < -1.0)
        reward = qos_score - 1.0;       // QoS catastrophic — ignore energy
    else
        reward = qos_score - energy_penalty;

    // FIX #7: zero-throughput penalty is gated on being INSIDE a traffic
    // phase.  The VarTraffic scenarios have deliberate gaps between phases
    // (e.g. phase 1 ends at 20 s, phase 2 starts at 20 s but may not have
    // delivered any packets yet in this tick).  We use a 1-second grace
    // window around the known phase boundaries instead of a single
    // global threshold.
    auto inTrafficGap = [](double t) -> bool {
        // Gap around phase transitions: [19.5, 20.5] and [39.5, 40.5]
        return (t > 19.5 && t < 20.5) || (t > 39.5 && t < 40.5);
    };
    double simT = omnetpp::simTime().dbl();
    if (simT > 0.5 && !inTrafficGap(simT) && thr < 1e-6)
        reward -= 1.0;

    // ── Build and send protobuf step request ──────────────────────────────────
    veinsgym::proto::Request req;
    req.set_id(stepId++);

    auto *step = req.mutable_step();
    auto *box  = step->mutable_observation()->mutable_box();
    box->add_values(safe(thr));
    box->add_values(safe(delay));
    box->add_values(safe(jitter));
    box->add_values(safe(loss));
    box->add_values(safe(numUe));
    box->add_values(safe(E));
    box->add_values(safe(currentTxPowerDbm));
    box->add_values(safe(sinr));

    step->mutable_reward()->mutable_box()->add_values(reward);

    auto reply = communicate(req);

    // ── Apply action ──────────────────────────────────────────────────────────
    if (reply.payload_case() == veinsgym::proto::Reply::kAction) {
        const auto &actSpace = reply.action();

        if (actSpace.value_case() == veinsgym::proto::Space::kBox) {
            double m = actSpace.box().values_size() > 0
                       ? actSpace.box().values(0) : 0.0;
            EV_INFO << "Gym Box action: " << m << "\n";
            applyMultiplier(m);
        }
        else if (actSpace.value_case() == veinsgym::proto::Space::kDiscrete) {
            int a = static_cast<int>(actSpace.discrete().value());
            EV_INFO << "Gym Discrete action: " << a << "\n";
            applyAction(a);
            double m = (a == 0) ? 2.0 : (a == 1) ? 1.0 : 0.0;
            applyMultiplier(m);
        }
    }

    scheduleAt(omnetpp::simTime() + par("tickInterval"), tick);
}

// ── applyMultiplier ───────────────────────────────────────────────────────────
//
// FIX #8: the gNB and UEs are independent power domains.
// The gNB uses the full [txPowerMin, txPowerMax] range.
// Each UE is limited to its own par("ueTxPower") maximum so we do not
// accidentally overshoot the UE power budget when scaling together.

// ── applyMultiplier ───────────────────────────────────────────────────────────
//
// Only the gNB (base station) TX power is adjusted by the RL agent.
// UE TX power is set statically by omnetpp.ini (**.ueTxPower = 26dBm) and
// is never touched here — controlling it is outside the project scope.

void GymConnection::applyMultiplier(double m)
{
    mAction = std::max(0.0, std::min(2.0, m));
    double alpha = mAction / 2.0;

    double txMin = par("txPowerMin").doubleValue();
    double txMax = par("txPowerMax").doubleValue();
    currentTxPowerDbm = txMin + alpha * (txMax - txMin);

    if (gnbPhy)
        gnbPhy->setTxPowerDbm(currentTxPowerDbm);
    // UE PHYs are intentionally not modified.
}

// ── applyAction ───────────────────────────────────────────────────────────────
//
// FIX #1: the new multi-phase / multi-UE scenarios have up to numUe*3 server
// apps.  Control ALL of them, not just app[0].

void GymConnection::applyAction(int a)
{
    auto *top    = getSystemModule();
    auto *server = top->getSubmodule("server");
    if (!server) {
        EV_WARN << "No server module found for traffic control.\n";
        return;
    }

    for (int j = 0; ; ++j) {
        auto *app = server->getSubmodule("app", j);
        if (!app) break;

        if (!app->hasPar("gymPaused") || !app->hasPar("samplingTime")) {
            EV_WARN << "server.app[" << j << "] missing gymPaused / samplingTime parameter; skipping.\n";
            continue;
        }

        if (a == 0) {                           // ACTIVE – full rate
            app->par("gymPaused").setBoolValue(false);
            app->par("samplingTime").setDoubleValue(0.02);
        } else if (a == 1) {                    // REDUCED – half rate
            app->par("gymPaused").setBoolValue(false);
            app->par("samplingTime").setDoubleValue(0.04);
        } else {                                // PAUSED
            app->par("gymPaused").setBoolValue(true);
        }
    }
}

// ── communicate ───────────────────────────────────────────────────────────────

veinsgym::proto::Reply GymConnection::communicate(const veinsgym::proto::Request &request)
{
    std::string msg = request.SerializeAsString();
    socket.send(zmq::message_t(msg.data(), msg.size()), zmq::send_flags::none);

    zmq::message_t response_msg;
    auto result = socket.recv(response_msg, zmq::recv_flags::none);
    if (!result) {
        EV_WARN << "ZMQ recv failed.\n";
        return veinsgym::proto::Reply();
    }

    veinsgym::proto::Reply reply;
    reply.ParseFromString(std::string(
        static_cast<char *>(response_msg.data()), response_msg.size()));
    return reply;
}

// ── Destructor / finish ───────────────────────────────────────────────────────

GymConnection::~GymConnection()
{
    if (tick) {
        cancelAndDelete(tick);
        tick = nullptr;
    }
    if (!shutdownSent) {
        veinsgym::proto::Request req;
        *(req.mutable_shutdown()) = {};
        communicate(req);
        shutdownSent = true;
    }
}

void GymConnection::finish()
{
    if (!shutdownSent) {
        veinsgym::proto::Request req;
        *(req.mutable_shutdown()) = {};
        communicate(req);
        shutdownSent = true;
    }
}