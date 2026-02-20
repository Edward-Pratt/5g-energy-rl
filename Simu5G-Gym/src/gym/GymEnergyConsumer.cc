//
// Simple energy consumer that accepts an externally-set power value.
//

#include "GymEnergyConsumer.h"

Define_Module(GymEnergyConsumer);

void GymEnergyConsumer::initialize(int stage)
{
    if (stage == inet::INITSTAGE_LOCAL) {
        const char *energySourceModule = par("energySourceModule");
        energySource = dynamic_cast<inet::power::IEpEnergySource *>(getModuleByPath(energySourceModule));
        if (!energySource)
            throw omnetpp::cRuntimeError("Energy source module '%s' not found", energySourceModule);
    }
    else if (stage == inet::INITSTAGE_POWER) {
        energySource->addEnergyConsumer(this);
        emit(inet::power::IEpEnergySource::powerConsumptionChangedSignal, powerConsumptionW);
    }
}

void GymEnergyConsumer::handleMessage(omnetpp::cMessage *msg)
{
    delete msg;
}

void GymEnergyConsumer::setPowerConsumptionW(double w)
{
    if (w < 0.0) w = 0.0;
    if (w == powerConsumptionW) return;
    powerConsumptionW = w;
    emit(inet::power::IEpEnergySource::powerConsumptionChangedSignal, powerConsumptionW);
}
