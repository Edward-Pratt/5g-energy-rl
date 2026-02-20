#pragma once

#include <omnetpp.h>
#include "inet/common/InitStages.h"
#include "inet/power/contract/IEpEnergyConsumer.h"
#include "inet/power/contract/IEpEnergySource.h"

class GymEnergyConsumer : public omnetpp::cSimpleModule, public inet::power::IEpEnergyConsumer {
public:
    void setPowerConsumptionW(double w);

    inet::power::IEnergySource *getEnergySource() const override { return energySource; }
    inet::power::W getPowerConsumption() const override { return inet::power::W(powerConsumptionW); }

protected:
    int numInitStages() const override { return inet::NUM_INIT_STAGES; }
    void initialize(int stage) override;
    void handleMessage(omnetpp::cMessage *msg) override;

private:
    inet::power::IEpEnergySource *energySource = nullptr;
    double powerConsumptionW = 0.0;
};
