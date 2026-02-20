
import inet.power.base.EpEnergyConsumerBase;

simple GymEnergyConsumer extends EpEnergyConsumerBase
{
    parameters:
        string energySourceModule = default("^.energyStorage");
        @class(GymEnergyConsumer);
}
