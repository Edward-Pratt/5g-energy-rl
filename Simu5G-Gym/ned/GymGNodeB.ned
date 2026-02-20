
import simu5g.nodes.gNodeB;
import inet.power.storage.IdealEpEnergyStorage;
import GymEnergyConsumer;

module GymGNodeB extends gNodeB
{
    submodules:
        energyStorage: IdealEpEnergyStorage {
            @display("p=90,520;is=s");
        }
        energyConsumer: GymEnergyConsumer {
            @display("p=220,520;is=s");
        }
}
