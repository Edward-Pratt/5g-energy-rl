
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

simple GymConnection {
	@class(GymConnection);
	string host = default("127.0.0.1"); // tcp port of the gym server
	int port = default(5555); // tcp port of the gym server
	string observation_space; // python code to set up the observation space in a gym
	string action_space; // python code to set up the observation space in a gym

	double tickInterval @unit(s) = default(0.1s);
	double pActive @unit(W) = default(100W);
	double pSleep @unit(W) = default(10W);
	double pComputeBase @unit(W) = default(20W);
	double pComputePerUe @unit(W) = default(2W);
	double pTxCoeff = default(0.001); // W per mW of tx power (scaling factor)
	double numResourceBlocks = default(50);  // 50 RBs = 10 MHz

	double txPowerMin @unit(dBm) = default(30dBm);
	double txPowerMax @unit(dBm) = default(46dBm);

	string gnbPhyPath = default("gnb.cellularNic.phy");
	string energyConsumerPath = default("gnb.energyConsumer");
	string sinrDlPath = default(""); // optional override; leave empty for auto-detect
	string sinrUlPath = default(""); // optional override; leave empty for auto-detect
}
