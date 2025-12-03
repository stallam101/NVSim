/*******************************************************************************
* Copyright (c) 2012-2013, The Microsystems Design Labratory (MDL)
* Department of Computer Science and Engineering, The Pennsylvania State University
* Exascale Computing Lab, Hewlett-Packard Company
* All rights reserved.
* 
* This source code is part of NVSim - An area, timing and power model for both 
* volatile (e.g., SRAM, DRAM) and non-volatile memory (e.g., PCRAM, STT-RAM, ReRAM, 
* SLC NAND Flash). The source code is free and you can redistribute and/or modify it
* by providing that the following conditions are met:
* 
*  1) Redistributions of source code must retain the above copyright notice,
*     this list of conditions and the following disclaimer.
* 
*  2) Redistributions in binary form must reproduce the above copyright notice,
*     this list of conditions and the following disclaimer in the documentation
*     and/or other materials provided with the distribution.
* 
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
* 
* Author list: 
*   Cong Xu	    ( Email: czx102 at psu dot edu 
*                     Website: http://www.cse.psu.edu/~czx102/ )
*   Xiangyu Dong    ( Email: xydong at cse dot psu dot edu
*                     Website: http://www.cse.psu.edu/~xydong/ )
*******************************************************************************/


#include "MemCell.h"
#include "formula.h"
#include "global.h"
#include "macros.h"
#include <math.h>
#include <random>
#include <vector>

/* Static random number generation infrastructure for stochastic sampling */
static std::random_device rd;
static std::mt19937 gen(rd());
static bool randomSeedSet = false;

/* Function to set seed for reproducible results */
static void SetRandomSeed(unsigned int seed) {
	gen.seed(seed);
	randomSeedSet = true;
}

/* Function to sample from truncated normal distribution */
static int SampleTruncatedNormal(double mean, double stddev, int minVal, int maxVal) {
	std::normal_distribution<double> distribution(mean, stddev);
	int sample;
	int attempts = 0;
	const int maxAttempts = 100; /* Prevent infinite loops */

	sample = (int)round(distribution(gen));
	
	/*do {
		sample = (int)round(distribution(gen));
		attempts++;
		if (attempts >= maxAttempts) {
			Fallback to bounds if distribution is too extreme
			sample = (sample < minVal) ? minVal : sample;
			sample = (sample > maxVal) ? maxVal : sample;
			break;
		}
	} while (sample < minVal || sample > maxVal); */
	
	return sample;
}

MemCell::MemCell() {
	// TODO Auto-generated constructor stub
	memCellType         = PCRAM;
	area                = 0;
	aspectRatio         = 0;
	resistanceOn        = 0;
	resistanceOff       = 0;
	readMode            = true;
	readVoltage         = 0;
	readCurrent         = 0;
	readPower           = 0;
        wordlineBoostRatio  = 1.0;
	resetMode           = true;
	resetVoltage        = 0;
	resetCurrent        = 0;
	minSenseVoltage     = 0.08;
	resetPulse          = 0;
	resetEnergy         = 0;
	setMode             = true;
	setVoltage          = 0;
	setCurrent          = 0;
	setPulse            = 0;
	accessType          = CMOS_access;
	processNode         = 0;
	setEnergy           = 0;

	/* Stochastic parameters - initialize to deterministic defaults */
	stochasticEnabled           = false;
	setPulseCountMean          = 1.0;
	setPulseCountStdDev        = 0.0;
	setPulseCountMin           = 1;
	setPulseCountMax           = 1;
	resetPulseCountMean        = 1.0;
	resetPulseCountStdDev      = 0.0;
	resetPulseCountMin         = 1;
	resetPulseCountMax         = 1;
	redundantPulseCountMean    = 1.0;
	redundantPulseCountStdDev  = 0.0;
	redundantPulseCountMin     = 1;
	redundantPulseCountMax     = 1;

	/* Optional */
	stitching         = 0;
	gateOxThicknessFactor = 2;
	widthSOIDevice = 0;
	widthAccessCMOS   = 0;
	voltageDropAccessDevice = 0;
	leakageCurrentAccessDevice = 0;
	capDRAMCell		  = 0;
	widthSRAMCellNMOS = 2.08;	/* Default NMOS width in SRAM cells is 2.08 (from CACTI) */
	widthSRAMCellPMOS = 1.23;	/* Default PMOS width in SRAM cells is 1.23 (from CACTI) */

	/*For memristors */
	readFloating = false;
	resistanceOnAtSetVoltage = 0;
	resistanceOffAtSetVoltage = 0;
	resistanceOnAtResetVoltage = 0;
	resistanceOffAtResetVoltage = 0;
	resistanceOnAtReadVoltage = 0;
	resistanceOffAtReadVoltage = 0;
	resistanceOnAtHalfReadVoltage = 0;
	resistanceOffAtHalfReadVoltage = 0;
}

MemCell::~MemCell() {
	// TODO Auto-generated destructor stub
}

void MemCell::ReadCellFromFile(const string & inputFile)
{
	FILE *fp = fopen(inputFile.c_str(), "r");
	char line[5000];
	char tmp[5000];

	if (!fp) {
		cout << inputFile << " cannot be found!\n";
		exit(-1);
	}

	while (fscanf(fp, "%[^\n]\n", line) != EOF) {
		if (!strncmp("-MemCellType", line, strlen("-MemCellType"))) {
			sscanf(line, "-MemCellType: %s", tmp);
			if (!strcmp(tmp, "SRAM"))
				memCellType = SRAM;
			else if (!strcmp(tmp, "DRAM"))
				memCellType = DRAM;
			else if (!strcmp(tmp, "eDRAM"))
				memCellType = eDRAM;
			else if (!strcmp(tmp, "MRAM"))
				memCellType = MRAM;
			else if (!strcmp(tmp, "PCRAM"))
				memCellType = PCRAM;
			else if (!strcmp(tmp, "FBRAM"))
				memCellType = FBRAM;
			else if (!strcmp(tmp, "memristor"))
				memCellType = memristor;
			else if (!strcmp(tmp, "SLCNAND"))
				memCellType = SLCNAND;
			else
				memCellType = MLCNAND;
			continue;
		}
		if (!strncmp("-ProcessNode", line, strlen("-ProcessNode"))) {
			sscanf(line, "-ProcessNode: %d", &processNode);
			continue;
		}
		if (!strncmp("-CellArea", line, strlen("-CellArea"))) {
			sscanf(line, "-CellArea (F^2): %lf", &area);
			continue;
		}
		if (!strncmp("-CellAspectRatio", line, strlen("-CellAspectRatio"))) {
			sscanf(line, "-CellAspectRatio: %lf", &aspectRatio);
			heightInFeatureSize = sqrt(area * aspectRatio);
			widthInFeatureSize = sqrt(area / aspectRatio);
			continue;
		}

		if (!strncmp("-ResistanceOnAtSetVoltage", line, strlen("-ResistanceOnAtSetVoltage"))) {
			sscanf(line, "-ResistanceOnAtSetVoltage (ohm): %lf", &resistanceOnAtSetVoltage);
			continue;
		}
		if (!strncmp("-ResistanceOffAtSetVoltage", line, strlen("-ResistanceOffAtSetVoltage"))) {
			sscanf(line, "-ResistanceOffAtSetVoltage (ohm): %lf", &resistanceOffAtSetVoltage);
			continue;
		}
		if (!strncmp("-ResistanceOnAtResetVoltage", line, strlen("-ResistanceOnAtResetVoltage"))) {
			sscanf(line, "-ResistanceOnAtResetVoltage (ohm): %lf", &resistanceOnAtResetVoltage);
			continue;
		}
		if (!strncmp("-ResistanceOffAtResetVoltage", line, strlen("-ResistanceOffAtResetVoltage"))) {
			sscanf(line, "-ResistanceOffAtResetVoltage (ohm): %lf", &resistanceOffAtResetVoltage);
			continue;
		}
		if (!strncmp("-ResistanceOnAtReadVoltage", line, strlen("-ResistanceOnAtReadVoltage"))) {
			sscanf(line, "-ResistanceOnAtReadVoltage (ohm): %lf", &resistanceOnAtReadVoltage);
			resistanceOn = resistanceOnAtReadVoltage;
			continue;
		}
		if (!strncmp("-ResistanceOffAtReadVoltage", line, strlen("-ResistanceOffAtReadVoltage"))) {
			sscanf(line, "-ResistanceOffAtReadVoltage (ohm): %lf", &resistanceOffAtReadVoltage);
			resistanceOff = resistanceOffAtReadVoltage;
			continue;
		}
		if (!strncmp("-ResistanceOnAtHalfReadVoltage", line, strlen("-ResistanceOnAtHalfReadVoltage"))) {
			sscanf(line, "-ResistanceOnAtHalfReadVoltage (ohm): %lf", &resistanceOnAtHalfReadVoltage);
			continue;
		}
		if (!strncmp("-ResistanceOffAtHalfReadVoltage", line, strlen("-ResistanceOffAtHalfReadVoltage"))) {
			sscanf(line, "-ResistanceOffAtHalfReadVoltage (ohm): %lf", &resistanceOffAtHalfReadVoltage);
			continue;
		}
		if (!strncmp("-ResistanceOnAtHalfResetVoltage", line, strlen("-ResistanceOnAtHalfResetVoltage"))) {
			sscanf(line, "-ResistanceOnAtHalfResetVoltage (ohm): %lf", &resistanceOnAtHalfResetVoltage);
			continue;
		}

		if (!strncmp("-ResistanceOn", line, strlen("-ResistanceOn"))) {
			sscanf(line, "-ResistanceOn (ohm): %lf", &resistanceOn);
			continue;
		}
		if (!strncmp("-ResistanceOff", line, strlen("-ResistanceOff"))) {
			sscanf(line, "-ResistanceOff (ohm): %lf", &resistanceOff);
			continue;
		}
		if (!strncmp("-CapacitanceOn", line, strlen("-CapacitanceOn"))) {
			sscanf(line, "-CapacitanceOn (F): %lf", &capacitanceOn);
			continue;
		}
		if (!strncmp("-CapacitanceOff", line, strlen("-CapacitanceOff"))) {
			sscanf(line, "-CapacitanceOff (F): %lf", &capacitanceOff);
			continue;
		}

		if (!strncmp("-GateOxThicknessFactor", line, strlen("-GateOxThicknessFactor"))) {
			sscanf(line, "-GateOxThicknessFactor: %lf", &gateOxThicknessFactor);
			continue;
		}

		if (!strncmp("-SOIDeviceWidth (F)", line, strlen("-SOIDeviceWidth (F)"))) {
			sscanf(line, "-SOIDeviceWidth (F): %lf", &widthSOIDevice);
			continue;
		}

		if (!strncmp("-ReadMode", line, strlen("-ReadMode"))) {
			sscanf(line, "-ReadMode: %s", tmp);
			if (!strcmp(tmp, "voltage"))
				readMode = true;
			else
				readMode = false;
			continue;
		}
		if (!strncmp("-ReadVoltage", line, strlen("-ReadVoltage"))) {
			sscanf(line, "-ReadVoltage (V): %lf", &readVoltage);
			continue;
		}
		if (!strncmp("-ReadCurrent", line, strlen("-ReadCurrent"))) {
			sscanf(line, "-ReadCurrent (uA): %lf", &readCurrent);
			readCurrent /= 1e6;
			continue;
		}
		if (!strncmp("-ReadPower", line, strlen("-ReadPower"))) {
			sscanf(line, "-ReadPower (uW): %lf", &readPower);
			readPower /= 1e6;
			continue;
		}
		if (!strncmp("-WordlineBoostRatio", line, strlen("-WordlineBoostRatio"))) {
			sscanf(line, "-WordlineBoostRatio: %lf", &wordlineBoostRatio);
			continue;
		}
		if (!strncmp("-MinSenseVoltage", line, strlen("-MinSenseVoltage"))) {
			sscanf(line, "-MinSenseVoltage (mV): %lf", &minSenseVoltage);
			minSenseVoltage /= 1e3;
			continue;
		}


		if (!strncmp("-ResetMode", line, strlen("-ResetMode"))) {
			sscanf(line, "-ResetMode: %s", tmp);
			if (!strcmp(tmp, "voltage"))
				resetMode = true;
			else
				resetMode = false;
			continue;
		}
		if (!strncmp("-ResetVoltage", line, strlen("-ResetVoltage"))) {
			sscanf(line, "-ResetVoltage (V): %lf", &resetVoltage);
			continue;
		}
		if (!strncmp("-ResetCurrent", line, strlen("-ResetCurrent"))) {
			sscanf(line, "-ResetCurrent (uA): %lf", &resetCurrent);
			resetCurrent /= 1e6;
			continue;
		}
		if (!strncmp("-ResetVoltage", line, strlen("-ResetVoltage"))) {
			sscanf(line, "-ResetVoltage (V): %lf", &resetVoltage);
			continue;
		}
		if (!strncmp("-ResetPulse (ns):", line, strlen("-ResetPulse (ns):"))) {
			sscanf(line, "-ResetPulse (ns): %lf", &resetPulse);
			resetPulse /= 1e9;
			continue;
		}
		if (!strncmp("-ResetEnergy", line, strlen("-ResetEnergy"))) {
			sscanf(line, "-ResetEnergy (pJ): %lf", &resetEnergy);
			resetEnergy /= 1e12;
			continue;
		}

		if (!strncmp("-SetMode", line, strlen("-SetMode"))) {
			sscanf(line, "-SetMode: %s", tmp);
			if (!strcmp(tmp, "voltage"))
				setMode = true;
			else
				setMode = false;
			continue;
		}
		if (!strncmp("-SetVoltage", line, strlen("-SetVoltage"))) {
			sscanf(line, "-SetVoltage (V): %lf", &setVoltage);
			continue;
		}
		if (!strncmp("-SetCurrent", line, strlen("-SetCurrent"))) {
			sscanf(line, "-SetCurrent (uA): %lf", &setCurrent);
			setCurrent /= 1e6;
			continue;
		}
		if (!strncmp("-SetVoltage", line, strlen("-SetVoltage"))) {
			sscanf(line, "-SetVoltage (V): %lf", &setVoltage);
			continue;
		}
		if (!strncmp("-SetPulse (ns):", line, strlen("-SetPulse (ns):"))) {
			sscanf(line, "-SetPulse (ns): %lf", &setPulse);
			setPulse /= 1e9;
			continue;
		}
		if (!strncmp("-SetEnergy", line, strlen("-SetEnergy"))) {
			sscanf(line, "-SetEnergy (pJ): %lf", &setEnergy);
			setEnergy /= 1e12;
			continue;
		}

		if (!strncmp("-AccessType", line, strlen("-AccessType"))) {
			sscanf(line, "-AccessType: %s", tmp);
			if (!strcmp(tmp, "CMOS"))
				accessType = CMOS_access;
			else if (!strcmp(tmp, "BJT"))
				accessType = BJT_access;
			else if (!strcmp(tmp, "diode"))
				accessType = diode_access;
			else
				accessType = none_access;
			continue;
		}

		if (!strncmp("-AccessCMOSWidth", line, strlen("-AccessCMOSWidth"))) {
			if (accessType != CMOS_access)
				cout << "Warning: The input of CMOS access transistor width is ignored because the cell is not CMOS-accessed." << endl;
			else
				sscanf(line, "-AccessCMOSWidth (F): %lf", &widthAccessCMOS);
			continue;
		}

		if (!strncmp("-VoltageDropAccessDevice", line, strlen("-VoltageDropAccessDevice"))) {
			sscanf(line, "-VoltageDropAccessDevice (V): %lf", &voltageDropAccessDevice);
			continue;
		}

		if (!strncmp("-LeakageCurrentAccessDevice", line, strlen("-LeakageCurrentAccessDevice"))) {
			sscanf(line, "-LeakageCurrentAccessDevice (uA): %lf", &leakageCurrentAccessDevice);
			leakageCurrentAccessDevice /= 1e6;
			continue;
		}

		if (!strncmp("-DRAMCellCapacitance", line, strlen("-DRAMCellCapacitance"))) {
			if (memCellType != DRAM && memCellType != eDRAM)
				cout << "Warning: The input of DRAM cell capacitance is ignored because the memory cell is not DRAM." << endl;
			else
				sscanf(line, "-DRAMCellCapacitance (F): %lf", &capDRAMCell);
			continue;
		}

		if (!strncmp("-SRAMCellNMOSWidth", line, strlen("-SRAMCellNMOSWidth"))) {
			if (memCellType != SRAM)
				cout << "Warning: The input of SRAM cell NMOS width is ignored because the memory cell is not SRAM." << endl;
			else
				sscanf(line, "-SRAMCellNMOSWidth (F): %lf", &widthSRAMCellNMOS);
			continue;
		}

		if (!strncmp("-SRAMCellPMOSWidth", line, strlen("-SRAMCellPMOSWidth"))) {
			if (memCellType != SRAM)
				cout << "Warning: The input of SRAM cell PMOS width is ignored because the memory cell is not SRAM." << endl;
			else
				sscanf(line, "-SRAMCellPMOSWidth (F): %lf", &widthSRAMCellPMOS);
			continue;
		}


		if (!strncmp("-ReadFloating", line, strlen("-ReadFloating"))) {
			sscanf(line, "-ReadFloating: %s", tmp);
			if (!strcmp(tmp, "true"))
				readFloating = true;
			else
				readFloating = false;
			continue;
		}

		if (!strncmp("-FlashEraseVoltage (V)", line, strlen("-FlashEraseVoltage (V)"))) {
			if (memCellType != SLCNAND && memCellType != MLCNAND)
				cout << "Warning: The input of programming/erase voltage is ignored because the memory cell is not flash." << endl;
			else
				sscanf(line, "-FlashEraseVoltage (V): %lf", &flashEraseVoltage);
			continue;
		}

		if (!strncmp("-FlashProgramVoltage (V)", line, strlen("-FlashProgramVoltage (V)"))) {
			if (memCellType != SLCNAND && memCellType != MLCNAND)
				cout << "Warning: The input of programming/program voltage is ignored because the memory cell is not flash." << endl;
			else
				sscanf(line, "-FlashProgramVoltage (V): %lf", &flashProgramVoltage);
			continue;
		}

		if (!strncmp("-FlashPassVoltage (V)", line, strlen("-FlashPassVoltage (V)"))) {
			if (memCellType != SLCNAND && memCellType != MLCNAND)
				cout << "Warning: The input of pass voltage is ignored because the memory cell is not flash." << endl;
			else
				sscanf(line, "-FlashPassVoltage (V): %lf", &flashPassVoltage);
			continue;
		}

		if (!strncmp("-FlashEraseTime", line, strlen("-FlashEraseTime"))) {
			if (memCellType != SLCNAND && memCellType != MLCNAND)
				cout << "Warning: The input of erase time is ignored because the memory cell is not flash." << endl;
			else {
				sscanf(line, "-FlashEraseTime (ms): %lf", &flashEraseTime);
				flashEraseTime /= 1e3;
			}
			continue;
		}

		if (!strncmp("-FlashProgramTime", line, strlen("-FlashProgramTime"))) {
			if (memCellType != SLCNAND && memCellType != MLCNAND)
				cout << "Warning: The input of erase time is ignored because the memory cell is not flash." << endl;
			else {
				sscanf(line, "-FlashProgramTime (us): %lf", &flashProgramTime);
				flashProgramTime /= 1e6;
			}
			continue;
		}

		if (!strncmp("-GateCouplingRatio", line, strlen("-GateCouplingRatio"))) {
			if (memCellType != SLCNAND && memCellType != MLCNAND)
				cout << "Warning: The input of gate coupling ratio (GCR) is ignored because the memory cell is not flash." << endl;
			else {
				sscanf(line, "-GateCouplingRatio: %lf", &gateCouplingRatio);
			}
			continue;
		}

		/* Stochastic modeling parameters */
		if (!strncmp("-StochasticEnabled", line, strlen("-StochasticEnabled"))) {
			char tmpStr[256];
			sscanf(line, "-StochasticEnabled: %s", tmpStr);
			if (!strcmp(tmpStr, "true") || !strcmp(tmpStr, "TRUE") || !strcmp(tmpStr, "1")) {
				stochasticEnabled = true;
			} else {
				stochasticEnabled = false;
			}
			continue;
		}

		/* SET transition (0→1) distribution parameters */
		if (!strncmp("-SetPulseCountMean", line, strlen("-SetPulseCountMean"))) {
			sscanf(line, "-SetPulseCountMean: %lf", &setPulseCountMean);
			continue;
		}
		if (!strncmp("-SetPulseCountStdDev", line, strlen("-SetPulseCountStdDev"))) {
			sscanf(line, "-SetPulseCountStdDev: %lf", &setPulseCountStdDev);
			continue;
		}
		if (!strncmp("-SetPulseCountMin", line, strlen("-SetPulseCountMin"))) {
			sscanf(line, "-SetPulseCountMin: %d", &setPulseCountMin);
			continue;
		}
		if (!strncmp("-SetPulseCountMax", line, strlen("-SetPulseCountMax"))) {
			sscanf(line, "-SetPulseCountMax: %d", &setPulseCountMax);
			continue;
		}

		/* RESET transition (1→0) distribution parameters */
		if (!strncmp("-ResetPulseCountMean", line, strlen("-ResetPulseCountMean"))) {
			sscanf(line, "-ResetPulseCountMean: %lf", &resetPulseCountMean);
			continue;
		}
		if (!strncmp("-ResetPulseCountStdDev", line, strlen("-ResetPulseCountStdDev"))) {
			sscanf(line, "-ResetPulseCountStdDev: %lf", &resetPulseCountStdDev);
			continue;
		}
		if (!strncmp("-ResetPulseCountMin", line, strlen("-ResetPulseCountMin"))) {
			sscanf(line, "-ResetPulseCountMin: %d", &resetPulseCountMin);
			continue;
		}
		if (!strncmp("-ResetPulseCountMax", line, strlen("-ResetPulseCountMax"))) {
			sscanf(line, "-ResetPulseCountMax: %d", &resetPulseCountMax);
			continue;
		}

		/* Redundant operation distribution parameters */
		if (!strncmp("-RedundantPulseCountMean", line, strlen("-RedundantPulseCountMean"))) {
			sscanf(line, "-RedundantPulseCountMean: %lf", &redundantPulseCountMean);
			continue;
		}
		if (!strncmp("-RedundantPulseCountStdDev", line, strlen("-RedundantPulseCountStdDev"))) {
			sscanf(line, "-RedundantPulseCountStdDev: %lf", &redundantPulseCountStdDev);
			continue;
		}
		if (!strncmp("-RedundantPulseCountMin", line, strlen("-RedundantPulseCountMin"))) {
			sscanf(line, "-RedundantPulseCountMin: %d", &redundantPulseCountMin);
			continue;
		}
		if (!strncmp("-RedundantPulseCountMax", line, strlen("-RedundantPulseCountMax"))) {
			sscanf(line, "-RedundantPulseCountMax: %d", &redundantPulseCountMax);
			continue;
		}
	}

	fclose(fp);
}


void MemCell::CellScaling(int _targetProcessNode) {
	if ((processNode > 0) && (processNode != _targetProcessNode)) {
		double scalingFactor = (double)processNode / _targetProcessNode;
		if (memCellType == PCRAM) {
			resistanceOn *= scalingFactor;
			resistanceOff *= scalingFactor;
			if (!setMode) {
				setCurrent /= scalingFactor;
			} else {
				setVoltage *= 1;
			}
			if (!resetMode) {
				resetCurrent /= scalingFactor;
			} else {
				resetVoltage *= 1;
			}
			if (accessType == diode_access) {
				capacitanceOn /= scalingFactor; //TO-DO
				capacitanceOff /= scalingFactor; //TO-DO
			}
		} else if (memCellType == MRAM){ //TO-DO: MRAM
			resistanceOn *= scalingFactor * scalingFactor;
			resistanceOff *= scalingFactor * scalingFactor;
			if (!setMode) {
				setCurrent /= scalingFactor;
			} else {
				setVoltage *= scalingFactor;
			}
			if (!resetMode) {
				resetCurrent /= scalingFactor;
			} else {
				resetVoltage *= scalingFactor;
			}
			if (accessType == diode_access) {
				capacitanceOn /= scalingFactor; //TO-DO
				capacitanceOff /= scalingFactor; //TO-DO
			}
		} else if (memCellType == memristor) { //TO-DO: memristor

		} else { //TO-DO: other RAMs

		}
		processNode = _targetProcessNode;
	}
}

double MemCell::GetMemristance(double _relativeReadVoltage) { /* Get the LRS resistance of memristor at log-linera region of I-V curve */
	if (memCellType == memristor) {
		double x1, x2, x3;  // x1: read voltage, x2: half voltage, x3: applied voltage
		if (readVoltage == 0) {
			x1 = readCurrent * resistanceOnAtReadVoltage;
		} else {
			x1 = readVoltage;
		}
		x2 = readVoltage / 2;
		x3 = _relativeReadVoltage * readVoltage;
		double y1, y2 ,y3; // y1:log(read current), y2: log(leakage current at half read voltage
		y1 = log2(x1/resistanceOnAtReadVoltage);
		y2 = log2(x2/resistanceOnAtHalfReadVoltage);
		y3 = (y2 - y1) / (x2 -x1) * x3 + (x2 * y1 - x1 * y2) / (x2 - x1);  //insertion
		return x3 / pow(2, y3);
	} else {  // not memristor, can't call the function
		cout <<"Warning[MemCell] : Try to get memristance from a non-memristor memory cell" << endl;
		return -1;
	}
}

void MemCell::CalculateWriteEnergy() {
	if (resetEnergy == 0) {
		if (resetMode) {
			if (memCellType == memristor)
				if (accessType == none_access)
					resetEnergy = fabs(resetVoltage) * (fabs(resetVoltage) - voltageDropAccessDevice) / resistanceOnAtResetVoltage * resetPulse;
				else
					resetEnergy = fabs(resetVoltage) * (fabs(resetVoltage) - voltageDropAccessDevice) / resistanceOn * resetPulse;
			else if (memCellType == PCRAM)
				resetEnergy = fabs(resetVoltage) * (fabs(resetVoltage) - voltageDropAccessDevice) / resistanceOn * resetPulse;	// PCM cells shows low resistance during most time of the switching
			else if (memCellType == FBRAM)
				resetEnergy = fabs(resetVoltage) * fabs(resetCurrent) * resetPulse;
			else
				resetEnergy = fabs(resetVoltage) * (fabs(resetVoltage) - voltageDropAccessDevice) / resistanceOn * resetPulse;
		} else {
			if (resetVoltage == 0){
				resetEnergy = tech->vdd * fabs(resetCurrent) * resetPulse; /*TO-DO consider charge pump*/
			} else {
				resetEnergy = fabs(resetVoltage) * fabs(resetCurrent) * resetPulse;
			}
			/* previous model seems to be problematic
			if (memCellType == memristor)
				if (accessType == none_access)
					resetEnergy = resetCurrent * (resetCurrent * resistanceOffAtResetVoltage + voltageDropAccessDevice) * resetPulse;
				else
					resetEnergy = resetCurrent * (resetCurrent * resistanceOff + voltageDropAccessDevice) * resetPulse;
			else if (memCellType == PCRAM)
				resetEnergy = resetCurrent * (resetCurrent * resistanceOn + voltageDropAccessDevice) * resetPulse;		// PCM cells shows low resistance during most time of the switching
			else if (memCellType == FBRAM)
				resetEnergy = fabs(resetVoltage) * fabs(resetCurrent) * resetPulse;
			else
				resetEnergy = resetCurrent * (resetCurrent * resistanceOff + voltageDropAccessDevice) * resetPulse;
		    */
		}
	}
	if (setEnergy == 0) {
		if (setMode) {
			if (memCellType == memristor)
				if (accessType == none_access)
					setEnergy = fabs(setVoltage) * (fabs(setVoltage) - voltageDropAccessDevice) / resistanceOnAtSetVoltage * setPulse;
				else
					setEnergy = fabs(setVoltage) * (fabs(setVoltage) - voltageDropAccessDevice) / resistanceOn * setPulse;
			else if (memCellType == PCRAM)
				setEnergy = fabs(setVoltage) * (fabs(setVoltage) - voltageDropAccessDevice) / resistanceOn * setPulse;			// PCM cells shows low resistance during most time of the switching
			else if (memCellType == FBRAM)
				setEnergy = fabs(setVoltage) * fabs(setCurrent) * setPulse;
			else
				setEnergy = fabs(setVoltage) * (fabs(setVoltage) - voltageDropAccessDevice) / resistanceOn * setPulse;
		} else {
			if (resetVoltage == 0){
				setEnergy = tech->vdd * fabs(setCurrent) * setPulse; /*TO-DO consider charge pump*/
			} else {
				setEnergy = fabs(setVoltage) * fabs(setCurrent) * setPulse;
			}
			/* previous model seems to be problematic
			if (memCellType == memristor)
				if (accessType == none_access)
					setEnergy = setCurrent * (setCurrent * resistanceOffAtSetVoltage + voltageDropAccessDevice) * setPulse;
				else
					setEnergy = setCurrent * (setCurrent * resistanceOff + voltageDropAccessDevice) * setPulse;
			else if (memCellType == PCRAM)
				setEnergy = setCurrent * (setCurrent * resistanceOn + voltageDropAccessDevice) * setPulse;		// PCM cells shows low resistance during most time of the switching
			else if (memCellType == FBRAM)
				setEnergy = fabs(setVoltage) * fabs(setCurrent) * setPulse;
			else
				setEnergy = setCurrent * (setCurrent * resistanceOff + voltageDropAccessDevice) * setPulse;
			*/
		}
	}
}

double MemCell::CalculateReadPower() { /* TO-DO consider charge pumped read voltage */
	if (readPower == 0) {
		if (cell->readMode) {	/* voltage-sensing */
			if (readVoltage == 0) { /* Current-in voltage sensing */
				return tech->vdd * readCurrent;
			}
			if (readCurrent == 0) { /*Voltage-divider sensing */
				double resInSerialForSenseAmp, maxBitlineCurrent;
				resInSerialForSenseAmp = sqrt(resistanceOn * resistanceOff);
				maxBitlineCurrent = (readVoltage - voltageDropAccessDevice) / (resistanceOn + resInSerialForSenseAmp);
				return tech->vdd * maxBitlineCurrent;
			}
		} else { /* current-sensing */
			double maxBitlineCurrent = (readVoltage - voltageDropAccessDevice) / resistanceOn;
			return tech->vdd * maxBitlineCurrent;
		}
	} else {
		return -1.0; /* should not call the function if read energy exists */
	}
	return -1.0;
}

void MemCell::PrintCell()
{
	switch (memCellType) {
	case SRAM:
		cout << "Memory Cell: SRAM" << endl;
		break;
	case DRAM:
		cout << "Memory Cell: DRAM" << endl;
		break;
	case eDRAM:
		cout << "Memory Cell: Embedded DRAM" << endl;
		break;
	case MRAM:
		cout << "Memory Cell: MRAM (Magnetoresistive)" << endl;
		break;
	case PCRAM:
		cout << "Memory Cell: PCRAM (Phase-Change)" << endl;
		break;
	case memristor:
		cout << "Memory Cell: RRAM (Memristor)" << endl;
		break;
	case FBRAM:
		cout << "Memory Cell: FBRAM (Floating Body)" <<endl;
		break;
	case SLCNAND:
		cout << "Memory Cell: Single-Level Cell NAND Flash" << endl;
		break;
	case MLCNAND:
		cout << "Memory Cell: Multi-Level Cell NAND Flash" << endl;
		break;
	default:
		cout << "Memory Cell: Unknown" << endl;
	}
	cout << "Cell Area (F^2)    : " << area << " (" << heightInFeatureSize << "Fx" << widthInFeatureSize << "F)" << endl;
	cout << "Cell Aspect Ratio  : " << aspectRatio << endl;

	if (memCellType == PCRAM || memCellType == MRAM || memCellType == memristor || memCellType == FBRAM) {
		if (resistanceOn < 1e3 )
			cout << "Cell Turned-On Resistance : " << resistanceOn << "ohm" << endl;
		else if (resistanceOn < 1e6)
			cout << "Cell Turned-On Resistance : " << resistanceOn / 1e3 << "Kohm" << endl;
		else
			cout << "Cell Turned-On Resistance : " << resistanceOn / 1e6 << "Mohm" << endl;
		if (resistanceOff < 1e3 )
			cout << "Cell Turned-Off Resistance: "<< resistanceOff << "ohm" << endl;
		else if (resistanceOff < 1e6)
			cout << "Cell Turned-Off Resistance: "<< resistanceOff / 1e3 << "Kohm" << endl;
		else
			cout << "Cell Turned-Off Resistance: "<< resistanceOff / 1e6 << "Mohm" << endl;

		if (readMode) {
			cout << "Read Mode: Voltage-Sensing" << endl;
			if (readCurrent > 0)
				cout << "  - Read Current: " << readCurrent * 1e6 << "uA" << endl;
			if (readVoltage > 0)
				cout << "  - Read Voltage: " << readVoltage << "V" << endl;
		} else {
			cout << "Read Mode: Current-Sensing" << endl;
			if (readCurrent > 0)
				cout << "  - Read Current: " << readCurrent * 1e6 << "uA" << endl;
			if (readVoltage > 0)
				cout << "  - Read Voltage: " << readVoltage << "V" << endl;
		}

		if (resetMode) {
			cout << "Reset Mode: Voltage" << endl;
			cout << "  - Reset Voltage: " << resetVoltage << "V" << endl;
		} else {
			cout << "Reset Mode: Current" << endl;
			cout << "  - Reset Current: " << resetCurrent * 1e6 << "uA" << endl;
		}
		cout << "  - Reset Pulse: " << TO_SECOND(resetPulse) << endl;

		if (setMode) {
			cout << "Set Mode: Voltage" << endl;
			cout << "  - Set Voltage: " << setVoltage << "V" << endl;
		} else {
			cout << "Set Mode: Current" << endl;
			cout << "  - Set Current: " << setCurrent * 1e6 << "uA" << endl;
		}
		cout << "  - Set Pulse: " << TO_SECOND(setPulse) << endl;

		switch (accessType) {
		case CMOS_access:
			cout << "Access Type: CMOS" << endl;
			break;
		case BJT_access:
			cout << "Access Type: BJT" << endl;
			break;
		case diode_access:
			cout << "Access Type: Diode" << endl;
			break;
		default:
			cout << "Access Type: None Access Device" << endl;
		}
	} else if (memCellType == SRAM) {
		cout << "SRAM Cell Access Transistor Width: " << widthAccessCMOS << "F" << endl;
		cout << "SRAM Cell NMOS Width: " << widthSRAMCellNMOS << "F" << endl;
		cout << "SRAM Cell PMOS Width: " << widthSRAMCellPMOS << "F" << endl;
	} else if (memCellType == SLCNAND) {
		cout << "Pass Voltage       : " << flashPassVoltage << "V" << endl;
		cout << "Programming Voltage: " << flashProgramVoltage << "V" << endl;
		cout << "Erase Voltage      : " << flashEraseVoltage << "V" << endl;
		cout << "Programming Time   : " << TO_SECOND(flashProgramTime) << endl;
		cout << "Erase Time         : " << TO_SECOND(flashEraseTime) << endl;
		cout << "Gate Coupling Ratio: " << gateCouplingRatio << endl;
	}
}

/* Stochastic modeling implementations */

TransitionType MemCell::ClassifyTransition(bool currentBit, bool targetBit) {
	if (!currentBit && targetBit) {
		return SET;         /* 0→1 transition */
	} else if (currentBit && !targetBit) {
		return RESET;       /* 1→0 transition */
	} else if (!currentBit && !targetBit) {
		return REDUNDANT_SET;   /* 0→0 redundant operation */
	} else {
		return REDUNDANT_RESET; /* 1→1 redundant operation */
	}
}

int MemCell::SamplePulseCount(TransitionType transitionType) {
	/* If stochastic mode is disabled, return fixed single pulse */
	if (!stochasticEnabled) {
		return 1;
	}
	
	/* Sample from appropriate distribution based on transition type */
	int pulseCount;
	switch (transitionType) {
		case SET:
			pulseCount = SampleTruncatedNormal(setPulseCountMean, setPulseCountStdDev, 
											   setPulseCountMin, setPulseCountMax);
			break;
		case RESET:
			pulseCount = SampleTruncatedNormal(resetPulseCountMean, resetPulseCountStdDev,
											   resetPulseCountMin, resetPulseCountMax);
			break;
		case REDUNDANT_SET:
		case REDUNDANT_RESET:
			pulseCount = SampleTruncatedNormal(redundantPulseCountMean, redundantPulseCountStdDev,
											   redundantPulseCountMin, redundantPulseCountMax);
			break;
		default:
			pulseCount = 1;
			break;
	}
	
	return pulseCount;
}

double MemCell::CalculateMultiPulseLatency(TransitionType transitionType, int pulseCount) {
	/* Calculate multi-pulse completion time */
	double singlePulseDuration;
	
	if (transitionType == SET || transitionType == REDUNDANT_SET) {
		singlePulseDuration = setPulse;
	} else {
		singlePulseDuration = resetPulse;
	}
	
	return pulseCount * singlePulseDuration;
}

/* Statistical validation functions */

void MemCell::ValidateDistributionSampling(TransitionType type, int sampleCount) {
	if (!stochasticEnabled) {
		cout << "Stochastic sampling validation skipped - stochastic mode disabled" << endl;
		return;
	}
	
	cout << "\n=== Validating Distribution Sampling for ";
	switch(type) {
		case SET: cout << "SET"; break;
		case RESET: cout << "RESET"; break;
		case REDUNDANT_SET: 
		case REDUNDANT_RESET: cout << "REDUNDANT"; break;
		case NONE: cout << "NONE"; break;
	}
	cout << " transition ===" << endl;
	
	/* Generate samples */
	std::vector<int> samples;
	double sum = 0.0;
	int minSample = 1000, maxSample = 0;
	
	for (int i = 0; i < sampleCount; i++) {
		int sample = SamplePulseCount(type);
		samples.push_back(sample);
		sum += sample;
		minSample = (sample < minSample) ? sample : minSample;
		maxSample = (sample > maxSample) ? sample : maxSample;
	}
	
	/* Calculate statistics */
	double sampleMean = sum / sampleCount;
	double sumSquaredDiff = 0.0;
	for (int sample : samples) {
		double diff = sample - sampleMean;
		sumSquaredDiff += diff * diff;
	}
	double sampleStdDev = sqrt(sumSquaredDiff / (sampleCount - 1));
	
	/* Get expected values */
	double expectedMean, expectedStdDev;
	int expectedMin, expectedMax;
	switch(type) {
		case SET:
			expectedMean = setPulseCountMean;
			expectedStdDev = setPulseCountStdDev;
			expectedMin = setPulseCountMin;
			expectedMax = setPulseCountMax;
			break;
		case RESET:
			expectedMean = resetPulseCountMean;
			expectedStdDev = resetPulseCountStdDev;
			expectedMin = resetPulseCountMin;
			expectedMax = resetPulseCountMax;
			break;
		case REDUNDANT_SET:
		case REDUNDANT_RESET:
			expectedMean = redundantPulseCountMean;
			expectedStdDev = redundantPulseCountStdDev;
			expectedMin = redundantPulseCountMin;
			expectedMax = redundantPulseCountMax;
			break;
		default:
			cout << "Unknown transition type" << endl;
			return;
	}
	
	/* Print results */
	printf("Samples: %d\n", sampleCount);
	printf("Expected Mean: %.2f, Actual Mean: %.2f (Error: %.1f%%)\n", 
		   expectedMean, sampleMean, fabs(sampleMean - expectedMean) / expectedMean * 100);
	printf("Expected StdDev: %.2f, Actual StdDev: %.2f (Error: %.1f%%)\n",
		   expectedStdDev, sampleStdDev, fabs(sampleStdDev - expectedStdDev) / expectedStdDev * 100);
	printf("Expected Range: [%d, %d], Actual Range: [%d, %d]\n",
		   expectedMin, expectedMax, minSample, maxSample);
	
	/* Validation checks */
	double meanError = fabs(sampleMean - expectedMean) / expectedMean * 100;
	double stddevError = fabs(sampleStdDev - expectedStdDev) / expectedStdDev * 100;
	bool meanOK = meanError < 5.0;  // Within 5%
	bool stddevOK = stddevError < 15.0;  // Within 15% (std dev has more variance)
	bool boundsOK = (minSample >= expectedMin) && (maxSample <= expectedMax);
	
	cout << "Validation Results:" << endl;
	cout << "  Mean: " << (meanOK ? "PASS" : "FAIL") << endl;
	cout << "  StdDev: " << (stddevOK ? "PASS" : "FAIL") << endl;
	cout << "  Bounds: " << (boundsOK ? "PASS" : "FAIL") << endl;
	cout << "  Overall: " << ((meanOK && stddevOK && boundsOK) ? "PASS" : "FAIL") << endl;
}

void MemCell::PrintStochasticParameters() {
	cout << "\n=== Stochastic Parameters ===" << endl;
	cout << "Stochastic Enabled: " << (stochasticEnabled ? "true" : "false") << endl;
	
	if (stochasticEnabled) {
		cout << "\nSET Transition (0→1):" << endl;
		printf("  Mean: %.2f pulses, StdDev: %.2f, Range: [%d, %d]\n",
			   setPulseCountMean, setPulseCountStdDev, setPulseCountMin, setPulseCountMax);
		
		cout << "RESET Transition (1→0):" << endl;
		printf("  Mean: %.2f pulses, StdDev: %.2f, Range: [%d, %d]\n",
			   resetPulseCountMean, resetPulseCountStdDev, resetPulseCountMin, resetPulseCountMax);
		
		cout << "Redundant Operations (0→0, 1→1):" << endl;
		printf("  Mean: %.2f pulses, StdDev: %.2f, Range: [%d, %d]\n",
			   redundantPulseCountMean, redundantPulseCountStdDev, redundantPulseCountMin, redundantPulseCountMax);
		
		cout << "Pulse Durations:" << endl;
		printf("  SET/Redundant-SET: %.2f ns\n", setPulse * 1e9);
		printf("  RESET/Redundant-RESET: %.2f ns\n", resetPulse * 1e9);
	}
}
