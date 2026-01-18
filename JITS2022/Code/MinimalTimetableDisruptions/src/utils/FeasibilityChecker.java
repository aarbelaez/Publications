package utils;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashSet;
import java.util.Set;

import core.InstanceMTD;
import heuristics.HeuristicsVariablesSet;
import heuristics.Stop;

public class FeasibilityChecker {
	
	private HeuristicsVariablesSet primaryVars;
	private HeuristicsVariablesSet backupVars;
	private InstanceMTD instance;
	PrintWriter checkingFile;
	
	public FeasibilityChecker(HeuristicsVariablesSet primary, InstanceMTD instance, String filename) {
		this.primaryVars = primary;
		this.instance = instance;
		try {
			checkingFile = new PrintWriter(new BufferedWriter(
					new FileWriter(String.format("../data/%s", filename), true)));
			checkingFile.println();
			checkingFile.println(new Date().toString());
			checkingFile.printf("city:%s, Cmax: %s, dt:%s\n\n", instance.inputFolder, instance.Cmax, instance.DTmax);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public FeasibilityChecker(HeuristicsVariablesSet primary, HeuristicsVariablesSet backup, InstanceMTD instance,
			String filename) {
		this.primaryVars = primary;
		this.backupVars = backup;
		this.instance = instance;
		try {
			checkingFile = new PrintWriter(new BufferedWriter(
					new FileWriter(String.format("../data/%s", filename), true)));
			checkingFile.println();
			checkingFile.println(new Date().toString());
			checkingFile.printf("city:%s, Cmax: %s, dt:%s\n\n", instance.inputFolder, instance.Cmax, instance.DTmax);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public boolean checkFeasibiliy(HeuristicsVariablesSet variables) {
		boolean isFeasible = true;
		for (int b = 0; b < instance.b; b++) {		
			for (int k = 0; k < instance.paths[b].length; k++) {
				double cRounded = Math.round(variables.c[b][k]);
				double eRounded = Math.round(variables.e[b][k]);
				double ctRounded = Math.round(variables.ct[b][k]);
				double arrivalRounded = Math.round(variables.arrivalTime[b][k]);
				double cPlusERounded = Math.round(variables.c[b][k] + variables.e[b][k]);
				
				if (cRounded < instance.Cmin) {
					checkingFile.printf("The arrival energy of bus %s in stop %s is less than allowed\n", b, k);
					checkingFile.printf("%s < %s\n", cRounded, instance.Cmin );
					isFeasible = false;
				}
				if (cRounded > instance.Cmax) {
					checkingFile.printf("The arrival energy of bus %s in stop %s is more than allowed\n", b, k);
					checkingFile.printf("%s > %s\n", cRounded, instance.Cmax);
					isFeasible = false;
				}
				double calculatedE = Math.round(instance.chargingRate*variables.ct[b][k]);
				if (cPlusERounded > instance.Cmax) {
					checkingFile.printf("Bus %s in stop %s is charging more than battery capacity\n", b, k);
					checkingFile.printf("%s > %s\n", cRounded + eRounded, instance.Cmax);
					isFeasible = false;
					// System.exit(0);
				}
				if (calculatedE < eRounded) {
					checkingFile.printf("The charging time does not correspond to the added energy (%s,%s)\n", b, k);
					checkingFile.printf("[%s][%s] %s < %s\n", b, k, instance.chargingRate*variables.ct[b][k],
							variables.e[b][k]);
					//System.exit(0);
					isFeasible = false;
				}
				double deltaTime = Math.abs(variables.arrivalTime[b][k] - arrivalRounded);
				if (deltaTime > instance.DTmax) {
					checkingFile.printf("The time disruption in bus %s stop %s is too high\n", b, k);
					checkingFile.printf("[%s][%s] %s > %s\n", b, k, deltaTime,
							instance.DTmax);
					//System.exit(0);
					isFeasible = false;
				}
				if (variables.xBStop[b][k]) {
					if (ctRounded > instance.maxChargingTime) {
						checkingFile.printf("Bus %s in stop %s is charging during too much time: %s > %s\n",
								b, k, ctRounded, instance.maxChargingTime);
						isFeasible = false;
					}
				}
				int i = instance.paths[b][k];
				if (variables.xBStop[b][k]) {
					if (!variables.x[i]) {
						checkingFile.printf("Stop is open while station %s is closed in bus %s stop %s\n", i, b, k);
						isFeasible = false;
						checkingFile.print("brokenSolution");
						//System.exit(0);
					}
				}
				if (k != instance.paths[b].length - 1) {
					int m = k + 1;
					int j = instance.paths[b][m];
					double roundedEArrival= Math.round(variables.c[b][m]);
					double roundedECalculatedArrival = Math.round(variables.c[b][k] + variables.e[b][k] - instance.D[i][j]);
					if (roundedEArrival > roundedECalculatedArrival) {
						checkingFile.printf("Bus %s has not enough energy to go from %s to %s\n", b, k, m);
						checkingFile.printf("%s > %s\n", variables.c[b][m], variables.c[b][k] + 
								variables.e[b][k] - instance.D[i][j]);
						isFeasible = false;
					}
					double roundedArrival= Math.round(variables.arrivalTime[b][m]*100)/100.0;
					double roundedCalculatedArrival = Math.round((variables.arrivalTime[b][k] + variables.ct[b][k] + 
							instance.T[b][k])*100)/100.0;
					if (roundedArrival < roundedCalculatedArrival) {
						checkingFile.printf("Bus %s cannot arrive so early at stop %s\n", b, m);
						checkingFile.printf("%s < %s\n", variables.arrivalTime[b][m], variables.arrivalTime[b][k] +
								variables.ct[b][k] + instance.T[b][k]);
						checkingFile.printf("%s < %s\n", roundedArrival, roundedCalculatedArrival);
						isFeasible = false;
					}
				}
				
				if (cRounded < 0) {
					checkingFile.printf("Arrival energy for bus %s in stop %s is negative\n", b, k);
					isFeasible = false;
				}
				if (eRounded < 0) {
					checkingFile.printf("Energy added by bus %s in stop %s is negative\n", b, k);
					isFeasible = false;
				}
				if (ctRounded < 0) {
					checkingFile.printf("Charging time for bus %s in stop %s is negative\n", b, k);
					isFeasible = false;
				}
				if (arrivalRounded < 0) {
					checkingFile.printf("Arrival time for bus %s in stop %s is negative\n", b, k);
					isFeasible = false;
				}
				
			}
		}
		boolean overlappingFeasibility = checkOverlappingConflicts(variables);
		isFeasible = isFeasible && overlappingFeasibility;
		checkingFile.printf("feasible: %s\n", isFeasible);
		return isFeasible;
	}
	
	public boolean checkOverlappingConflicts(HeuristicsVariablesSet variables) {
		boolean isFeasible = true;
		for (int i = 0; i < instance.n; i++) {
			//System.out.println(i);
			Set<Stop> conflicts = new HashSet<Stop>();
			for(int b = 0; b < instance.b; b++) {
				for(int k = 0; k < instance.paths[b].length; k++) {
					int bStation = instance.paths[b][k];
					if (bStation != i || variables.ct[b][k] < 0.05) {
						//System.out.printf("b=%s, k=%s\n", b, k);
						continue;
					}
					if (b != instance.b - 1) {
						for(int b2 = b + 1; b2 < instance.b; b2++) {
							for(int m = 0; m < instance.paths[b2].length; m++) {
								int b2Station = instance.paths[b2][m];
								if (b2Station != i || variables.ct[b2][m] < 0.05) {
									continue;
								}
								double departureTimeB = ToolsMTD.round(variables.arrivalTime[b][k] + variables.ct[b][k]);
								double departureTimeB2 = ToolsMTD.round(variables.arrivalTime[b2][m] + variables.ct[b2][m]);
								double arrivalB = ToolsMTD.round(variables.arrivalTime[b][k]);
								double arrivalB2 = ToolsMTD.round(variables.arrivalTime[b2][m]);
								if (arrivalB >= arrivalB2 &&
										arrivalB < departureTimeB2 ||
										departureTimeB > arrivalB2 
												&& departureTimeB <= departureTimeB2) {
									conflicts.add(new Stop(b, k, variables.arrivalTime[b][k]));
									conflicts.add(new Stop(b2, m, variables.arrivalTime[b2][m]));
									/*
									System.out.printf("%s >= %s && %s < %s || %s > %s && %s <= %s",
											arrivalB, arrivalB2, arrivalB, departureTimeB2,
											departureTimeB, arrivalB2, departureTimeB, departureTimeB2);
									*/
								}
								/*
								System.out.printf("%s >= %s && %s < %s || %s >= %s && %s < %s",
										variables.arrivalTime[b][k], variables.arrivalTime[b2][m], variables.arrivalTime[b][k], departureTimeB2,
										departureTimeB, variables.arrivalTime[b2][m], departureTimeB, departureTimeB2);
								*/
							}
						}
					}
				}
			}
			
			
			ArrayList<Stop> conflictsToOrder = new ArrayList<Stop>(conflicts);
			conflictsToOrder.sort((s1, s2) -> s1.arrivalTime.compareTo(s2.arrivalTime));
			if (conflicts.size() > 0) {
				isFeasible = false;
				checkingFile.printf("Number of overlapping constraints in station %s: %s\n", i, conflicts.size());
				for (Stop stop: conflictsToOrder) {
					checkingFile.printf("b=%s, k=%s, t=%s, ct=%s\n", stop.bus, stop.stop, stop.arrivalTime,
							variables.ct[stop.bus][stop.stop]);		
				}
				checkingFile.println();
			}
		}
		return isFeasible;
	}
	
	public boolean checkRobustFeasibility() {
		boolean primaryFeasibility = checkFeasibiliy(primaryVars);
		boolean backupFeasibility = checkFeasibiliy(backupVars);
		boolean switchingFeasibility = checkSwitchingFeasibility();
		boolean overlappingBackupPrimaryFeasibility = checkOverlappingBackupPrimaryFeasibility();
		boolean pathFeasibility = checkPathConstraintFeasibility();
		boolean feasible = primaryFeasibility && backupFeasibility && switchingFeasibility && 
				overlappingBackupPrimaryFeasibility && pathFeasibility;
		checkingFile.printf("feasible: \n", feasible);
		return feasible;
	}
	
	public boolean checkSwitchingFeasibility() {
		boolean isFeasible = true;
		for (int b = 0; b < instance.b; b++) {
			for (int k = 0; k < instance.paths[b].length; k++) {
				double primaryC = ToolsMTD.round(primaryVars.c[b][k]);
				double backupC = ToolsMTD.round(backupVars.c[b][k]);
				double primaryT = ToolsMTD.round(primaryVars.arrivalTime[b][k]);
				double backupT = ToolsMTD.round(backupVars.arrivalTime[b][k]);
				if (primaryVars.xBStop[b][k]) {
					// Switch to the backup route
					if (backupC > primaryC) {
						isFeasible = false;
						checkingFile.printf("Bus %s cannot switch to backup in stop %s because of its energy\n", b, k);
						checkingFile.printf("%s > %s\n", backupVars.c[b][k], primaryVars.c[b][k]);
					}
					if (backupT < primaryT) {
						isFeasible = false;
						checkingFile.printf("Bus %s cannot switch to backup in stop %s because of its time\n", b, k);
						checkingFile.printf("%s < %s\n", backupVars.arrivalTime[b][k], primaryVars.arrivalTime[b][k]);
					}
				}
				// Back to the primary route
				if (k != 0 && backupVars.xBStop[b][k-1]) {
					if (backupC < primaryC) {
						isFeasible = false;
						checkingFile.printf("Bus %s cannot back to primary in stop %s because of its energy\n", b, k);
						checkingFile.printf("%s < %s\n", backupVars.c[b][k], primaryVars.c[b][k]);
					}
					if (backupT > primaryT) {
						isFeasible = false;
						checkingFile.printf("Bus %s cannot back to primary in stop %s because of its time\n", b, k);
						checkingFile.printf("%s > %s\n", backupVars.arrivalTime[b][k], primaryVars.arrivalTime[b][k]);
					}
				}
			}
		}
		return isFeasible;
	}
	
	public boolean checkOverlappingBackupPrimaryFeasibility() {
		boolean isFeasible = true;
		for (int i = 0; i < instance.n; i++) {
			//System.out.println(i);
			Set<Stop> conflicts = new HashSet<Stop>();
			for(int bp = 0; bp < instance.b; bp++) {
				for(int k = 0; k < instance.paths[bp].length; k++) {
					int bStation = instance.paths[bp][k];
					if (bStation != i || ToolsMTD.round(primaryVars.ct[bp][k]) == 0) {
						//System.out.printf("b=%s, k=%s\n", b, k);
						continue;
					}
					for(int bb = 0; bb < instance.b; bb++) {
						if (bp == bb) {
							continue;
						}
						for(int m = 0; m < instance.paths[bb].length; m++) {
							int b2Station = instance.paths[bb][m];
							if (b2Station != i || ToolsMTD.round(backupVars.ct[bb][m]) == 0) {
								continue;
							}
							double departureTimeB = primaryVars.arrivalTime[bp][k] + primaryVars.ct[bp][k];
							double departureTimeB2 = backupVars.arrivalTime[bb][m] + backupVars.ct[bb][m];
							if (primaryVars.arrivalTime[bp][k] >= backupVars.arrivalTime[bb][m] && 
									primaryVars.arrivalTime[bp][k] < departureTimeB2 ||
									departureTimeB > backupVars.arrivalTime[bb][m] && departureTimeB <= departureTimeB2) {
								conflicts.add(new Stop(bp, k, primaryVars.arrivalTime[bp][k]));
								conflicts.add(new Stop(bb, m, backupVars.arrivalTime[bb][m]));		
							}
							/*
							System.out.printf("%s >= %s && %s < %s || %s >= %s && %s < %s",
									variables.arrivalTime[b][k], variables.arrivalTime[b2][m], variables.arrivalTime[b][k], departureTimeB2,
									departureTimeB, variables.arrivalTime[b2][m], departureTimeB, departureTimeB2);
							*/
						}
					}
				}
			}
			
			
			ArrayList<Stop> conflictsToOrder = new ArrayList<Stop>(conflicts);
			conflictsToOrder.sort((s1, s2) -> s1.arrivalTime.compareTo(s2.arrivalTime));
			if (conflicts.size() > 0) {
				isFeasible = false;
				checkingFile.printf("Number of ROBUST overlapping constraints in station %s: %s\n", i, conflicts.size());
				for (Stop stop: conflictsToOrder) {
					double ct = 0;
					if (stop.arrivalTime == primaryVars.arrivalTime[stop.bus][stop.stop]) {
						ct = primaryVars.ct[stop.bus][stop.stop];
					} else if (stop.arrivalTime == backupVars.arrivalTime[stop.bus][stop.stop]) {
						ct = backupVars.ct[stop.bus][stop.stop];
					}
					checkingFile.printf("b=%s, k=%s, t=%s, ct=%s\n", stop.bus, stop.stop, stop.arrivalTime,
							ct);		
				}
				//System.out.println();
			}
		}
		return isFeasible;
	}
	
	/**
	 * Check that a station is not backup of itself, i.e., a station can be work as backup and primary, only when
	 * backup comes first than primary
	 * @return
	 */
	public boolean checkPathConstraintFeasibility() {
		boolean feasible = true;
		for (int b = 0; b < instance.b; b++) {
			HashSet<Integer> primaryChargingStations= new HashSet<Integer>();
			for (int k = 0; k < instance.paths[b].length; k++) {
				int i = instance.paths[b][k];
				/*
				if (primaryVars.e[b][k] > 0.1) {
					primaryChargingStations.add(i);
				}
				if (backupVars.e[b][k] > 0.1 && primaryChargingStations.contains(i)) {
					checkingFile.printf("Station %s is backup of itself for bus %s, stop %s\n", i, b, k);
					feasible = false;
					break;
				}
				*/
				//Basic path constarint
				if (primaryVars.e[b][k] > 0.1 && backupVars.e[b][k] > 0.1) {
					feasible = false;
					break;
				}
				// For repeated stops
				if (k != instance.paths[b].length - 1) {
					int j = instance.paths[b][k + 1];
					boolean backupCharging = backupVars.e[b][k] >= 0.1;
					boolean primaryCharging = primaryVars.e[b][k+1] >= 0.1;
					boolean repeatedStops = i == j && instance.originalTimetable[b][k] == instance.originalTimetable[b][k+1];
					if (repeatedStops) {
						//System.out.printf("Repeated stops for bus %s: %s and %s\n",
						//	 b, k, k+1);
					}
					if (repeatedStops && primaryCharging && backupCharging) {
						checkingFile.printf("Station %s is backup of itself for bus %s, at repeated stops %s and %s\n",
								i, b, k, k+1);
						feasible = false;
					}
				}
				
			}
		}
		return feasible;
	}
	
	public void closeCheckingFile() {
		checkingFile.close();
	}

}
