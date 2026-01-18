package heuristics;

import java.util.LinkedList;
import java.util.ListIterator;
import java.util.Random;
import java.util.logging.Logger;

import core.InstanceMTD;
import ilog.cplex.IloCplex.CutType;
import utils.ToolsMTD;

public class AddOperator {
	HeuristicsVariablesSet currentSolution;
	InstanceMTD instance;
	
	Random rn;
	

	OpenStationsStructure openStationsStructure;
	ListIterator<OpenStation> openStations;
	
	double nextS;
	double nextAddedEnergy;
	double nextAddedCtime;
	double newE;
	double prevAddednergy;
	double prevAddedCtime;
	
	OverlappingConflictSolver overlappingConflictSolver;
	
	private final static Logger LOGGER = Logger.getLogger(AddOperator.class.getName());
	
	
	public AddOperator(CompleteSolution currentSolution, InstanceMTD instance, Random rn) {
		this.currentSolution = currentSolution.normalVars;
		this.instance = instance;
		this.openStationsStructure = currentSolution.openStationsStructure;
		this.rn = rn;
		overlappingConflictSolver = new OverlappingConflictSolver(currentSolution);
		
		LOGGER.setLevel(instance.loggerLevel);
	}
	
	public void resetOperator(CompleteSolution currentSolution) {
		this.currentSolution = currentSolution.normalVars;
		this.openStationsStructure = currentSolution.openStationsStructure;
		overlappingConflictSolver = new OverlappingConflictSolver(currentSolution);
		nextS = 0;
		nextAddedEnergy = 0;
		nextAddedCtime = 0;
		newE = 0;
		prevAddednergy = 0;
		prevAddedCtime = 0;
		
	}
	
	public boolean checkFeasibility(Stop stop, int b, int i, OpenStation nextOpenStation) {
		boolean isFeasible = true;
		
		if (nextOpenStation != null) {	
			
			//System.out.println("Has next");
			
			int k = stop.stop;
			
			nextS = 100000;
			nextAddedEnergy = 0;
			
			double p = rn.nextDouble();
			int nextK = nextOpenStation.stop;
			
			// Max amount of energy that can be added
			//nextAddedEnergy = currentSolution.c[b][nextK] - instance.Cmin; 
			
			//nextAddedEnergy = Math.min(p*currentSolution.e[b][nextK], nextAddedEnergy);
			nextAddedEnergy = p*currentSolution.e[b][nextK];
			
			//System.out.printf("nextAddedEnergy: %s\n", nextAddedEnergy);
			nextAddedCtime =  nextAddedEnergy / instance.chargingRate;
			
			for (int j = k + 1; j <= nextK; j++ ) {
				double delayAvailable = instance.DTmax - (currentSolution.getDeltaT(b, j) + nextAddedCtime);
				nextS = delayAvailable < nextS ? delayAvailable : nextS;
			}
			
			isFeasible = ToolsMTD.round(nextS) >= 0;
			
			boolean energyFeasibility = ToolsMTD.round(currentSolution.c[b][k] + nextAddedEnergy) <= instance.Cmax;
			
			
			boolean nextStillOpen = currentSolution.e[b][nextK] - nextAddedEnergy >= 1;
			
			isFeasible = isFeasible && energyFeasibility && nextStillOpen;
			
			openStations.previous();
			
			// Inserting first stop
			if (!openStations.hasPrevious()) {
				/**
				 * Overlapping conflicts
				 */	
				double newArrival = currentSolution.arrivalTime[b][k];
				double newCT = nextAddedCtime;
				boolean isOverlappingFeasible = overlappingConflictSolver.checkConflicts(
						newArrival, newCT, b, k, i);
				if (!isOverlappingFeasible) {
					return false;
				}
			}
		} 
		
		if (!isFeasible) {
			return false;
		}
		
		/**
		 * Overlapping conflicts
		 */
		
		/*
		double newArrival = currentSolution.arrivalTime[b][k];
		double newCT = currentSolution.ct[b][k] + nextAddedCtime;
		boolean isOverlappingFeasible = overlappingConflictSolver.checkConflicts(
				newArrival, newCT, b, k, i);
		*/
		
		
		int k = stop.stop;
		
		boolean hadPrevious = false;
		
		if (openStations.hasPrevious()) {
			
			//System.out.println("has previous");
			
			OpenStation prevOpenStation = openStations.previous();
			int prevK = prevOpenStation.stop;
			
			if (prevK >= k) {
				System.out.printf("In basic: prevk=%s, k= %s\n", prevK, k);
				System.exit(1);
			}
			
			for (int j = prevK + 1; j <= k; j++ ) {
				newE += instance.D[instance.paths[b][j-1]][instance.paths[b][j]];
			}
			
			//System.out.println("p prevK = " + prevK);
			// Max charge that can be transfered from prevK to new stop k
			double maxNewCharge = currentSolution.c[b][prevK] + currentSolution.e[b][prevK] - newE - instance.Cmin;
			//double maxNewChargeByLeft = currentSolution.c[b][prevK] - instance.Cmin;
			maxNewCharge = Math.min(maxNewCharge, currentSolution.e[b][prevK]);
			//System.out.println("p newE " + newE);
			//System.out.println("p maxNewCharge " + maxNewCharge);
			//System.out.printf("Transferred energy: %s. CurrentEnergy: %s\n", maxNewCharge, currentSolution.e[b][prevK]);
			//System.out.printf("c[%s][%s]: %s, e[][]: %s, newE: %s\n", b, prevK, currentSolution.c[b][prevK],
			//		currentSolution.e[b][prevK], newE);
			double minP = maxNewCharge / currentSolution.e[b][prevK];	
			double preP = 0 + (rn.nextDouble() * (minP - 0));
			/*
			System.out.println("minP: " + minP);
			System.out.println("p: " + p);
			System.out.println("maxNewCharge: " + maxNewCharge);
			System.out.println("nextAddedEnergy: " + nextAddedEnergy);
			*/
			//double p = rn.nextDouble();
			
			/*
			if (currentSolution.e[b][prevK] < 0.01) {
				System.out.println("WARNING. Open stops without charging!!!! " + currentSolution.e[b][prevK]);
				System.exit(0);
			}
			*/
			
			prevAddednergy = preP*currentSolution.e[b][prevK];
			prevAddedCtime = prevAddednergy / instance.chargingRate;
			
			//System.out.println("p prevAddednergy " + prevAddednergy);
			
			if (prevAddednergy + nextAddedEnergy < 1 || 
					currentSolution.e[b][prevK] - prevAddednergy < 1) {
				return false;
			}
			if (!checkFeasibilityFromLeft(stop, b, i, prevAddednergy)) {
				//System.out.println("SuperHello");
				//System.exit(0);
				return false;
			}
			
			/**
			 * Overlapping conflicts
			 */
			
			double newArrival = currentSolution.arrivalTime[b][k] - prevAddedCtime;
			double newCT = prevAddedCtime + nextAddedCtime;
			boolean isOverlappingFeasible = overlappingConflictSolver.checkConflicts(
					newArrival, newCT, b, k, i);
			if (!isOverlappingFeasible) {
				return false;
			}
			
			hadPrevious = true;

			openStations.next();
		}
		
		if (!hadPrevious && nextAddedEnergy < 1 ) {
			return false;
		}
		
		// Checking max time in a single recharge
		double newCT = prevAddedCtime + nextAddedCtime;
		if (newCT > instance.maxChargingTime) {
			return false;
		}
		
		if (nextOpenStation != null) {	
			openStations.next();
		}
		
		//System.out.println("reach final");
		//System.out.println("isFeasible " + isFeasible);
		
		return isFeasible;
	}
	
	public void addStation(int i) {
		if (openStationsStructure.getNumOpenStopsPerStations(i) == 0) {
			int b = -1;
			for (Stop stop: instance.stopsPerStation[i]) {
				nextAddedEnergy = 0;
				nextAddedCtime = 0;
				newE = 0;
				prevAddednergy = 0;
				prevAddedCtime = 0;
				b = stop.bus;
				openStations = openStationsStructure.getListIteratorOpenStations(b);
				boolean willBeLastOpenStation = true;
				while (openStations.hasNext()) {
					OpenStation openStation = openStations.next();
					if (openStation.stop > stop.stop) {
						if (checkFeasibility(stop, b, i, openStation)) {
							openStations.previous();
							updateAfterAdd(stop, b, i);
						}
						willBeLastOpenStation = false;
						break;			
					}
				}
				if (willBeLastOpenStation) {
					LOGGER.info("Adding last stop");
					//nextAddedEnergy = 0;
					//nextAddedCtime = 0;
					//openStations.previous();
					if (checkFeasibility(stop, b, i, null)) {
						updateAfterAdd(stop, b, i);
					}
				}
				//currentSolution.print(String.format("solutionAfterAdd%s_%s_%s", i, b, stop.stop));
				
				/*
				if (newE <= 0) {
					System.out.println("newE <= 0!!!");
					System.exit(1);
				}
				*/
				/*
				if (!currentSolution.checkFeasibiliy()) {
					currentSolution.print("overlapped");
					System.exit(1);;
				};
				*/
			}
		}
		
	}
	
	public boolean checkFeasibilityFromLeft(Stop stop, int b, int i, double energyBorrowedFromPrevious) {
		boolean feasible = true;
		feasible = ToolsMTD.round(currentSolution.c[b][stop.stop] - energyBorrowedFromPrevious) >= instance.Cmin;
		//System.out.printf("%s - %s >= %s\n", currentSolution.c[b][stop.stop], energyBorrowedFromPrevious,
		//		instance.Cmin);
		return feasible;
		
	}
	
	public void updateAfterAdd(Stop stop, int b, int i) {
		boolean hadPrevious = false;
		int k = stop.stop;
		
		LOGGER.info(String.format("Adding bus %s, stop %s", b, k));
		
		if (openStations.hasPrevious()) {
			OpenStation prevOpenStation = openStations.previous();
			int prevK = prevOpenStation.stop;		
			
			//System.out.println("The stop has previous");
				
			currentSolution.e[b][prevK] -= prevAddednergy;
			currentSolution.ct[b][prevK] -= prevAddedCtime;
			currentSolution.e[b][k] = prevAddednergy;
			currentSolution.ct[b][k] = prevAddedCtime;
			openStations.next();
			//openStations.next();
			
			currentSolution.xBStop[b][k] = true;
			currentSolution.x[i] = true;
			//System.out.printf("Hola? k:%s, prevK: %s\n", k, prevK);
			
			updateOpenStationsStructure(prevK + 1, b, k, i, prevAddednergy, prevAddedCtime);
			
			hadPrevious = true;
			
			/*
			System.out.printf("Energy added from left [%s][%s] to [%s][%s]: %s\n", b, prevK, b, k,
					energyNewCharge);
			*/
	
		}
		
		
		/**
		 * We could check if until new open station was not the max deviation of s 
		 * and avoid this step
		 */
		if (openStations.hasNext()) {
			//System.out.println("The stop has next");
			OpenStation nextOpenStation = openStations.next();
			
			if (!hadPrevious) {
				
				LOGGER.info("Added at the beginning as a first station");
											
				/**
				 * TODO 
				 * Improve this for. It should not be necessary
				 */
				for (int j = 0 + 1; j <= k; j++ ) {
					newE += instance.D[instance.paths[b][j-1]][instance.paths[b][j]];
				}
				
				currentSolution.xBStop[b][k] = true;
				currentSolution.x[i] = true;
				
				openStations.previous();	
				updateOpenStationsStructure(0 + 1, b, k, i, 0, 0);
				
				
			}
			nextOpenStation.E -= newE;
			//System.out.printf("newE[%s][%s] = %s\n", b, k, newE);
			int nextK = nextOpenStation.stop;
			//System.out.printf("nextE[%s][%s] = %s\n", b, nextK, nextOpenStation.E);
			currentSolution.e[b][nextK] -= nextAddedEnergy;
			currentSolution.ct[b][nextK] -= nextAddedCtime;
			currentSolution.e[b][k] += nextAddedEnergy;
			currentSolution.ct[b][k] += nextAddedCtime;
			double nextS = 100000;
			for (int j = k + 1; j <= nextK; j++ ) {
				currentSolution.c[b][j] += nextAddedEnergy;
				currentSolution.arrivalTime[b][j] += nextAddedCtime;
				double delayAvailable = instance.DTmax - currentSolution.getDeltaT(b, j);
				nextS = delayAvailable < nextS ? delayAvailable : nextS;
			}
			nextOpenStation.s = nextS;
			//openStations.previous();
			
			/*
			System.out.printf("Energy added from right [%s][%s] to [%s][%s]: %s\n", b, nextK, b, k,
					nextAddedEnergy);
			/*
			if (nextOpenStation.E < 0) {
				System.out.println("nextE < 0!!!");
				System.exit(1);
			}
			*/
		}
			
	}
	
	public void updateOpenStationsStructure(int indexFromToUpdateE, int b, int k, int i, double prevAddedEnergy,
			double prevAddedCTime) {
		double newS = 100000;
		for (int j = indexFromToUpdateE; j <= k; j++ ) {
			currentSolution.c[b][j] -= prevAddedEnergy;
			currentSolution.arrivalTime[b][j] -= prevAddedCTime;
			double delayAvailable = instance.DTmax - currentSolution.getDeltaT(b, j);
			newS = delayAvailable < newS ? delayAvailable : newS;
		}
		
		OpenStation newOpenStation = new OpenStation(newE, newS, k, b);
		openStations.add(newOpenStation);
		if (openStationsStructure.openStopsPerStation.containsKey(i)) {
			openStationsStructure.openStopsPerStation.get(i).add(newOpenStation);
		} else {
			LinkedList<OpenStation> osPerS = new LinkedList<OpenStation>();
			osPerS.add(newOpenStation);
			openStationsStructure.openStopsPerStation.put(i, osPerS);	
		}
	}
	
}
