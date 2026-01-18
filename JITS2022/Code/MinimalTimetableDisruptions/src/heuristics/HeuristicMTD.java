package heuristics;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.ListIterator;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

import core.InstanceMTD;
import utils.FeasibilityChecker;

public class HeuristicMTD {
	//HeuristicsVariablesSet currentSolution;
	CompleteSolution currentSolution;
	CompleteSolution bestCurrentSolution;
	public CompleteSolution overallBestCurrentSolution;
	//OpenStationsStructure bestOpenStationsStructure;
	InstanceMTD instance;
	//OpenStationsStructure openStationsStructure;
	ArrayList<Integer> objectives;
	//ArrayList<Integer> addedEnergy;
	//ArrayList<Integer> addedEnergyFromStructure;
	
	AddOperator addOperator;
	RemoveOperator removeOperator;
	
	
	
	Random rn;
	
	private final static Logger LOGGER = Logger.getLogger(HeuristicMTD.class.getName());
	
	public HeuristicMTD(CompleteSolution initialSolution, InstanceMTD instance, AddOperator addOperator,
			RemoveOperator removeOperator, Random rn) {
		super();
		this.currentSolution = initialSolution;
		this.instance = instance;
		this.addOperator = addOperator;
		this.removeOperator= removeOperator;
		this.rn = rn;
		//openStationsStructure = new OpenStationsStructure(initialSolution, instance);
		//currentSolution.openStationsStructure.print();
		//openStationsStructure.printOpenStopsPerStation();
		LOGGER.setLevel(instance.loggerLevel);	
		//System.out.println("Hola");
	}
	
	public void runHeuristic(int execTime) throws FileNotFoundException {
		//PrintWriter writer = new PrintWriter("../data/heuristicOutput.txt");
		double timeBestObj = -1;
		int obj = currentSolution.getNumberOpenStations();
		
		//RemoveOperator removeOperator = new RemoveOperator(currentSolution, instance, openStationsStructure, rn);
		
		int bestSolutionObj = instance.n;
		objectives = new ArrayList<Integer>();
		objectives.add(obj);
		//addedEnergy = new ArrayList<Integer>(); 
		//addedEnergyFromStructure = new ArrayList<Integer>();
		
		//openStationsStructure.checkFeasibility();
		/*
		if (!currentSolution.checkLinkedListConsistencyOpenStopsPerBus()) {
			LOGGER.info("Inconsistent structure");
			System.exit(0);
		}
		*/
		
		ArrayList<Integer> stationIds = new ArrayList<Integer>();
		for (int i = 0; i < instance.n; i++) {
			stationIds.add(i);
		}
		
		long startTime = System.currentTimeMillis();
		int maxExecTime = execTime*60*1000;
		
		
		ArrayList<Long> iterationTimesBig = new ArrayList<Long>(); 
		long startIterBig = System.currentTimeMillis();
		long maxIters = (long)instance.n * 1000000 * 100;
		
		int numCurrentSolutionSelected = 0;
		
		int prevObj = obj;

		for (int i = 0; i < maxIters; i++) {
			
			LOGGER.info("********** Iteration " + i + " *************\n");
			
			if (stationIds.isEmpty()) {

				
				
								
				if (prevObj <= obj && i > 0) {				
					
					/*
					if (!currentSolution.checkOpenStructureIntegrity()) {
						break;
					}
					*/
					
					/*
					if (!currentSolution.checkFeasibility(instance, "partial.log")) {
						LOGGER.info("Unfeasible after a block of removals");
						break;
					};
					*/
					/*
					if (currentSolution.checkForNegativeSlacks()) {
						System.out.println("Negative slacks after block of removals");
						break;
					};
					*/
					//currentSolution.printOpenStationStructure();
					
					/*
					if (!currentSolution.checkBackups()) {
						LOGGER.info("No backups after a block of removals");
						break;
					};
					*/
										
							
					
					
					/*
					if (currentSolution.getNumberOpenStops() != currentSolution.getNumberOpenStopsPerStation()) {
						System.out.println("INCONSISTENCY!!");
						break;
					}
					*/
					/*
					if(!currentSolution.areAllOpenStationsCharging()) {
						break;//System.exit(0);;
					};
					*/
					
					
								
					if (System.currentTimeMillis() - startTime >= maxExecTime) {
						break;
					}
					obj = currentSolution.getNumberOpenStations();					
					int bestObj = bestCurrentSolution == null ? 1000 : bestCurrentSolution.getNumberOpenStations();
					//System.out.println("Current solution: " + obj);
					//System.out.println("Best solution: " + bestObj);
					double threshold = 0.05;
					double p = rn.nextDouble();

					if (p > threshold) {
						LOGGER.info("Best solution selected");
						if (obj < bestObj) {
							//System.out.println("current solution is the best");
							bestCurrentSolution = currentSolution.createCopy(instance);
							//bestOpenStationsStructure = new OpenStationsStructure(bestCurrentSolution, instance);
							
						} else {
							//System.out.println("replacing current solution for the the best");
							//System.out.printf("Is feasible the solution: %s\n", bestCurrentSolution.checkFeasibility(instance));
							currentSolution = bestCurrentSolution.createCopy(instance);
							//openStationsStructure = new OpenStationsStructure(currentSolution, instance);
							addOperator.resetOperator(currentSolution);
							removeOperator.resetOperator(currentSolution);
							obj = bestObj;
						}
					} else {
						LOGGER.info("current solution selected");
						bestCurrentSolution = currentSolution.createCopy(instance);
						//bestOpenStationsStructure = new OpenStationsStructure(bestCurrentSolution, instance);
						numCurrentSolutionSelected++;
					}
					
					
					
					//System.out.println("new current solution: " + openStationsStructure.getNumberOpenStations());
					
					objectives.add(obj);
					//addedEnergy.add(currentSolution.getWastedEnergy());
					//addedEnergy.add(currentSolution.getTotalChargingTime());
					//addedEnergyFromStructure.add(currentSolution.getTotalAddedEnergyFromOpenStructure());
					/*
					if (objectives.size() > 235) {
						try {
							System.out.println("obj: " + obj);
							Thread.sleep(5*1000);
						} catch (InterruptedException e) {
							// TODO Auto-generated catch block
							e.printStackTrace();
						}
					}
					*/
					
					
					
					int addingStations = (int) Math.round(obj*0.2);
					//System.out.printf("Adding %s stations\n", addingStations);
					
					for (int j = 0; j < addingStations; j++) {
						int selectedStation2 = rn.nextInt(instance.n);
						LOGGER.info("adding station " + selectedStation2);
						/*
						if (i == 301080) {
							currentSolution.print("finalSolution_" + selectedStation2);
						}
						*/
						
						//currentSolution.printOpenStationStructure(27);
						
						addOperator.addStation(selectedStation2);
						//currentSolution.print("partialSolutionAfterAdd_" + i);
						/*
						if (i == 260550 && !currentSolution.areAllOpenStationsCharging()) {
							break;//System.exit(0);;
						};
						*/
						/*
						if (!currentSolution.checkFeasibility(instance, "partial")) {
							break;
						};
						*/
						
					}
					/*
					if (!currentSolution.checkOpenStructureIntegrity()) {
						break;
					}
					*/
					
					//objectives.add(currentSolution.getNumberOpenStations());
					
					/*
					if (!currentSolution.checkFeasibility(instance, "partial.log")) {
						LOGGER.info("Exiting after perturbation");
						break;
					};
					*/
					/*
					if (currentSolution.checkForNegativeSlacks()) {
						System.out.println("Negative slacks after block of addings");
						break;
					};
					*/
					//currentSolution.printOpenStationStructure();
					
					
					/*
					if (!currentSolution.checkBackups()) {
						LOGGER.info("No backups after a block of additions");
						break;
					};
					*/
					
					
					/*
					if (!currentSolution.checkLinkedListConsistencyOpenStopsPerBus()) {
						LOGGER.info("Inconsistent structure afte perturbation");
						System.exit(0);
					}
					*/
					
					
					
									
					/*
					if(!currentSolution.areAllOpenStationsCharging()) {
						break;//System.exit(0);;
					};
					*/
					
					
					
					//System.out.println("new current solution after insert: " + openStationsStructure.getNumberOpenStations());				
					iterationTimesBig.add(System.currentTimeMillis() - startIterBig);
					startIterBig = System.currentTimeMillis();
				}
				//Old way
				for (int k = 0; k < instance.n; k++) {
					
					/*
					I did not find any inconsistence among different structures
					if (currentSolution.getNumOpenStopsPerStations(k) != currentSolution.getNumOpenStopsPerStationsFromNormalVars(k)) {
						System.out.println("Inconsistency!!!");
						System.exit(0);
					}
					if (currentSolution.getNumOpenStopsPerStations(k) != currentSolution.getNumOpenStopsPerStationsFromMainStructure(k)) {
						System.out.println("Inconsistency!!!");
						System.exit(0);
					}
					*/
					if (currentSolution.getNumOpenStopsPerStations(k) > 0) {
						stationIds.add(k);
						//System.out.printf("Station %s number of stops: %s\n", k, currentSolution.getNumOpenStopsPerStations(k));
					}
				}		
				//Collections.shuffle(stationIds, rn);
				
				
				stationIds.sort(new Comparator<Integer>() {
					public int compare(Integer o1, Integer o2) {
						Integer numberBuses1 = currentSolution.getNumOpenStopsPerStations(o1);
						Integer numberBuses2 = currentSolution.getNumOpenStopsPerStations(o2);
				        return numberBuses2.compareTo(numberBuses1);
				    }
				});
				
			
				/*
				System.out.println("Stations in order");
				for (int s: stationIds) {
					System.out.println(s);
				}
				try {
					Thread.sleep(5*1000);
				} catch (InterruptedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				*/
				
				prevObj = obj;
			}
			
			
			
			
			
			//int selectedBus = rn.nextInt(instance.b);
			//int selectedStation = rn.nextInt(instance.n);
			//Old way
			//int selectedStation = stationIds.get(i % instance.n);
			//Random selection
			int selectedStation = stationIds.remove(stationIds.size() - 1);
			//int selectedStation = stationIds.remove(0);
			
			
			
			LOGGER.info("Station selected: " + selectedStation);
			
			//int numOpenStations = openStationsStructure.getNumOpenStations(selectedBus);
			
			
			//int numOpenStopsPerStations = currentSolution.getNumOpenStopsPerStations(selectedStation);	
			//if ( numOpenStopsPerStations > 0) {
				
				
				/*
				removeOperators = new ArrayList<RemoveOperator>();
				for (int j = 0; j < numOpenStopsPerStations; j++) {
					removeOperators.add(new RemoveOperator(currentSolution, instance, openStationsStructure, rn));
				}
				*/
				
				//System.out.printf("Number of open stops in bus %s: %s\n", selectedBus, numOpenStations);
				//int selectedStopIndex = rn.nextInt(numOpenStations);
				//int selectedStopIndex = 6;
				//System.out.println("Random stop: " + selectedStopIndex);
			//int prevObj = currentSolution.getNumberOpenStations();
			//boolean feas = false;
			
			//currentSolution.printOpenStationStructure(8);
			//int seed = rn.nextInt(100000000);
			//if (removeOperator.checkAllBuses(selectedStation, seed)) {
				//System.exit(0);
				//CompleteSolution oldSolution = currentSolution.createCopy(instance);
				if (!removeOperator.removeAllBuses(selectedStation, i)) {
					//currentSolution = oldSolution;
					/*
					try {
						Thread.sleep(3*1000);
					} catch (InterruptedException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
					*/
				};
				//feas = true;
			//}
					//System.out.print("Feasible movement\n");
					//currentSolution.print("prevRemoveSolution" + i);
					//removeStation(removeOperators, i);
					//currentSolution.print("prevAddSolution" + i);
					
					//int selectedStation = rn.nextInt(instance.n);currentSolution.print("postAddSolution" + i);
					//currentSolution.print("partialSolution" + i);
					//System.out.println("feasible?" + currentSolution.checkFeasibiliy());
			//}
			
			obj = currentSolution.getNumberOpenStations();
			/*
			if (feas) {
				System.out.println("PREV " + prevObj);
				System.out.println("AFTE " + obj);
				try {
					Thread.sleep(3*1000);
				} catch (InterruptedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
			*/
			if (obj < bestSolutionObj) {
				bestSolutionObj = obj;
				//overallBestCurrentSolution = new HeuristicsVariablesSet(instance, currentSolution.normalVars);
				overallBestCurrentSolution = currentSolution.createCopy(instance);
				timeBestObj = ((System.currentTimeMillis() - startTime) / 1000) / 60.0;
			}
			
			LOGGER.info("Number of open stations: " + obj + "\n");
			//System.out.print("Number of open stops 1: " + openStationsStructure.getNumberOpenStops1() + "\n");
			//System.out.print("Number of open stops 2: " + openStationsStructure.getNumberOpenStops2() + "\n");
			//objectives.add(obj);
			//writableOutput += iterOutput + openStationsStructure.stopsPerStationToString();
			//LOGGER.info("\n\n");
			//writableOutput += "\n\n";
			
			//currentSolution.print("partialSolution" + i);
			
			//openStationsStructure.checkFeasibility();
			
			/*
			if (i == 301080 || i == 301079) {
				//currentSolution.print("prevFinalSolution");
				//break;
			}
			*/
			
			
			/*
			if (!currentSolution.checkFeasibility(instance, "partial.log")) {
				LOGGER.info("Unfeasible after a specific station remove");
				break;
			};
			*/
			
			
			
			
			
					
			//currentSolution.stopsPerStationToString();
			//currentSolution.printOpenStationStructure();
			/*
			try {
				Thread.sleep(2*1000);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			*/
			
			/*
			if (i == 81408) {
				break;
			}
			*/
			
			/*
			if (obj == 75) {
				currentSolution.normalVars.writeJson();
				break;
			}
			*/
			/*
			if(!currentSolution.areAllOpenStationsCharging()) {
				break;//System.exit(0);;
			};
			*/
			/*
			if (!currentSolution.checkOpenStructureIntegrity()) {
				break;
			}
			*/
			/* No inconsistences detected
			if (!currentSolution.checkOpenStopsConsistency()) {
				System.exit(0);
				break;
			}
			*/
			
			
							
		}
		
		System.out.printf("Current Selected %s times\n", numCurrentSolutionSelected);
		System.out.printf("Number of buses: %s/%s \n", instance.b, currentSolution.normalVars.c.length);
		System.out.printf("Number of buses with open stations: %s\n", currentSolution.getNumBusesWithOpenStations());
		
		System.out.print("Curent Number of open stations: " + currentSolution.getNumberOpenStations() + "\n");
		if (bestCurrentSolution != null) {
			System.out.println("Overall Best Solution: " + bestSolutionObj + "\n");
			System.out.println("Current Best Solution: " + bestCurrentSolution.getNumberOpenStations() + "\n");
		}
		
		/*
		int o = 0;
		for (Integer bs: bjs) {
			//System.out.print(bs + ",");
			/*
			if (o == 1000) {
				break;
			}
			o++;
		}
		System.out.println();
		*/
		
		//writer.print(writableOutput);
		
		currentSolution.computeNumberOpenStations();
		currentSolution.print("finalSolution");
		System.out.println("Final solution is Feasible: " + currentSolution.checkFeasibility(instance, "feasibility_final.log"));
		
		if (bestCurrentSolution != null) {
			bestCurrentSolution.computeNumberOpenStations();
			bestCurrentSolution.print("finalBestSolution");
			System.out.println("Final solution is Feasible: " + bestCurrentSolution.checkFeasibility(instance, "feasibility_current_best.log"));
			
			overallBestCurrentSolution.computeNumberOpenStations();
			if(!overallBestCurrentSolution.checkOpenStationsConsistency()) {
				System.out.println("Inconsistency in the number of open stations");
				System.exit(0);
			}
			overallBestCurrentSolution.print("overallBestSolution");
			System.out.println("Overall best solution is Feasible: " + 
					overallBestCurrentSolution.checkFeasibility(instance, "feasibility_overall_best.log"));

			
			overallBestCurrentSolution.writeStationsUse();
			overallBestCurrentSolution.writeResults(maxExecTime/1000, objectives.size());
			overallBestCurrentSolution.writeReducedResults(maxExecTime/1000, bestSolutionObj, timeBestObj);
			
			//overallBestCurrentSolution.printOpenStationStructure();
			//overallBestCurrentSolution.stopsPerStationToString();
		}
		
		writeEvolution(objectives, "objectives.csv");
		//writeEvolution(addedEnergy, "wastedEnergy.csv");
		//writeEvolution(addedEnergyFromStructure, "addedEnergyFromStructure.csv");
		writeIterationTimes(iterationTimesBig, "extended-iters");
		
		//writer.close();
	}
	
	
	/**
	 * @deprecated
	 * @param removeOperators
	 * @param iteration
	 */
	public void removeStation(ArrayList<RemoveOperator> removeOperators, int iteration) {
		for (int i = 0; i < removeOperators.size(); i++) {
			removeOperators.get(i).removeStation();
			currentSolution.print("partialSolution" + iteration + "_" + i);
		}
	}
	
	public void writeEvolutionObjectives() {
		try {
			PrintWriter writer = new PrintWriter("../data/objectives.csv");
			for (Integer obj: objectives) {
				writer.println(obj);
			}
			writer.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public void writeEvolution(ArrayList<Integer> array, String name) {
		try {
			PrintWriter writer = new PrintWriter("../data/" + name);
			for (Integer elm: array) {
				writer.println(elm);
			}
			writer.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public void writeIterationTimes(ArrayList<Long> times, String name) {
		try {
			PrintWriter writer = new PrintWriter("../data/" + name +".csv");
			for (Long t: times) {
				writer.println(t);
			}
			writer.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	
	
}
