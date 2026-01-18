package core;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;

import greedy.EasiestGreedyHeuristic;
import greedy.GreedyManager;
import greedy.StrategyRechargeMaxAllowed;
import greedy.StrategyRechargeOnlyNeeded;
import heuristics.AddOperator;
import heuristics.CompleteSolution;
import heuristics.HeuristicMTD;
import heuristics.HeuristicsVariablesSet;
import heuristics.RemoveOperator;
import heuristics.RobustAddOperator;
import heuristics.RobustCompleteSolution;
import heuristics.RobustRemoveOperator;
import ilog.concert.IloException;
import modelimplementations.ArrivingWithoutChargingOverlappingModelExtension;
import modelimplementations.ArrivingWithoutChargingRouteModelExtension;
import modelimplementations.BasicOverlappingOnlyPerStopModelExtension;
import modelimplementations.BasicRouteModelExtension;
import modelimplementations.HardWarmstartModelExtension;
import modelimplementations.LaurasModelExtension;
import modelimplementations.MaxEnergyAddedModelExtension;
import modelimplementations.MinChargesModelExtension;
import modelimplementations.MinChargingTimeRouteModelExtension;
import modelimplementations.MinEnergyAddedModelExtension;
import modelimplementations.MinNumberChargersModelExtension;
import modelimplementations.SecurityMarginTimeOverlappingModelExtension;
import modelimplementations.SecurityMarginTimeSequenceModelExtension;
import modelimplementations.StationChargeRouteModelExtension;
import modelimplementations.InstallChargerRouteModelExtension;
import modelimplementations.ResilientBasicOverlappingOnlyPerStopModelExtension;
import modelimplementations.ResilientRobustAllignChargeRouteModelExtension;
import modelimplementations.ResilientRobustTimeRouteModelExtension;
import modelimplementations.ResilientSkipChargeRouteModelExtension;
import modelimplementations.ResilientAllignChargeRouteModelExtension;
import modelimplementations.ResilientTimeOverlappingOnlyPerStopModelExtension;
import modelimplementations.ResilientTimeRouteModelExtension;
import modelimplementations.RobustChargeBackBackupModelExtension;
import modelimplementations.RobustChargeRouteModelExtension;
import modelimplementations.RobustSkipChargeRouteModelExtension;
import modelimplementations.RobustTimeBackBackupModelExtension;
import modelimplementations.RobustTimeRouteModelExtension;
import modelimplementations.SecurityMarginTimeOverlappingOnlyPerStationModelExtension;
import modelimplementations.SecurityMarginTimeSequenceModelExtension;
import modelinterface.ModelExtension;
import modelinterface.OverlappingModelExtension;
import modelinterface.RouteModelExtension;
import utils.FeasibilityChecker;
import utils.OutputForWarm;
import variables.BasicVariablesSet;
import variables.OverlappingOnlyPerStopVariablesSet;
import variables.OverlappingVariablesSet;
import variables.StartingChargeTimeVariablesSet;
import variables.StationChargeVariablesSet;
import variables.StationVariablesSet;

public class Executor {
	
	String[] inputFolders = {"limerick-7-lines"};
	int execTime = 2;
	// Method used to solve the problem
	String method = "cplex";
	
	Map<String, Integer> basicParameterCodes;
	int[][] parameters;
	
	String experimentsFilename = "../data/experiment_parameters.txt";
	
	String experimentSessionTime;
	
	static String checkingFilename;
	
	InstanceMTD instanceMTD;

	public static void main(String[] args) {
		String expFilename = args[0];
		Executor exec = new Executor(expFilename);
		//exec.experimentSessionTime = new Date().toString();
		exec.experimentSessionTime = args[1];
		if (args.length > 2) {
			checkingFilename = args[2];
		}
		exec.readBasicParameters();
		exec.runExperiments();
		//exec.runSingleModel(120000, 30, 30, 0, 15 * 60, 1, 2);
		//exec.runModelInChain(60000, 30, 10);
	}
	
	public Executor(String experimentsFilename) {
		basicParameterCodes = new HashMap<String, Integer>();
		basicParameterCodes.put("cMaxes", 0);
		basicParameterCodes.put("minSpeeds", 1);
		basicParameterCodes.put("insSpeeds", 2);
		basicParameterCodes.put("restTimes", 3);
		basicParameterCodes.put("dtTimes", 4);
		basicParameterCodes.put("lauras", 5);
		basicParameterCodes.put("arrivingWithoutCharging", 6);
		basicParameterCodes.put("robust", 7);
		basicParameterCodes.put("backToPrimary", 8);
		basicParameterCodes.put("minCt", 9);
		basicParameterCodes.put("obj", 10);
		basicParameterCodes.put("threads", 11);
		basicParameterCodes.put("cMins", 12);
		basicParameterCodes.put("securityMargins", 13);
		basicParameterCodes.put("chargingRates", 14);
		basicParameterCodes.put("warmingStart", 15);
		basicParameterCodes.put("solvingMethod", 16);
		this.experimentsFilename = experimentsFilename;
	}
	
	public OutputForWarm runSingleModel(int cMax, int modelSpeed, int instanceSpeed, int restTime, int DTmax,
			int lauras, int maxExecTime, int arrivingWithoutCharging, int robust, int backToPrimary,
			int minCt, int objective, int threads, int cMin, int SM, int chargingRate, String inputFolder,
			OutputForWarm initialSolution, boolean mustFix, int iter) {
		//int objective = ModelMTD.CHARGERS_OBJ;
		
		//instanceMTD.setInstance8();
		OutputForWarm output = null;
		
		if (method.equals("cplex") || method.equals("cplex-heuristic")) {
		
			BasicVariablesSet routeVariables = new BasicVariablesSet();
			routeVariables.setInstance(instanceMTD);
			//OverlappingVariablesSet overlappingVariables = new OverlappingVariablesSet();
			OverlappingOnlyPerStopVariablesSet overlappingVariables = new OverlappingOnlyPerStopVariablesSet();
			overlappingVariables.setInstance(instanceMTD);
			StationVariablesSet stationVariables = new StationVariablesSet();
			stationVariables.setInstance(instanceMTD);
			//StationChargeVariablesSet stationChargeVariables = new StationChargeVariablesSet();
			//stationChargeVariables.setInstance(instanceMTD);
			
			ArrayList<RouteModelExtension> routeModelExtensions = new ArrayList<RouteModelExtension>();
			ArrayList<OverlappingModelExtension> overlappingModelExtensions = new ArrayList<OverlappingModelExtension>();
			routeModelExtensions.add(new BasicRouteModelExtension(routeVariables, stationVariables,  objective));
			routeModelExtensions.add(new MinChargingTimeRouteModelExtension(routeVariables));
			routeModelExtensions.add(new InstallChargerRouteModelExtension(routeVariables, stationVariables));
			//routeModelExtensions.add(new StationChargeRouteModelExtension(routeVariables, stationVariables, stationChargeVariables));
			
			//overlappingModelExtensions.add(new BasicOverlappingModelExtension(overlappingVariables, routeVariables));
			overlappingModelExtensions.add(new BasicOverlappingOnlyPerStopModelExtension(overlappingVariables, routeVariables));
			
			if (arrivingWithoutCharging == 0) {
				routeModelExtensions.add(new SecurityMarginTimeSequenceModelExtension(routeVariables));
				overlappingModelExtensions.add(new SecurityMarginTimeOverlappingOnlyPerStationModelExtension(overlappingVariables,
						routeVariables));
			}
			
			
			if (lauras == 1) {
				System.out.println("Laura's extension activated");
				routeModelExtensions.add(new LaurasModelExtension(routeVariables));
			}			
			
			if (arrivingWithoutCharging == 1) {
				// FOR ARRIVING WITHOUT CHARGING
				System.out.println("Arriving without charging extension activated");
				StartingChargeTimeVariablesSet startingChargeVariables = new StartingChargeTimeVariablesSet();
				startingChargeVariables.setInstance(instanceMTD);
				
				/*
				routeModelExtensions.add(new ArrivingWithoutChargingRouteModelExtension(routeVariables, startingChargeVariables));
				overlappingModelExtensions.add(new ArrivingWithoutChargingOverlappingModelExtension(routeVariables, overlappingVariables,
								startingChargeVariables));
				*/
				/////////////////////////////////////////////////////////////////////////////////////////		
			}
			
			if (initialSolution != null) {
					stationVariables.initialSolution = initialSolution;
					stationVariables.mustFix = mustFix;
					//The warm start is disabled for route variables. See Lauras constraint Class
					routeVariables.initialSolution = initialSolution.routeOutputs.get("");
					routeVariables.mustFix = mustFix;
					
			}
			
			
			if (robust == 1) {
			
				//FOR RESILIENT MODEL
				BasicVariablesSet backupRouteVariables = new BasicVariablesSet();
				backupRouteVariables.setInstance(instanceMTD);
				OverlappingOnlyPerStopVariablesSet backupOverlappingVariables = new OverlappingOnlyPerStopVariablesSet();
				backupOverlappingVariables.setInstance(instanceMTD);
				OverlappingOnlyPerStopVariablesSet backupPrimaryOverlappingVariables = new OverlappingOnlyPerStopVariablesSet();
				backupPrimaryOverlappingVariables.setInstance(instanceMTD);
				backupPrimaryOverlappingVariables.breakBusSymmetries = false;
				
				//For the backup route
				BasicRouteModelExtension backupRouteModel = new BasicRouteModelExtension(backupRouteVariables, stationVariables, objective);
				backupRouteModel.setPrefixVarname("backup");
				backupRouteModel.setSetObjectiveStation(false);
				routeModelExtensions.add(backupRouteModel);
				routeModelExtensions.add(new SecurityMarginTimeSequenceModelExtension(backupRouteVariables));
				routeModelExtensions.add(new MinChargingTimeRouteModelExtension(backupRouteVariables));
				routeModelExtensions.add(new InstallChargerRouteModelExtension(backupRouteVariables, stationVariables));
				overlappingModelExtensions.add(new BasicOverlappingOnlyPerStopModelExtension(backupOverlappingVariables, backupRouteVariables));
				overlappingModelExtensions.add(new SecurityMarginTimeOverlappingOnlyPerStationModelExtension(backupOverlappingVariables,
						backupRouteVariables));
				
				//Combined between backup and primary variables 
				//StationChargeVariablesSet backupStationChargeVariables = new StationChargeVariablesSet();
				//backupStationChargeVariables.setInstance(instanceMTD);
				//routeModelExtensions.add(new StationChargeRouteModelExtension(backupRouteVariables, stationVariables, backupStationChargeVariables));
				////routeModelExtensions.add(new ResilientSkipChargeRouteModelExtension(stationChargeVariables, backupStationChargeVariables));
				routeModelExtensions.add(new RobustSkipChargeRouteModelExtension(routeVariables, backupRouteVariables));
				System.out.println("Robust extension activated");
				// Primary to backup
				routeModelExtensions.add(new RobustChargeRouteModelExtension(routeVariables, backupRouteVariables));
				routeModelExtensions.add(new RobustTimeRouteModelExtension(routeVariables, backupRouteVariables));
				// Backup to primary
				if (backToPrimary == 1) { 
					routeModelExtensions.add(new RobustChargeBackBackupModelExtension(routeVariables, backupRouteVariables));
					routeModelExtensions.add(new RobustTimeBackBackupModelExtension(routeVariables, backupRouteVariables));
				}
				
				overlappingModelExtensions.add(new ResilientBasicOverlappingOnlyPerStopModelExtension(backupPrimaryOverlappingVariables, 
						routeVariables, backupRouteVariables));
				overlappingModelExtensions.add(new ResilientTimeOverlappingOnlyPerStopModelExtension(backupPrimaryOverlappingVariables, 
						routeVariables, backupRouteVariables));
				
				
				if (lauras == 1) {
					routeModelExtensions.add(new LaurasModelExtension(backupRouteVariables));
				}
				//////////////////////////////////////////////////////
				//routeModelExtensions.add(new MinEnergyAddedModelExtension(backupRouteVariables));
				if (initialSolution != null) {
					//The warm start is disabled for route variables. See Lauras constraint Class
					backupRouteVariables.initialSolution = initialSolution.routeOutputs.get("backup");
					backupRouteVariables.mustFix = mustFix;
				}
			}
			
			
			ArrayList<ModelExtension> otherModelExtensions = new ArrayList<ModelExtension>();
			if (instanceMTD.minChargersExtension) {
				System.out.println("Min number of chargers extension activated");
				otherModelExtensions.add(new MinNumberChargersModelExtension(stationVariables));
			}
			
			// To get the minAddedEnergy array
			instanceMTD.computeMinEnergyAddedPerBus();
			routeModelExtensions.add(new MinEnergyAddedModelExtension(routeVariables));
			//routeModelExtensions.add(new MaxEnergyAddedModelExtension(routeVariables));
			//routeModelExtensions.add(new MinChargesModelExtension(routeVariables));
			
			//Hard warmstart
			//otherModelExtensions.add(new HardWarmstartModelExtension(stationVariables));
					
			
			
			LinkedHashMap<String, Integer> enabledModelExtensions = new LinkedHashMap<String, Integer>();
			enabledModelExtensions.put("lauras", lauras);
			enabledModelExtensions.put("arrivingWithoutCharging", arrivingWithoutCharging);
			enabledModelExtensions.put("robust", robust);
			enabledModelExtensions.put("backToPrimary", backToPrimary);
			enabledModelExtensions.put("objective", objective);
			enabledModelExtensions.put("threads", threads);
			
			ModelMTD mdtExecution = new ModelMTD(objective, instanceMTD, maxExecTime, routeModelExtensions,
					overlappingModelExtensions, otherModelExtensions, threads);
			mdtExecution.experimentSessionTime = experimentSessionTime;
			mdtExecution.enabledModelExtensions = enabledModelExtensions;
			double objOut = mdtExecution.solveModel();
			
			
			boolean wasSolved = objOut != -1;
			try {
				mdtExecution.printer.parametersLabel = mdtExecution.printer.parametersLabel + "_" + instanceMTD.b + "_" + iter;
				mdtExecution.writeOutput(wasSolved, iter);
			} catch (IloException e) {
				System.out.println("Error while printing");
				e.printStackTrace();
			}
			
			mdtExecution.end();
			
			ArrayList<Integer> actualBuses = new ArrayList<Integer>();
			for (int b = 0; b < instanceMTD.originalB; b++) {
				if (!instanceMTD.discardedBuses.contains(b)) {
					actualBuses.add(b);
				}
			}		
			output = new OutputForWarm(mdtExecution.routeOutputs, actualBuses, instanceMTD.paths);
		} else if (method.equals("heuristic")) {
			
			CompleteSolution heuInitialSolution;
			instanceMTD.computeMinEnergyAddedPerBus();
			
			HeuristicsVariablesSet heuInitialSolutionVars = new HeuristicsVariablesSet(instanceMTD);
			
			
			
			
			/*
			HeuristicsVariablesSet heuInitialSolutionVars = new HeuristicsVariablesSet(instanceMTD);
			heuInitialSolutionVars.readSolution("../data/result.json", "path");
			//heuInitialSolutionVars.readSolution("../data/worstHeuSolution.json", "path");
			heuInitialSolutionVars.print("readedSolution");
			//heuInitialSolutionVars.computeNumberOpenStations();
			//System.out.println("Read obj: " + heuInitialSolutionVars.numberOpenStations);
			//initialSolution.checkFeasibiliy();
			//FeasibilityChecker rc = new FeasibilityChecker(heuInitialSolutionVars, instanceMTD, "feasibility_output");
			//System.out.printf("Feasible: %s\n", rc.checkFeasibiliy(heuInitialSolutionVars));
			//rc.closeCheckingFile();
			*/
						
			Random rn = new Random(123456789L);
			//Random rn = new Random(456L);
			//Random rn = new Random(1L);
			//Random rn = new Random();
			AddOperator addOperator;
			RemoveOperator removeOperator;
			
			//heuInitialSolutionVars.writeJson();
			
			//System.exit(0);
			
			
			
			if (robust == 1) {
				HeuristicsVariablesSet backupVars = new HeuristicsVariablesSet(instanceMTD);
				
				//backupVars.readSolution("../data/result.json", "bpath");	
				
				GreedyManager greedyManager = new GreedyManager(instanceMTD, heuInitialSolutionVars, backupVars);
				greedyManager.run();
				heuInitialSolutionVars.computeNumberOpenStations();
				backupVars.computeNumberOpenStations();
				heuInitialSolutionVars.print("robustInitialPrimary");
				backupVars.print("robustInitialBackup");
				heuInitialSolutionVars.fixXs();
				backupVars.fixXs();
				System.out.printf("Initial added energy: %s\n", heuInitialSolutionVars.getTotalAddedEnergy());
				FeasibilityChecker rc = new FeasibilityChecker(heuInitialSolutionVars, backupVars, instanceMTD, "feasibility_greedy_robust.txt");
				System.out.printf("Feasible: %s", rc.checkRobustFeasibility());
				rc.closeCheckingFile();
				heuInitialSolution = new RobustCompleteSolution(heuInitialSolutionVars, backupVars, instanceMTD);
				RobustCompleteSolution rcs = (RobustCompleteSolution) heuInitialSolution;
				rcs.writeJson();
				rcs.linkPrimaryBackupOpenStations();
				//System.exit(0);
				addOperator = new RobustAddOperator((RobustCompleteSolution) heuInitialSolution, instanceMTD, rn);	
				removeOperator = new RobustRemoveOperator((RobustCompleteSolution) heuInitialSolution, instanceMTD, rn);
			} else {
				StrategyRechargeOnlyNeeded onlyNeededStrategy = new StrategyRechargeOnlyNeeded(heuInitialSolutionVars, instanceMTD);
				//StrategyRechargeMaxAllowed maxAllowedStrategy = new StrategyRechargeMaxAllowed(heuInitialSolutionVars, instanceMTD);
				EasiestGreedyHeuristic easiestGreedy = new EasiestGreedyHeuristic(instanceMTD, onlyNeededStrategy, heuInitialSolutionVars);
				easiestGreedy.greedyAlgorithm();
				FeasibilityChecker checker = new FeasibilityChecker(heuInitialSolutionVars, instanceMTD, "feasibility_output.txt");
				boolean isFeasible = checker.checkFeasibiliy(heuInitialSolutionVars);
				checker.closeCheckingFile();
				System.out.printf("Initial added energy: %s\n", heuInitialSolutionVars.getTotalAddedEnergy());
				if (!isFeasible) {
					System.out.println("The initial solution is not feasible!!!");
					System.exit(0);
				}
				
				
				//System.exit(0);
				
				heuInitialSolution = new CompleteSolution(heuInitialSolutionVars, instanceMTD);
				addOperator = new AddOperator(heuInitialSolution, instanceMTD, rn);	
				removeOperator = new RemoveOperator(heuInitialSolution, instanceMTD, rn);
			}
			
			System.out.printf("Number of buses with open stations: %s\n", heuInitialSolution.getNumBusesWithOpenStations());
			
			System.out.printf("Wasted energy: %s\n", heuInitialSolution.getWastedEnergy());
			
			//System.exit(0);
			

			try {
				
				heuInitialSolution.checkFeasibility(instanceMTD, "feasibility_initial.log");
				HeuristicMTD onlyRemove = new HeuristicMTD(heuInitialSolution, instanceMTD, addOperator, removeOperator, rn);
				onlyRemove.runHeuristic(maxExecTime);
				//System.out.println("Super final obj: " + onlyRemove.overallBestCurrentSolution.numberOpenStations);
				//output = onlyRemove.overallBestCurrentSolution.toCplexOutput();
				//System.exit(0);
			} catch (Exception e) {
				e.printStackTrace();
			}
		} else if (method.equals("checker")) {
			System.out.println("Checking a solution...");
			HeuristicsVariablesSet primarySolution = new HeuristicsVariablesSet(instanceMTD);
			primarySolution.readSolution(checkingFilename, "path");
			//primarySolution.print("ddddd");
			primarySolution.computeNumberOpenStations();
			System.out.println("Read obj: " + primarySolution.numberOpenStations);
			if (robust == 1) {
				HeuristicsVariablesSet backupSolution = new HeuristicsVariablesSet(instanceMTD);
				backupSolution.readSolution(checkingFilename, "bpath");
				FeasibilityChecker rc = new FeasibilityChecker(primarySolution, backupSolution, instanceMTD, "feasibility_output");
				System.out.printf("Feasible: %s", rc.checkRobustFeasibility());
				rc.closeCheckingFile();
			} else {
				FeasibilityChecker rc = new FeasibilityChecker(primarySolution, instanceMTD, "feasibility_output");
				System.out.printf("Feasible: %s", rc.checkFeasibiliy(primarySolution));
				rc.closeCheckingFile();
			}
		}
		
		return output;
		
	}
	
	/*
	public void runModelInChain(int cMax, int speed, int restTime, int DTmax) {
		int objective = ModelMTD.CHARGERS_OBJ;
		InstanceMTD instanceMTD = new InstanceMTD(cMax, speed, restTime, DTmax, inputFolder);
		instanceMTD.readInput();
		ModelMTD mdtExecution = new ModelMTD(objective, instanceMTD, 180);
		double objOut = mdtExecution.solveModel();
		if (objOut != -1) {
			int[] stationsOutput = mdtExecution.getResult1Index();
			mdtExecution.end();
			for (int i : stationsOutput) {
				//System.out.println(i);
			}
			mdtExecution.setFixedXs(stationsOutput);
			mdtExecution.setObjectiveType(ModelMTD.CHARGES_OBJ);
			mdtExecution.setMaxExcecTime(60);
			mdtExecution.solveModel();
			mdtExecution.end();
		}	
	}
	
	*/
	
	public OutputForWarm runModelInChain(int cMax, int modelSpeed, int instanceSpeed, int restTime, int DTmax,
			int lauras, int maxExecTime, int arrivingWithoutCharging, int robust, int backToPrimary,
			int minCt, int objective, int threads, int cMin, int SM, int chargingRate,
			int warmStart, String inputFolder, boolean mustFix) {
		
		//instanceMTD = new InstanceMTD(cMax, modelSpeed, instanceSpeed, restTime, DTmax, minCt, inputFolder,
		//		cMin, SM, chargingRate);
		//instanceMTD.readInput();
		
		robust = warmStart == 1 ? 0 : robust;
		
		long startTime = System.currentTimeMillis();
		
		OutputForWarm routeOutputs = runSingleModel(cMax, modelSpeed, instanceSpeed,
				restTime, DTmax, lauras, maxExecTime, arrivingWithoutCharging, robust, backToPrimary, minCt,
				objective, threads, cMin, SM, chargingRate, inputFolder, null, mustFix, 0);
		
		if (!this.method.equals("cplex-heuristic")) {
			maxExecTime = (int) Math.round(maxExecTime - ((System.currentTimeMillis() - startTime) / 1000.0) / 60.0);
			System.out.println("newMaxTime: " + maxExecTime);
		}
		
		
		
		
		if (warmStart == 1 && (method.equals("cplex") || method.equals("cplex-heuristic"))) {
			routeOutputs = runSingleModel(cMax, modelSpeed, instanceSpeed,
					restTime, DTmax, lauras, maxExecTime, arrivingWithoutCharging, 1, backToPrimary, minCt,
					objective, threads, cMin, SM, chargingRate, inputFolder, routeOutputs, mustFix, 0);
		}
		/*
		****** For a warm start from a heuristic solution to a cplex execution 
		method = "heuristic";
		HashMap<String, HashMap<String, Double>[][]> routeOutputs = runSingleModel(cMax, modelSpeed, instanceSpeed,
				restTime, DTmax, lauras, maxExecTime, arrivingWithoutCharging, robust, backToPrimary, minCt,
				objective, threads, cMin, SM, chargingRate, inputFolder, null);
		method = "cplex";
		if (warmStart == 1 && method.equals("cplex")) {
			runSingleModel(cMax, modelSpeed, instanceSpeed,
					restTime, DTmax, lauras, maxExecTime, arrivingWithoutCharging, robust, backToPrimary, minCt,
					objective, threads, cMin, SM, chargingRate, inputFolder, routeOutputs);
		}
		*/
		
		/*
		runSingleModel(cMax, modelSpeed, instanceSpeed,
				restTime, DTmax, lauras, maxExecTime, arrivingWithoutCharging, 1, backToPrimary, minCt,
				objective, threads, cMin, SM, chargingRate, null);
		*/
		
		return routeOutputs;
		
		
	}
	
	
	public void runCplexHeuristic(int cMax, int modelSpeed, int instanceSpeed, int restTime, int DTmax,
			int lauras, int maxExecTime, int arrivingWithoutCharging, int robust, int backToPrimary,
			int minCt, int objective, int threads, int cMin, int SM, int chargingRate,
			int warmStart, String inputFolder) {
		
		
		
		ArrayList originalDiscardedList = (ArrayList) instanceMTD.discardedBuses.clone();
		HashMap<Integer, Integer> originalBusesId = (HashMap<Integer, Integer>) instanceMTD.originalBusesId.clone();
		
		int originalNumBuses = instanceMTD.b;
		//originalNumBuses = 367;
		
		OutputForWarm resultingSolution = null;
		HashMap<Integer, Double> firstResultingSolution = null;
		
		Random rd = new Random(938377);
		
		int heuMaxExecTime = 10*60*1000;
		int iterTime = maxExecTime*60*1000;
		boolean shouldKeepGoing = true;
		long startTime = System.currentTimeMillis();
		//for (int i = 0; i < 3; i++) {
		int i = 0;
		while (true) {
			
			
			
			boolean mustFix = false;
			double p = 0.5;
			ArrayList<Integer> discardedByHeuristic = new ArrayList<Integer>();
			int originalNumDiscarded = (int) Math.round(originalNumBuses * (1 - p));
			
			//Random rd = new Random(123456789L);
			//Random rd = new Random(9);
			//Random rd = new Random(36);
			//Random rd = new Random(0);			
			
			while(discardedByHeuristic.size() < originalNumDiscarded) {
				int disB = rd.nextInt(originalNumBuses);
				if (!discardedByHeuristic.contains(disB)) {
					discardedByHeuristic.add(disB);
				}		
			}
			
			//System.out.printf("Number of discarding when want %s of the original: %s\n", p, discardedByHeuristic.size());
			
			
			
			
			
			for ( ; p <= 1; p += 0.25) {
				
				if ((System.currentTimeMillis() - startTime) + iterTime >= heuMaxExecTime) {
					maxExecTime = (int) Math.round(((heuMaxExecTime - (System.currentTimeMillis() - startTime)) / 1000.0) / 60.0);
					System.out.println("newMaxTime: " + maxExecTime);
					if ((System.currentTimeMillis() - startTime) >= heuMaxExecTime || maxExecTime < 1) {
						return;
					}
				}
				
				
				int numDiscarded = (int) (Math.round(originalNumBuses * (1 - p)));
				//int numRemovalsInDiscarded = discardedByHeuristic.size() - numDiscarded;
				
				while(discardedByHeuristic.size() > numDiscarded) {
					int indexB = rd.nextInt(discardedByHeuristic.size());
					discardedByHeuristic.remove(indexB);
				}
				System.out.printf("Number of discarding when want %s of the original: %s\n", p, discardedByHeuristic.size());
				
				ArrayList<Integer> newDiscardedBuses = new ArrayList<Integer>();
				newDiscardedBuses.addAll(originalDiscardedList);
				for (int rawId: discardedByHeuristic) {
					newDiscardedBuses.add(originalBusesId.get(rawId));
				}
				System.out.printf("Discarding size: %s\n", newDiscardedBuses.size());
				instanceMTD.discardedBuses = newDiscardedBuses;
				try {
					instanceMTD.readBuses();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				
				
				
				if (resultingSolution != null) {
					resultingSolution.filterByNewInstance(instanceMTD);
				}
				
				
				OutputForWarm newResultingSolution = null;
				
				if (p == 0.5 && i == 0) {
					newResultingSolution = runModelInChain(cMax, modelSpeed, instanceSpeed,
							restTime, DTmax, lauras, maxExecTime, arrivingWithoutCharging, 1, backToPrimary, minCt,
							objective, threads, cMin, SM, chargingRate, warmStart, inputFolder, mustFix);							
					 
				} else {
					newResultingSolution = runSingleModel(cMax, modelSpeed, instanceSpeed,
							restTime, DTmax, lauras, maxExecTime, arrivingWithoutCharging, 1, backToPrimary, minCt,
							objective, threads, cMin, SM, chargingRate, inputFolder, resultingSolution, mustFix, i);
				}	
				if (newResultingSolution != null && newResultingSolution.routeOutputs != null &&
						newResultingSolution.routeOutputs.size() > 0) {
					resultingSolution = newResultingSolution;
				} else {
					if (resultingSolution != null) {
						resultingSolution.routeOutputs = resultingSolution.routeOutsClone;
					}
				}
				
				mustFix = true;
			}
			i++;
		}
		
	}
	
	public void runCplexHeuristicPerLine(int cMax, int modelSpeed, int instanceSpeed, int restTime, int DTmax,
			int lauras, int maxExecTime, int arrivingWithoutCharging, int robust, int backToPrimary,
			int minCt, int objective, int threads, int cMin, int SM, int chargingRate,
			int warmStart, String inputFolder) {
		
		Set<Entry<String, ArrayList<Integer>>> linesBusesESet = instanceMTD.linesBuses.entrySet();
		List<Entry<String, ArrayList<Integer>>> orderedLines = new ArrayList<Entry<String, ArrayList<Integer>>>(linesBusesESet);
		
			
		ArrayList<Integer> originalDiscardedList = (ArrayList<Integer>) instanceMTD.discardedBuses.clone();
		
		for (int busId: originalDiscardedList) {
			for (Entry<String, ArrayList<Integer>> lineBuses: orderedLines) {
					lineBuses.getValue().remove(Integer.valueOf(busId));
			}
		}
		
		orderedLines.removeIf(line -> line.getValue().size() == 0);
		
		System.out.println("After removing irrelevant lines");
		for (Entry<String, ArrayList<Integer>> line: orderedLines) {
			System.out.printf("%s: %s buses\n", line.getKey(), line.getValue().size());
		}
		
		orderedLines.sort(new Comparator<Entry<String, ArrayList<Integer>>>() {
			@Override
			public int compare(Entry<String, ArrayList<Integer>> arg0, Entry<String, ArrayList<Integer>> arg1) {
				Integer size0 = new Integer(arg0.getValue().size());
				Integer size1 = new Integer(arg1.getValue().size());
				return size0.compareTo(size1);
			}
		});
		
		int heuMaxExecTime = 10*60*1000;
		int iterTime = maxExecTime*60*1000;
		long startTime = System.currentTimeMillis();
		
		OutputForWarm resultingSolution = null;
		
		for (int m = 0; true; m++) {
			boolean mustFix = false;
			int k = 0;
			int i = orderedLines.size() - 1;
			ArrayList<Integer> newDiscardedBuses = new ArrayList<Integer>();
			do {
				
				
				if ((System.currentTimeMillis() - startTime) + iterTime >= heuMaxExecTime) {
					maxExecTime = (int) Math.round(((heuMaxExecTime - (System.currentTimeMillis() - startTime)) / 1000.0) / 60.0);
					System.out.println("newMaxTime: " + maxExecTime);
					if ((System.currentTimeMillis() - startTime) >= heuMaxExecTime || maxExecTime < 1) {
						return;
					}
				}
				
				System.out.printf("Iteration %s\n", k);
				newDiscardedBuses = new ArrayList<Integer>();
				newDiscardedBuses.addAll(originalDiscardedList);
				for (int j = 0; j < i; j++) {
					Entry<String, ArrayList<Integer>> lineBuses = orderedLines.get(j);
					System.out.println("Removing...");
					System.out.println(lineBuses.getKey());
					for (int id: lineBuses.getValue()) {
						System.out.println(id);
						if (!newDiscardedBuses.contains(id)) {
							newDiscardedBuses.add(id);
						}
					}
				}
				instanceMTD.discardedBuses = newDiscardedBuses;
				try {
					instanceMTD.readBuses();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				
				if (resultingSolution != null) {
					resultingSolution.filterByNewInstance(instanceMTD);
				}
				
				
				OutputForWarm newResultingSolution = null;
				if (k == 0 && m == 0) {
					newResultingSolution = runModelInChain(cMax, modelSpeed, instanceSpeed,
							restTime, DTmax, lauras, maxExecTime, arrivingWithoutCharging, 1, backToPrimary, minCt,
							objective, threads, cMin, SM, chargingRate, warmStart, inputFolder, mustFix);
				} else {
					newResultingSolution = runSingleModel(cMax, modelSpeed, instanceSpeed,
							restTime, DTmax, lauras, maxExecTime, arrivingWithoutCharging, 1, backToPrimary, minCt,
							objective, threads, cMin, SM, chargingRate, inputFolder, resultingSolution, mustFix, m);
				}
				if (newResultingSolution != null && newResultingSolution.routeOutputs != null &&
						newResultingSolution.routeOutputs.size() > 0) {
					resultingSolution = newResultingSolution;
				} else {
					if (resultingSolution != null) {
						resultingSolution.routeOutputs = resultingSolution.routeOutsClone;
					}
				}
				k++;
				i = i - k;
				mustFix = true;
			} while (newDiscardedBuses.size() > originalDiscardedList.size());
		}

		
	}
	
	public void runExperiments() {
		
		for (String inputFolder : inputFolders) {
		for (int cMax : parameters[basicParameterCodes.get("cMaxes")]) {
			for (int rTime : parameters[basicParameterCodes.get("restTimes")]) {
			for (int mSpeed : parameters[basicParameterCodes.get("minSpeeds")]) {
				for (int dt : parameters[basicParameterCodes.get("dtTimes")]) {
				for (int iSpeed : parameters[basicParameterCodes.get("insSpeeds")]) {
					for (int lau : parameters[basicParameterCodes.get("lauras")]) {
					for (int awc : parameters[basicParameterCodes.get("arrivingWithoutCharging")]) {
						for (int bck: parameters[basicParameterCodes.get("backToPrimary")]) {
						for (int rbt: parameters[basicParameterCodes.get("robust")]) {
							for (int minCt : parameters[basicParameterCodes.get("minCt")]) {
							for (int obj : parameters[basicParameterCodes.get("obj")]) {
								for (int thr : parameters[basicParameterCodes.get("threads")]) {
								for (int cMin : parameters[basicParameterCodes.get("cMins")]) {
									for (int sms : parameters[basicParameterCodes.get("securityMargins")]) {
									for (int crs : parameters[basicParameterCodes.get("chargingRates")]) {
										for (int wst : parameters[basicParameterCodes.get("warmingStart")]) {
											//runModelInChain(cMax, mSpeed, rTime);	
											if (mSpeed >= iSpeed) {												
												instanceMTD = new InstanceMTD(cMax, mSpeed, iSpeed, rTime, dt * 60, minCt, inputFolder,
														cMin, sms, crs);
												instanceMTD.readBusesIdsPerLine();
												instanceMTD.readInput();
												if (method.equals("cplex-heuristic")) {
													runCplexHeuristic(cMax, mSpeed, iSpeed, rTime, dt * 60, lau,
															execTime, awc, rbt, bck, minCt, obj, thr, cMin,
															sms, crs, wst, inputFolder);
												} else {
													runModelInChain(cMax, mSpeed, iSpeed, rTime, dt * 60, lau,
																execTime, awc, rbt, bck, minCt, obj, thr, cMin,
																sms, crs, wst, inputFolder, false);
												}												
											}
										}
									}}
								}}				
							}}
						}}
					}}
				}}
			}}
		}}
	}
	
	public void readBasicParameters() {
		parameters = new int[basicParameterCodes.size()][];
		try {
			FileReader fr = new FileReader(experimentsFilename);
			BufferedReader br = new BufferedReader(fr);
			String line;
			line = br.readLine();
			String inputFoldersString = line.split("#")[0].trim();
			String[] rawInputFolders = inputFoldersString.split(",");
			inputFolders = new String[rawInputFolders.length];
			for (int i = 0; i < rawInputFolders.length; i++) {
				inputFolders[i] = rawInputFolders[i].trim();
			}
			line = br.readLine();
			execTime = Integer.parseInt(line.split("#")[0].trim());
			line = br.readLine();
			method = line.split("#")[0].trim();

			int i = 0;
			
			while((line = br.readLine()) != null) {
				String[] values = line.split("#")[0].split(",");
				parameters[i] = new int[values.length];
				fillParameter(values, parameters[i]);
				i++;
			}
			br.close();
		
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	private void fillParameter(String[] values, int[] parameter) {
		for (int i = 0; i < parameter.length; i++ ) {
			parameter[i] = Integer.parseInt(values[i].trim());
		}
	}

}
