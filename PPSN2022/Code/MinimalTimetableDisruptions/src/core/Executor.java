package core;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

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
	
	public HashMap<String, HashMap<String, Double>[][]> runSingleModel(int cMax, int modelSpeed, int instanceSpeed, int restTime, int DTmax,
			int lauras, int maxExecTime, int arrivingWithoutCharging, int robust, int backToPrimary,
			int minCt, int objective, int threads, int cMin, int SM, int chargingRate, String inputFolder,
			HashMap<String, HashMap<String, Double>[][]> initialSolution) {
		//int objective = ModelMTD.CHARGERS_OBJ;
		InstanceMTD instanceMTD = new InstanceMTD(cMax, modelSpeed, instanceSpeed, restTime, DTmax, minCt, inputFolder,
				cMin, SM, chargingRate);
		instanceMTD.readInput();
		//instanceMTD.setInstance8();
		HashMap<String, HashMap<String, Double>[][]> output = null;
		
		if (method.equals("cplex")) {
		
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
					routeVariables.initialSolution = initialSolution;
					
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
			mdtExecution.end();
			output = mdtExecution.routeOutputs;
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
			FeasibilityChecker rc = new FeasibilityChecker(heuInitialSolutionVars, instanceMTD, "feasibility_output");
			System.out.printf("Feasible: %s\n", rc.checkFeasibiliy(heuInitialSolutionVars));
			rc.closeCheckingFile();
			*/
						
			//Random rn = new Random(123456789L);
			//Random rn = new Random(456L);
			//Random rn = new Random(1L);
			Random rn = new Random();
			AddOperator addOperator;
			RemoveOperator removeOperator;
			
			//heuInitialSolutionVars.writeJson();
			
			//System.exit(0);
			
			
			
			if (robust == 1) {
				HeuristicsVariablesSet backupVars = new HeuristicsVariablesSet(instanceMTD);
				/*
				backupVars.readSolution("../data/result.json", "bpath");	
				*/
				GreedyManager greedyManager = new GreedyManager(instanceMTD, heuInitialSolutionVars, backupVars);
				greedyManager.run();
				heuInitialSolutionVars.print("robustInitialPrimary");
				backupVars.print("robustInitialBackup");
				System.out.printf("Initial added energy: %s\n", heuInitialSolutionVars.getTotalAddedEnergy());
				FeasibilityChecker rc = new FeasibilityChecker(heuInitialSolutionVars, backupVars, instanceMTD, "feasibility_greedy_robust.txt");
				System.out.printf("Feasible: %s", rc.checkRobustFeasibility());
				rc.closeCheckingFile();
				//System.exit(0);
				heuInitialSolution = new RobustCompleteSolution(heuInitialSolutionVars, backupVars, instanceMTD);
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
				output = onlyRemove.overallBestCurrentSolution.toCplexOutput();
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
	
	public void runModelInChain(int cMax, int modelSpeed, int instanceSpeed, int restTime, int DTmax,
			int lauras, int maxExecTime, int arrivingWithoutCharging, int robust, int backToPrimary,
			int minCt, int objective, int threads, int cMin, int SM, int chargingRate,
			int warmStart, String inputFolder) {
		
		robust = warmStart == 1 ? 0 : robust;
		
		HashMap<String, HashMap<String, Double>[][]> routeOutputs = runSingleModel(cMax, modelSpeed, instanceSpeed,
				restTime, DTmax, lauras, maxExecTime, arrivingWithoutCharging, robust, backToPrimary, minCt,
				objective, threads, cMin, SM, chargingRate, inputFolder, null);
		
		if (warmStart == 1 && method.equals("cplex")) {
			runSingleModel(cMax, modelSpeed, instanceSpeed,
					restTime, DTmax, lauras, maxExecTime, arrivingWithoutCharging, 1, backToPrimary, minCt,
					objective, threads, cMin, SM, chargingRate, inputFolder, routeOutputs);
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
												runModelInChain(cMax, mSpeed, iSpeed, rTime, dt * 60, lau,
															execTime, awc, rbt, bck, minCt, obj, thr, cMin,
															sms, crs, wst, inputFolder);	
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
