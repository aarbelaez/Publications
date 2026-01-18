package core;


import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.internal.Streams;
import com.google.gson.stream.JsonReader;

import heuristics.HeuristicsVariablesSet;
import ilog.concert.IloException;
import ilog.concert.IloIntVar;
import ilog.concert.IloLinearNumExpr;
import ilog.concert.IloNumExpr;
import ilog.concert.IloNumVar;
import ilog.cplex.IloCplex;
import ilog.cplex.IloCplex.UnknownObjectException;
import modelinterface.ModelExtension;
import modelinterface.OverlappingModelExtension;
import modelinterface.RouteModelExtension;
import utils.PrinterMTD;
import utils.FeasibilityChecker;
import utils.ToolsMTD;
import ilog.cplex.NumVarAlreadyInLPMatrixException;


public class ModelMTD {
	
	IloCplex cplex;
	
	InstanceMTD instance;
	
	// Execution configuration
	/**
	 * Execution's duration in minutes
	 */
	int maxExcecTime = 5;
	
	/**
	 * For minimizing x variable
	 */
	public static int CHARGERS_OBJ = 0;
	/**
	 * For minimizing deviationTime variable
	 */
	public static int TIME_OBJ = 1;
	/**
	 * For minimizing xBStop variable
	 */
	public static int CHARGES_OBJ = 2;
	/**
	 * For minimizing xBStation variable
	 */
	public static int CHARGERS_PER_STATION_OBJ = 3;
	/**
	 * For minimizing both x and xBStop variables
	 */
	public static int BOTH_CHARGERS_CHARGES_OBJ = 4;
	/**
	 * For minimizing the max deviation of each bus
	 */
	public static int MINIMAX_TIME_OBJ = 5;
	
	/**
	 * Fixed values for xs
	 */
	int[] fixedXs = null;
		
	String experimentSessionTime;
	
	int objectiveType;
	
	
	
	ArrayList<RouteModelExtension> routeModelExtensions;
	ArrayList<OverlappingModelExtension> overlappingModelExtensions;
	ArrayList<ModelExtension> otherModelExtensions;
	
	/**
	 * Whether model extensions are on or off
	 */
	LinkedHashMap<String, Integer> enabledModelExtensions;
	
	HashMap<String, HashMap<String, Double>[][]> routeOutputs;
	
	PrinterMTD printer;
	
	/**
	 *Number of threads to execute cplex
	 */
	int threads;
	
	int numPairsForOverlapping = 0;
	

	
	/*
	public static void main(String[] args) {
		
		
		
		int objective = ModelMTD.CHARGES_OBJ;
		InstanceMTD instanceMTD = new InstanceMTD();
		instanceMTD.readInput();
		ModelMTD mdtExecution = new ModelMTD(objective, instanceMTD);
		//mdtExecution.readInput();
		// TOY INSTANCES
		//mdtExecution.setInstance1();
		//mdtExecution.setInstance2();
		//mdtExecution.setInstance3();		
		double objOut = mdtExecution.solveModel();
		if (objOut != -1) {
			int[] stationsOutput = mdtExecution.getResult1Index();
			mdtExecution.end();
			for (int i : stationsOutput) {
				System.out.println(i);
			}
			mdtExecution.setFixedXs(stationsOutput);
			mdtExecution.setObjectiveType(ModelMTD.CHARGES_OBJ);
			mdtExecution.solveModel();
			mdtExecution.end();
		}
		
		
		
		
	}
	*/
	
	public ModelMTD(int objType, InstanceMTD instanceMTD, int maxExcecTime, ArrayList<RouteModelExtension> rme,
			ArrayList<OverlappingModelExtension> ome, ArrayList<ModelExtension> otherMe, int threads) {
		this.objectiveType = objType;
		this.instance = instanceMTD;
		this.maxExcecTime = maxExcecTime;
		this.routeModelExtensions = rme;
		this.overlappingModelExtensions = ome;
		this.otherModelExtensions = otherMe;
		this.threads = threads;
		
		for (int bu = 0; bu < instance.b; bu++) {
			System.out.printf("Bus %s: %s\n", bu, instance.paths[bu].length);
		}
		
	}
	
public double solveModel() {
		
		double objOut = -1;
		// MODEL
		try {
	    	cplex = new IloCplex();
	    	
			//BASIC CONSTRAINTS
			System.out.println("Building basic constraints");
			for (RouteModelExtension rme : routeModelExtensions) {
				rme.setCplex(cplex);
				rme.setInstance(instance);
				rme.defineVariables(cplex);
				for (int b = 0; b < instance.b; b++) {
					rme.addConstraintsPerBus(b);
					for (int k = 1; k < instance.paths[b].length; k++) {
						int m = k - 1;
						int i = instance.paths[b][k];
						int j = instance.paths[b][m];
						rme.addConstraintsPerStop(b, k, m, i, j);
					}	
					for (int i = 0; i < instance.n; i++) {
						rme.addConstraintsPerStation(b, i);
					}
					for (int k = 0; k < instance.paths[b].length; k++) {
						for (int m = 0; m < instance.paths[b].length; m++) {
							int i = instance.paths[b][k];
							int j = instance.paths[b][m];
							rme.addConstraintsBetweenSeparatedStops(b, k, m, i, j);
						}
					}
				}		
				System.out.println(rme.getClass());
				rme.setWarmStarts();
			}
			
			//NON-OVERLAPPING CONSTRAINTS
			System.out.println("Building non-overlapping constraints");
			for (OverlappingModelExtension ome : overlappingModelExtensions) {
				ome.setCplex(cplex);
				ome.setInstance(instance);
				//Variables Z and z
				ome.defineVariables(cplex);
				//System.out.println(ome.getClass().getName());
				//System.out.println(ome.getBreakBusSymmetries());
				//For each pair of buses
				for (int b = 0; b < instance.b; b++) {
					for (int d = 0; d < (ome.getBreakBusSymmetries() ? b : instance.b); d++) {
						//Constraints (8) to (10)
						for (int i = 0; i < instance.n; i++) {
							for (int j = i; j <= i; j++) {
								if (j == i && b != d) {
									if (ToolsMTD.stationInPath(b, i, instance.paths) && ToolsMTD.stationInPath(d, j, instance.paths) ) {
										ome.addConstraintsPerStation(b, d, i, j);
									}
								}
							}
						}
						//Constraints (11) to (13)
						for (int k = 0; k < instance.paths[b].length; k++) {
							for (int m = 0; m < instance.paths[d].length; m++) {
								int i = instance.paths[b][k];
								int j = instance.paths[d][m];
								if (i == j && b != d && 
										(Math.abs(instance.originalTimetable[b][k] - instance.originalTimetable[d][m]) <= 
											(instance.overlappingTimeDistance))) {
									ome.addConstraintsPerStop(b, d, k, m, i, j);
									numPairsForOverlapping++;
									/*
									int timeDistance = Math.abs(instance.originalTimetable[b][k] - instance.originalTimetable[d][m]);
									if (timeDistance > 28*60) {
										int mins = (int) Math.round(instance.originalTimetable[b][k] / 60);
										System.out.printf("b:%s, k:%s, i:%s, ot:%s\n", b, k, i,
												ToolsMTD.minutesToStringTime(mins));
										mins = (int) Math.round(instance.originalTimetable[d][m] / 60);
										System.out.printf("b:%s, k:%s, i:%s, ot:%s\n\n", d, m, j,
												ToolsMTD.minutesToStringTime(mins));
									}
									*/
								}
							}
						}
					}
				}	
			}
			
			//GLOBAL CONSTRAINTS
			for (ModelExtension otherMe : otherModelExtensions) {
	    		otherMe.setCplex(cplex);
	    		otherMe.setInstance(instance);
	    		otherMe.addConstraintsGlobally();
	    	}
			
			//setWarmStart();
			// OBJECTIVE
			IloLinearNumExpr objective = cplex.linearNumExpr();
			for (RouteModelExtension rme : routeModelExtensions) {
				rme.addObjective(objective);
			}
			cplex.addMinimize(objective);
			
			
			printer = new PrinterMTD(instance, enabledModelExtensions, objectiveType, cplex, experimentSessionTime);
			String logFileName = String.format("../data/cplex-%s.log", experimentSessionTime.replaceAll("\\W", "-"));
			String logHeader = printer.logHeader();
			System.out.println(logHeader);
			OutputStream logFile = null;
			if (System.getProperty("os.name").startsWith("Windows")) {
				logFile = new FileOutputStream(logFileName, true);
				printer.writeInputInfoLog(logFile, logHeader);
				cplex.setOut(logFile);
			}
			cplex.setParam(IloCplex.Param.TimeLimit, maxExcecTime*60);	
			//cplex.setParam(IloCplex.Param.MIP.Tolerances.Integrality, 0.00000000000001) ;
			//cplex.setParam(IloCplex.Param.Simplex.Tolerances.Optimality, 0.000000001) ;
			if (threads != 0) {
				cplex.setParam(IloCplex.Param.Threads, threads);
			}
			System.out.println("Solving...");
			double start = cplex.getCplexTime();
			
			double elapsed = 0;
			boolean wasSolved = false;
        	if (cplex.solve()) {
        		objOut = cplex.getObjValue();
        		System.out.println("obj = " + objOut);
        		wasSolved = true;
        			
        	} else {
        		System.out.println("Not solved?");
        		wasSolved = false;
        	}   
        	elapsed = cplex.getCplexTime() - start;
        	writeOutput(elapsed, wasSolved);  
        	if (logFile != null) {
        		logFile.close();
        	}
        	//cplex.close();
			
		} catch (IloException | IOException e) {
			e.printStackTrace();
		}	
		
		return objOut;
		
	}
	
	
	
	
	/*
	private void setFixedVariables() throws IloException {
		if (fixedXs != null) {
			for (int i : fixedXs) {
				if (i == 1) {
					x[i].setLB(1);
				} else {
					x[i].setUB(0);
				}
			}
		}
	}
	*/
	
	/*
	private void setWarmStart() throws IloException {
		if (fixedXs != null) {
			IloNumVar[] startVar = new IloNumVar[instance.n];
		    double[] startVal = new double[instance.n];
		    int numOpened = 0;
		    for (int i = 0; i < instance.n; i++) {
		    	startVar[i] = x[i];
		    	startVal[i] = fixedXs[i];
		    	numOpened += fixedXs[i];
			}    
		    System.out.println("Warming start with " + numOpened + " chargers");
		    IloLinearNumExpr warmStartConstraint = cplex.linearNumExpr();
		    for (int i = 0; i < instance.n; i++) {
		    	warmStartConstraint.addTerm(1.0, x[i]);
		    }
			cplex.addLe(warmStartConstraint, numOpened); 
			
			cplex.addMIPStart(startVar, startVal);
		
		}
	}
	
	*/
	
	
	
	/*
	public int[] getResult1Index() {
		int[] result = null;
		if (objectiveType == CHARGERS_OBJ) {
			result = new int[instance.n];
			for (int i = 0; i < instance.n; i++) {
				try {
					result[i] = (int) Math.round(cplex.getValue(x[i]));
				} catch (UnknownObjectException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				} catch (IloException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		} 
		return result;
	}
	
	
	public int[][] getResult2Indices() {
		int[][] result = null;
		result = new int[instance.b][];
		for (int i = 0; i < instance.b; i++) {
			result[i] = new int[instance.paths[i].length];
			for (int j = 0; j < instance.paths[i].length; j++) {
				try {
					if (objectiveType == TIME_OBJ) {	
						result[i][j] = (int) Math.round(cplex.getValue(deviationTime[i][j])); 
					} else if (objectiveType == CHARGES_OBJ) {
						result[i][j] = (int) Math.round(cplex.getValue(xBStation[i][j]));
					}
				} catch (IloException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}		
			}
		}
		return result;
	}
	
	*/
	
	public void writeOutput(double elapsedTime, boolean solved) throws UnknownObjectException, IloException {
		
			
		/*
		writer.println("\nStations opened");
		
		try {
			for (int i = 0; i < instance.n; i++) {
				
				int opened = (int) Math.round(cplex.getValue(x[i]));
				if (opened == 1)
					writer.printf("x[%s]=%s\n", i, opened);
			}
		} catch (UnknownObjectException e) {
			System.out.println("Error printing xs");
			//e.printStackTrace();
		}
		writer.println();
		
		*/
	
		if (solved) {
			routeOutputs = new HashMap<String, HashMap<String, Double>[][]>();
			
			for (RouteModelExtension rme: routeModelExtensions) {
				if (!routeOutputs.containsKey(rme.getPrefixVarname())) {
					HashMap<String, Double>[][] allVarVals = new HashMap[instance.b][];
					for (int b = 0; b < instance.b; b++) {
		    			allVarVals[b] = new HashMap[instance.paths[b].length];
		    			for (int k = 0; k < instance.paths[b].length; k++) {
		    				int i = instance.paths[b][k];
	    					HashMap<String, Double> varVals = rme.getPrintableVarVals(b, k, i);
	    					allVarVals[b][k] = varVals;
		    			}
					}
		    		routeOutputs.put(rme.getPrefixVarname(), allVarVals);
					
				}
			}
		
			printer.setRouteOutputs(routeOutputs);
			printer.printRawRoutes();
			printer.writeChargingStatistics();
			printer.writeGephiOutput();
			printer.writeStationsUse();
			
			
			String filename = "checker-" + printer.parametersLabel;
			boolean feasible = true;
			if (enabledModelExtensions.get("robust") == 0) {
				System.out.println("Checking solution feasibility");
				HeuristicsVariablesSet heuristicVars = new HeuristicsVariablesSet(instance);
				heuristicVars.readSolutionFromCplexOutput(routeOutputs, "");
				FeasibilityChecker checker = new FeasibilityChecker(heuristicVars, instance, filename);
				feasible = checker.checkFeasibiliy(heuristicVars);
				checker.closeCheckingFile();
				heuristicVars.print("yet-another-output");
				heuristicVars.writeJson();
			} else {
				System.out.println("Checking solution feasibility for robust model");
				HeuristicsVariablesSet primaryVars = new HeuristicsVariablesSet(instance);
				primaryVars.readSolutionFromCplexOutput(routeOutputs, "");
				HeuristicsVariablesSet backupVars = new HeuristicsVariablesSet(instance);
				backupVars.readSolutionFromCplexOutput(routeOutputs, "backup");
				backupVars.print("backupSolution");
				FeasibilityChecker robustChecker = new FeasibilityChecker(primaryVars, backupVars, instance, filename);
				feasible = robustChecker.checkRobustFeasibility();
				robustChecker.closeCheckingFile();
			}
			System.out.println("Feasible: " + feasible);
			if (!feasible) {
				System.out.println("######## NOT Feasible ##########");
				System.out.println("check on " + filename);
			};	
			
			
			
			
		}		
		printer.writeResults(elapsedTime, solved, numPairsForOverlapping);
		printer.writeReducedResults(elapsedTime, solved, numPairsForOverlapping);
		
		
		
		
		/*
		System.out.println("\n");
		for (int bu = 0; bu < instance.b; bu++) {
			for (int d = 0; d < bu; d++) {
				for (int k = 0; k < instance.n; k++) {
					for (int m = 0; m < instance.n; m++) {
						if (k == m && bu != d && 
								(ToolsMTD.stationInPath(bu, k, instance.paths) &&
										ToolsMTD.stationInPath(d, m, instance.paths))) {
							writer.printf("Z[%s][%s][%s][%s] = ", bu, d, k, m);
							writer.printf(cplex.getValue(Z.get(bu).get(d).get(k).get(m)) + "\n");
						}
					}
				}
			}
		}
		
		writer.println("\nCritical Stops");
		
		for (int bu = 0; bu < instance.b; bu++) {
			for (int k = 0; k < instance.paths[bu].length; k++) {
				boolean bShown = false;
				for (int d = 0; d < bu; d++) {	
					for (int m = 0; m < instance.paths[d].length; m++) {
						int i = instance.paths[bu][k];
						int j = instance.paths[d][m];
						if (i == j && bu != d && 
								(Math.abs(instance.originalTimetable[bu][k] - instance.originalTimetable[d][m]) <= 
									2*instance.DTmax)) {
							int bCharges = (int) Math.round(cplex.getValue(xBStation[bu][i]));
							int dCharges = (int) Math.round(cplex.getValue(xBStation[d][j]));
							if (bCharges==1 && dCharges==1) {
								if (!bShown) {
									writer.println();
									printAStop(bu, k, writer, "");
									bShown = true;
								}
								printAStop(d, m, writer, "\t");
							}
							
						}
					}
				}
			}
		}	
		*/	
	}
	
	
	
	public void setFixedXs(int[] fixedXs) {
		this.fixedXs = fixedXs;
	}
	
	public void setObjectiveType(int obj) {
		this.objectiveType = obj;
	}
	
	public void end() {
		cplex.end();
	}

	public void setMaxExcecTime(int maxExcecTime) {
		this.maxExcecTime = maxExcecTime;
	}
	

}

	
