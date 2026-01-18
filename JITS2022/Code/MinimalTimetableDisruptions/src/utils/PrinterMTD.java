package utils;

import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

import core.InstanceMTD;
import ilog.concert.IloException;
import ilog.cplex.IloCplex;
import modelinterface.RouteModelExtension;

public class PrinterMTD {
	
	/**
	 * The key indicates if the variables are of primary or backup route.
	 * The inner keys are the names of the variables
	 */
	HashMap<String, HashMap<String, Double>[][]> routeOutputs;
	InstanceMTD instance;
	IloCplex cplex;
	Map<Integer, String> objNames;
	int objectiveType;
	String dateWithoutSpaces;
	public String parametersLabel;
	
	/**
	 * Whether model extensions are on or off
	 */
	LinkedHashMap<String, Integer> enabledModelExtensions;
	
	String experimentSessionTime;
	
	public PrinterMTD(InstanceMTD instance, LinkedHashMap<String, Integer> enabledModelExtensions, 
			int objectiveType, IloCplex cplex, String experimentSessionTime) {
		this.enabledModelExtensions = enabledModelExtensions;
		this.instance = instance;
		this.objectiveType = objectiveType;
		this.cplex = cplex;
		this.experimentSessionTime = experimentSessionTime;
		objNames = new HashMap<Integer, String>();
		objNames.put(0, "chargers");
		objNames.put(1, "time");
		objNames.put(2, "charges");
		objNames.put(3, "chargers-station");
		objNames.put(4, "chargers-charges");
		dateWithoutSpaces = experimentSessionTime.replaceAll("\\W", "-");
		
		parametersLabel = String.format("%s_%s_%s_%s_%s_%s_%s_", instance.inputFolder, objNames.get(objectiveType), instance.Cmax,
				instance.modelMinSpeedKmH, instance.restTime / 60, instance.DTmax / 60, instance.minCt / 60);
		String modelExtensions = "";
		for (String key: enabledModelExtensions.keySet()) {
			modelExtensions += enabledModelExtensions.get(key) + "_";
		}
		parametersLabel += modelExtensions + dateWithoutSpaces;
		
	}
	
	public void printRawRoutes() {
		
		Set<Integer> openStations = new HashSet<Integer>();
		
		try {
			PrintWriter writer = new PrintWriter(String.format("../data/output_%s.txt", parametersLabel));
			writer.println("obj = "+ cplex.getObjValue());
			writer.println("gap = "+ cplex.getMIPRelativeGap());
			
			writer.println("\nNumber of stops");
			for (int bu = 0; bu < instance.b; bu++) {
				writer.printf("Bus %s: %s\n", bu, instance.paths[bu].length);
			}
		
			for (int b = 0; b < instance.b; b++) {
				writer.printf("\nBus %s\n", b);
				for (int k = 0; k < instance.paths[b].length; k++) {
					try {
						int i = instance.paths[b][k];
							
								
							writer.println("Bus " + b + ", Stop " + k );
							writer.println("Station " + i + " / " + 
									instance.stationNames.getOrDefault(i, ""));
		
							writer.printf("original[%s][%s]=%s / %s\n",
									b,k,instance.strOriginalTimetable[b][k],instance.originalTimetable[b][k]);
						for (String varsType : routeOutputs.keySet()) {
							String startingCharacter = varsType; 	
						
							HashMap<String, Double> varVals = routeOutputs.get(varsType)[b][k];
							for (String varname: varVals.keySet()) {
								if (varname == "arrivalTime") {
									String arrival = String.valueOf(varVals.get(varname));
									int minutes = (int) Math.round(varVals.get(varname).doubleValue() / 60);
									arrival = ToolsMTD.minutesToStringTime(minutes);
									writer.printf(startingCharacter + " arrival=%s / %s\n",arrival,varVals.get(varname));
								} else {
									writer.printf(startingCharacter + " %s=%s\n",varname,varVals.get(varname));
								}
								if (varname == "x") {
									if (varVals.get(varname) == 1) {
										openStations.add(i);
									}
								}
								
							}
						}
						writer.println();
					} catch (Exception ex) {
						System.out.println("error in " + b + " " + k);
						ex.printStackTrace();
					}
				}
			}
			
			writer.println("Open stations:");
			for (int oStation: openStations) {
				writer.println(oStation);
			}
			
			writer.close();
		} catch (FileNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} catch (Exception ex) {
			ex.printStackTrace();
		}
	}
	
	public String logHeader() {
		String inputInfo = "\n" + new Date().toString() + "\n";
		String modelExtensions = "";
		for (String key: enabledModelExtensions.keySet()) {
			modelExtensions += ", " + key + ": " + enabledModelExtensions.get(key);
		}
		inputInfo += String.format("objective type: %s, CMax: %s, model speed: %s, instance speed: %s\n"
				+ "rest time: %s, DTmax: %s, minCt: %s%s\n",
				objNames.get(objectiveType), instance.Cmax, instance.modelMinSpeedKmH, instance.instanceMinSpeedKmH,
				instance.restTimeMins, instance.DTmax / 60, instance.minCt / 60, modelExtensions);
		return inputInfo;
	}
	
	public void writeInputInfoLog(OutputStream file, String content) throws IOException {
		byte[] strToBytes = content.getBytes();
		file.write(strToBytes);
	}
	
	public void writeResults(double elapsedTime, boolean solved, int numPairsForOverlapping) throws IloException {
		
		String filename = String.format("../data/results-%s.csv", dateWithoutSpaces);
		FileWriter fw;
		try {
			fw = new FileWriter(filename, true);
		
			String modelExtensions = "";
			for (String key: enabledModelExtensions.keySet()) {
				modelExtensions += enabledModelExtensions.get(key) + ",";
			}
			String inputs =  String.format("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,", new Date().toString(),
					instance.inputFolder, instance.Cmax, instance.Cmin,
					instance.modelMinSpeedKmH, instance.instanceMinSpeedKmH, instance.restTimeMins, instance.DTmax / 60, 
					instance.minCt / 60, instance.b, modelExtensions);
			String outputs = "NO SOLUTION,-,-," + elapsedTime + "\n";
			if (solved) {
				double lowerBound = cplex.getObjValue() * (1 - cplex.getMIPRelativeGap());
				outputs = String.format("%s,%s,%s,%s,,%s,%s,%s\n", cplex.getObjValue(), cplex.getMIPRelativeGap(), lowerBound, 
						elapsedTime, parametersLabel, instance.numStops, numPairsForOverlapping);
			}
			String results = inputs + outputs;
			fw.write(results);
			fw.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
public void writeReducedResults(double elapsedTime, boolean solved, int numPairsForOverlapping) throws IloException {
		
		String filename = String.format("../data/reduced-results-%s.csv", dateWithoutSpaces);
		FileWriter fw;
		try {
			fw = new FileWriter(filename, true);
		
			String inputs =  String.format("%s,%s,%s,",
					instance.inputFolder, instance.Cmax, instance.DTmax / 60);
			String outputs = "NO SOLUTION,-," + elapsedTime + "\n";
			if (solved) {
				outputs = String.format("%s,%s\n", cplex.getObjValue(), elapsedTime / 60.0);
			}
			String results = inputs + outputs;
			fw.write(results);
			fw.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	
	public void writeChargingStatistics() {
		try {
			PrintWriter writer = new PrintWriter(String.format("../data/statistics_output_%s.txt", parametersLabel));
			
			for (int b = 0; b < instance.b; b++) {
				HashMap<String, Set<Integer>> data =  new HashMap<String, Set<Integer>>();
				for (String varsType : routeOutputs.keySet()) {
					data.put(varsType, new HashSet<Integer>());
				}
				data.put("all", new HashSet<Integer>());
				for (int k = 0; k < instance.paths[b].length; k++) {
					int i = instance.paths[b][k];
					for (String varsType : routeOutputs.keySet()) {
						if (routeOutputs.get(varsType)[b][k].get("e") >= 0.005) {
							data.get(varsType).add(i);
						}
						if (routeOutputs.get(varsType)[b][k].get("x") == 1.0) {
							data.get("all").add(i);
						}
						
					}
				}
				
				writer.println("Bus " + b);
				for (String varsType : routeOutputs.keySet()) {
					writer.print(varsType + ": ");
					for (Integer station : data.get(varsType)) {
						writer.print(station + " ");
					}
					//writer.print("- ");
				}
				/*
				writer.print("all: ");
				for (Integer station : data.get("all")) {
					writer.print(station + " ");
				}
				*/
				writer.println();
				
				
			}
			writer.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
			
	}
	
	public void writeGephiOutput() {
		
		//Set of backup and primary stations
		HashMap<String, Set<Integer>> data =  new HashMap<String, Set<Integer>>();
		data.put("backup", new HashSet<Integer>());
		data.put("primary", new HashSet<Integer>());
		for (int b = 0; b < instance.b; b++) {
			for (int k = 0; k < instance.paths[b].length; k++) {
				int i = instance.paths[b][k];
				for (String varsType : routeOutputs.keySet()) {			
					if (routeOutputs.get(varsType)[b][k].get("e") >= 0.5) {
						if (varsType.equals("")) {
							data.get("primary").add(i);
						} else if (varsType.equals("backup")) {
							data.get("backup").add(i);
						}
					}
					
				}
			}
		}
		
		//For each station its set of buses
		HashMap<Integer, Set<Integer>> busesStation =  new HashMap<Integer, Set<Integer>>();
		String busHeaders = "";
		for (int b = 0; b < instance.b; b++) {
			
			for (int k = 0; k < instance.paths[b].length; k++) {
				int i = instance.paths[b][k];
				if (!busesStation.containsKey(i)) {
					busesStation.put(i, new HashSet<Integer>());
				}
				busesStation.get(i).add(b);
			}
			String busEnd = ",";
			if (b == instance.b - 1) {
				busEnd = "";
			}
			busHeaders += "bus" + b + busEnd;
		}
		
		/*
		HashMap<Integer, Integer> typeStation =  new HashMap<Integer, Integer>();
		HashMap<Integer, Integer> busStation =  new HashMap<Integer, Integer>();
		*/
		try {
			PrintWriter writer = new PrintWriter(String.format("../data/gephi_nodes_%s.csv", parametersLabel));
			writer.println("id,station,bus," + busHeaders);
			for (int i = 0; i < instance.n; i++) {
				
				boolean isPrimary = data.get("primary").contains(i);
				boolean isBackup = data.get("backup").contains(i);
				int type = 0;
				if (isPrimary && isBackup ) {
					type = 10;
				} else if (isPrimary) {
					type = 8;
				} else if (isBackup) {
					type = 5;
				}
				//typeStation.put(i, type);
				
				int bus = -1;
				if (busesStation.containsKey(i)) {
					if (busesStation.get(i).size() > 1) {
						bus = 1000;
					} else {
						for(int b : busesStation.get(i)) {
							bus = b;
						}
					}
				}
				String busCommunities = "";
				for (int b = 0; b < instance.b; b++) {
					String end = ",";
					if (b == instance.b - 1) {
						end = "";
					}
					if (busesStation.containsKey(i) && busesStation.get(i).contains(b)) {
						busCommunities += "1" + end;
					} else {
						busCommunities += "0" + end;
					}
				}
				
				//busStation.put(i, bus);
				writer.println(String.format("%s,%s,%s,%s", i, type, bus, busCommunities));
			}
			writer.close();
			
			PrintWriter writer2 = new PrintWriter(String.format("../data/gephi_edges_%s.csv", parametersLabel));
			writer2.println("Source,Target,distance");
			for (int b = 0; b < instance.b; b++) {
				for (int k = 0; k < instance.paths[b].length - 1; k++) {
					int j = instance.paths[b][k];
					int i = instance.paths[b][k + 1];
					writer2.println(String.format("%s,%s,%s", j, i, instance.D[j][i]));
				}
			}
			writer2.close();
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public void writeStationsUse() {
		
		
		HashMap<Integer, ArrayList<ChargesRange>> chargeRanges = new HashMap<Integer, ArrayList<ChargesRange>>();
		for (int b = 0; b < instance.b; b++) {
			for (int k = 0; k < instance.paths[b].length; k++) {
				int stationId = instance.paths[b][k];
				HashMap<String, Double> varVals = routeOutputs.get("")[b][k];
				double ct = varVals.get("ct").doubleValue();
				double t = varVals.get("arrivalTime").doubleValue();
				if (ToolsMTD.round(ct) > 0) {
					double iniTime = t;
					double finTime = t + ct;
					ChargesRange range = new ChargesRange(iniTime, finTime);
					if (!chargeRanges.containsKey(stationId)) {
						chargeRanges.put(stationId, new ArrayList<ChargesRange>());
					}
					chargeRanges.get(stationId).add(range);
				}
			}
		}
		try {
			PrintWriter writer = new PrintWriter("../data/station_use.csv");
			for (int station: chargeRanges.keySet()) {
				for (ChargesRange cr: chargeRanges.get(station)) {
					writer.printf("%s,%s,%s\n", station, cr.initialTime, cr.finishTime);
				}
			}
			writer.close();
			
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public void setRouteOutputs(HashMap<String, HashMap<String, Double>[][]> routeOutputs) {
		this.routeOutputs = routeOutputs;
	}
	
	

}
