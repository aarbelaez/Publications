package heuristics;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import java.util.TreeSet;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonIOException;
import com.google.gson.JsonObject;
import com.google.gson.internal.Streams;
import com.google.gson.stream.JsonReader;

import core.InstanceMTD;
import ilog.concert.IloException;
import ilog.concert.IloIntVar;
import ilog.concert.IloNumVar;
import utils.ChargesRange;
import utils.ToolsMTD;

public class HeuristicsVariablesSet {
	
	InstanceMTD instance;
	//BASIC VARIABLES
	public double[][] arrivalTime;
	public double[][] deviationTime;
	/**
	 * remaining battery 
	 */
	public double[][] c;
	/**
	 * Energy added during charging
	 */
	public double[][] e;
	/**
	 * charging time
	 */
	public double[][] ct;
	/**
	 * xBStation[i][j] determines whether bus i charges at station j 
	 */
	public boolean[][] xBStation;
	/**
	 * xBStation[i][j] determines whether bus i charges at stop j 
	 */
	public boolean[][] xBStop;
	/**
	 * x[i] determines whether a charger is installed in station i
	 */
	public boolean[] x;
	/**
	 * Number of open stations
	 */
	/**
	 * Deviation time
	 */
	public double[][] dt;
	public int numberOpenStations;
	
	public HeuristicsVariablesSet(InstanceMTD instance) {
		this.instance = instance;
		numberOpenStations = instance.n;
		defineVariables();
	}
	
	public HeuristicsVariablesSet(InstanceMTD instance, HeuristicsVariablesSet toCopy) {
		this.instance = instance;
		numberOpenStations = instance.n;
		defineVariables();
		
		for (int b = 0; b < instance.b; b++) {
			arrivalTime[b] = toCopy.arrivalTime[b].clone();
			deviationTime[b] = toCopy.deviationTime[b].clone();
			c[b] = toCopy.c[b].clone();
			e[b] = toCopy.e[b].clone();
			ct[b] = toCopy.ct[b].clone();
			xBStation[b] = toCopy.xBStation[b].clone();
			xBStop[b] = toCopy.xBStop[b].clone();
		}
		
		x = toCopy.x.clone();
	}
	
	public void defineBasicVariablesArrays() {
		//VARIABLES		
		arrivalTime =  new double[instance.b][];
		deviationTime = new double[instance.b][];
		// remaining battery 
		c = new double[instance.b][];
		// Energy added during charging
		e = new double[instance.b][];
		// charging time
		ct = new double[instance.b][];
		// charging station decision per bus 
		xBStation = new boolean[instance.b][];
		for (int i = 0; i < instance.b; i++) {
			xBStation[i] = new boolean[instance.n]; 
		}
		// charging stop decision per bus 
		xBStop = new boolean[instance.b][];
		x = new boolean[instance.n];
		dt = new double[instance.b][];
		
	}
	
	public void defineVariables() {
		defineBasicVariablesArrays();
		for (int b = 0; b < instance.b; b++) {		
			defineBasicVariablesPerBus(b);
			for (int i = 0; i < instance.paths[b].length; i++) {
				defineBasicVariablesPerBusPerStop(b, i);
			}
		}	
		
	}
	
	public void defineBasicVariablesPerBus(int b) {
		arrivalTime[b] = new double[instance.paths[b].length]; 
		deviationTime[b] = new double[instance.paths[b].length];
		c[b] = new double[instance.paths[b].length]; // CONSTRAINT (1)
		e[b] = new double[instance.paths[b].length]; 
		ct[b] = new double[instance.paths[b].length]; 
		xBStop[b] = new boolean[instance.paths[b].length]; 
		dt[b] = new double[instance.paths[b].length];
		
	}
	
	public void print(String filename) {
		try {
			PrintWriter writer = new PrintWriter("../data/" + filename + ".txt");
			writer.printf("obj=%s\n\n", numberOpenStations);
			instance.addedEnergies = new ArrayList<Double>();
			double[] addedEnergiesPerStation = new double[instance.n];
			for (int b = 0; b < instance.b; b++) {		
				double busEnergy = 0;
				for (int k = 0; k < instance.paths[b].length; k++) {
					int i = instance.paths[b][k];
					writer.printf("x[%s]=%s\n", i,  x[i]);
					writer.printf("xBStation[%s][%s]=%s\n", b, i,  xBStation[b][i]);
					writer.printf("xBStop[%s][%s]=%s\n", b, k,  xBStop[b][k]);
					writer.printf("c[%s][%s]=%s\n", b, k,  c[b][k]);
					writer.printf("e[%s][%s]=%s\n", b, k,  e[b][k]);
					writer.printf("ct[%s][%s]=%s\n", b, k,  ct[b][k]);	
					writer.printf("arrivalTime[%s][%s]=%s\n", b, k,  arrivalTime[b][k]);	
					writer.printf("originalTime[%s][%s]=%s\n\n", b, k,  instance.originalTimetable[b][k]);
					busEnergy += e[b][k];
					addedEnergiesPerStation[i] += e[b][k]; 
				}
				instance.addedEnergies.add(Math.floor(busEnergy));
			}
			for (int b = 0; b < instance.b; b++) {	
				writer.printf("Energy added in the route of bus %s: %s\n", b, instance.addedEnergies.get(b));
			}
			writer.println();
			for (int i = 0; i < instance.n; i++) {	
				writer.printf("Energy added in station %s: %s\n", i, addedEnergiesPerStation[i]);
			}
			writer.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
		
	public double getDeltaT(int b, int k) {
		return arrivalTime[b][k] - instance.originalTimetable[b][k];
	}
	
	public void readSolution(String filename, String pathType) {
		try {
			
			double energyAdded = 0;
			
			JsonElement tmpJsonElement = Streams.parse(new JsonReader(new FileReader(filename)));
			
			JsonObject lineJsonObject = tmpJsonElement.getAsJsonObject();
			JsonArray lineBuses = lineJsonObject.get("buses").getAsJsonArray();
			
			//Set<Integer> openStations = new TreeSet<Integer>();
			
			int newBusId = 0;
					
			for (JsonElement busObject : lineBuses) {
				//System.out.println(busObject);
				JsonObject busJsonObject = busObject.getAsJsonObject();
				int busId = busJsonObject.get("bus").getAsInt();
				// The following statement must be executed only when the reading solution does not have discarded buses
				if (instance.discardedBuses.contains(busId)) {
					continue;
				}
				JsonArray busPath = busJsonObject.get(pathType).getAsJsonArray();
				
				
				for (int k = 0; k < busPath.size(); k++) {
					//System.out.println(busId + " " + k);
					JsonElement stop = busPath.get(k);
					JsonObject stopJsonObject = stop.getAsJsonObject();
					arrivalTime[newBusId][k] = stopJsonObject.get("t").getAsDouble();
					c[newBusId][k] = stopJsonObject.get("c").getAsDouble();
					e[newBusId][k] = stopJsonObject.get("e").getAsDouble();
					//ct[busId][k] = stopJsonObject.get("ct").getAsDouble();
					//xBStop[busId][k] = stopJsonObject.get("xBStop").getAsInt() == 1 ? true : false;
					if (e[newBusId][k] > 0.01) {
						int station = instance.paths[newBusId][k];
						xBStop[newBusId][k] = true;
						xBStation[newBusId][station] = true;
						x[station] = true;
						ct[newBusId][k] = e[newBusId][k] / instance.chargingRate;
						energyAdded += e[newBusId][k];
						//System.out.printf("[%s][%s], station %s\n", newBusId, k, station);
					} else {
						e[newBusId][k] = 0;
					}
					/*
					if (stopJsonObject.get("x").getAsInt() == 1) {
						int station = instance.paths[busId][k];
						//openStations.add(station);
						System.out.printf("Open station: %s\n", station);
					}	
					*/
					
				}
				newBusId++;
			}
			//System.out.println("sum(x_i = true) = " + openStations.size());
			System.out.printf("Energy added in %s: %s\n", pathType, energyAdded);
		} catch (Exception e) {
			System.out.println("Error reading partial solution");
			e.printStackTrace();
		}
	}
	
	public void writeJson() {
			
			Gson gson = new Gson();
			JsonObject rootObject = new JsonObject();
			JsonArray busesArray = new JsonArray();
			for (int b = 0; b < instance.b; b++) {		
				JsonObject busObject = new JsonObject(); 
				busObject.addProperty("bus", b);
				JsonArray pathArray = new JsonArray();
				for (int k = 0; k < instance.paths[b].length; k++) {
					int i = instance.paths[b][k];
					JsonObject stopObject = new JsonObject();
					stopObject.addProperty("station", i);
					stopObject.addProperty("x", x[i] ? 1 : 0);
					stopObject.addProperty("xBStop", xBStop[b][k] ? 1 : 0);
					stopObject.addProperty("c", c[b][k]);
					stopObject.addProperty("e", e[b][k]);
					stopObject.addProperty("ct", ct[b][k]);
					stopObject.addProperty("t", arrivalTime[b][k]);
					stopObject.addProperty("ot", instance.originalTimetable[b][k]);
					//stopObject.addProperty("dt", dt[b][k]);
					//stopObject.addProperty("altDt", Math.abs(instance.originalTimetable[b][k] - arrivalTime[b][k]));
					pathArray.add(stopObject);
				}
				busObject.add("path", pathArray);
				busesArray.add(busObject);
			}
			rootObject.add("buses", busesArray);
			try {
				FileWriter fileWriter = new FileWriter("../data/ini_solu.json");
				gson.toJson(rootObject, fileWriter);
				fileWriter.close();
			} catch (JsonIOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
	}

	public void readSolutionFromCplexOutput(HashMap<String, HashMap<String, Double>[][]> cplexVars, String vType) {
		for (String type: cplexVars.keySet()) {
			if (type.equals(vType)) {
				for (int b = 0; b < instance.b; b++) {
					for (int k = 0; k < instance.paths[b].length; k++) {
						for (String varName: cplexVars.get(type)[b][k].keySet()) {
							HashMap<String, Double> varsPerStop = cplexVars.get(type)[b][k];
							//System.out.printf("%s[%s][%s]:%s\n", varName, b, k, varsPerStop.get(varName));
							switch (varName) {
								case "arrivalTime": 
									arrivalTime[b][k] = varsPerStop.get(varName);
									break;
								case "c":
									c[b][k] = varsPerStop.get(varName);
									break;
								case "e":
									e[b][k] = varsPerStop.get(varName);
									break;
								case "ct":
									ct[b][k] = varsPerStop.get(varName);
									break;
								case "xBStop":
									xBStop[b][k] = ToolsMTD.round(varsPerStop.get(varName)) != 0;
									break;
								case "x":
									x[instance.paths[b][k]] = ToolsMTD.round(varsPerStop.get(varName)) != 0;
									break;
								case "xs":
									xBStation[b][instance.paths[b][k]] = ToolsMTD.round(varsPerStop.get(varName)) != 0;
									break;
								/*
								case "dt":
									dt[b][k] = varsPerStop.get(varName);
									break;
								*/
								default:
									System.out.println("STRANGE VARIABLE IS APPEARING: " + varName);				
							}
						}
					}
				}
			}
		}
		
		
	}
	
	public HashMap<String, HashMap<String, Double>[][]> toCplexOutput() {
		HashMap<String, HashMap<String, Double>[][]> cplexVars = new HashMap<String, HashMap<String, Double>[][]>();
		HashMap<String, Double>[][] allVarVals = new HashMap[instance.b][];
		for (int b = 0; b < instance.b; b++) {
			allVarVals[b] = new HashMap[instance.paths[b].length];
			for (int k = 0; k < instance.paths[b].length; k++) {
				int i = instance.paths[b][k];
				HashMap<String, Double> varVals = new HashMap<String, Double>();
				varVals.put("arrivalTime", arrivalTime[b][k]); 
				varVals.put("c", c[b][k]);
				varVals.put("e", e[b][k]);
				varVals.put("ct", ct[b][k]);
				varVals.put("xBStop", xBStop[b][k] ? 1.0 : 0.0);
				varVals.put("x", x[instance.paths[b][k]] ? 1.0 : 0.0);
				varVals.put("xs", xBStation[b][instance.paths[b][k]] ? 1.0 : 0.0);
				allVarVals[b][k] = varVals;
			}
		}
		cplexVars.put("", allVarVals);
		return cplexVars;
	}
	
	public void computeNumberOpenStations() {
		numberOpenStations = 0;
		for (int i = 0; i < instance.n; i++) {
			if (x[i]) {
				numberOpenStations++;
				//System.out.println(i);
			}
			
		}
		//System.out.println(numberOpenStations);
		//return numberOpenStations;
	}
	
	public int computeNumberOpenStationsFromStops() {
		int numberOpenStations = 0;
		ArrayList<Integer> numOpenStopsPerStation = new ArrayList<Integer>();
		for (int i = 0; i < instance.n; i++) {
			numOpenStopsPerStation.add(0);
		}
		for (int b = 0; b < instance.b; b++) {
			for (int k = 0; k < instance.paths[b].length; k++) {
				int i = instance.paths[b][k];
				if (xBStop[b][k]) {
					numOpenStopsPerStation.set(i, numOpenStopsPerStation.get(i) + 1);
				}
			}
		}
		for (int i = 0; i < instance.n; i++) {
			if (numOpenStopsPerStation.get(i) > 0) {
				numberOpenStations++;
			}
		}
		return numberOpenStations;
	}
	
	public void writeStationsUse() {
		HashMap<Integer, ArrayList<ChargesRange>> chargeRanges = new HashMap<Integer, ArrayList<ChargesRange>>();
		for (int b = 0; b < instance.b; b++) {
			for (int k = 0; k < instance.paths[b].length; k++) {
				int stationId = instance.paths[b][k];
				if (ToolsMTD.round(ct[b][k]) > 0) {
					double iniTime = arrivalTime[b][k];
					double finTime = arrivalTime[b][k] + ct[b][k];
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
	
	public void writeResults(double elapsedTime, int numLocalMins) {
		
		String filename = String.format("../data/results-%s.csv", "heuristic");
		FileWriter fw;
		try {
			fw = new FileWriter(filename, true);
		

			String inputs =  String.format("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,,,,,,,,", new Date().toString(),
					instance.inputFolder, instance.Cmax, instance.Cmin,
					instance.modelMinSpeedKmH, instance.instanceMinSpeedKmH, instance.restTimeMins, instance.DTmax / 60, 
					instance.minCt / 60, instance.b);
			String outputs = String.format("%s,,,%s,,,%s,%s\n", numberOpenStations, elapsedTime,
					instance.numStops, numLocalMins);
			String results = inputs + outputs;
			fw.write(results);
			fw.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
public void writeReducedResults(double elapsedTime, int obj, double timeObj) {
		
		String filename = String.format("../data/reduced-results-%s.csv", "heuristic");
		FileWriter fw;
		try {
			fw = new FileWriter(filename, true);
		

			String inputs =  String.format("%s,%s,%s,",
					instance.inputFolder, instance.Cmax, instance.DTmax / 60);

			String outputs = String.format("%s,%s,%s\n", obj, elapsedTime / 60, timeObj);
			
			String results = inputs + outputs;
			fw.write(results);
			fw.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	
	
	public void defineBasicVariablesPerBusPerStop(int b, int k) {
		
	}
	
	public void setInstance(InstanceMTD instance) {
		this.instance = instance;
	}
	
	public int getTotalAddedEnergy() {
		double energy = 0;
		for (int b = 0; b < instance.b; b++) {
			for (int k = 0; k < instance.paths[b].length; k++) {
				energy += e[b][k];
			}
		}
		return (int) ToolsMTD.round(energy);
	}
	
	public int getTotalArrivalEnergy() {
		double energy = 0;
		for (int b = 0; b < instance.b; b++) {
			for (int k = 0; k < instance.paths[b].length; k++) {
				energy += c[b][k];
			}
		}
		return (int) ToolsMTD.round(energy);
	}
	
	public int getTotalChargingTime() {
		double time = 0;
		for (int b = 0; b < instance.b; b++) {
			for (int k = 0; k < instance.paths[b].length; k++) {
				time += ct[b][k];
			}
		}
		return (int) ToolsMTD.round(time);
	}
	
	public int getWastedEnergy() {
		double wastedEnergy = 0;
		for (int b = 0; b < instance.b; b++) {
			for (int k = 0; k < instance.paths[b].length - 1; k++) {
				int nextStation = instance.paths[b][k + 1];
				int currentStation = instance.paths[b][k];
				double waste = (c[b][k] + e[b][k] - c[b][k + 1]) - instance.D[currentStation][nextStation];
				//System.out.printf("waste[%s][%s]=%s\n", b, k, waste);
				wastedEnergy += waste;
			}
		}
		return (int) ToolsMTD.round(wastedEnergy);
	}
	
	public int getNumberOpenStops() {
		int openStops = 0;
		for (int b = 0; b < instance.b; b++) {
			for (int k = 0; k < instance.paths[b].length - 1; k++) {
				if (xBStop[b][k]) {
					openStops++;
				}
			}
		}
		return openStops;
	}
	
	public int getNumberOpenStopsPerStation(int i) {
		int openStops = 0;
		for (int b = 0; b < instance.b; b++) {
			for (int k = 0; k < instance.paths[b].length - 1; k++) {
				if (i == instance.paths[b][k] && xBStop[b][k]) {
					openStops++;
				}
			}
		}
		return openStops;
	}
	
	/**
	 * Fix x and xStop values depending on actual recharges (e vars)
	 */
	public void fixXs() {
		for (int i = 0; i < instance.n; i++) {
			x[i]= false;
		}
		for (int b = 0; b < instance.b; b++) {
			for (int k = 0; k < instance.paths[b].length; k ++) {
				if (e[b][k] > 0) {
					int i = instance.paths[b][k];
					xBStop[b][k] = true;
					x[i] = true;
				} else {
					xBStop[b][k] = false;
				}
			}
		}
	}
}
