package core;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Level;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.internal.Streams;
import com.google.gson.stream.JsonReader;

import heuristics.Stop;
import utils.ToolsMTD;

/**
 * This is an instance for the Minimal Timetable Disruptions problem
 * @author cdloaiza
 *
 */
public class InstanceMTD {
	
	public Map<Integer,String> stationNames;
	
	// PARAMETERS
	/**
	 * Max deviation time allowed
	 */
	public int DTmax;
	/**
	 * Last time allowed
	 */
	public int Tmax;
	/**
	 * number of stations
	 */
	public int n;
	/**
	 * number of buses
	 */
	public int b;
	public int Cmax;
	public int Cmin;
	public int maxChargingTime;
	public double maxAddingEnergy;
	/**
	 * energy necessary to go from i to j
	 */
	public int [][] D;
	/**
	 * T[b][i] is the time necessary for b to go from i to i+1
	 */
	public int [][] T;
	/**
	 * originalTimetable[i][j] is the timetable of the bus i at stop j
	 */
	public int[][] originalTimetable;
	/**
	 * strOriginalTimetable[i][j] is the timetable of the bus i at stop j as it is read in the json
	 */
	public String[][] strOriginalTimetable;
	/**
	 * paths[i][j] Station in stop j of bus i 
	 */
	public int[][] paths;
	/**
	 * speeds[i][j] is the speed with which the bus i depart from stop j
	 */
	public double[][] speeds;
	/**
	 * units of charging per minute
	 */
	public double chargingRate;
	/**
	 * Arbitrary big number
	 */
	/**
	 * Security margin time between charges in seconds
	 */
	public int SM;
	/**
	 * The biggest value of D in the route
	 */
	public int maxD;
	/**
	 * The biggest value of T in the route
	 */
	public int maxT;
	
	public double M = 100;
	
	public double MOverlapping;
	
	public double MAX_SPEED;
	public double MIN_SPEED;
	public int modelMinSpeedKmH;
	public int instanceMinSpeedKmH;
	
	/**
	 * Minimum charging time allowed
	 */
	public int minCt;
	
	/**
	 * The minimum number of chargers that the model has to install
	 */
	public int minNumberChargers;
	public boolean minChargersExtension = false;
	
	
	
	/**
	 * Rest time of the driver in seconds
	 */
	public int restTime;
	/**
	 * Rest time of the driver in minutes
	 */
	public int restTimeMins;
	
	
	public String inputFolder = "cork";
	
	public String busesFilename;
	
	/**
	 * Store all stops per station
	 */
	public ArrayList<Stop>[] stopsPerStation;
	
	/**
	 * Min energy added required per bus
	 */
	public ArrayList<Double> addedEnergies;
	/**
	 * chargesRequiredPerBus
	 */
	public ArrayList<Integer> requiredXStops;
	
	public int numStops = 0;
	
	/**
	 * Mapping bus ids after removing irrelevant to original ones
	 */
	public HashMap<Integer, Integer> originalBusesId;
	
	
	/**
	 * Buses ignored by the model
	 */
	public ArrayList<Integer> discardedBuses;
	
	/**
	 * Min time distance since two stops are considered for overlapping constraints
	 */
	public double overlappingTimeDistance;
	
	public Level loggerLevel = Level.SEVERE; 
	
	public InstanceMTD() {
		Cmax = 60000;
		instanceMinSpeedKmH = 30;
		modelMinSpeedKmH = 30;
		MIN_SPEED = ToolsMTD.kmHourToMtSeconds(modelMinSpeedKmH);	
		this.inputFolder = "cork-220";
		this.restTime = 10 * 60;
		setSecondaryParameters();
	}
	
	/**
	 * 
	 * @param cmax in mts
	 * @param speed in km/h
	 * @param restTime in minutes
	 */
	public InstanceMTD(int cmax, int modelSpeed, int instanceSpeed, int restTime, int DTmax, int minCt,
			String inputFolder, int cmin, int SM, int chargingRate) {
		this.Cmax = cmax;
		this.modelMinSpeedKmH = modelSpeed;
		MIN_SPEED = ToolsMTD.kmHourToMtSeconds(modelMinSpeedKmH);
		this.instanceMinSpeedKmH = instanceSpeed;
		this.inputFolder = inputFolder;
		this.restTimeMins = restTime;
		this.restTime = restTime * 60;
		this.DTmax = DTmax;
		this.minCt = minCt * 60;
		this.Cmin = cmin;
		this.SM = SM * 60;
		this.chargingRate = (chargingRate*1000) / 60.0;
		setSecondaryParameters();
	}
	
	private void setSecondaryParameters() {
		busesFilename = String.format("../data/%s/buses_input_%s_%s.json", inputFolder, instanceMinSpeedKmH, restTimeMins);
		Tmax = 100000;
		//DTmax = 5 * 60;
		//Cmin = 15000;
		//maxChargingTime = 12 * 60;
		//maxChargingTime = (int) Math.round(Cmax / chargingRate);
		maxChargingTime = (int) Math.round(((Cmax / chargingRate)*0.8) - (Cmin / chargingRate));
		maxAddingEnergy = maxChargingTime * chargingRate;
		System.out.printf("Max ct: %s\n", maxChargingTime);
		//System.exit(0);
		// units of charging per second
		//chargingRate = 10000 / 60; // ;
		// M = 2*DTmax + maxChargingTime;
		M = Tmax;
		MAX_SPEED = ToolsMTD.kmHourToMtSeconds(60);
		//SM = 60;
		MOverlapping = 4*DTmax + 2*maxChargingTime + 2*SM;
		
		// Including SM is important since SM constraints are also implemented 
		// in overlapping constraints
		overlappingTimeDistance = 2*DTmax + maxChargingTime + SM;
		//overlappingTimeDistance = (2*DTmax + maxChargingTime + SM) * 0.5;
		discardedBuses = new ArrayList<Integer>();
		originalBusesId = new HashMap<Integer, Integer>();
	}
	
	public void setInstance1() {
		stationNames = new HashMap<Integer,String>();
		Tmax = 50;
		DTmax = 50;
		// number of stations
		n = 4;
		// number of buses
		b = 3;
		Cmax = 10;
		Cmin = 0;
		maxChargingTime = 10;
		// energy necessary to go from i to j
		D = new int[][]{
				{0, 3, 3, 9},
				{3, 0, 4, 10},
				{3, 4, 0, 3},
				{9, 10, 3, 0}
		};
		// time necessary to go from i to j
		int[][] Ti = new int[][]{
				{0, 3, 3, 9},
				{3, 0, 4, 10},
				{3, 4, 0, 3},
				{9, 10, 3, 0}
		};
		originalTimetable = new int[][]{
			{0, 4, 8, 12},
			{0, 3, 14, 60},
			{0, 10, 14, 22}
		};
		strOriginalTimetable = new String[originalTimetable.length][];
		for (int i = 0; i < originalTimetable.length; i++) {
			strOriginalTimetable[i] = new String[originalTimetable[i].length];
			for (int j = 0; j < originalTimetable[i].length; j++) {
				strOriginalTimetable[i][j] = String.valueOf(originalTimetable[i][j]);
			}
		}	
		paths = new int[][] {
				{0, 1, 2, 3},
				{2, 1, 3, 0},
				{3, 0, 2, 1}
		};
		// units of charging per minute
		chargingRate = 1;
		
		setTForToyInstances(Ti);
		
		M = Tmax;
		SM = 0;
	}
	
	public void setInstance2() {
		stationNames = new HashMap<Integer,String>();
		Tmax = 100;
		DTmax = 100;
		// number of stations
		n = 4;
		// number of buses
		b = 3;
		Cmax = 10;
		Cmin = 2;
		maxChargingTime = 10;
		// energy necessary to go from i to j
		D = new int[][]{
			{0, 3, 7, 1},
			{3, 0, 5, 90},
			{7, 5, 0, 4},
			{1, 90, 4, 0}
		};
		// time necessary to go from i to j
		int[][] Ti = new int[][]{
			{0, 3, 7, 9},
			{3, 0, 5, 90},
			{7, 5, 0, 10},
			{9, 90, 10, 0}
		};
		originalTimetable = new int[][] {
			{0, 3, 11, 21},
			{0, 9, 19},
			{0, 11}
		};
		strOriginalTimetable = new String[originalTimetable.length][];
		for (int i = 0; i < originalTimetable.length; i++) {
			strOriginalTimetable[i] = new String[originalTimetable[i].length];
			for (int j = 0; j < originalTimetable[i].length; j++) {
				strOriginalTimetable[i][j] = String.valueOf(originalTimetable[i][j]);
			}
		}
		paths = new int[][] {
			{0, 1, 2, 3},
			{3, 0, 2},
			{2, 3}
		};
		// units of charging per minute
		chargingRate = 1;
		
		setTForToyInstances(Ti);
		
		M = Tmax;
		SM = 0;
	}
	
	public void setInstance3() {
		stationNames = new HashMap<Integer,String>();
		Tmax = 50;
		DTmax = 50;
		// number of stations
		n = 4;
		// number of buses
		b = 2;
		Cmax = 10;
		Cmin = 2;
		maxChargingTime = 10;
		// energy necessary to go from i to j
		D = new int[][]{
			{0, 3, 7, 8},
			{3, 0, 5, 3},
			{7, 5, 0, 8},
			{8, 3, 8, 0}
		};
		// time necessary to go from i to j
		int[][] Ti = new int[][]{
			{0, 3, 7, 8},
			{3, 0, 5, 3},
			{7, 5, 0, 8},
			{8, 3, 8, 0}
		};
		originalTimetable = new int[][]{
			{0, 3, 11, 19},
			{1, 4, 13, 20},
		};
		strOriginalTimetable = new String[originalTimetable.length][];
		for (int i = 0; i < originalTimetable.length; i++) {
			strOriginalTimetable[i] = new String[originalTimetable[i].length];
			for (int j = 0; j < originalTimetable[i].length; j++) {
				strOriginalTimetable[i][j] = String.valueOf(originalTimetable[i][j]);
			}
		}
		paths = new int[][] {
			{0, 1, 2, 3},
			{3, 1, 2, 0},
		};
		// units of charging per minute
		chargingRate = 1;
		
		setTForToyInstances(Ti);
		
		M = Tmax;
		SM = 0;
	}
	
	public void setInstance4() {
		stationNames = new HashMap<Integer,String>();
		Tmax = 100;
		DTmax = 100;
		// number of stations
		n = 4;
		// number of buses
		b = 1;
		Cmax = 10;
		Cmin = 2;
		maxChargingTime = 10;
		// energy necessary to go from i to j
		D = new int[][]{
			{0, 3, 7, 1},
			{3, 0, 5, 90},
			{7, 5, 0, 4},
			{1, 90, 4, 0}
		};
		// time necessary to go from i to j
		int[][] Ti = new int[][]{
			{0, 3, 7, 9},
			{3, 0, 5, 90},
			{7, 5, 0, 10},
			{9, 90, 10, 0}
		};
		originalTimetable = new int[][] {
			{0, 3, 11, 21},
			{0, 9, 19},
			{0, 11}
		};
		strOriginalTimetable = new String[originalTimetable.length][];
		for (int i = 0; i < originalTimetable.length; i++) {
			strOriginalTimetable[i] = new String[originalTimetable[i].length];
			for (int j = 0; j < originalTimetable[i].length; j++) {
				strOriginalTimetable[i][j] = String.valueOf(originalTimetable[i][j]);
			}
		}
		paths = new int[][] {
			//{0, 1, 2, 3},
			{3, 0, 2},
			//{2, 3}
		};
		// units of charging per minute
		chargingRate = 1;
		
		setTForToyInstances(Ti);
		
		M = Tmax;
		SM = 0;
	}
	
	public void setInstance5() {
		stationNames = new HashMap<Integer,String>();
		Tmax = 100;
		DTmax = 100;
		// number of stations
		n = 11;
		// number of buses
		b = 1;
		Cmax = 10;
		Cmin = 0;
		maxChargingTime = 10;
		// energy necessary to go from i to j
		D = new int[][]{
			{100, 5, 100, 100, 100, 100, 100, 100, 100, 100, 100},
			{5, 100, 3, 100, 100, 100, 100, 100, 100, 100, 100},
			{100, 3, 100, 2, 100, 100, 100, 100, 100, 100, 100},
			{100, 100, 2, 100, 1, 100, 100, 100, 100, 100, 100},
			{100, 100, 100, 1, 100, 3, 100, 100, 100, 100, 100},
			{100, 100, 100, 100, 3, 100, 3, 100, 100, 100, 100},
			{100, 100, 100, 100, 100, 3, 100, 1, 100, 100, 100},
			{100, 100, 100, 100, 100, 100, 1, 100, 1, 100, 100},
			{100, 100, 100, 100, 100, 100, 100, 1, 100, 1, 100},
			{100, 100, 100, 100, 100, 100, 100, 100, 1, 100, 2},
			{100, 100, 100, 100, 100, 100, 100, 100, 100, 2, 100}
		};
		originalTimetable = new int[][] {
			{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
		};
		strOriginalTimetable = new String[originalTimetable.length][];
		for (int i = 0; i < originalTimetable.length; i++) {
			strOriginalTimetable[i] = new String[originalTimetable[i].length];
			for (int j = 0; j < originalTimetable[i].length; j++) {
				strOriginalTimetable[i][j] = String.valueOf(originalTimetable[i][j]);
			}
		}
		paths = new int[][] {
			{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
		};
		T = new int[][] {
			{5, 3, 2, 1, 3, 3, 1, 1, 1, 2, 100}
		};
		// units of charging per minute
		chargingRate = 1;
		
		M = Tmax;
		SM = 0;
	}
	
	public void setInstance6() {
		stationNames = new HashMap<Integer,String>();
		Tmax = 100;
		DTmax = 100;
		// number of stations
		n = 14;
		// number of buses
		b = 2;
		Cmax = 10;
		Cmin = 0;
		maxChargingTime = 10;
		// energy necessary to go from i to j
		D = new int[][]{
			{100, 5, 100, 100, 100, 100, 100, 100, 100, 100, 100,    100, 100, 100},
			{5, 100, 3, 100, 100, 100, 100, 100, 100, 100, 100,      100, 100, 100},
			{100, 3, 100, 2, 100, 100, 100, 100, 100, 100, 100,      100, 100, 100},
			{100, 100, 2, 100, 1, 100, 100, 100, 100, 100, 100,      100, 100, 100},
			{100, 100, 100, 1, 100, 3, 100, 100, 100, 100, 100,      100, 100, 100},
			{100, 100, 100, 100, 3, 100, 3, 100, 100, 100, 100,      100, 100, 100},
			{100, 100, 100, 100, 100, 3, 100, 1, 100, 100, 100,      100, 3, 7},
			{100, 100, 100, 100, 100, 100, 1, 100, 1, 100, 100,      100, 100, 100},
			{100, 100, 100, 100, 100, 100, 100, 1, 100, 1, 100,      100, 100, 100},
			{100, 100, 100, 100, 100, 100, 100, 100, 1, 100, 2,      100, 100, 100},
			{100, 100, 100, 100, 100, 100, 100, 100, 100, 2, 100,     100, 100, 100},
			
			{100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,    100, 2, 100},
			{100, 100, 100, 100, 100, 100, 3, 100, 100, 100, 100,    2, 100, 100},
			{100, 100, 100, 100, 100, 100, 7, 100, 100, 100, 100,    100, 100, 100}
		};
		originalTimetable = new int[][] {
			{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
			{11, 12, 13, 14}
		};
		strOriginalTimetable = new String[originalTimetable.length][];
		for (int i = 0; i < originalTimetable.length; i++) {
			strOriginalTimetable[i] = new String[originalTimetable[i].length];
			for (int j = 0; j < originalTimetable[i].length; j++) {
				strOriginalTimetable[i][j] = String.valueOf(originalTimetable[i][j]);
			}
		}
		paths = new int[][] {
			{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
			{11, 12, 6, 13}
		};
		T = new int[][] {
			{5, 3, 2, 1, 3, 3, 1, 1, 1, 2, 100},
			{2, 3, 7, 100}
		};
		// units of charging per minute
		chargingRate = 1;
		
		M = Tmax;
		SM = 0;
	}
	
	public void setInstance7() {
		stationNames = new HashMap<Integer,String>();
		Tmax = 100;
		DTmax = 100;
		// number of stations
		n = 5;
		// number of buses
		b = 1;
		Cmax = 10;
		Cmin = 0;
		maxChargingTime = 10;
		// energy necessary to go from i to j
		D = new int[][]{
			{100, 7, 100, 100, 100},
			{7, 100, 4, 100, 100},
			{100, 4, 100, 2, 100},
			{100, 100, 2, 100, 7},
			{100, 100, 100, 7, 100},
		};
		originalTimetable = new int[][] {
			{0, 1, 2, 3, 4}
		};
		strOriginalTimetable = new String[originalTimetable.length][];
		for (int i = 0; i < originalTimetable.length; i++) {
			strOriginalTimetable[i] = new String[originalTimetable[i].length];
			for (int j = 0; j < originalTimetable[i].length; j++) {
				strOriginalTimetable[i][j] = String.valueOf(originalTimetable[i][j]);
			}
		}
		paths = new int[][] {
			{0, 1, 2, 3, 4}
		};
		T = new int[][] {
			{7, 4, 2, 7, 100}
		};
		// units of charging per minute
		chargingRate = 1;
		
		M = Tmax;
		SM = 0;
	}
	
	public void setInstance8() {
		stationNames = new HashMap<Integer,String>();
		Tmax = 100;
		DTmax = 5;
		// number of stations
		n = 14;
		// number of buses
		b = 2;
		Cmax = 10;
		Cmin = 0;
		maxChargingTime = 10;
		// energy necessary to go from i to j
		D = new int[][]{
			{100, 5, 100, 100, 100, 100, 100, 100, 100, 100, 100,    100, 100, 100},
			{5, 100, 3, 100, 100, 100, 100, 100, 100, 100, 100,      100, 100, 100},
			{100, 3, 100, 2, 100, 100, 100, 100, 100, 100, 100,      100, 100, 100},
			{100, 100, 2, 100, 1, 100, 100, 100, 100, 100, 100,      100, 100, 100},
			{100, 100, 100, 1, 100, 3, 100, 100, 100, 100, 100,      100, 100, 100},
			{100, 100, 100, 100, 3, 100, 3, 100, 100, 100, 100,      100, 100, 100},
			{100, 100, 100, 100, 100, 3, 100, 1, 100, 100, 100,      100, 3, 7},
			{100, 100, 100, 100, 100, 100, 1, 100, 1, 100, 100,      100, 100, 100},
			{100, 100, 100, 100, 100, 100, 100, 1, 100, 1, 100,      100, 100, 100},
			{100, 100, 100, 100, 100, 100, 100, 100, 1, 100, 2,      100, 100, 100},
			{100, 100, 100, 100, 100, 100, 100, 100, 100, 2, 100,     100, 100, 100},
			
			{100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,    100, 2, 100},
			{100, 100, 100, 100, 100, 100, 3, 100, 100, 100, 100,    2, 100, 100},
			{100, 100, 100, 100, 100, 100, 7, 100, 100, 100, 100,    100, 100, 100}
		};
		originalTimetable = new int[][] {
			{0, 5, 13, 15, 16, 19, 30, 31, 32, 33, 34},
			{0, 27, 30, 39}
		};
		strOriginalTimetable = new String[originalTimetable.length][];
		for (int i = 0; i < originalTimetable.length; i++) {
			strOriginalTimetable[i] = new String[originalTimetable[i].length];
			for (int j = 0; j < originalTimetable[i].length; j++) {
				strOriginalTimetable[i][j] = String.valueOf(originalTimetable[i][j]);
			}
		}
		paths = new int[][] {
			{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
			{11, 12, 6, 13}
		};
		T = new int[][] {
			{5, 3, 2, 1, 3, 3, 1, 1, 1, 2, 100},
			{2, 3, 7, 100}
		};
		// units of charging per minute
		chargingRate = 1;
		
		M = Tmax;
		SM = 0;
		
		stopsPerStation = (ArrayList<Stop>[])new ArrayList[n];
		for (int i = 0; i < n; i++ ) {
			stopsPerStation[i] = new ArrayList<Stop>();
		}
		stopsPerStation[0].add(new Stop(0, 0, 0));
		stopsPerStation[1].add(new Stop(0, 1, 0));
		stopsPerStation[2].add(new Stop(0, 2, 0));
		stopsPerStation[3].add(new Stop(0, 3, 0));
		stopsPerStation[4].add(new Stop(0, 4, 0));
		stopsPerStation[5].add(new Stop(0, 5, 0));
		stopsPerStation[6].add(new Stop(0, 6, 0));
		stopsPerStation[7].add(new Stop(0, 7, 0));
		stopsPerStation[8].add(new Stop(0, 8, 0));
		stopsPerStation[9].add(new Stop(0, 9, 0));
		stopsPerStation[10].add(new Stop(0, 10, 0));
		
		stopsPerStation[11].add(new Stop(1, 0, 0));
		stopsPerStation[12].add(new Stop(1, 1, 0));
		stopsPerStation[6].add(new Stop(1, 2, 0));
		stopsPerStation[13].add(new Stop(1, 3, 0));
		
		minNumberChargers = 5;
		minChargersExtension = true;
	}
	
	public void setInstance9() {
		stationNames = new HashMap<Integer,String>();
		Tmax = 100;
		DTmax = 5;
		// number of stations
		n = 14;
		// number of buses
		b = 3;
		Cmax = 10;
		Cmin = 0;
		maxChargingTime = 10;
		// energy necessary to go from i to j
		D = new int[][]{
			{100, 5, 100, 100, 100, 100, 100, 100, 100, 100, 100,    100, 100, 100},
			{5, 100, 3, 100, 100, 100, 100, 100, 100, 100, 100,      100, 100, 100},
			{100, 3, 100, 2, 100, 100, 100, 100, 100, 100, 100,      100, 100, 100},
			{100, 100, 2, 100, 1, 100, 100, 100, 100, 100, 100,      100, 100, 100},
			{100, 100, 100, 1, 100, 3, 100, 100, 100, 100, 100,      100, 100, 100},
			{100, 100, 100, 100, 3, 100, 3, 100, 100, 100, 100,      100, 100, 100},
			{100, 100, 100, 100, 100, 3, 100, 1, 100, 100, 100,      100, 3, 7},
			{100, 100, 100, 100, 100, 100, 1, 100, 1, 100, 100,      100, 100, 100},
			{100, 100, 100, 100, 100, 100, 100, 1, 100, 1, 100,      100, 100, 100},
			{100, 100, 100, 100, 100, 100, 100, 100, 1, 100, 2,      100, 100, 100},
			{100, 100, 100, 100, 100, 100, 100, 100, 100, 2, 100,     100, 100, 100},
			
			{100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,    100, 2, 100},
			{100, 100, 100, 100, 100, 100, 3, 100, 100, 100, 100,    2, 100, 100},
			{100, 100, 100, 100, 100, 100, 7, 100, 100, 100, 100,    100, 100, 100}
		};
		originalTimetable = new int[][] {
			{0, 5, 13, 15, 16, 19, 30, 31, 32, 33, 34},
			{0, 27, 30, 39},
			{16, 19, 30, 39}
		};
		strOriginalTimetable = new String[originalTimetable.length][];
		for (int i = 0; i < originalTimetable.length; i++) {
			strOriginalTimetable[i] = new String[originalTimetable[i].length];
			for (int j = 0; j < originalTimetable[i].length; j++) {
				strOriginalTimetable[i][j] = String.valueOf(originalTimetable[i][j]);
			}
		}
		paths = new int[][] {
			{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
			{11, 12, 6, 13},
			{4, 5, 6, 13}
		};
		T = new int[][] {
			{5, 3, 2, 1, 3, 3, 1, 1, 1, 2, 100},
			{2, 3, 7, 100},
			{3, 3, 7, 100}
		};
		// units of charging per minute
		chargingRate = 1;
		
		M = Tmax;
		SM = 0;
		
		minNumberChargers = 5;
		minChargersExtension = true;
	}
	
	public void setTForToyInstances(int[][] Ti) {
		T =  new int[paths.length][];
		for (int b = 0; b < paths.length; b++) {
			T[b] = new int[paths[b].length];
			for (int k = 0; k < paths[b].length - 1; k++) {
				T[b][k] = Ti[paths[b][k]][paths[b][k + 1]];
			}
		}
	}
	
	public void readStations() throws IOException {
		stationNames = new HashMap<Integer,String>();
		FileReader fileReader = new FileReader(String.format("../data/%s/stations_input.csv", inputFolder));
		BufferedReader br = new BufferedReader(fileReader);
		String line;
		// Headers
		br.readLine();
		int i = 0;
		while ((line = br.readLine()) != null) {
			String[] station = line.split(",");
			stationNames.put(Integer.parseInt(station[1]), station[0]);
			i++;
		}
		br.close();
		this.n = i;
		System.out.println("Number of stations: " + this.n);
	}
	
	public int readDistances() throws IOException {
		D = new int[this.n][this.n];
		FileReader fileReader = new FileReader(String.format("../data/%s/distances_input.csv", inputFolder));
		BufferedReader br = new BufferedReader(fileReader);
		String line;
		int i = 0;
		while ((line = br.readLine()) != null) {
			if (i != 0) {
				String[] record = line.split(",");
				int from = Integer.parseInt(record[0]); 
				int to = Integer.parseInt(record[1]);
				Double distance =  Double.parseDouble(record[2]);
				// System.out.println(distance);
				double distanceMts = distance*1000;
				// Change for the actual discharging amount which anyways must be proportional to distance
				D[from][to] = (int) distanceMts;
				D[to][from] = (int) distanceMts;
			}
			i++;
		}
		br.close();
		return i;
	}
	
	public void readBuses() throws IOException {
		
		stopsPerStation = (ArrayList<Stop>[])new ArrayList[n];
		for (int i = 0; i < n; i++ ) {
			stopsPerStation[i] = new ArrayList<Stop>();
		}
		
		maxT = 0;
		
		this.b = 0;
		
		numStops = 0;
		
		JsonElement tmpJsonElement = Streams.parse(new JsonReader(new FileReader(busesFilename)));
		JsonArray linesList = tmpJsonElement.getAsJsonArray();
		
		for (JsonElement lineObject : linesList) {
			JsonObject lineJsonObject = lineObject.getAsJsonObject();
			JsonArray lineBuses = lineJsonObject.get("buses").getAsJsonArray();
			b = b + lineBuses.size();
		}
		
		b = b - discardedBuses.size();
		
		System.out.println("Number of buses: " + b);
		
		paths = new int[b][];
		originalTimetable = new int[b][];
		strOriginalTimetable = new String[b][];
		speeds = new double[b][];
		T = new int[b][];
		
		PrintWriter writer = new PrintWriter("../data/path.txt");
		
		int newBusId = 0;
		
		for (JsonElement lineObject : linesList) {
			JsonObject lineJsonObject = lineObject.getAsJsonObject();
			JsonArray lineBuses = lineJsonObject.get("buses").getAsJsonArray();
			
			for (JsonElement busObject : lineBuses) {
				//System.out.println(busObject);
				JsonObject busJsonObject = busObject.getAsJsonObject();
				int busId = busJsonObject.get("bus").getAsInt();
				if (discardedBuses.contains(busId)) {
					//System.out.println(busId);
					continue;
				}
				JsonArray busPath = busJsonObject.get("path").getAsJsonArray();
				//System.out.println(busId);
				paths[newBusId] = new int[busPath.size()];
				originalTimetable[newBusId] = new int[busPath.size()];
				strOriginalTimetable[newBusId] = new String[busPath.size()];
				speeds[newBusId] = new double[busPath.size()];
				T[newBusId] = new int[busPath.size()];
				
				writer.println("\n"+newBusId);
				
				System.out.printf("Bus %s ----> %s\n", newBusId, busId);
				originalBusesId.put(newBusId, busId);
				
				int previousStationId = -1;
				JsonObject previousStopJsonObject = new JsonObject();
				
				for (int i = 0; i < busPath.size(); i++) {
					JsonElement stop = busPath.get(i);
					JsonObject stopJsonObject = stop.getAsJsonObject();
					int stationId = stopJsonObject.get("station_id").getAsInt();
					// System.out.println(stopId);
					paths[newBusId][i] = stationId;
					String strTime = stopJsonObject.get("time").getAsString();
					strOriginalTimetable[newBusId][i] = strTime;
					boolean isToyInstance = instanceMinSpeedKmH == 0 ? true : false;
					int arrivalInMinutes = ToolsMTD.stringTimeToMinutes(strTime, !isToyInstance);
					int arrivalInSeconds = arrivalInMinutes * 60;
					originalTimetable[newBusId][i] = arrivalInSeconds;
					
					stopsPerStation[stationId].add(new Stop(newBusId, i, 0));
					
					// recording speeds
					if (i > 0) {
						int timeNeededForPrevious = originalTimetable[newBusId][i] - originalTimetable[newBusId][i-1];
						// when arrival times are the same, we say the time difference is 30 seconds
						timeNeededForPrevious = timeNeededForPrevious == 0 ? 30 : timeNeededForPrevious;
						double requiredSpeed =  D[previousStationId][stationId] / (double)timeNeededForPrevious;
						if (requiredSpeed > MAX_SPEED) {
							requiredSpeed = MAX_SPEED;
						}
						speeds[newBusId][i-1] = Math.max(requiredSpeed, MIN_SPEED);
						// Time to go from i-1 to i
						T[newBusId][i-1] = (int) Math.round(D[previousStationId][stationId] / speeds[newBusId][i-1]);
						if (previousStopJsonObject.has("rest")) {
							T[newBusId][i-1] = T[newBusId][i-1] + restTime;
							writer.printf("Resting %s seconds\n", restTime);
						}
						if (T[newBusId][i-1] > maxT) {
							maxT = T[newBusId][i-1];
						}
						if (D[previousStationId][stationId] > maxD) {
							maxD = D[previousStationId][stationId];
						}
						if (stationId == previousStationId && 
								originalTimetable[newBusId][i] == originalTimetable[newBusId][i-1]) {
							//System.out.println("Repeated stop!");
							//System.out.printf("id: %s, time:%s\n", stationId, 
							//		ToolsMTD.minutesToStringTime(originalTimetable[busId][i]/60));
						}
					}
					
					if (previousStationId != -1) {
						writer.printf("T:%s, D:%s\n", T[newBusId][i-1], D[previousStationId][stationId]);
						writer.printf("speeds:%s\n", speeds[newBusId][i-1]);
					}
					writer.printf("bus:%s, stop:%s, station:%s/%s, time:%s, timeSeconds:%s\n",
							newBusId, i, stationId, stationNames.get(stationId), strTime, arrivalInSeconds);
					
					previousStationId = stationId;
					previousStopJsonObject = stopJsonObject;
					
					numStops++;
					
				}
				newBusId++;
			}
			
		}
		System.out.println("Number of stops: " + numStops);
		writer.close();	
	};
	
	
	public void readInput() {
		
		try {
			readStations();
			readDistances();
			readBuses();
			computeMinEnergyAddedPerBus();
			identifyIrrelevantBusesByAddedEnergy();
			readBuses();
			//identifyCustomIrrelevantBuses();
			//readBuses();
			//System.exit(0);
			
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} 
		
	}
	
	public void computeMinEnergyAddedPerBus() {
		addedEnergies = new ArrayList<Double>();
		requiredXStops = new ArrayList<Integer>();		
		double totalMinEnergy = 0;
		//double timeRequired = 0;
		for (int bu = 0; bu < b; bu++) {
			double energy = Cmax * -1 + Cmin;
			for (int k = 0; k < paths[bu].length - 1; k++) {
				energy += D[paths[bu][k]][paths[bu][k+1]];
				//timeRequired += T[bu][k];
				/*
				if (k == 66) {
					System.out.println("required energy: " + energy);
					System.out.println("required time: " + timeRequired);
					//System.exit(0);
				}
				*/
			}
			//System.out.printf("b: %s, raw-sum(e):%s\n", bu, Math.max(energy, 0));
			//System.out.printf("b: %s, sum(e):%s\n", bu, Math.floor(Math.max(energy, 0)));
			double addedEnergy = Math.floor(Math.max(energy, 0));
			addedEnergies.add(addedEnergy);
			totalMinEnergy += addedEnergy;
			int minReqCharges = (int) Math.ceil(addedEnergy / (maxChargingTime*chargingRate));
			requiredXStops.add(minReqCharges);
		}
		System.out.println("Total min energy: " + totalMinEnergy);
	}
	
    public void identifyIrrelevantBusesByAddedEnergy () {
		discardedBuses = new ArrayList<Integer>();
		//System.out.println("Discarding ");
		for (int bu = 0; bu < b; bu++) {
			if (addedEnergies.get(bu) <= 0) {
				//System.out.println(bu);
				discardedBuses.add(bu);
			}
        }
		System.out.println("Number of buses discarded: " + discardedBuses.size());
    }
    
    public void identifyCustomIrrelevantBuses () {
    	int[] discardedBusesArr = {0, 1, 2, 3, 8, 16, 17, 18, 20, 21, 23, 24, 25, 28, 29, 30, 32, 33, 34, 35, 63, 66, 68, 80, 81, 82, 83, 84, 85, 86, 87, 91, 95, 97, 100, 101, 102, 106, 143, 167, 170, 198, 199, 200, 201, 202, 203, 204, 211, 212};
    	//discardedBuses = new ArrayList<Integer>();
    	for (int bus: discardedBusesArr) {
    		discardedBuses.add(originalBusesId.get(bus));
    	}
		System.out.println("Number of buses discarded: " + discardedBuses.size());
    }

}
