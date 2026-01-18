
package heuristics;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Set;
import java.util.TreeSet;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonIOException;
import com.google.gson.JsonObject;

import core.InstanceMTD;
import utils.FeasibilityChecker;

public class RobustCompleteSolution extends CompleteSolution {
	
	//HeuristicsVariablesSet backupNormalVars; 
	//OpenStationsStructure backupOpenStationsStructure;
	CompleteSolution robustRoute;
	InstanceMTD instance;
	
	public RobustCompleteSolution(HeuristicsVariablesSet originalNormalVars, HeuristicsVariablesSet backupNormalVars,  
			InstanceMTD instance) {
		super(originalNormalVars, instance);
		//this.backupNormalVars = backupNormalVars;
		//backupOpenStationsStructure = new OpenStationsStructure(backupNormalVars, instance);
		robustRoute = new CompleteSolution(backupNormalVars, instance);
		this.instance = instance;
	}
	
	/**
	 * This constructor works for creating a copy of the input solution
	 * @param formerSolution
	 * @param instance
	 */
	public RobustCompleteSolution(RobustCompleteSolution formerSolution, InstanceMTD instance) {
		super(formerSolution, instance);
		//backupNormalVars = new HeuristicsVariablesSet(instance, formerSolution.backupNormalVars);
		//backupOpenStationsStructure = new OpenStationsStructure(backupNormalVars, instance);
		robustRoute = new CompleteSolution(formerSolution.robustRoute, instance);
		this.instance = instance;
		linkPrimaryBackupOpenStations();
	}
	
	public int getNumberOpenStations() {
		Set<Integer> openStations = new TreeSet<Integer>();
		int numOpenStations = 0;
		for (int i: openStationsStructure.openStopsPerStation.keySet()) {
			//System.out.println(i);
			if (openStationsStructure.openStopsPerStation.get(i).size() > 0) {
				openStations.add(i);
			}
		}
		//System.out.println("Primary open stations: " + openStations.size());
		//System.out.println("Robust");
		int openBackup = 0;
		for (int i: robustRoute.openStationsStructure.openStopsPerStation.keySet()) {
			//System.out.println(i);
			if (robustRoute.openStationsStructure.openStopsPerStation.get(i).size() > 0 ) {
				openStations.add(i);
				openBackup++;
			}
		}
		//System.out.println("Backup open stations: " + openBackup);
		//System.out.println("Total open stations: " + openStations.size());
		//System.exit(0);
		numOpenStations = openStations.size();
		return numOpenStations;
	}
	
	public CompleteSolution createCopy(InstanceMTD instance) {
		return new RobustCompleteSolution(this, instance);
	}
	
	public boolean checkFeasibility(InstanceMTD instance, String filename) {
		FeasibilityChecker robustChecker = new FeasibilityChecker(normalVars, robustRoute.normalVars, instance, filename);
		boolean feasible = robustChecker.checkRobustFeasibility();
		robustChecker.closeCheckingFile();
		return feasible;
	}
	
	public void printOpenStationStructure(int b) {
		System.out.println("Primary-------------------\n");
		openStationsStructure.print(b);
		System.out.println("backup--------------------\n");
		robustRoute.openStationsStructure.print(b);
	}
	
	public void writeJson() {
		
		Gson gson = new Gson();
		JsonObject rootObject = new JsonObject();
		JsonArray busesArray = new JsonArray();
		for (int b = 0; b < instance.b; b++) {		
			JsonObject busObject = new JsonObject(); 
			busObject.addProperty("bus", b);
			JsonArray pathArray = new JsonArray();
			JsonArray bPathArray = new JsonArray();
			for (int k = 0; k < instance.paths[b].length; k++) {
				int i = instance.paths[b][k];
				JsonObject stopObject = new JsonObject();
				stopObject.addProperty("station", i);
				stopObject.addProperty("x", normalVars.x[i] ? 1 : 0);
				stopObject.addProperty("xBStop", normalVars.xBStop[b][k] ? 1 : 0);
				stopObject.addProperty("c", normalVars.c[b][k]);
				stopObject.addProperty("e", normalVars.e[b][k]);
				stopObject.addProperty("ct", normalVars.ct[b][k]);
				stopObject.addProperty("t", normalVars.arrivalTime[b][k]);
				stopObject.addProperty("ot", instance.originalTimetable[b][k]);
				pathArray.add(stopObject);
				
				JsonObject bStopObject = new JsonObject();
				bStopObject.addProperty("station", i);
				bStopObject.addProperty("x", robustRoute.normalVars.x[i] ? 1 : 0);
				bStopObject.addProperty("xBStop", robustRoute.normalVars.xBStop[b][k] ? 1 : 0);
				bStopObject.addProperty("c", robustRoute.normalVars.c[b][k]);
				bStopObject.addProperty("e", robustRoute.normalVars.e[b][k]);
				bStopObject.addProperty("ct", robustRoute.normalVars.ct[b][k]);
				bStopObject.addProperty("t", robustRoute.normalVars.arrivalTime[b][k]);
				bStopObject.addProperty("ot", instance.originalTimetable[b][k]);
				bPathArray.add(bStopObject);
			}
			busObject.add("path", pathArray);
			busObject.add("bpath", bPathArray);
			busesArray.add(busObject);
		}
		rootObject.add("buses", busesArray);
		try {
			FileWriter fileWriter = new FileWriter("../data/worstHeuSolution.json");
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
	
	public void print(String filename) {
		normalVars.print(filename + "-primary");
		robustRoute.normalVars.print(filename + "-backup");
	}
	
	public int getNumberOpenStationsByX() {
		int totalOpenStations = 0;
		for (int i = 0; i < instance.n; i++) {
			if (normalVars.x[i] || robustRoute.normalVars.x[i]) {
				totalOpenStations++;
			}
		}
		return totalOpenStations;
	}
	
	public boolean checkOpenStationsConsistency() {
		boolean isConsistent = true;
		isConsistent = getNumberOpenStationsByX() == getNumberOpenStations();
		//System.out.printf("%s=%s\n", normalVars.numberOpenStations, getNumberOpenStations());
		return isConsistent; 
	}
	
	public void linkPrimaryBackupOpenStations() {
		OpenStation currentPrimary = null;
		OpenStation currentBackup = null;
		for (int b = 0; b < instance.b; b++) {
			for (int k = 0; k < instance.paths[b].length; k++) {
				if (normalVars.xBStop[b][k]) {
					currentPrimary = openStationsStructure.openStopsPerBus.get(b).get(k);
				}
				if (robustRoute.normalVars.xBStop[b][k]) {
					currentBackup = robustRoute.openStationsStructure.openStopsPerBus.get(b).get(k);
					if (currentPrimary != null) {
						currentPrimary.setNextBackup(currentBackup);
						currentBackup.setPreviousPrimary(currentPrimary);
					}
				}
			}
		}
	}
	
	public boolean checkBackups() {
		for (int head: openStationsStructure.headsIdPerBus.keySet()) {
			int stop = openStationsStructure.headsIdPerBus.get(head);
			OpenStation currentOS = openStationsStructure.openStopsPerBus.get(head).get(stop);
			/*
			if (head == 0) {
				System.out.println("primary: " + currentOS.stop);
				System.out.println("backup: " + currentOS.nextBackup().stop);
			}
			*/
			if (!currentOS.hasBackup()) {
				return false;
			}
			while (currentOS.hasNext()) { 
				currentOS = currentOS.next();
				if (!currentOS.hasBackup()) {
					return false;
				}
			} 
		}
		return true;
	}
	
	public boolean checkForNegativeSlacks() {
		return openStationsStructure.checkForNegativeSlacks() || robustRoute.openStationsStructure.checkForNegativeSlacks();
	}
	
	public void printOpenStationStructure() {
		openStationsStructure.print();
		robustRoute.openStationsStructure.print();
	}
	

}

