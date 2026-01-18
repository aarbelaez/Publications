package heuristics;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;

import core.InstanceMTD;
import utils.FeasibilityChecker;
import utils.ToolsMTD;

public class CompleteSolution {

	HeuristicsVariablesSet normalVars; 
	OpenStationsStructure openStationsStructure;
	
	public CompleteSolution(HeuristicsVariablesSet originalNormalVars, InstanceMTD instance) {
		normalVars = originalNormalVars;
		openStationsStructure = new OpenStationsStructure(normalVars, instance);
	}
	
	/**
	 * This constructor works for creating a copy of the input solution
	 * @param formerSolution
	 * @param instance
	 */
	public CompleteSolution(CompleteSolution formerSolution, InstanceMTD instance) {
		normalVars = new HeuristicsVariablesSet(instance, formerSolution.normalVars);
		openStationsStructure = new OpenStationsStructure(normalVars, instance);
	}
	
	public CompleteSolution createCopy(InstanceMTD instance) {
		return new CompleteSolution(this, instance);
	}
	
	public int getNumberOpenStations() {
		return openStationsStructure.getNumberOpenStations();
	}
	
	public int getNumOpenStopsPerStations(int i) {
		return openStationsStructure.getNumOpenStopsPerStations(i);
	}
	
	public int getNumOpenStopsPerStationsFromNormalVars(int i) {
		return normalVars.getNumberOpenStopsPerStation(i);
	}
	
	public int getNumOpenStopsPerStationsFromMainStructure(int i) {
		return openStationsStructure.getNumOpenStopsPerStationsFromMainStructure(i);
	}
	
	public void print(String filename) {
		normalVars.print(filename);
	}
	
	public void computeNumberOpenStations() {
		//System.out.println(openStationsStructure.stopsPerStationToString());
		normalVars.computeNumberOpenStations();
	}
	
	public boolean checkFeasibility(InstanceMTD instance, String filename) {
		FeasibilityChecker checker = new FeasibilityChecker(normalVars, instance, filename);
		boolean feasible = checker.checkFeasibiliy(normalVars);
		checker.closeCheckingFile();
		return feasible;
	}
	
	public void stopsPerStationToString() {
		System.out.println(openStationsStructure.stopsPerStationToString());
	}
	
	public void printOpenStationStructure() {
		openStationsStructure.print();
	}
	
	public void printOpenStationStructure(int b) {
		openStationsStructure.print(b);
	}
	
	public int getNumBusesWithOpenStations() {
		return openStationsStructure.getNumBusesWithOpenStations();
	}
	
	public boolean areAllOpenStationsCharging() {
		boolean areCharging = true;
		for (LinkedList<OpenStation> busPath: openStationsStructure.openStations) {
			for (OpenStation os: busPath) {
				int b = os.bus;
				int k = os.stop;
				if (normalVars.e[b][k] <= 0.01) {
					System.out.printf("Open station without charging e[%s][%s]=%s\n", b, k, normalVars.e[b][k]);
					return false;
				}
			}
		}
		return areCharging;
	}
	
	public int getTotalAddedEnergy() {
		return normalVars.getTotalAddedEnergy();
	}
	
	public int getTotalArrivalEnergy() {
		return normalVars.getTotalArrivalEnergy();
	}
	
	public int getWastedEnergy() {
		return normalVars.getWastedEnergy();
	}
	
	public int getTotalChargingTime() {
		return normalVars.getTotalChargingTime();
	}
	
	public int getTotalAddedEnergyFromOpenStructure() {
		double energy = 0;
		for (LinkedList<OpenStation> busPath: openStationsStructure.openStations) {
			for (OpenStation os: busPath) {
				int b = os.bus;
				int k = os.stop;
				energy += normalVars.e[b][k];
			}
		}
		return (int) ToolsMTD.round(energy);
	}
	
	public boolean checkOpenStructureIntegrity() {
		return openStationsStructure.checkIntegrety();
	}
	
	public int getNumberOpenStops() {
		return openStationsStructure.getNumberOpenStops1();
	}
	
	public int getNumberOpenStopsPerStation() {
		return openStationsStructure.getNumberOpenStops2();
	}
	
	public boolean checkOpenStopsConsistency() {
		boolean isConsistent = true;
		isConsistent = getNumberOpenStops() == getNumberOpenStopsPerStation();
		isConsistent = isConsistent && getNumberOpenStops() == normalVars.getNumberOpenStops();
		isConsistent = isConsistent && getNumberOpenStopsPerStation() == normalVars.getNumberOpenStops();
		return isConsistent;
	}
	
	public boolean checkOpenStationsConsistency() {
		boolean isConsistent = true;
		normalVars.computeNumberOpenStations();
		isConsistent = normalVars.numberOpenStations == getNumberOpenStations();
		//System.out.printf("%s=%s\n", normalVars.numberOpenStations, getNumberOpenStations());
		return isConsistent; 
	}
	
	public HashMap<String, HashMap<String, Double>[][]> toCplexOutput() {
		return normalVars.toCplexOutput();
	}
	
	public void writeStationsUse() {
		normalVars.writeStationsUse();
	}
	
	public void writeResults(double elapsedTime) {
		normalVars.writeResults(elapsedTime);
	}
	
	public void writeReducedResults(double elapsedTime, int obj, double timeObj) {
		normalVars.writeReducedResults(elapsedTime, obj, timeObj);
	}
}
