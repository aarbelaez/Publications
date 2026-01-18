package utils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

import core.InstanceMTD;

/**
 * To wrap data needed to pass a solution to a different execution
 * @author smartebuses1
 *
 */
public class OutputForWarm {
	
	public OutputForWarm(HashMap<String, HashMap<String, Double>[][]> routeOutputs, ArrayList<Integer> actualBuses, int[][] paths) {
		super();
		this.routeOutputs = routeOutputs;
		this.actualBuses = actualBuses;
		this.paths = paths;
		routeOutsClone = routeOutputs;
	}
	public HashMap<String, HashMap<String, Double>[][]> routeOutputs;
	public HashMap<String, HashMap<String, Double>[][]> routeOutsClone;
	
	public ArrayList<Integer> actualBuses;
	public int[][] paths;
	
	/**
	 * This should be the original set of open stations
	 */
	public Set<Integer> getOpenStations() {
		Set<Integer> openStations = new HashSet<Integer>();
		for (int b = 0; b < routeOutsClone.get("").length; b++) {
			if (routeOutsClone.get("")[b] != null) {
				for (int k = 1; k < routeOutsClone.get("")[b].length; k++) {
					int stationId = paths[b][k];
					HashMap<String, Double> varVals = routeOutsClone.get("")[b][k];
					boolean openPrimary = varVals.get("x") == 1;
					boolean openBackup = false;
					try {
						HashMap<String, Double> bVarVals = routeOutsClone.get("backup")[b][k];
						openBackup = bVarVals.get("x") == 1;
					} catch (NullPointerException e) {
						//System.out.println("Exception. No backup!");
					}
					if (openPrimary || openBackup) {
						openStations.add(stationId);
					}
				}
			}
		}
		return openStations;
	}
	
	public void filterByNewInstance (InstanceMTD instanceMTD) {
		this.routeOutsClone = new HashMap<String, HashMap<String, Double>[][]>();
		
		System.out.printf("Given solution buses originally: %s\n", this.routeOutputs.get("").length);
		
		HashMap<String, HashMap<String, Double>[][]> routeOuts = this.routeOutputs;
		
		for (String route : routeOuts.keySet()) {
		
			HashMap<String, Double>[][] outputPerBus = routeOuts.get(route);
			HashMap<String, Double>[][] newOutputPerBus = new HashMap[instanceMTD.b][];
			
			ArrayList<Integer> thisActualBuses = new ArrayList<Integer>();
			for (int b = 0; b < instanceMTD.originalB; b++) {
				if (!instanceMTD.discardedBuses.contains(b)) {
					thisActualBuses.add(b);
				}
			}	
			
			int g = 0;
			int h = 0;
			while (h < instanceMTD.b) {
				int thisOrB = thisActualBuses.get(h);
				int warmOrB = 10000;
				try {
					warmOrB = this.actualBuses.get(g);
				} catch (IndexOutOfBoundsException e) {
					
				}
				if (thisOrB == warmOrB) {
					newOutputPerBus[h] = outputPerBus[g];
					g++;
					h++;
				} else if (thisOrB > warmOrB) {
					g++;
				} else if (thisOrB < warmOrB) {
					newOutputPerBus[h] = null;
					h++;
				}
			}
			
			routeOuts.put(route, newOutputPerBus);
			this.routeOutsClone.put(route, outputPerBus);
			//outputPerBus = newOutputPerBus;
			System.out.printf("outputPerBus length: %s\n", outputPerBus.length);
			System.out.printf("newOutputPerBus length: %s\n", newOutputPerBus.length);
			System.out.printf("Given solution buses after process: %s\n", this.routeOutputs.get("").length);
		}
	}
}
