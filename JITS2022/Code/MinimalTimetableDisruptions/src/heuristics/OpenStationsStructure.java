package heuristics;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.ListIterator;
import java.util.Map;

import core.InstanceMTD;

/**
 * A collection of sequences of open stations, which have additional information and are often modified
 * @author cedaloaiza
 *
 */
public class OpenStationsStructure {
	
	private HeuristicsVariablesSet variables;
	private InstanceMTD instance;
	public ArrayList<HashMap<Integer, OpenStation>> openStopsPerBus;
	public Map<Integer,LinkedList<OpenStation>> openStopsPerStation;
	public HashMap<Integer, Integer> headsIdPerBus;
	
	public OpenStationsStructure(HeuristicsVariablesSet variables, InstanceMTD instance) {
		this.variables = variables;
		this.instance = instance;
		openStopsPerBus = new ArrayList<HashMap<Integer, OpenStation>>();
		openStopsPerStation = new HashMap<Integer, LinkedList<OpenStation>>(instance.n);
		headsIdPerBus = new HashMap<Integer, Integer>();
		build();
	}
	
	public void build() {
		for (int b = 0; b < instance.b; b++) {
			HashMap<Integer, OpenStation> bOpenStations = new HashMap<Integer, OpenStation>(instance.paths[b].length);
			double nextE = 0;
			double nextS = 100000;
			OpenStation prev = null;
			for (int k = 0; k < instance.paths[b].length; k++) {
				int station = instance.paths[b][k];
				if (k > 0) {
					int prevStation = instance.paths[b][k-1];
					nextE += instance.D[prevStation][station];
					double delayAvailable = instance.DTmax - variables.getDeltaT(b, k);
					nextS = delayAvailable < nextS ? delayAvailable : nextS;
				}
				if (variables.xBStop[b][k]) {
					OpenStation openStation = new OpenStation(nextE, nextS, k, b);
					bOpenStations.put(k, openStation);
					openStation.setPrevious(prev);
					if (prev != null) {
						prev.setNext(openStation);
					} else {
						headsIdPerBus.put(b, k);
					}
					try {
						openStopsPerStation.get(station).add(openStation);
					} catch (NullPointerException e) {
						LinkedList<OpenStation> sequenceStops = new LinkedList<OpenStation>();
						sequenceStops.add(openStation);
						openStopsPerStation.put(station, sequenceStops);
					}
					nextE = 0;
					nextS = 100000;
					prev = openStation;
				}
			}
			openStopsPerBus.add(bOpenStations);
		}
	}
	
	public boolean checkIntegrety() {
		boolean isGood = true;
		for (int b = 0; b < instance.b; b++) {
			int stopIndex = 0;
			OpenStation bOpenStations = openStopsPerBus.get(b).get(0);
			double nextE = 0;
			double nextS = 100000;
			for (int k = 0; k < instance.paths[b].length; k++) {
				int station = instance.paths[b][k];
				if (k > 0) {
					int prevStation = instance.paths[b][k-1];
					nextE += instance.D[prevStation][station];
					double delayAvailable = instance.DTmax - variables.getDeltaT(b, k);
					nextS = delayAvailable < nextS ? delayAvailable : nextS;
				}
				if (variables.xBStop[b][k]) {
					OpenStation openStation = bOpenStations.next();
					bOpenStations = openStation;
					
					if (b != openStation.bus || k != openStation.stop ) {
						System.out.printf("The order in open stations is incorrect. k[regular]=%s, k[structure]=%s\n",
								k, openStation.stop);
						return false;
					}
					if (nextE != openStation.E) {
						System.out.printf("E variable is not consistent. E[%s, %s][regular]=%s, E[%s, %s][structure]=%s\n",
								b, k, nextE, openStation.bus, openStation.stop, openStation.E);
						return false;
					}
					if (stopIndex > 0) {
						if (nextS != openStation.s) {
							System.out.printf("s variable is not consistent. s[%s, %s][regular]=%s, s[%s, %s][structure]=%s\n",
									b, k, nextS, openStation.bus, openStation.stop, openStation.s);
							return false;
						}
					}
					
					stopIndex++;
					
					//bOpenStations.add(new OpenStation(nextE, nextS, k, b));
					nextE = 0;
					nextS = 100000;
					
					ListIterator<OpenStation> sequenceStops = getListIteratorOpenStopsPerStation(station);
					/*
					for (OpenStation op: )
			
					
					try {
						openStopsPerStation.get(station).add(new OpenStation(nextE, nextS, k, b));
					} catch (NullPointerException e) {
						LinkedList<OpenStation> sequenceStops = new LinkedList<OpenStation>();
						sequenceStops.add(new OpenStation(nextE, nextS, k, b));
						openStopsPerStation.put(station, sequenceStops);
					}
							*/
					
				}
			}
		}
		return isGood;
	}
	
	public int getNumOpenStations(int b) {
		return openStopsPerBus.get(b).size();
	}
	
	public int getNumBusesWithOpenStations() {
		int num = 0;
		for (HashMap<Integer, OpenStation> l: openStopsPerBus) {
			if (l.size() > 0) {
				num++;
			}
		}
		return num;
	}
	
	public int getNumOpenStopsPerStations(int i) {
		return openStopsPerStation.containsKey(i) ? openStopsPerStation.get(i).size() : 0;
	}
	
	/*
	public ListIterator<OpenStation> getListIteratorOpenStations(int b) {
		return openStations.get(b).listIterator();
	}
	*/
	
	public ListIterator<OpenStation> getListIteratorOpenStopsPerStation(int i) {
		return openStopsPerStation.get(i).listIterator();
	}
	
	public int getNumOpenStopsPerStationsFromMainStructure(int i) {
		int num = 0;
		for (HashMap<Integer, OpenStation> l: openStopsPerBus) {
			for (Integer osk: l.keySet()) {
				if (instance.paths[l.get(osk).bus][l.get(osk).stop] == i) {
					num++;
				}
			}
		}
		return num;
	}
	
	public void print() {
		int b = 0;
		for (HashMap<Integer, OpenStation> busSequence: openStopsPerBus) {
			System.out.println("Bus " + b + "--------------\n");
			for (Integer openS: busSequence.keySet()) {
				busSequence.get(openS).print();
				System.out.println();
			}
			b++;
			System.out.println();
		}
	}
	
	public void print(int b) {
		HashMap<Integer, OpenStation> busSequence = openStopsPerBus.get(b);
		System.out.println("Bus " + b + "--------------\n");
		for (Integer openS: busSequence.keySet()) {
			busSequence.get(openS).print();
			System.out.println();
		}
		System.out.println("Open stations: " + busSequence.size());
	}
	
	public void newPrint(int b) {
		HashMap<Integer, OpenStation> busSequence = openStopsPerBus.get(b);
		System.out.println("Bus " + b + "--------------\n");
		OpenStation head = getHeadInOpenStopPerBus(b);
		head.print();
		int i = 1;
		while (head.hasNext()) {
			head = head.next();
			head.print();
			System.out.println();
			i++;
		}
		System.out.println("Open stations: " + i);
	}
	
	public String stopsPerStationToString() {
		String output = "";
		for (int i = 0; i < instance.n; i++) {
			if (openStopsPerStation.get(i) != null && openStopsPerStation.size() > 0) {
				output += "Station " + i + " has " + openStopsPerStation.get(i).size() + " stops || ";
				for (OpenStation s: openStopsPerStation.get(i)) {
					output += String.format("- [%s][%s]", s.bus, s.stop);
				}
				output += "\n";
				//System.out.println(output);
			}
		}
		return output;
	}
	
	public int getNumberOpenStations() {
		int numOpenStations = 0;
		for (int b: openStopsPerStation.keySet()) {
			if (openStopsPerStation.get(b).size() > 0) {
				numOpenStations++;
			}
		}
		return numOpenStations;
	}
	
	public void removeStopFromOpenStopsPerStation(int b, int i, int k) {
		//System.out.println("Number of stations open stops: " +  openStopsPerStation.size());
		ListIterator<OpenStation> stopsIterator = openStopsPerStation.get(i).listIterator();
		while (stopsIterator.hasNext()) {
			OpenStation stop = stopsIterator.next();
			if (stop.bus == b && stop.stop == k ) {
				stopsIterator.remove();
				break;
			}
		}
	}
	
	public void removeStopFromOpenStopsPerBus(int b, int k) {
		//System.out.println("Number of stations open stops: " +  openStopsPerStation.size());
		openStopsPerBus.get(b).remove(k);
	}
	
	public boolean checkFeasibility() {
		boolean feasible = true;
		for (HashMap<Integer, OpenStation> lop : openStopsPerBus) {
			for (Integer op: lop.keySet()) {
				if (lop.get(op).E <= 0) {
					System.out.println("E is <= 0");
					lop.get(op).print();
					System.exit(1);
				}
			}
		}
		return feasible;
	}
	
	public int getNumberOpenStops1() {
		int numOpenStations = 0;
		for (HashMap<Integer, OpenStation> stopsPerBus: openStopsPerBus) {
				numOpenStations += stopsPerBus.size();
		}
		return numOpenStations;
	}
	
	public int getNumberOpenStops2() {
		int numOpenStations = 0;
		for (int b: openStopsPerStation.keySet()) {
				numOpenStations += openStopsPerStation.get(b).size();
		}
		return numOpenStations;
	}
	
	public OpenStation getHeadInOpenStopPerBus(int b) {
		int min = 10000;
		for (int k: openStopsPerBus.get(b).keySet()) {
			min = k < min ? k : min;
		}
		OpenStation head = openStopsPerBus.get(b).get(min);
		return head;		
	}
	
	/*
	public boolean checkLinkedListConsistencyOpenStopsPerBus() {
		boolean consistent = true;
		for (int bu = 0; bu < instance.b; bu++) {
			OpenStation head = getHeadInOpenStopPerBus(bu);
			OpenStation currentStation = head;
			int prevK = head.stop;
			while (currentStation.hasNext()) {
				if (currentStation.hasPrevious()) {
					consistent = prevK == currentStation.previous().stop;
					if (!consistent) {
						System.out.printf("Inconsistency %s!=%s\n", prevK, currentStation.previous().stop);
						System.exit(0);
						return false;
					}
				}
				prevK = currentStation.stop;
				currentStation = currentStation.next();
			}
		}
		return consistent;
	}
	*/
	
	public boolean checkForNegativeSlacks() {
		boolean are = false;
		for (int b = 0; b < instance.b; b++) {
			OpenStation stop = openStopsPerBus.get(b).get(headsIdPerBus.get(b));
			while (stop != null) {
				if (stop.s < 0) {
					return true;
				}
				stop = stop.next();
			}
		}
		return are;
	}
	

}
