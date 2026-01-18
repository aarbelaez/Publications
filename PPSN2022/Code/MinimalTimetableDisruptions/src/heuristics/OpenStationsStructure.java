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
	public ArrayList<LinkedList<OpenStation>> openStations;
	public Map<Integer,LinkedList<OpenStation>> openStopsPerStation;
	
	public OpenStationsStructure(HeuristicsVariablesSet variables, InstanceMTD instance) {
		this.variables = variables;
		this.instance = instance;
		openStations = new ArrayList<LinkedList<OpenStation>>();
		openStopsPerStation = new HashMap<Integer, LinkedList<OpenStation>>(instance.n);
		build();
	}
	
	public void build() {
		for (int b = 0; b < instance.b; b++) {
			LinkedList<OpenStation> bOpenStations = new LinkedList<OpenStation>();
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
					OpenStation openStation = new OpenStation(nextE, nextS, k, b);
					bOpenStations.add(openStation);
					try {
						openStopsPerStation.get(station).add(openStation);
					} catch (NullPointerException e) {
						LinkedList<OpenStation> sequenceStops = new LinkedList<OpenStation>();
						sequenceStops.add(openStation);
						openStopsPerStation.put(station, sequenceStops);
					}
					nextE = 0;
					nextS = 100000;
					
				}
			}
			openStations.add(bOpenStations);
		}
	}
	
	public boolean checkIntegrety() {
		boolean isGood = true;
		for (int b = 0; b < instance.b; b++) {
			int stopIndex = 0;
			ListIterator<OpenStation> bOpenStations = getListIteratorOpenStations(b);
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
		return openStations.get(b).size();
	}
	
	public int getNumBusesWithOpenStations() {
		int num = 0;
		for (LinkedList<OpenStation> l: openStations) {
			if (l.size() > 0) {
				num++;
			}
		}
		return num;
	}
	
	public int getNumOpenStopsPerStations(int i) {
		return openStopsPerStation.containsKey(i) ? openStopsPerStation.get(i).size() : 0;
	}
	
	public ListIterator<OpenStation> getListIteratorOpenStations(int b) {
		return openStations.get(b).listIterator();
	}
	
	public ListIterator<OpenStation> getListIteratorOpenStopsPerStation(int i) {
		return openStopsPerStation.get(i).listIterator();
	}
	
	public int getNumOpenStopsPerStationsFromMainStructure(int i) {
		int num = 0;
		for (LinkedList<OpenStation> l: openStations) {
			for (OpenStation os: l) {
				if (instance.paths[os.bus][os.stop] == i) {
					num++;
				}
			}
		}
		return num;
	}
	
	public void print() {
		int b = 0;
		for (LinkedList<OpenStation> busSequence: openStations) {
			System.out.println("Bus " + b + "--------------\n");
			for (OpenStation openS: busSequence) {
				openS.print();
				System.out.println();
			}
			b++;
			System.out.println();
		}
	}
	
	public void print(int b) {
		LinkedList<OpenStation> busSequence = openStations.get(b);
		System.out.println("Bus " + b + "--------------\n");
		for (OpenStation openS: busSequence) {
			openS.print();
			System.out.println();
		}
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
	
	public boolean checkFeasibility() {
		boolean feasible = true;
		for (LinkedList<OpenStation> lop : openStations) {
			for (OpenStation op: lop) {
				if (op.E <= 0) {
					System.out.println("E is <= 0");
					op.print();
					System.exit(1);
				}
			}
		}
		return feasible;
	}
	
	public int getNumberOpenStops1() {
		int numOpenStations = 0;
		for (LinkedList<OpenStation> stopsPerBus: openStations) {
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

}
