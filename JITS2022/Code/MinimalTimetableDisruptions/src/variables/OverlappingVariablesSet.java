package variables;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import ilog.concert.IloException;
import ilog.concert.IloIntVar;

public class OverlappingVariablesSet extends VariablesSet {
	
	public boolean breakBusSymmetries = true;
	
	//OVERLAPPING
	/**
	 * buses b and d are using the same charging station
	 */
	public Map<Integer, Map<Integer, Map<Integer, IloIntVar>>> Z;
	/**
	 * !zbdi = b arrive to station when d has left
	 */
	public Map<Integer, Map<Integer, Map<Integer, Map<Integer, IloIntVar>>>> zbd;
	/**
	 * !zbdi = d arrive to station when b has left
	 */
	public Map<Integer, Map<Integer, Map<Integer, Map<Integer, IloIntVar>>>> zdb;
	
	@Override
	public void defineVariables() throws IloException {
		// buses b and d are using the same charging station
		Z = new HashMap<Integer, Map<Integer, Map<Integer, IloIntVar>>>();
		zbd = new HashMap<Integer, Map<Integer, Map<Integer, Map<Integer, IloIntVar>>>>();
		zdb = new HashMap<Integer, Map<Integer, Map<Integer, Map<Integer, IloIntVar>>>>();
		for (int b = 0; b < instance.b; b++) {
			Z.put(b, new HashMap<Integer, Map<Integer, IloIntVar>>());
			zdb.put(b, new HashMap<Integer, Map<Integer, Map<Integer, IloIntVar>>>());
			zbd.put(b, new HashMap<Integer, Map<Integer, Map<Integer, IloIntVar>>>());
			for (int d = 0; d < (breakBusSymmetries ? b : instance.b); d++) {		
				
				if (b != d) {
					
				
					Z.get(b).put(d, new HashMap<Integer, IloIntVar>());
					for (int i = 0; i < instance.n; i++) {
						for (int j = i; j <= i; j++) {
							if (i == j && b != d) {
								Z.get(b).get(d).put(i, cplex.boolVar());
							}
						}
					}
					
					zbd.get(b).put(d, new HashMap<Integer, Map<Integer, IloIntVar>>());
					zdb.get(b).put(d, new HashMap<Integer, Map<Integer, IloIntVar>>());
					for (int k = 0; k < instance.paths[b].length; k++) {
						zbd.get(b).get(d).put(k, new HashMap<Integer, IloIntVar>());
						zdb.get(b).get(d).put(k, new HashMap<Integer, IloIntVar>());
						for (int m = 0; m < instance.paths[d].length; m++) {
							int i = instance.paths[b][k];
							int j = instance.paths[d][m];
							if (i == j  && 
									(Math.abs(instance.originalTimetable[b][k] - instance.originalTimetable[d][m]) <= 2*instance.DTmax)) {
								zbd.get(b).get(d).get(k).put(m, cplex.boolVar());
								zdb.get(b).get(d).get(k).put(m, cplex.boolVar());
							}
								
							
						}
					}
				 	
					
				}
				
				
			}
		}
	}

}
