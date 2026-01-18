package variables;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

import ilog.concert.IloException;
import ilog.concert.IloIntVar;
import ilog.concert.IloLinearNumExpr;
import ilog.concert.IloNumVar;
import utils.OutputForWarm;

public class StationVariablesSet extends RouteVariablesSet {

	/**
	 * x[i] determines whether a charger is installed in station i
	 */
	public IloIntVar[] x;
	
	public OutputForWarm initialSolution = null;
	
	public boolean warmSetted = false;
	
	public boolean mustFix = false;
	
	@Override
	public void defineBasicVariablesArrays() throws IloException {
		// charging station decision 
		x = cplex.boolVarArray(instance.n);

	}

	@Override
	public void defineBasicVariablesPerBus(int b) throws IloException {
		// TODO Auto-generated method stub

	}

	@Override
	public void defineBasicVariablesPerBusPerStop(int b, int k) throws IloException {
		// TODO Auto-generated method stub

	}
	
	/**
	 * Invoked by InstallChargerRouteModelExtension
	 * @throws IloException
	 */
	public void setWarmStart() throws IloException {
		
		if (!warmSetted) {
			
			//HashMap<Integer, Double> iniStartVal = initialSolution;
			HashMap<Integer, Double> iniStartVal = new HashMap<Integer, Double>();
			
		    int numOpened = 0;
		    
		    //This is when we are getting the routeOutputs as initial solution
		    int numStops = 0;
		    
		    for (int b = 0; b < instance.b; b++) {
				for (int k = 0; k < instance.paths[b].length; k++) {
					numStops++;
				}
		    }

		    IloNumVar[] startVarAT = new IloNumVar[numStops];
		    double[] startValAT = new double[numStops];
		    IloNumVar[] startVarCT = new IloNumVar[numStops];
		    double[] startValCT = new double[numStops];
		    
		    
		    /*
		    for (int b = 0; b < instance.b; b++) {
				for (int k = 1; k < instance.paths[b].length; k++) {
					int stationId = instance.paths[b][k];
					if (initialSolution.get("")[b] != null) {
						HashMap<String, Double> varVals = initialSolution.get("")[b][k];
						boolean openPrimary = varVals.get("x") == 1;
						boolean openBackup = false;
						try {
							HashMap<String, Double> bVarVals = initialSolution.get("backup")[b][k];
							openBackup = bVarVals.get("x") == 1;
						} catch (NullPointerException e) {
							//System.out.println("Exception. No backup!");
						}
						if (openPrimary || openBackup) {
							iniStartVal.put(stationId, 1.0);
						}
					}
				}
		    }
		    */
		    
		    for (int station: initialSolution.getOpenStations()) {
		    	iniStartVal.put(station, 1.0);
		    }
		    
		    
		    
		    HashSet<Integer> realStations = new HashSet<Integer>();
		    for (int b = 0; b < instance.b; b++) {
		    	for (int k = 1; k < instance.paths[b].length; k++) {
		    		realStations.add(instance.paths[b][k]);
		    	}
		    }
		    
		    
		    iniStartVal.entrySet().removeIf(station -> !realStations.contains(station.getKey()));
		    
		    
		    IloNumVar[] startVar = new IloNumVar[iniStartVal.size()];
		    double[] startVal = new double[iniStartVal.size()];
		    	    
		    int i = 0;
		    for (Integer stationId: iniStartVal.keySet()) {
		    	startVar[i] = x[stationId];
		    	startVal[i] = iniStartVal.get(stationId);
		    	numOpened += startVal[i];
		    	i++;
		    }
			  
		    System.out.println("Warming start with " + numOpened + " chargers");
		    System.out.println("Printing them...");
		    for (Integer stationId: iniStartVal.keySet()) {
		    	System.out.println(stationId);
		    }
			
			cplex.addMIPStart(startVar, startVal);
			
			
			if (mustFix) {
				int fixingChargers = 6;
				int j = 0;
				System.out.println("Fixing " + numOpened + " chargers");
				System.out.println("Printing them...");
				for (int stationId : iniStartVal.keySet()) {
					x[stationId].setLB(1);
					j++;
					//if (j == 6) {
						//break;
					//}
				}
			}
			
			
		}
		
		warmSetted = true;
		
	}
	/*
	public void setWarmStart() throws IloException {
		if (!warmSetted) {
		    int[] vals = new int[instance.n];
		    vals[84] = 1;
		    vals[241] = 1;
		    vals[539] = 1;
		    
		    IloNumVar[] startVar = new IloNumVar[instance.n];
		    double[] startVal = new double[instance.n];
		    	  
		    int numOpened = 0;
		    for (int i = 0; i < instance.n; i++) {
		    	startVar[i] = x[i];
		    	startVal[i] = vals[i];
		    	numOpened += vals[i];
		    }
			  
		    System.out.println("Warming start with " + numOpened + " chargers");
		    IloLinearNumExpr warmStartConstraint = cplex.linearNumExpr();
		    for (int i = 0; i < instance.n; i++) {
		    	warmStartConstraint.addTerm(1.0, x[i]);
		    }
			cplex.addLe(warmStartConstraint, 6); 
	
			
			cplex.addMIPStart(startVar, startVal);
		
			warmSetted = true;
		}
	}
	*/
	
}
