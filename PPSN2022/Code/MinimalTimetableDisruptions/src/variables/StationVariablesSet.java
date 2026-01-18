package variables;

import java.util.HashMap;

import ilog.concert.IloException;
import ilog.concert.IloIntVar;
import ilog.concert.IloLinearNumExpr;
import ilog.concert.IloNumVar;

public class StationVariablesSet extends RouteVariablesSet {

	/**
	 * x[i] determines whether a charger is installed in station i
	 */
	public IloIntVar[] x;
	
	public HashMap<String, HashMap<String, Double>[][]> initialSolution = null;
	
	public boolean warmSetted = false;
	
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
	
	public void setWarmStart() throws IloException {
		
		if (!warmSetted) {
			HashMap<Integer, Double> iniStartVal = new HashMap<Integer, Double>();
			
			
		    int numOpened = 0;
		    
		    for (int b = 0; b < instance.b; b++) {
				for (int k = 1; k < instance.paths[b].length; k++) {
					int stationId = instance.paths[b][k];
					HashMap<String, Double> varVals = initialSolution.get("")[b][k];
					if (varVals.get("x") == 1) {
						iniStartVal.put(stationId, varVals.get("x"));
					}
					
				}
		    }
		    
		    
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
	
			
			cplex.addMIPStart(startVar, startVal);
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
