package variables;
import java.util.HashMap;

import ilog.concert.IloException;
import ilog.concert.IloIntVar;
import ilog.concert.IloNumVar;

public class BasicVariablesSet extends RouteVariablesSet {
	
	//BASIC VARIABLES
	public IloNumVar[][] arrivalTime;
	public IloNumVar[][] deviationTime;
	/**
	 * remaining battery 
	 */
	public IloNumVar[][] c;
	/**
	 * Energy added during charging
	 */
	public IloNumVar[][] e;
	/**
	 * charging time
	 */
	public IloNumVar[][] ct;
	/**
	 * xBStation[i][j] determines whether bus i charges at stop j 
	 */
	public IloIntVar[][] xBStop;
	
	public HashMap<String, HashMap<String, Double>[][]> initialSolution = null;
	
	@Override
	public void defineBasicVariablesArrays() throws IloException {
		//VARIABLES		
		arrivalTime =  new IloNumVar[instance.b][];
		deviationTime = new IloNumVar[instance.b][];
		// remaining battery 
		c = new IloNumVar[instance.b][];
		// Energy added during charging
		e = new IloNumVar[instance.b][];
		// charging time
		ct = new IloNumVar[instance.b][];
		// charging stop decision per bus 
		xBStop = new IloIntVar[instance.b][];
		
	}

	@Override
	public void defineBasicVariablesPerBus(int b) throws IloException {
		arrivalTime[b] = new IloNumVar[instance.paths[b].length]; //cplex.numVarArray(paths[bu].length, 0, Tmax);
		deviationTime[b] = cplex.numVarArray(instance.paths[b].length, 0, instance.DTmax);
		c[b] = cplex.numVarArray(instance.paths[b].length, instance.Cmin, instance.Cmax); // CONSTRAINT (1)
		e[b] = cplex.numVarArray(instance.paths[b].length, 0, instance.Cmax); 
		ct[b] = cplex.numVarArray(instance.paths[b].length, 0, instance.maxChargingTime); 
		xBStop[b] = cplex.boolVarArray(instance.paths[b].length); 
		arrivalTime[b][0] = cplex.numVar(Math.max(instance.originalTimetable[b][0] - instance.DTmax, 0),
				Math.min(instance.originalTimetable[b][0] + instance.DTmax, instance.Tmax));
		
	}

	@Override
	public void defineBasicVariablesPerBusPerStop(int b, int k) throws IloException {
		arrivalTime[b][k] = cplex.numVar(Math.max(instance.originalTimetable[b][k] - instance.DTmax, 0),
				Math.min(instance.originalTimetable[b][k] + instance.DTmax, instance.Tmax));
		
	}
	
public void setWarmStart() throws IloException {
		
		
	    int numStops = 0;
	    
	    for (int b = 0; b < instance.b; b++) {
			for (int k = 0; k < instance.paths[b].length; k++) {
				numStops++;
			}
	    }
	    
	    IloNumVar[] startVarXB = new IloNumVar[numStops];
	    double[] startValXB = new double[numStops];
	    IloNumVar[] startVarAT = new IloNumVar[numStops];
	    double[] startValAT = new double[numStops];
	    IloNumVar[] startVarC = new IloNumVar[numStops];
	    double[] startValC = new double[numStops];
	    IloNumVar[] startVarE = new IloNumVar[numStops];
	    double[] startValE = new double[numStops];
	    IloNumVar[] startVarCT = new IloNumVar[numStops];
	    double[] startValCT = new double[numStops];
	    
	    int m = 0;
	    for (int b = 0; b < instance.b; b++) {
			for (int k = 0; k < instance.paths[b].length; k++) {
				int stationId = instance.paths[b][k];
				HashMap<String, Double> varVals = initialSolution.get("")[b][k];
				startVarXB[m] = xBStop[b][k]; 
				startValXB[m] = varVals.get("xBStop");
				startVarAT[m] = arrivalTime[b][k]; 
				startValAT[m] = varVals.get("arrivalTime");
				startVarC[m] = c[b][k]; 
				startValC[m] = varVals.get("c");
				startVarE[m] = e[b][k]; 
				startValE[m] = varVals.get("e");
				startVarCT[m] = ct[b][k]; 
				startValCT[m] = varVals.get("ct");
				m++;
			}
	    }
	    
	    System.out.println("Warming start basic...");
		
		cplex.addMIPStart(startVarXB, startValAT);
		cplex.addMIPStart(startVarAT, startValXB);
		cplex.addMIPStart(startVarC, startValC);
		cplex.addMIPStart(startVarE, startValE);
		cplex.addMIPStart(startVarCT, startValCT);
		

		
	}




}
