package variables;

import ilog.concert.IloException;
import ilog.concert.IloIntVar;

public class StationChargeVariablesSet extends RouteVariablesSet {

	
	/**
	 * xBStation[i][j] determines whether bus i charges at station j 
	 */
	public IloIntVar[][] xBStation;	

	@Override
	public void defineBasicVariablesArrays() throws IloException {
		// charging station decision per bus 
		xBStation = new IloIntVar[instance.b][];
		for (int i = 0; i < instance.b; i++) {
			xBStation[i] = cplex.boolVarArray(instance.n); 
		}
		
	}

	@Override
	public void defineBasicVariablesPerBus(int b) throws IloException {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void defineBasicVariablesPerBusPerStop(int b, int k) throws IloException {
		// TODO Auto-generated method stub
		
	}

}
