package variables;
import ilog.concert.IloException;
import ilog.concert.IloNumVar;

public class StartingChargeTimeVariablesSet extends RouteVariablesSet {

	//Starting Charge Time (The time in which the bus starts to charge)
	public IloNumVar[][] st;
	
	@Override
	public void defineBasicVariablesArrays() throws IloException {
		st = new IloNumVar[instance.b][];
	}

	@Override
	public void defineBasicVariablesPerBus(int b) throws IloException {
		st[b] = cplex.numVarArray(instance.paths[b].length, 0, instance.Tmax);
	}

	@Override
	public void defineBasicVariablesPerBusPerStop(int b, int k) throws IloException {
		// TODO Auto-generated method stub

	}

}
