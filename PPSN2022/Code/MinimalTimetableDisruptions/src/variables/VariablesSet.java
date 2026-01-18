package variables;
import core.InstanceMTD;
import ilog.concert.IloException;
import ilog.cplex.IloCplex;

public abstract class VariablesSet {
	
	protected InstanceMTD instance;
	protected IloCplex cplex;
	protected boolean areVariablesDefined = false;
	
	public abstract void defineVariables() throws IloException;

	public void setCplex(IloCplex cplex) {
		this.cplex = cplex;
	}

	public InstanceMTD getInstance() {
		return instance;
	}

	public void setInstance(InstanceMTD instance) {
		this.instance = instance;
	}

	public void setAreVariablesDefined(boolean areVariablesDefined) {
		this.areVariablesDefined = areVariablesDefined;
	} 
	
	
	

}
