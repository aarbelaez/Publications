package modelinterface;
import java.util.HashMap;

import core.InstanceMTD;
import ilog.concert.IloException;
import ilog.concert.IloLinearNumExpr;
import ilog.cplex.IloCplex;

public abstract class ModelExtension {
	
	protected IloCplex cplex;
	protected InstanceMTD instance;
	
	
	public void addConstraintsGlobally() throws IloException {};

	public void addObjective(IloLinearNumExpr objective) throws IloException {
		System.out.println("No objective in extension " + this.getClass());
	}

	public void setCplex(IloCplex cplex) {
		this.cplex = cplex;
	}
	
	public void setInstance(InstanceMTD instance) {
		this.instance = instance;
	}
	
	public void defineVariables(IloCplex cplex) throws IloException {
	};
	
	public void setWarmStarts() throws IloException {
	};
	

}
