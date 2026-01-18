package modelinterface;
import java.util.HashMap;
import java.util.LinkedHashMap;

import ilog.concert.IloException;
import ilog.cplex.IloCplex.UnknownObjectException;

public abstract class RouteModelExtension extends ModelExtension {
	
	protected String prefixVarname = "";
	
	public void addConstraintsPerBus(int b) throws IloException {};
	
	/**
	 * For a particular bus b and station k, it
	 * Builds the basic constraints which must exist per stop. From (2) to (6) and the added ones
	 * @param b
	 * @throws IloException
	 */
	public abstract void addConstraintsPerStop(int b, int k, int m, int i, int j) throws IloException;
	
	/**
	 * For a particular bus b and station i, it
	 * Builds the basic constraints which must exist per station. (7)
	 * @param b
	 * @throws IloException
	 */
	public abstract void addConstraintsPerStation(int b, int i) throws IloException;
	
	public void addConstraintsBetweenSeparatedStops(int b, int k, int m, int i, int j) throws IloException {};
	
	public HashMap<String, Double> getPrintableVarVals(int b, int k, int i) throws UnknownObjectException, IloException {
		//System.out.println("No printable var/vals in extension " + this.getClass());
		return new LinkedHashMap<String, Double>();
	}
	
	public void setPrefixVarname(String prefixVarname) {
		this.prefixVarname = prefixVarname;
	}

	public String getPrefixVarname() {
		return prefixVarname;
	}
	
	
	
	
	
	
	
}
