package modelinterface;
import ilog.concert.IloException;

public abstract class OverlappingModelExtension extends ModelExtension {
	
	
	protected boolean breakBusSymmetries = true;
	
	/**
	 * For a pair of buses bu and d, it
	 * Builds the overlapping constraints which must exist per stop. 
	 * @param bu
	 * @throws IloException
	 */
	public abstract void addConstraintsPerStop(int b, int d, int k, int m, int i, int j) throws IloException;
	/**
	 * For a pair of buses bu and d, it
	 * Builds the overlapping constraints which must exist per station. 
	 * @param bu
	 * @throws IloException
	 */
	public abstract void addConstraintsPerStation(int b, int d, int i, int j) throws IloException;

	public boolean getBreakBusSymmetries() {
		return breakBusSymmetries;
	}
}
