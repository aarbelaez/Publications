package heuristics;

public class OpenStation {
	
	/**
	 * The energy needed to come here from the previous open station 
	 */
	public double E = 0;
	/**
	 * The max delay allowed between the previous open station and this (previous, this]
	 */
	public double s = 0;
	public int stop;
	public int bus;
	
	private OpenStation previousStation;
	private OpenStation nextStation;
	
	private OpenStation nextBackup;
	private OpenStation previousPrimary;
	
	public OpenStation(double e, double s, int stop, int b) {
		super();
		E = e;
		this.s = s;
		this.stop = stop;
		this.bus = b;
	}
	
	public void print() {
		System.out.printf("bus=%s\n", bus);
		System.out.printf("stop=%s\n", stop);
		System.out.printf("E=%s\n", E);
		System.out.printf("s=%s\n", s);
	}
	
	public boolean hasPrevious() {
		return previousStation != null;
	}
	
	public boolean hasNext() {
		return nextStation != null;
	}
	
	public boolean hasBackup() {
		return nextBackup != null;
	}
	
	public boolean hasPrimary() {
		return previousPrimary != null;
	}
	
	public void setNext(OpenStation os) {
		this.nextStation = os;
	}
	
	public void setPrevious(OpenStation os) {
		this.previousStation = os;
	}
	
	public void setNextBackup(OpenStation os) {
		this.nextBackup = os;
	}
	
	public void setPreviousPrimary(OpenStation os) {
		this.previousPrimary = os;
	}
	
	public void closeOpenStation() {
		if (hasPrevious()) {
			previousStation.setNext(nextStation);
		}
		if (hasNext()) {
			nextStation.setPrevious(previousStation);
		}
	}
	
	public void openClosedStation(OpenStation previousStation, OpenStation nextStation) {
		this.previousStation = previousStation;
		this.nextStation = nextStation;
		if (previousStation != null) {
			this.previousStation.setNext(this);
		}
		if (nextStation != null) {
			this.nextStation.setPrevious(this);
		}
	}
	
	public OpenStation previous() {
		return previousStation;
	}
	
	public OpenStation next() {
		return nextStation;
	}
	
	public OpenStation nextBackup() {
		return nextBackup;
	}
	
	public OpenStation previousPrimary() {
		return previousPrimary;
	}
	
	
	

}
