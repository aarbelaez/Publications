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
	
	

}
