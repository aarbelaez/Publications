package utils;
/**
 * A collection of useful methods
 * @author cdloaiza
 *
 */

public class ToolsMTD {
	
	public static int stringTimeToMinutes(String strTime, boolean change0000) {
		String[] splitedTime = strTime.split(":");
		int minutesInHours = Integer.parseInt(splitedTime[0])*60;
		int minutesMinutes = Integer.parseInt(splitedTime[1]);
		int totalMinutes = minutesInHours + minutesMinutes;
		if (change0000 && strTime.equals("00:00")) {
			System.out.println("00:00 found!");	
			totalMinutes = 24 * 60;
		}
		return totalMinutes;
	}
	
	public static String minutesToStringTime(int minutes) {
		int hours = minutes / 60;
		int minutesLeft = minutes % 60;
		String stringTime = hours + ":" + minutesLeft;
		return stringTime;
	}
	
	
	/**
	 * Check if possibleStation is the path of busId
	 * @param busId
	 * @param possibleStation
	 * @return
	 */
	public static boolean stationInPath(int busId, int possibleStation, int[][] paths) {
		boolean found = false;
		for (int stationId : paths[busId]) {
			if (possibleStation == stationId) {
				found = true;
				break;
			}
		}
		return found;
	}
	
	public static double kmHourToMtSeconds(int kmHour) {
		int mtHour = kmHour * 1000;
		double mtSeconds = mtHour / 3600.0;
		return mtSeconds;
	}
	
	public static double round(double number) {
		//double rounded = Math.round(number * 100) / 100.0;
		double rounded = Math.round(number);
		return rounded;
	}

}
