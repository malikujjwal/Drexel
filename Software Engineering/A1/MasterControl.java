import java.util.List;

public class MasterControl {
	public static void main(String[] args) {
		MasterControl masterController = new MasterControl();
		masterController.start();
	}

	public void start() {
		Input readFile = new Input();
		List<String> lines = readFile.read();

		CircularShifter lineShifter = new CircularShifter();
		List<String> shiftedLines = lineShifter.shiftLines(lines);

		Alphabetizer lineSorter = new Alphabetizer();
		List<String> sortedLines = lineSorter.sort(shiftedLines);

		Output writeFile = new Output();
		writeFile.write(sortedLines);
	}
}