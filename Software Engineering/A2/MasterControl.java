import java.util.List;

public class MasterControl {
	public static void main(String[] args) {
		MasterControl masterController = new MasterControl();
		masterController.start();
	}

	public void start() {
		Input readFile = new Input();
		CircularShifter lineShifter = new CircularShifter();
		Alphabetizer lineSorter = new Alphabetizer();
		Output writeFile = new Output();

		readFile.attach(lineShifter);
		lineShifter.attach(lineSorter);

		readFile.read();
		List<String> sortedLines = lineSorter.getAlphabetizedLines();
		writeFile.write(sortedLines);
	}
}