public class MasterControl {
	public static void main(String[] args) {
		MasterControl masterControl = new MasterControl();
		masterControl.start("mc.txt", "mc_output.txt");
	}

	public void start(String inputFileName, String outputFileName) {
		Pipe inputToShifter = new Pipe();
		Pipe shifterToSorter = new Pipe();
		Pipe sorterToOutput = new Pipe();

		Input readFile = new Input(null, inputToShifter, inputFileName);
		CircularShifter lineShifter = new CircularShifter(inputToShifter, shifterToSorter);
		Alphabetizer lineSorter = new Alphabetizer(shifterToSorter, sorterToOutput);
		Output writeFile = new Output(sorterToOutput, null, outputFileName);

		readFile.start();
		lineShifter.start();
		lineSorter.start();
		writeFile.start();
	}
}