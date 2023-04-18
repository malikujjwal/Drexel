public class MasterControl {
	public static void main(String[] args) {
		MasterControl masterControl = new MasterControl();
		masterControl.start(new PipeList(), new PipeQueue(), new PipeList(), "kwic.txt", "kwic_output.txt");
	}

	public void start(Pipe inputToShifter, Pipe shifterToSorter, Pipe sorterToOutput, String inputFileName,
			String outputFileName) {
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