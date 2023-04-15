import java.util.List;

public class MediatorMasterControl implements Mediator {
	private Input readFile;
	private CircularShifter lineShifter;
	private Alphabetizer lineSorter;
	private Output writeFile;

	public MediatorMasterControl() {
		readFile = new Input(this);
		lineShifter = new CircularShifter(this);
		lineSorter = new Alphabetizer();
		writeFile = new Output();
	}

	public void start() {
		readFile.read();
	}

	@Override
	public void inputProvidesNextLine(String line) {
		lineShifter.shiftLine(line);
	}

	@Override
	public void inputIsFinished() {
		List<String> lines = lineSorter.getAlphabetizedLines();
		writeFile.write(lines);
	}

	@Override
	public void shiftedLine(String line) {
		lineSorter.addToAlphabetizedSet(line);
	}
}
