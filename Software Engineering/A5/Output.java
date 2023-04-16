import java.io.IOException;
import java.io.PrintWriter;

public class Output extends Filter {
	private FileHandler fileHandler;
	private PrintWriter printWriter;

	Output(Pipe inPipe, Pipe outPipe, String fileName) {
		super(inPipe, outPipe);
		fileHandler = new FileHandler();
		fileHandler.setFileName(fileName);
		createFileWriter();
	}

	@Override
	void filter() {
		if (inPipe.isNotEmptyOrIsNotClosed()) {
			if (inPipe.hasNext()) {
				writeToFile(inPipe.read());
			}
		} else {
			closeFile();
			stop();
		}
	}

	private void writeToFile(String line) {
		try {
			printWriter.println(line);
		} catch (Exception e) {
			fileHandler.logError(e);
		}
	}

	private void createFileWriter() {
		try {
			printWriter = new PrintWriter(fileHandler.getFileName());
		} catch (IOException e) {
			fileHandler.logError(e);
		}
	}

	private void closeFile() {
		printWriter.close();
	}
}