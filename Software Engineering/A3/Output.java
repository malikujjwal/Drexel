import java.io.IOException;
import java.io.PrintWriter;

public class Output implements Observer {
	private FileHandler fileHandler;
	private PrintWriter printWriter;

	Output() {
		fileHandler = new FileHandler();
		fileHandler.setFileName("kwic_output.txt");
		createFileWriter();
	}

	@Override
	public void update(Message message) {
		try {
			if (!message.isFinished()) {
				writeToFile(message.getString());
			} else {
				closeFile();
			}
		} catch (IOException e) {
			fileHandler.logError(e);
		}
	}

	private void writeToFile(String line) throws IOException {
		printWriter.println(line);
	}

	private void createFileWriter() {
		try {
			printWriter = new PrintWriter(fileHandler.getFileName());
		} catch (IOException e) {
			fileHandler.logError(e);
		}
	}

	private void closeFile() throws IOException {
		printWriter.close();
	}
}