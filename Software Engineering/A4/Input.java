import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class Input extends Colleague {
	private FileHandler fileHandler;

	Input(Mediator mediator) {
		super(mediator);
		fileHandler = new FileHandler();
		fileHandler.setFileName("kwic.txt");
	}

	public void read() {
		try {
			readFileFromName();
			mediator.inputIsFinished();
		} catch (IOException e) {
			fileHandler.logError(e);
		}
	}

	private void readFileFromName() throws IOException {
		String fileName = fileHandler.getFileName();
		BufferedReader reader = Files.newBufferedReader(Paths.get(fileName));
		String line;
		while ((line = reader.readLine()) != null) {
			mediator.inputProvidesNextLine(line);
		}
		reader.close();
	}
}