import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class Input extends Subject {
	private FileHandler fileHandler;
	private BufferedReader reader;

	Input() {
		fileHandler = new FileHandler();
		fileHandler.setFileName("kwic.txt");
	}

	public void read() {
		try {
			createFileReader();
			readFileFromName();
			notifyAllObservers(new Message(true));
		} catch (IOException e) {
			fileHandler.logError(e);
		}
	}

	private void readFileFromName() throws IOException {
		String line;
		while ((line = reader.readLine()) != null) {
			notifyAllObservers(new Message(line));
		}
		reader.close();
	}

	private void createFileReader() throws IOException {
		String fileName = fileHandler.getFileName();
		reader = Files.newBufferedReader(Paths.get(fileName));
	}
}