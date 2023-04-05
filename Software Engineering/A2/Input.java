import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class Input extends Subject {
	private FileHandler fileHandler;

	Input() {
		fileHandler = new FileHandler();
		fileHandler.setFileName("kwic.txt");
	}

	public void read() {
		try {
			readFileFromName();
		} catch (IOException e) {
			fileHandler.logError(e);
		}
	}

	private void readFileFromName() throws IOException {
		String fileName = fileHandler.getFileName();
		BufferedReader reader = Files.newBufferedReader(Paths.get(fileName));
		String line;
		while ((line = reader.readLine()) != null) {
			notifyAllObservers(line);
		}
		reader.close();
	}
}