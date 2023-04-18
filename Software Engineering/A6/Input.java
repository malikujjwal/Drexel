import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class Input extends Filter {
	private FileHandler fileHandler;
	private BufferedReader reader;

	Input(Pipe inPipe, Pipe outPipe, String fileName) {
		super(inPipe, outPipe);
		fileHandler = new FileHandler();
		fileHandler.setFileName(fileName);
		createFileReader();
	}

	@Override
	void filter() {
		try {
			readLineFromFile();
		} catch (IOException e) {
			fileHandler.logError(e);
		}
	}

	private void readLineFromFile() throws IOException {
		String line;
		if ((line = reader.readLine()) != null) {
			outPipe.write(line);
		} else {
			outPipe.close();
			reader.close();
			stop();
		}
	}

	private void createFileReader() {
		try {
			String fileName = fileHandler.getFileName();
			reader = Files.newBufferedReader(Paths.get(fileName));
		} catch (IOException e) {
			fileHandler.logError(e);
		}
	}
}