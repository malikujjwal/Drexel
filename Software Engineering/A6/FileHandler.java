import java.util.logging.Level;
import java.util.logging.Logger;

public class FileHandler {
	private static final Logger LOGGER = Logger.getLogger(Logger.GLOBAL_LOGGER_NAME);
	private String fileName;

	public String getFileName() {
		return fileName;
	}

	public void setFileName(String fileName) {
		this.fileName = fileName;
	}

	public void logError(Exception e) {
		LOGGER.log(Level.INFO, "Exception :: ", e);
	}
}