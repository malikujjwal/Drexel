import java.io.Serializable;

public class Message implements Serializable {
	private final String line;
	private final boolean isProcessTerminated;

	Message(String line) {
		this.line = line;
		this.isProcessTerminated = false;
	}

	Message(boolean status) {
		this.isProcessTerminated = status;
		this.line = null;
	}

	public boolean isFinished() {
		return isProcessTerminated;
	}

	public String getString() {
		if (line == null) {
			throw new IllegalStateException("Method has been invoked at an illegal or inappropriate time.");
		}
		return line;
	}
}
