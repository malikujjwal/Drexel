import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.Socket;

public abstract class ObjectStreamer {
	protected ObjectInputStream reader;
	protected ObjectOutputStream writer;

	ObjectStreamer(Socket socket) {
		this.writer = createWriterFromSocket(socket);
		this.reader = createReaderFromSocket(socket);
	}

	private ObjectInputStream createReaderFromSocket(Socket socket) {
		try {
			return new ObjectInputStream(socket.getInputStream());
		} catch (Exception e) {
			System.out.println("Reader creation failed");
			return null;
		}
	}

	private ObjectOutputStream createWriterFromSocket(Socket socket) {
		try {
			return new ObjectOutputStream(socket.getOutputStream());
		} catch (IOException e) {
			System.out.println("Writer creation failed: " + e.getMessage());
			return null;
		}
	}
}
