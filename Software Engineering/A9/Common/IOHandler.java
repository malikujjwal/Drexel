import java.io.IOException;
import java.net.Socket;

public class IOHandler extends ObjectStreamer {

	IOHandler(Socket socket) {
		super(socket);
	}

	public void sendMessage(Message message) {
		try {
			this.writer.writeObject(message);
		} catch (IOException e) {
			System.out.println("Could not send message");
		}
	}

	public Message receiveMessage() {
		try {
			return (Message) this.reader.readObject();
		} catch (IOException | ClassNotFoundException e) {
			System.out.println("Could not read message");
		}
		return null;
	}
}
