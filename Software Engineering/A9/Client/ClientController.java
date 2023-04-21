import java.util.ArrayList;

public class ClientController {
	private final String clientName;
	ArrayList<String> lines;

	ClientController(String clientName) {
		this.clientName = clientName;
		lines = new ArrayList<>();
	}

	private void sendLinesToHandler() {
		ClientHandler client = new ClientHandler(clientName);
		Client.log("Client created: " + clientName);
		for (String s : this.lines) {
			client.sendMessageToServer(new Message(s));
		}
		client.sendMessageToServer(new Message(true));
	}

	public void startClient() {
		new Thread(this::sendLinesToHandler).start();
	}

	public void addLines(ArrayList<String> lines) {
		this.lines.addAll(lines);
	}
}
