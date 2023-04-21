import java.net.Socket;

public abstract class Client {
	protected Socket clientSocket;
	protected SocketConfig socketConfig;
	protected IOHandler ioHandler;

	Client(SocketConfigParser socketConfigParser) {
		this.socketConfig = socketConfigParser;
		this.clientSocket = socketConfig.getClientSocket();
		this.ioHandler = new IOHandler(clientSocket);
	}

	static void log(String message) {
		System.out.println(message);
	}

	static void log(String clientName, String message) {
		log(clientName + " -> " + message);
	}

	public void closeConnectionToServer(Socket socket) {
		this.socketConfig.closeSocketConnection(socket);
	}

	public abstract void receiveServerResponse(Message serverResponse);

	public abstract void sendMessageToServer(Message message);
}
