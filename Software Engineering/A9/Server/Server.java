import java.net.ServerSocket;
import java.net.Socket;

public abstract class Server {
	protected SocketConfig socketConfig;
	protected ServerSocket serverSocket;

	Server() {
		this.socketConfig = new SocketConfigParser();
		this.serverSocket = socketConfig.getServerSocket();
	}

	public abstract void acceptConnections();

	public abstract void startNewWorker(Socket socket);
}
