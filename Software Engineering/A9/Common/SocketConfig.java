import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;

public abstract class SocketConfig {
	private String hostName;
	private int port;

	SocketConfig(String host, int port) {
		this.hostName = host;
		this.port = port;
	}

	public ServerSocket getServerSocket() {
		try {
			return new ServerSocket(this.port);
		} catch (IOException e) {
			System.out.println("Error creating client socket: " + e.getMessage());
		}
		return null;
	}

	public Socket getClientSocket() {
		try {
			return new Socket(this.hostName, this.port);
		} catch (IOException e) {
			System.out.println("Error creating client socket: " + e.getMessage());
		}
		return null;
	}

	public void closeSocketConnection(Socket socket) {
		try {
			socket.close();
		} catch (IOException e) {
			System.out.println("Error while closing socket: " + e.getMessage());
		}
	}
}
