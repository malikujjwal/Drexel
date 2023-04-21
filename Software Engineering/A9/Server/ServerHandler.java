import java.net.Socket;

public class ServerHandler extends Server {

	@Override
	public void acceptConnections() {
		try {
			while (true) {
				this.startNewWorker(serverSocket.accept());
			}
		} catch (Exception e) {
			System.out.println("Error while making a connection with client");
		}
	}

	@Override
	public void startNewWorker(Socket socket) {
		Worker worker = new Worker(new IOHandler(socket));
		Thread startNewClient = new Thread(worker);
		startNewClient.start();
	}
}
