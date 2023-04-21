public class ClientHandler extends Client {
	private String clientName;
	private boolean isClientWorking;

	ClientHandler(String clientName) {
		super(new SocketConfigParser());
		this.clientName = clientName;
		this.isClientWorking = true;
	}

	@Override
	public void receiveServerResponse(Message message) {
		if (message.isFinished()) {
			closeConnectionToServer(clientSocket);
			Client.log(clientName, "Killing client");
			this.isClientWorking = false;
		} else {
			Client.log(clientName, "Server Response: " + message.getString());
		}
	}

	@Override
	public void sendMessageToServer(Message message) {
		ioHandler.sendMessage(message);
		if (message.isFinished()) {
			Client.log(clientName, "waiting for server response....");
			while (this.isClientWorking) {
				this.receiveServerResponse(ioHandler.receiveMessage());
			}
		} else {
			Client.log(clientName, "message sent to server");
		}
	}
}
