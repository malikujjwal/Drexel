public abstract class Component {
	protected String type;

	protected Component(String type) {
		this.type = type;
	}

	public String getType() {
		return this.type;
	}

}