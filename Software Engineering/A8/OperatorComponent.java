public abstract class OperatorComponent implements Component {
	private Component left;
	private Component right;

	public OperatorComponent(Component left, Component right) {
		this.left = left;
		this.right = right;
	}

	public Component getLeft() {
		return left;
	}

	public Component getRight() {
		return right;
	}
}