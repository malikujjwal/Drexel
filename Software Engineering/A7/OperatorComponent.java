public abstract class OperatorComponent extends Component {
	private Component left;
	private Component right;

	public OperatorComponent(String type, Component left, Component right) {
		super(type);
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