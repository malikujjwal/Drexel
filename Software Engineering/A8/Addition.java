public class Addition extends OperatorComponent {
	public Addition(Component left, Component right) {
		super(left, right);
	}

	@Override
	public int accept(Visitor visitor) {
		return visitor.visit(this);
	}
}